from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from semble import (
    BaselineIdentity,
    BaselineRegistry,
    ChangeKind,
    GitChangeClassifier,
    SearchScope,
    fetch_remote_baseline,
    open_git_workspace,
    resolve_revision,
    resolve_workspace_binding,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _init_repository(root: Path) -> str:
    subprocess.run(["git", "init", "-b", "main", str(root)], check=True, capture_output=True)
    _git(root, "config", "user.name", "Semble Test")
    _git(root, "config", "user.email", "semble@example.invalid")
    (root / "auth.py").write_text("def authenticate():\n    return 'base auth'\n")
    (root / "database.py").write_text("def database():\n    return 'base database'\n")
    (root / "worker.py").write_text(
        "def worker():\n    return 'base worker with enough content for rename detection'\n"
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base")
    return _git(root, "rev-parse", "HEAD")


def test_git_classifier_covers_committed_worktree_and_untracked_delta(tmp_path: Path) -> None:
    """Git classification spans committed, deleted, renamed, untracked, and reverted paths."""
    repo = tmp_path / "repo"
    base = _init_repository(repo)
    classifier = GitChangeClassifier(repo, base)

    (repo / "auth.py").write_text("def authenticate():\n    return 'committed branch auth'\n")
    _git(repo, "add", "auth.py")
    _git(repo, "commit", "-m", "branch change")
    (repo / "database.py").unlink()
    (repo / "worker.py").rename(repo / "renamed_worker.py")
    (repo / "new_feature.py").write_text("def new_feature():\n    return 'untracked addition'\n")

    changes = classifier.all_changes()
    by_path = {change.path: change for change in changes}
    assert by_path["auth.py"].kind is ChangeKind.MODIFIED
    assert by_path["database.py"].kind is ChangeKind.DELETED
    assert by_path["renamed_worker.py"].kind is ChangeKind.RENAMED
    assert by_path["renamed_worker.py"].previous_path == "worker.py"
    assert by_path["new_feature.py"].kind is ChangeKind.ADDED

    (repo / "auth.py").write_text("def authenticate():\n    return 'base auth'\n")
    classified = classifier.classify({repo / "auth.py", repo / "database.py"})
    classified_by_path = {change.path: change for change in classified}
    assert classified_by_path["auth.py"].kind is ChangeKind.UNCHANGED
    assert classified_by_path["database.py"].kind is ChangeKind.DELETED


def test_fetch_remote_baseline_pins_remote_commit_without_moving_checkout(tmp_path: Path) -> None:
    """Fetching pins the new remote commit without moving the caller's checkout."""
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(remote)], check=True, capture_output=True)
    first = _init_repository(seed)
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "main")
    subprocess.run(["git", "clone", str(remote), str(checkout)], check=True, capture_output=True)
    checkout_head = _git(checkout, "rev-parse", "HEAD")
    assert checkout_head == first

    (seed / "auth.py").write_text("def authenticate():\n    return 'new remote main'\n")
    _git(seed, "add", "auth.py")
    _git(seed, "commit", "-m", "advance remote")
    _git(seed, "push", "origin", "main")
    remote_head = _git(seed, "rev-parse", "HEAD")

    identity = fetch_remote_baseline(checkout, remote="origin", branch="main")
    assert identity.repository == str(remote)
    assert identity.revision == remote_head
    assert resolve_revision(checkout, "origin/main") == remote_head
    assert _git(checkout, "rev-parse", "HEAD") == checkout_head


def test_resolve_workspace_binding_uses_clean_primary_checkout(tmp_path: Path) -> None:
    """A Paseo-style linked worktree reuses its clean immutable primary checkout."""
    primary = tmp_path / "primary"
    worktree = tmp_path / "worktree"
    revision = _init_repository(primary)
    _git(primary, "remote", "add", "origin", "https://github.com/example/project.git")
    _git(primary, "worktree", "add", "-b", "agent-task", str(worktree), revision)

    binding = resolve_workspace_binding(worktree, tmp_path / "cache")

    assert binding.workspace_root == worktree.resolve()
    assert binding.baseline_root == primary.resolve()
    assert binding.identity == BaselineIdentity("https://github.com/example/project.git", revision)
    assert binding.branch == "agent-task"
    assert binding.baseline_source == "primary_checkout"


def test_resolve_workspace_binding_materializes_exact_nonprimary_revision(tmp_path: Path) -> None:
    """A branch commit gets one reusable clean baseline without mutating the worktree."""
    primary = tmp_path / "primary"
    worktree = tmp_path / "worktree"
    base = _init_repository(primary)
    _git(primary, "remote", "add", "origin", "https://github.com/example/project.git")
    _git(primary, "worktree", "add", "-b", "agent-task", str(worktree), base)
    (worktree / "auth.py").write_text("def authenticate():\n    return 'committed branch state'\n")
    _git(worktree, "add", "auth.py")
    _git(worktree, "commit", "-m", "branch baseline")
    revision = _git(worktree, "rev-parse", "HEAD")

    first = resolve_workspace_binding(worktree, tmp_path / "cache")
    second = resolve_workspace_binding(worktree, tmp_path / "cache")

    assert first == second
    assert first.baseline_source == "materialized_revision"
    assert first.baseline_root not in {primary.resolve(), worktree.resolve()}
    assert _git(first.baseline_root, "rev-parse", "HEAD") == revision
    assert _git(first.baseline_root, "status", "--porcelain") == ""
    assert _git(worktree, "rev-parse", "HEAD") == revision


@pytest.mark.anyio
async def test_open_git_workspace_requires_clean_base_and_indexes_existing_delta(tmp_path: Path, mock_model) -> None:
    """A session verifies its base, attaches it once, and indexes the current Git delta before readiness."""
    baseline = tmp_path / "baseline"
    worktree = tmp_path / "worktree"
    revision = _init_repository(baseline)
    subprocess.run(["git", "clone", str(baseline), str(worktree)], check=True, capture_output=True)
    (worktree / "auth.py").write_text("def authenticate():\n    return 'workspace violet marker'\n")
    (worktree / "new.py").write_text("def new_path():\n    return 'workspace addition'\n")
    identity = BaselineIdentity("example/repository", revision)
    registry = BaselineRegistry()

    with (
        patch("semble.index.index.load_model", return_value=(mock_model, "test-model")),
        patch("semble.cache.save_index_to_cache") as mock_save,
    ):
        session = open_git_workspace(
            registry,
            identity,
            baseline_root=baseline,
            workspace_root=worktree,
            model_path="test-model",
        )
    mock_save.assert_called_once()
    assert session.classifier.base_revision == revision
    assert session.index.changed_paths == {
        "auth.py": ChangeKind.MODIFIED,
        "new.py": ChangeKind.ADDED,
    }
    changed = session.index.search("workspace violet marker", scope=SearchScope.CHANGED)
    assert changed.delta_results[0].result.chunk.file_path == "auth.py"
    (worktree / "auth.py").write_text("def authenticate():\n    return 'immediate synchronized marker'\n")
    await session.synchronize()
    synchronized = session.index.search("immediate synchronized marker", scope=SearchScope.CHANGED)
    assert synchronized.delta_results[0].result.chunk.file_path == "auth.py"
    assert registry.references(identity) == 1
    await session.close()
    assert registry.references(identity) == 0


@pytest.mark.parametrize("failure", ["wrong_revision", "dirty_baseline"])
def test_open_git_workspace_rejects_nonimmutable_baseline(
    failure: str,
    tmp_path: Path,
    mock_model,
) -> None:
    """A baseline must be clean and exactly match the identity's immutable commit."""
    baseline = tmp_path / "baseline"
    worktree = tmp_path / "worktree"
    revision = _init_repository(baseline)
    shutil.copytree(baseline, worktree)
    identity = BaselineIdentity("example/repository", "f" * 40 if failure == "wrong_revision" else revision)
    if failure == "dirty_baseline":
        (baseline / "auth.py").write_text("def authenticate():\n    return 'dirty baseline'\n")
    with (
        patch("semble.index.index.load_model", return_value=(mock_model, "test-model")),
        pytest.raises(ValueError, match="Baseline checkout"),
    ):
        open_git_workspace(
            BaselineRegistry(),
            identity,
            baseline_root=baseline,
            workspace_root=worktree,
            model_path="test-model",
        )


@pytest.mark.parametrize(
    ("remote", "branch"),
    [
        ("--upload-pack=malicious", "main"),
        ("origin", "--force"),
        ("", "main"),
        ("origin", ""),
    ],
)
def test_fetch_remote_baseline_rejects_option_like_or_empty_names(
    remote: str,
    branch: str,
    tmp_path: Path,
) -> None:
    """Remote and branch values cannot be interpreted as Git command options."""
    with pytest.raises(ValueError, match="cannot start"):
        fetch_remote_baseline(tmp_path, remote=remote, branch=branch)
