from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from watchfiles import Change

from semble import (
    BaselineIdentity,
    BaselineRegistry,
    ChangeKind,
    SearchOrigin,
    SearchScope,
    SembleIndex,
    WorkspaceFileChange,
    WorkspaceIndex,
    WorkspaceWatcher,
)
from semble.index.dense import embed_chunks as real_embed_chunks


def _write_project(root: Path) -> None:
    root.mkdir()
    (root / "auth.py").write_text("def baseline_authentication():\n    return 'legacy amber token'\n")
    (root / "database.py").write_text("def database_checkpoint():\n    return 'durable cobalt record'\n")
    (root / "worker.py").write_text("def background_worker():\n    return 'silver queue marker'\n")


def _baseline(root: Path, mock_model) -> SembleIndex:
    with patch("semble.index.index.load_model", return_value=(mock_model, "test-model")):
        return SembleIndex.from_path(root, model_path="test-model")


def _workspace(tmp_path: Path, mock_model):
    base = tmp_path / "base"
    worktree = tmp_path / "worktree"
    _write_project(base)
    shutil.copytree(base, worktree)
    registry = BaselineRegistry()
    identity = BaselineIdentity("example/repository", "a" * 40)
    baseline = _baseline(base, mock_model)
    lease = registry.acquire(identity, lambda: baseline)
    return registry, identity, baseline, worktree, WorkspaceIndex(lease, worktree)


def _paths(results) -> list[str]:
    return [hit.result.chunk.file_path for hit in results]


def test_workspace_scopes_preserve_baseline_delta_and_provenance(tmp_path: Path, mock_model) -> None:
    """Search facets expose current content, immutable base content, and provenance."""
    registry, identity, _baseline_index, worktree, workspace = _workspace(tmp_path, mock_model)
    try:
        initial = workspace.search("legacy amber token", scope=SearchScope.WORKSPACE)
        assert initial.baseline == identity
        assert initial.changed_file_count == 0
        assert "auth.py" in _paths(initial.base_results)
        assert not initial.delta_results
        assert all(
            hit.origin is SearchOrigin.BASE and hit.change is ChangeKind.UNCHANGED for hit in initial.base_results
        )

        (worktree / "auth.py").write_text("def current_authentication():\n    return 'violet access beacon'\n")
        workspace.apply_changes([WorkspaceFileChange("auth.py", ChangeKind.MODIFIED)])

        changed = workspace.search("violet access beacon", scope=SearchScope.CHANGED)
        assert _paths(changed.delta_results)[0] == "auth.py"
        assert changed.delta_results[0].origin is SearchOrigin.DELTA
        assert changed.delta_results[0].change is ChangeKind.MODIFIED
        assert not changed.base_results

        unchanged = workspace.search("legacy amber token", scope=SearchScope.UNCHANGED)
        assert "auth.py" not in _paths(unchanged.base_results)
        base = workspace.search("legacy amber token", scope=SearchScope.BASE)
        auth = next(hit for hit in base.base_results if hit.result.chunk.file_path == "auth.py")
        assert auth.origin is SearchOrigin.BASE
        assert auth.change is ChangeKind.MODIFIED

        effective = workspace.search("violet access beacon", scope=SearchScope.WORKSPACE)
        assert "auth.py" in _paths(effective.delta_results)
        assert "auth.py" not in _paths(effective.base_results)
    finally:
        workspace.close()
    assert registry.references(identity) == 0


def test_workspace_handles_add_delete_rename_and_revert(tmp_path: Path, mock_model) -> None:
    """Added, deleted, renamed, and reverted paths maintain correct facets."""
    _registry, _identity, _baseline_index, worktree, workspace = _workspace(tmp_path, mock_model)
    try:
        (worktree / "database.py").unlink()
        (worktree / "new_feature.py").write_text("def quartz_feature():\n    return 'new horizon signal'\n")
        (worktree / "worker.py").rename(worktree / "renamed_worker.py")
        workspace.apply_changes(
            [
                WorkspaceFileChange("database.py", ChangeKind.DELETED),
                WorkspaceFileChange("new_feature.py", ChangeKind.ADDED),
                WorkspaceFileChange("renamed_worker.py", ChangeKind.RENAMED, previous_path="worker.py"),
            ]
        )

        changed = workspace.search("new horizon signal", scope=SearchScope.CHANGED)
        assert _paths(changed.delta_results)[0] == "new_feature.py"
        assert changed.delta_results[0].change is ChangeKind.ADDED

        renamed = workspace.search("silver queue marker", scope=SearchScope.WORKSPACE)
        assert "renamed_worker.py" in _paths(renamed.delta_results)
        assert "worker.py" not in _paths(renamed.base_results)
        assert (
            next(hit for hit in renamed.delta_results if hit.result.chunk.file_path == "renamed_worker.py").change
            is ChangeKind.RENAMED
        )

        deleted = workspace.search("durable cobalt record", scope=SearchScope.WORKSPACE)
        assert "database.py" not in _paths(deleted.base_results)
        original = workspace.search("durable cobalt record", scope=SearchScope.BASE)
        assert (
            next(hit for hit in original.base_results if hit.result.chunk.file_path == "database.py").change
            is ChangeKind.DELETED
        )

        (worktree / "auth.py").write_text("def changed_authentication():\n    return 'temporary marker'\n")
        workspace.apply_changes([WorkspaceFileChange("auth.py", ChangeKind.MODIFIED)])
        (worktree / "auth.py").write_text("def baseline_authentication():\n    return 'legacy amber token'\n")
        workspace.apply_changes([WorkspaceFileChange("auth.py", ChangeKind.UNCHANGED)])
        reverted = workspace.search("legacy amber token", scope=SearchScope.UNCHANGED)
        assert "auth.py" in _paths(reverted.base_results)
        assert "auth.py" not in workspace.changed_paths
    finally:
        workspace.close()


def test_workspace_prepares_query_once_across_base_and_delta(tmp_path: Path, mock_model) -> None:
    """A workspace query embeds once even when both physical indexes are searched."""
    _registry, _identity, _baseline_index, worktree, workspace = _workspace(tmp_path, mock_model)
    try:
        (worktree / "auth.py").write_text("def changed_authentication():\n    return 'single encoding marker'\n")
        workspace.apply_changes([WorkspaceFileChange("auth.py", ChangeKind.MODIFIED)])
        mock_model.encode.reset_mock()
        result = workspace.search("single encoding marker", scope=SearchScope.WORKSPACE)
        assert "auth.py" in _paths(result.delta_results)
        mock_model.encode.assert_called_once()
    finally:
        workspace.close()


def test_shared_baseline_isolated_between_worktrees_and_reference_counted(tmp_path: Path, mock_model) -> None:
    """Worktrees share immutable baselines while retaining private delta state."""
    base = tmp_path / "base"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_project(base)
    shutil.copytree(base, first_root)
    shutil.copytree(base, second_root)
    identity = BaselineIdentity("example/repository", "b" * 40)
    baseline = _baseline(base, mock_model)
    registry = BaselineRegistry()
    builds = 0

    def build() -> SembleIndex:
        nonlocal builds
        builds += 1
        return baseline

    first = WorkspaceIndex(registry.acquire(identity, build), first_root)
    second = WorkspaceIndex(registry.acquire(identity, build), second_root)
    assert builds == 1
    assert first.baseline.index is second.baseline.index
    assert registry.references(identity) == 2

    (first_root / "auth.py").write_text("def first_only():\n    return 'isolated workspace marker'\n")
    first.apply_changes([WorkspaceFileChange("auth.py", ChangeKind.MODIFIED)])
    assert "auth.py" in _paths(first.search("isolated workspace marker", scope=SearchScope.CHANGED).delta_results)
    assert not second.search("isolated workspace marker", scope=SearchScope.CHANGED).delta_results
    assert "auth.py" in _paths(second.search("legacy amber token", scope=SearchScope.UNCHANGED).base_results)

    first.close()
    assert registry.references(identity) == 1
    assert registry.evict_unused() == 0
    second.close()
    assert registry.references(identity) == 0
    assert registry.evict_unused() == 1


def test_rapid_edit_retries_until_file_snapshot_is_stable(tmp_path: Path, mock_model) -> None:
    """An edit that changes during embedding retries and publishes only the latest bytes."""
    _registry, _identity, _baseline_index, worktree, workspace = _workspace(tmp_path, mock_model)
    target = worktree / "auth.py"
    calls = 0

    def changing_embed(model, chunks):
        nonlocal calls
        calls += 1
        vectors = real_embed_chunks(model, chunks)
        if calls == 1:
            target.write_text("def newest_authentication():\n    return 'latest generation marker expanded'\n")
        return vectors

    try:
        target.write_text("def stale_authentication():\n    return 'intermediate generation'\n")
        with patch("semble.workspace.embed_chunks", side_effect=changing_embed):
            workspace.apply_changes([WorkspaceFileChange("auth.py", ChangeKind.MODIFIED)])
        assert calls == 2
        newest = workspace.search("latest generation marker expanded", scope=SearchScope.CHANGED)
        assert _paths(newest.delta_results)[0] == "auth.py"
        stale = workspace.search("intermediate generation", scope=SearchScope.CHANGED)
        assert all("intermediate generation" not in hit.result.chunk.content for hit in stale.delta_results)
    finally:
        workspace.close()


def test_workspace_with_every_baseline_path_shadowed_returns_only_delta(tmp_path: Path, mock_model) -> None:
    """Shadowing every baseline path yields an empty base facet without backend errors."""
    base = tmp_path / "base"
    worktree = tmp_path / "worktree"
    base.mkdir()
    (base / "only.py").write_text("def original():\n    return 'old marker'\n")
    shutil.copytree(base, worktree)
    registry = BaselineRegistry()
    identity = BaselineIdentity("example/one-file", "c" * 40)
    lease = registry.acquire(identity, lambda: _baseline(base, mock_model))
    workspace = WorkspaceIndex(lease, worktree)
    try:
        (worktree / "only.py").write_text("def replacement():\n    return 'new marker'\n")
        workspace.apply_changes([WorkspaceFileChange("only.py", ChangeKind.MODIFIED)])
        result = workspace.search("new marker", scope=SearchScope.WORKSPACE)
        assert not result.base_results
        assert _paths(result.delta_results) == ["only.py"]
    finally:
        workspace.close()


def test_authoritative_reconcile_removes_vanished_delta_membership(tmp_path: Path, mock_model) -> None:
    """A full Git snapshot removes overlay state for paths no longer different from base."""
    _registry, _identity, _baseline_index, worktree, workspace = _workspace(tmp_path, mock_model)
    try:
        added = worktree / "transient.py"
        added.write_text("def transient():\n    return 'short lived delta'\n")
        workspace.reconcile_changes([WorkspaceFileChange("transient.py", ChangeKind.ADDED)])
        assert "transient.py" in workspace.changed_paths
        added.unlink()
        workspace.reconcile_changes([])
        assert "transient.py" not in workspace.changed_paths
        assert not workspace.search("short lived delta", scope=SearchScope.CHANGED).delta_results
    finally:
        workspace.close()


@pytest.mark.anyio
async def test_workspace_watcher_applies_classified_event_and_releases_baseline(
    tmp_path: Path,
    mock_model,
) -> None:
    """Watcher events update the overlay and shutdown releases its baseline lease."""
    registry, identity, _baseline_index, worktree, workspace = _workspace(tmp_path, mock_model)
    events: asyncio.Queue[set[tuple[Change, str]]] = asyncio.Queue()
    watcher_started = asyncio.Event()

    async def fake_awatch(*_paths: Path, **_kwargs: object):  # type: ignore[no-untyped-def]
        watcher_started.set()
        yield await events.get()
        await asyncio.Future()

    def classify(paths):
        assert paths == {worktree / "auth.py"}
        return [WorkspaceFileChange("auth.py", ChangeKind.MODIFIED)]

    (worktree / "auth.py").write_text("def watched_auth():\n    return 'watcher delta signal'\n")
    with patch("semble.workspace.awatch", new=fake_awatch):
        watcher = WorkspaceWatcher(workspace, classify)
        watcher.start()
        await asyncio.wait_for(watcher_started.wait(), timeout=1.0)
        await events.put({(Change.modified, str(worktree / "auth.py"))})
        for _attempt in range(100):
            if "auth.py" in workspace.changed_paths:
                break
            await asyncio.sleep(0)
        assert "auth.py" in workspace.changed_paths
        assert "auth.py" in _paths(workspace.search("watcher delta signal", scope=SearchScope.CHANGED).delta_results)
        await watcher.close()
    assert registry.references(identity) == 0


def test_layered_workspace_matches_full_index_behavioral_oracle(tmp_path: Path, mock_model) -> None:
    """Layered facets and a complete worktree index retrieve the same expected current files."""
    _registry, _identity, _baseline_index, worktree, workspace = _workspace(tmp_path, mock_model)
    try:
        (worktree / "auth.py").write_text("def current_authentication():\n    return 'violet current access'\n")
        (worktree / "database.py").unlink()
        (worktree / "new_feature.py").write_text("def current_feature():\n    return 'quartz current feature'\n")
        workspace.apply_changes(
            [
                WorkspaceFileChange("auth.py", ChangeKind.MODIFIED),
                WorkspaceFileChange("database.py", ChangeKind.DELETED),
                WorkspaceFileChange("new_feature.py", ChangeKind.ADDED),
            ]
        )
        full = _baseline(worktree, mock_model)
        cases = [
            ("violet current access", "auth.py"),
            ("quartz current feature", "new_feature.py"),
            ("silver queue marker", "worker.py"),
        ]
        for query, expected in cases:
            layered = workspace.search(query, scope=SearchScope.WORKSPACE, top_k=5)
            layered_paths = {*_paths(layered.base_results), *_paths(layered.delta_results)}
            full_paths = {result.chunk.file_path for result in full.search(query, top_k=5)}
            assert expected in layered_paths
            assert expected in full_paths

        deleted = workspace.search("durable cobalt record", scope=SearchScope.WORKSPACE)
        assert "database.py" not in {*_paths(deleted.base_results), *_paths(deleted.delta_results)}
    finally:
        workspace.close()


@pytest.mark.anyio
async def test_workspace_watcher_applies_empty_authoritative_reconcile(tmp_path: Path, mock_model) -> None:
    """An ignore-rule event may clear the complete delta even when classification returns no paths."""
    _registry, _identity, _baseline_index, worktree, workspace = _workspace(tmp_path, mock_model)
    (worktree / "auth.py").write_text("def changed():\n    return 'temporary delta'\n")
    workspace.apply_changes([WorkspaceFileChange("auth.py", ChangeKind.MODIFIED)])
    assert workspace.changed_paths

    async def fake_awatch(*_paths: Path, **_kwargs: object):  # type: ignore[no-untyped-def]
        yield {(Change.modified, str(worktree / ".gitignore"))}
        await asyncio.Future()

    with patch("semble.workspace.awatch", new=fake_awatch):
        watcher = WorkspaceWatcher(workspace, lambda _paths: [], lambda _paths: True)
        watcher.start()
        for _attempt in range(100):
            if not workspace.changed_paths:
                break
            await asyncio.sleep(0)
        assert not workspace.changed_paths
        await watcher.close()


@pytest.mark.anyio
async def test_workspace_watcher_continues_after_transient_refresh_failure(tmp_path: Path, mock_model) -> None:
    """One classifier failure is logged and the following filesystem event still updates the delta."""
    _registry, _identity, _baseline_index, worktree, workspace = _workspace(tmp_path, mock_model)
    (worktree / "auth.py").write_text("def recovered():\n    return 'watcher recovered marker'\n")
    calls = 0

    async def fake_awatch(*_paths: Path, **_kwargs: object):  # type: ignore[no-untyped-def]
        event = {(Change.modified, str(worktree / "auth.py"))}
        yield event
        yield event
        await asyncio.Future()

    def classify(_paths):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient git lock")
        return [WorkspaceFileChange("auth.py", ChangeKind.MODIFIED)]

    with (
        patch("semble.workspace.awatch", new=fake_awatch),
        patch("semble.workspace.logger") as logger,
    ):
        watcher = WorkspaceWatcher(workspace, classify)
        watcher.start()
        for _attempt in range(100):
            if "auth.py" in workspace.changed_paths:
                break
            await asyncio.sleep(0)
        assert "auth.py" in workspace.changed_paths
        logger.warning.assert_called_once()
        await watcher.close()
