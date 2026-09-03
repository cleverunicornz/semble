from __future__ import annotations

import os
import subprocess
from collections.abc import Collection, Sequence
from pathlib import Path

from semble.index import SembleIndex
from semble.types import ContentType
from semble.workspace import (
    BaselineIdentity,
    BaselineRegistry,
    ChangeKind,
    WorkspaceFileChange,
    WorkspaceIndex,
    WorkspaceWatcher,
)

_INDEX_RULE_FILES = frozenset({".gitignore", ".sembleignore"})
_GIT_TIMEOUT_SECONDS = 60


def _git(root: Path, *args: str) -> bytes:
    """Run one bounded Git command and return its raw stdout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        raise RuntimeError("git is not installed or not on PATH") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git command timed out after {_GIT_TIMEOUT_SECONDS}s") from None
    if result.returncode != 0:
        detail = os.fsdecode(result.stderr).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def resolve_revision(root: str | Path, revision: str) -> str:
    """Resolve a Git revision to one immutable 40-hex commit ID."""
    resolved_root = Path(root).resolve()
    output = _git(resolved_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    commit = os.fsdecode(output).strip()
    if len(commit) != 40 or not all(character in "0123456789abcdefABCDEF" for character in commit):
        raise RuntimeError(f"Git revision did not resolve to a commit: {revision}")
    return commit.lower()


def fetch_remote_baseline(
    root: str | Path,
    *,
    remote: str = "origin",
    branch: str,
) -> BaselineIdentity:
    """Fetch one remote branch and return its canonical URL plus pinned commit."""
    if not remote or not branch or remote.startswith("-") or branch.startswith("-"):
        raise ValueError("Git remote and branch must be non-empty and cannot start with '-'")
    resolved_root = Path(root).resolve()
    _git(resolved_root, "fetch", "--no-tags", remote, branch)
    revision = resolve_revision(resolved_root, f"refs/remotes/{remote}/{branch}")
    repository = os.fsdecode(_git(resolved_root, "remote", "get-url", remote)).strip()
    if not repository:
        raise RuntimeError(f"Git remote has no URL: {remote}")
    return BaselineIdentity(repository=repository, revision=revision)


def _decode_path(value: bytes) -> str:
    return os.fsdecode(value)


def _parse_name_status(raw: bytes) -> list[WorkspaceFileChange]:
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    changes: list[WorkspaceFileChange] = []
    index = 0
    while index < len(fields):
        status = _decode_path(fields[index])
        index += 1
        code = status[:1]
        if code in {"R", "C"}:
            if index + 1 >= len(fields):
                raise RuntimeError("Malformed Git rename/copy status output")
            previous = _decode_path(fields[index])
            path = _decode_path(fields[index + 1])
            index += 2
            changes.append(
                WorkspaceFileChange(
                    path=path,
                    kind=ChangeKind.RENAMED if code == "R" else ChangeKind.ADDED,
                    previous_path=previous if code == "R" else None,
                )
            )
            continue
        if index >= len(fields):
            raise RuntimeError("Malformed Git name-status output")
        path = _decode_path(fields[index])
        index += 1
        kind = {
            "A": ChangeKind.ADDED,
            "D": ChangeKind.DELETED,
        }.get(code, ChangeKind.MODIFIED)
        changes.append(WorkspaceFileChange(path=path, kind=kind))
    return changes


class GitChangeClassifier:
    """Derive current workspace path states relative to one pinned Git commit."""

    def __init__(self, root: str | Path, base_revision: str) -> None:
        """Pin the classifier to a repository root and resolved base commit."""
        self.root = Path(root).resolve()
        self.base_revision = resolve_revision(self.root, base_revision)

    def all_changes(self) -> list[WorkspaceFileChange]:
        """Return every tracked and untracked current difference from the base."""
        return self._collect_changes()

    def _collect_changes(self, pathspecs: Collection[str] = ()) -> list[WorkspaceFileChange]:
        """Return current differences, optionally bounded to explicit repository paths."""
        paths = sorted(pathspecs)
        tracked = _parse_name_status(
            _git(
                self.root,
                "diff",
                "--name-status",
                "-z",
                "--find-renames",
                self.base_revision,
                "--",
                *paths,
            )
        )
        untracked = [
            _decode_path(path)
            for path in _git(
                self.root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *paths,
            ).split(b"\0")
            if path
        ]
        renames = self._pair_unstaged_renames(tracked, untracked)
        renamed_from = {change.previous_path for change in renames}
        renamed_to = {change.path for change in renames}
        tracked = [change for change in tracked if change.path not in renamed_from]
        tracked.extend(renames)
        tracked.extend(
            WorkspaceFileChange(path=path, kind=ChangeKind.ADDED) for path in untracked if path not in renamed_to
        )
        return tracked

    def _pair_unstaged_renames(
        self,
        tracked: Sequence[WorkspaceFileChange],
        untracked: Sequence[str],
    ) -> list[WorkspaceFileChange]:
        """Pair unique deleted and untracked files whose Git blob IDs are identical."""
        deleted_by_hash: dict[str, list[str]] = {}
        for change in tracked:
            if change.kind is not ChangeKind.DELETED:
                continue
            try:
                digest = _decode_path(_git(self.root, "rev-parse", f"{self.base_revision}:{change.path}")).strip()
            except RuntimeError:
                continue
            deleted_by_hash.setdefault(digest, []).append(change.path)

        untracked_by_hash: dict[str, list[str]] = {}
        for path in untracked:
            try:
                digest = _decode_path(_git(self.root, "hash-object", "--", path)).strip()
            except RuntimeError:
                continue
            untracked_by_hash.setdefault(digest, []).append(path)

        return [
            WorkspaceFileChange(path=destinations[0], kind=ChangeKind.RENAMED, previous_path=sources[0])
            for digest, sources in deleted_by_hash.items()
            if len(sources) == 1 and len(destinations := untracked_by_hash.get(digest, [])) == 1
        ]

    def classify(self, paths: Collection[Path]) -> list[WorkspaceFileChange]:
        """Classify changed filesystem paths, returning UNCHANGED for paths reverted to base."""
        requested = {self._relative(path) for path in paths}
        if self.requires_full_reconcile(paths):
            return self.all_changes()
        current = self._collect_changes(requested)
        covered = {member for change in current for member in (change.path, change.previous_path) if member is not None}
        current.extend(
            WorkspaceFileChange(path=path, kind=ChangeKind.UNCHANGED) for path in sorted(requested - covered)
        )
        return current

    @staticmethod
    def requires_full_reconcile(paths: Collection[Path]) -> bool:
        """Return whether changed ignore rules can affect paths beyond the event batch."""
        return any(path.name in _INDEX_RULE_FILES for path in paths)

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(self.root))
        except ValueError as exc:
            raise ValueError(f"Git change path is outside repository: {path}") from exc


class GitWorkspaceSession:
    """Pinned Git classifier, layered index, and watcher owned as one lifecycle."""

    def __init__(self, index: WorkspaceIndex, classifier: GitChangeClassifier) -> None:
        """Create a stopped session around an initialized workspace index."""
        self.index = index
        self.classifier = classifier
        self.watcher = WorkspaceWatcher(index, classifier.classify, classifier.requires_full_reconcile)

    def start(self) -> None:
        """Start watcher-driven delta maintenance."""
        self.watcher.start()

    async def close(self) -> None:
        """Stop watching, drop the overlay, and release the shared baseline."""
        await self.watcher.close()


def open_git_workspace(
    registry: BaselineRegistry,
    identity: BaselineIdentity,
    *,
    baseline_root: str | Path,
    workspace_root: str | Path,
    content: Sequence[ContentType] = (ContentType.CODE,),
    model_path: str | None = None,
) -> GitWorkspaceSession:
    """Build or attach a clean pinned baseline and initialize the complete workspace delta."""
    baseline_path = Path(baseline_root).resolve()
    workspace_path = Path(workspace_root).resolve()
    if resolve_revision(baseline_path, "HEAD") != identity.revision:
        raise ValueError("Baseline checkout HEAD does not match the pinned baseline revision")
    if GitChangeClassifier(baseline_path, identity.revision).all_changes():
        raise ValueError("Baseline checkout must be clean at the pinned revision")
    classifier = GitChangeClassifier(workspace_path, identity.revision)

    def build_baseline() -> SembleIndex:
        from semble.cache import save_index_to_cache

        index = SembleIndex.from_path(baseline_path, content=content, model_path=model_path)
        save_index_to_cache(index, str(baseline_path))
        return index

    lease = registry.acquire(identity, build_baseline)
    try:
        workspace = WorkspaceIndex(lease, workspace_path)
        workspace.reconcile_changes(classifier.all_changes())
    except Exception:
        lease.close()
        raise
    return GitWorkspaceSession(workspace, classifier)
