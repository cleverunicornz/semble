from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from vicinity.backends.basic import BasicArgs
from watchfiles import awatch

from semble.chunking import chunk_source
from semble.index import SembleIndex
from semble.index.bm25 import BM25
from semble.index.dense import SelectableBasicBackend, embed_chunks
from semble.index.files import FileStatus, detect_language, get_extensions, get_file_status, read_file_text
from semble.index.sparse import enrich_for_bm25
from semble.index.types import make_chunk_id
from semble.search import PreparedQuery, search_prepared
from semble.tokens import tokenize
from semble.types import Chunk, ContentType, EmbeddingMatrix, SearchResult

WORKSPACE_WATCH_DEBOUNCE_MS = 200
WORKSPACE_WATCH_STEP_MS = 50
_STABLE_READ_ATTEMPTS = 3
_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="semble-workspace-search")
logger = logging.getLogger(__name__)


class SearchScope(str, Enum):
    """Workspace search view selected by the caller."""

    WORKSPACE = "workspace"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    BASE = "base"


class SearchOrigin(str, Enum):
    """Physical index that supplied a workspace search result."""

    BASE = "base"
    DELTA = "delta"


class ChangeKind(str, Enum):
    """Current path state relative to the workspace's immutable baseline."""

    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass(frozen=True, slots=True)
class BaselineIdentity:
    """Stable repository and revision identity for one immutable baseline."""

    repository: str
    revision: str

    def __post_init__(self) -> None:
        """Validate that both identity components are non-empty."""
        if not self.repository.strip() or not self.revision.strip():
            raise ValueError("Baseline repository and revision must be non-empty")


@dataclass(frozen=True, slots=True)
class WorkspaceFileChange:
    """One authoritative current path state relative to the baseline."""

    path: str
    kind: ChangeKind
    previous_path: str | None = None

    def __post_init__(self) -> None:
        """Validate path and rename metadata."""
        if not self.path.strip():
            raise ValueError("Workspace change path must be non-empty")
        if self.kind is ChangeKind.RENAMED and not self.previous_path:
            raise ValueError("Renamed workspace changes require previous_path")
        if self.kind is not ChangeKind.RENAMED and self.previous_path is not None:
            raise ValueError("previous_path is valid only for renamed workspace changes")


@dataclass(frozen=True, slots=True)
class WorkspaceSearchHit:
    """Search result annotated with baseline/delta provenance and change state."""

    result: SearchResult
    origin: SearchOrigin
    change: ChangeKind

    def to_dict(self) -> dict[str, object]:
        """Return an MCP-friendly result object."""
        return {
            "file_path": self.result.chunk.file_path,
            "start_line": self.result.chunk.start_line,
            "end_line": self.result.chunk.end_line,
            "content": self.result.chunk.content,
            "score": self.result.score,
            "origin": self.origin.value,
            "change": self.change.value,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceSearchResponse:
    """Scoped search response preserving separate baseline and delta facets."""

    baseline: BaselineIdentity
    scope: SearchScope
    changed_file_count: int
    base_results: tuple[WorkspaceSearchHit, ...]
    delta_results: tuple[WorkspaceSearchHit, ...]

    def to_dict(self) -> dict[str, object]:
        """Return an MCP-friendly response object."""
        return {
            "baseline": {
                "repository": self.baseline.repository,
                "revision": self.baseline.revision,
            },
            "scope": self.scope.value,
            "changed_file_count": self.changed_file_count,
            "base_results": [result.to_dict() for result in self.base_results],
            "delta_results": [result.to_dict() for result in self.delta_results],
        }


@dataclass(slots=True)
class _BaselineEntry:
    index: SembleIndex
    references: int = 0


_BaselineKey = tuple[BaselineIdentity, tuple[ContentType, ...]]


class BaselineLease:
    """Reference-counted handle to one immutable baseline index."""

    def __init__(
        self,
        registry: BaselineRegistry,
        key: _BaselineKey,
        index: SembleIndex,
    ) -> None:
        """Create one live handle owned by the registry."""
        self._registry = registry
        self._key = key
        self.identity = key[0]
        self.index = index
        self._closed = False

    def close(self) -> None:
        """Release this handle exactly once without evicting the baseline cache."""
        if self._closed:
            return
        self._closed = True
        self._registry._release(self._key)

    def __enter__(self) -> BaselineLease:
        """Return this lease as a context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release this lease when its context exits."""
        self.close()


class BaselineRegistry:
    """Process-local immutable baseline cache keyed by identity and content variant."""

    def __init__(self) -> None:
        """Create an empty process-local baseline registry."""
        self._entries: dict[_BaselineKey, _BaselineEntry] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        identity: BaselineIdentity,
        build: Callable[[], SembleIndex],
        content: Sequence[ContentType] = (),
    ) -> BaselineLease:
        """Return one shared exact-content baseline and increment its reference count."""
        key = (identity, tuple(content))
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _BaselineEntry(index=build())
                self._entries[key] = entry
            entry.references += 1
            return BaselineLease(self, key, entry.index)

    def _release(self, key: _BaselineKey) -> None:
        """Release one exact baseline reference while retaining its cached index."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.references == 0:
                raise ValueError(f"Baseline {key!r} has no active reference")
            entry.references -= 1

    def references(self, identity: BaselineIdentity) -> int:
        """Return active references across every content variant for an identity."""
        with self._lock:
            return sum(entry.references for key, entry in self._entries.items() if key[0] == identity)

    def evict_unused(self) -> int:
        """Remove every unreferenced baseline variant and return the number evicted."""
        with self._lock:
            unused = [key for key, entry in self._entries.items() if entry.references == 0]
            for key in unused:
                del self._entries[key]
            return len(unused)


@dataclass(frozen=True, slots=True)
class _OverlayFile:
    chunks: tuple[Chunk, ...]
    vectors: EmbeddingMatrix
    change: ChangeKind
    mtime_ns: int
    size: int


@dataclass(frozen=True, slots=True)
class _OverlaySnapshot:
    chunks: list[Chunk]
    semantic_index: SelectableBasicBackend
    bm25_index: BM25


class WorkspaceIndex:
    """Immutable baseline plus a mutable, path-shadowing worktree delta index."""

    def __init__(self, baseline: BaselineLease, root: str | Path) -> None:
        """Attach a worktree root to a leased immutable baseline."""
        self.baseline = baseline
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self._content = baseline.index.content
        self._extensions = frozenset(get_extensions(self._content))
        self._baseline_paths = baseline.index.file_paths
        self._overlay_files: dict[str, _OverlayFile] = {}
        self._changes: dict[str, ChangeKind] = {}
        self._shadowed_paths: set[str] = set()
        self._overlay: _OverlaySnapshot | None = None
        self._generation = 0
        self._lock = threading.RLock()
        self._closed = False

    @property
    def identity(self) -> BaselineIdentity:
        """Return the immutable baseline identity used by this workspace."""
        return self.baseline.identity

    @property
    def changed_paths(self) -> dict[str, ChangeKind]:
        """Return a copy of current path states relative to the baseline."""
        with self._lock:
            return dict(self._changes)

    @property
    def content(self) -> tuple[ContentType, ...]:
        """Return the content classes covered by both physical indexes."""
        return self._content

    @property
    def baseline_file_count(self) -> int:
        """Return the number of files represented by the immutable baseline."""
        return len(self._baseline_paths)

    @property
    def baseline_chunk_count(self) -> int:
        """Return the number of chunks represented by the immutable baseline."""
        return len(self.baseline.index.chunks)

    @property
    def delta_file_count(self) -> int:
        """Return the number of changed files with searchable delta content."""
        with self._lock:
            return len(self._overlay_files)

    @property
    def delta_chunk_count(self) -> int:
        """Return the number of chunks represented by the mutable delta."""
        with self._lock:
            return 0 if self._overlay is None else len(self._overlay.chunks)

    @property
    def generation(self) -> int:
        """Return the monotonically increasing published delta generation."""
        with self._lock:
            return self._generation

    @property
    def shadowed_file_count(self) -> int:
        """Return the number of baseline paths hidden from the effective workspace."""
        with self._lock:
            return len(self._shadowed_paths)

    def _relative_path(self, value: str | Path) -> str:
        path = Path(value)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"Workspace change path is outside root: {value}") from exc
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Workspace change path is outside root: {value}")
        return str(path)

    def _index_file(self, relative: str, change: ChangeKind) -> _OverlayFile | None:
        path = self.root / relative
        if path.suffix.lower() not in self._extensions:
            return None
        for _attempt in range(_STABLE_READ_ATTEMPTS):
            before = path.stat()
            if get_file_status(path, None) is not FileStatus.VALID:
                return None
            source = read_file_text(path)
            chunks = chunk_source(source, relative, detect_language(path))
            vectors = embed_chunks(self.baseline.index.model, chunks)
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
                return _OverlayFile(tuple(chunks), vectors, change, after.st_mtime_ns, after.st_size)
        raise RuntimeError(f"Workspace file kept changing while indexed: {relative}")

    @staticmethod
    def _build_overlay(overlay_files: dict[str, _OverlayFile]) -> _OverlaySnapshot | None:
        """Build one immutable retrieval snapshot from cached per-file delta vectors."""
        ordered = [overlay_files[path] for path in sorted(overlay_files)]
        chunks = [chunk for entry in ordered for chunk in entry.chunks]
        if not chunks:
            return None
        vectors = np.vstack([entry.vectors for entry in ordered])
        bm25_index = BM25()
        chunk_ids = []
        for entry in ordered:
            for slot, chunk in enumerate(entry.chunks):
                chunk_id = make_chunk_id(chunk.file_path, slot)
                bm25_index.add_document(chunk_id, tokenize(enrich_for_bm25(chunk)))
                chunk_ids.append(chunk_id)
        bm25_index.set_doc_order(chunk_ids)
        return _OverlaySnapshot(
            chunks=chunks,
            semantic_index=SelectableBasicBackend(vectors, BasicArgs()),
            bm25_index=bm25_index,
        )

    def apply_changes(self, changes: Sequence[WorkspaceFileChange]) -> None:
        """Apply one authoritative batch and atomically publish the resulting delta snapshot."""
        with self._lock:
            self._require_open()
            overlay_files = dict(self._overlay_files)
            current_changes = dict(self._changes)
            shadowed_paths = set(self._shadowed_paths)
            overlay_changed = False
            for change in changes:
                relative = self._relative_path(change.path)
                if change.kind is ChangeKind.UNCHANGED:
                    current_changes.pop(relative, None)
                    shadowed_paths.discard(relative)
                    overlay_changed = overlay_files.pop(relative, None) is not None or overlay_changed
                    continue
                if change.kind is ChangeKind.RENAMED:
                    previous = self._relative_path(change.previous_path or "")
                    current_changes[previous] = ChangeKind.DELETED
                    shadowed_paths.add(previous)
                    overlay_changed = overlay_files.pop(previous, None) is not None or overlay_changed
                if change.kind is ChangeKind.DELETED:
                    current_changes[relative] = ChangeKind.DELETED
                    shadowed_paths.add(relative)
                    overlay_changed = overlay_files.pop(relative, None) is not None or overlay_changed
                    continue
                current_changes[relative] = change.kind
                if relative in self._baseline_paths:
                    shadowed_paths.add(relative)
                existing = overlay_files.get(relative)
                path = self.root / relative
                with contextlib.suppress(OSError):
                    stat = path.stat()
                    if (
                        existing is not None
                        and existing.change is change.kind
                        and existing.mtime_ns == stat.st_mtime_ns
                        and existing.size == stat.st_size
                    ):
                        continue
                overlay_file = self._index_file(relative, change.kind)
                if overlay_file is None:
                    overlay_changed = overlay_files.pop(relative, None) is not None or overlay_changed
                else:
                    overlay_files[relative] = overlay_file
                    overlay_changed = True
            overlay = self._build_overlay(overlay_files) if overlay_changed else self._overlay
            state_changed = current_changes != self._changes or shadowed_paths != self._shadowed_paths
            self._overlay_files = overlay_files
            self._changes = current_changes
            self._shadowed_paths = shadowed_paths
            self._overlay = overlay
            if overlay_changed or state_changed:
                self._generation += 1

    def reconcile_changes(self, changes: Sequence[WorkspaceFileChange]) -> None:
        """Replace current delta membership with one authoritative change snapshot."""
        desired: set[str] = set()
        for change in changes:
            if change.kind is ChangeKind.UNCHANGED:
                continue
            desired.add(self._relative_path(change.path))
            if change.previous_path:
                desired.add(self._relative_path(change.previous_path))
        with self._lock:
            reverted = [
                WorkspaceFileChange(path=path, kind=ChangeKind.UNCHANGED) for path in self._changes.keys() - desired
            ]
        self.apply_changes([*reverted, *changes])

    def _search_base(
        self,
        prepared: PreparedQuery,
        scope: SearchScope,
        top_k: int,
    ) -> list[SearchResult]:
        excluded = None if scope is SearchScope.BASE else self.baseline.index.indices_for_paths(self._shadowed_paths)
        return self.baseline.index.search_prepared(
            prepared,
            top_k,
            excluded=excluded,
            record_stats=False,
        )

    def _search_delta(self, prepared: PreparedQuery, top_k: int) -> list[SearchResult]:
        overlay = self._overlay
        if overlay is None:
            return []
        return search_prepared(
            prepared,
            overlay.semantic_index,
            overlay.bm25_index,
            overlay.chunks,
            top_k,
            rerank=ContentType.CODE in self._content,
        )

    def search(
        self,
        query: str,
        *,
        scope: SearchScope = SearchScope.WORKSPACE,
        top_k: int = 10,
    ) -> WorkspaceSearchResponse:
        """Search one explicit workspace facet while preserving result provenance."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        with self._lock:
            self._require_open()
            prepared = self.baseline.index.prepare_query(query)
            base_results: list[SearchResult] = []
            delta_results: list[SearchResult] = []
            searches_base = scope in {SearchScope.BASE, SearchScope.UNCHANGED, SearchScope.WORKSPACE}
            searches_delta = scope in {SearchScope.CHANGED, SearchScope.WORKSPACE}
            if scope is SearchScope.WORKSPACE and self._overlay is not None:
                base_future = _SEARCH_EXECUTOR.submit(self._search_base, prepared, scope, top_k)
                delta_future = _SEARCH_EXECUTOR.submit(self._search_delta, prepared, top_k)
                base_results = base_future.result()
                delta_results = delta_future.result()
            else:
                if searches_base:
                    base_results = self._search_base(prepared, scope, top_k)
                if searches_delta:
                    delta_results = self._search_delta(prepared, top_k)
            return WorkspaceSearchResponse(
                baseline=self.identity,
                scope=scope,
                changed_file_count=len(self._changes),
                base_results=tuple(self._base_hit(result) for result in base_results),
                delta_results=tuple(self._delta_hit(result) for result in delta_results),
            )

    def _base_hit(self, result: SearchResult) -> WorkspaceSearchHit:
        return WorkspaceSearchHit(
            result=result,
            origin=SearchOrigin.BASE,
            change=self._changes.get(result.chunk.file_path, ChangeKind.UNCHANGED),
        )

    def _delta_hit(self, result: SearchResult) -> WorkspaceSearchHit:
        return WorkspaceSearchHit(
            result=result,
            origin=SearchOrigin.DELTA,
            change=self._changes[result.chunk.file_path],
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Workspace index is closed")

    def close(self) -> None:
        """Drop the private overlay and release its shared baseline reference."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._overlay_files.clear()
            self._changes.clear()
            self._shadowed_paths.clear()
            self._overlay = None
            self.baseline.close()

    def __enter__(self) -> WorkspaceIndex:
        """Return this workspace as a context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close this workspace when its context exits."""
        self.close()


class WorkspaceWatcher:
    """Debounced filesystem watcher applying caller-classified Git/path changes."""

    def __init__(
        self,
        workspace: WorkspaceIndex,
        classify: Callable[[Collection[Path]], Sequence[WorkspaceFileChange]],
        full_reconcile: Callable[[Collection[Path]], bool] | None = None,
        reconcile: Callable[[], Sequence[WorkspaceFileChange]] | None = None,
    ) -> None:
        """Create a watcher using the caller's authoritative change classifier."""
        self.workspace = workspace
        self.classify = classify
        self.full_reconcile = full_reconcile
        self.reconcile = reconcile
        self._task: asyncio.Task[None] | None = None
        self._refresh_lock = asyncio.Lock()

    def start(self) -> None:
        """Start watching exactly once."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name=f"semble-workspace-watch:{self.workspace.root}")

    async def _run(self) -> None:
        async for changes in awatch(
            self.workspace.root,
            debounce=WORKSPACE_WATCH_DEBOUNCE_MS,
            step=WORKSPACE_WATCH_STEP_MS,
        ):
            try:
                paths = {Path(path) for _change, path in changes}
                async with self._refresh_lock:
                    classified = await asyncio.to_thread(self.classify, paths)
                    if self.full_reconcile is not None and self.full_reconcile(paths):
                        await asyncio.to_thread(self.workspace.reconcile_changes, classified)
                    elif classified:
                        await asyncio.to_thread(self.workspace.apply_changes, classified)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Failed to refresh workspace delta; watching continues", exc_info=True)

    async def synchronize(self) -> None:
        """Publish all current Git changes before a read-your-writes search."""
        if self.reconcile is None:
            return
        async with self._refresh_lock:
            classified = await asyncio.to_thread(self.reconcile)
            await asyncio.to_thread(self.workspace.reconcile_changes, classified)

    async def close(self) -> None:
        """Stop the watcher and release the workspace overlay."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(self.workspace.close)
