from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from watchfiles import awatch

from semble.cache import resolve_cache_folder, save_index_to_cache
from semble.git_workspace import (
    GitWorkspaceSession,
    WorkspaceBinding,
    open_git_workspace,
    resolve_revision,
    resolve_workspace_binding,
)
from semble.index import SembleIndex
from semble.index.dense import load_model
from semble.index.files import get_extensions, get_max_file_bytes
from semble.types import ContentType
from semble.utils import is_git_url
from semble.workspace import (
    WORKSPACE_WATCH_DEBOUNCE_MS,
    WORKSPACE_WATCH_STEP_MS,
    BaselineIdentity,
    BaselineRegistry,
    ChangeKind,
    SearchScope,
    WorkspaceSearchHit,
    WorkspaceSearchResponse,
)

logger = logging.getLogger(__name__)

_CACHE_MAX_SIZE = 10
_WATCH_DEBOUNCE_MS = 200
_INDEX_RULE_FILES = frozenset({".gitignore", ".sembleignore"})
ContentSelection = Literal["code", "docs", "config", "all"]
SemanticFacet = Literal["workspace", "changed", "unchanged", "base"]
_CacheKey = tuple[str, tuple[ContentType, ...]]
_WorkspaceKey = tuple[str, str, BaselineIdentity, tuple[ContentType, ...]]


async def _get_index(repo: str, cache: _IndexCache, content: Sequence[ContentType]) -> SembleIndex:
    """Return a cached index for an explicit repository source."""
    if is_git_url(repo) and not repo.startswith(("https://", "http://")):
        raise ValueError(f"Only https://, http://, or local directory paths are accepted as `repo`. Got: {repo!r}")
    try:
        return await cache.get(repo, content=content)
    except Exception as exc:
        raise ValueError(f"Failed to index {repo!r}: {exc}") from exc


def _resolve_content_selection(
    content: ContentSelection | None, default_content: Sequence[ContentType]
) -> tuple[ContentType, ...]:
    """Resolve one exact content-index variant."""
    if content is None:
        return tuple(default_content)
    if content == "all":
        return tuple(ContentType)
    return (ContentType(content),)


def _format_workspace_hit(hit: WorkspaceSearchHit, max_snippet_lines: int | None) -> dict[str, object]:
    """Format one result with baseline/delta and change provenance."""
    chunk = hit.result.chunk
    payload: dict[str, object] = {
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "score": hit.result.score,
        "origin": hit.origin.value,
        "change": hit.change.value,
    }
    if max_snippet_lines != 0:
        payload["content"] = (
            chunk.content if max_snippet_lines is None else "\n".join(chunk.content.splitlines()[:max_snippet_lines])
        )
    return payload


def _index_context(
    binding: WorkspaceBinding,
    session: GitWorkspaceSession,
    content: Sequence[ContentType],
    facet: SearchScope,
    *,
    session_acquire_ms: float,
    synchronization_ms: float,
) -> dict[str, object]:
    index = session.index
    changes = index.changed_paths
    counts = Counter(change.value for change in changes.values())
    return {
        "repository": binding.identity.repository,
        "workspace_root": str(binding.workspace_root),
        "branch": binding.branch,
        "baseline_revision": binding.identity.revision,
        "baseline_source": binding.baseline_source,
        "facet": facet.value,
        "content": [item.value for item in content],
        "index_state": "current",
        "read_your_writes": True,
        "generation": index.generation,
        "session_acquire_ms": round(session_acquire_ms, 3),
        "synchronization_ms": round(synchronization_ms, 3),
        "watcher": {
            "quiet_window_ms": WORKSPACE_WATCH_STEP_MS,
            "maximum_batch_window_ms": WORKSPACE_WATCH_DEBOUNCE_MS,
            "behavior": (
                "Filesystem bursts are grouped, but every semantic_search synchronizes Git-visible changes "
                "before reading either index."
            ),
        },
        "baseline_index": {
            "immutable": True,
            "files": index.baseline_file_count,
            "chunks": index.baseline_chunk_count,
        },
        "delta_index": {
            "immutable": False,
            "changed_paths": len(changes),
            "searchable_files": index.delta_file_count,
            "chunks": index.delta_chunk_count,
            "shadowed_baseline_paths": index.shadowed_file_count,
            "changes_by_kind": {kind.value: counts.get(kind.value, 0) for kind in ChangeKind},
        },
        "coverage": {
            "indexed_extensions": sorted(get_extensions(content)),
            "maximum_file_bytes": get_max_file_bytes(),
            "included": "Supported files inside this Paseo workspace for the selected content classes.",
            "excluded": [
                "paths outside the current workspace",
                "Git-ignored and .sembleignore paths",
                "unsupported and binary files",
                "empty files",
                "files above the configured size limit",
                "content classes not selected for this index",
            ],
        },
    }


def _format_semantic_response(
    query: str,
    response: WorkspaceSearchResponse,
    context: dict[str, object],
    max_snippet_lines: int | None,
) -> dict[str, object]:
    changed_results: list[dict[str, object]] = []
    unchanged_results: list[dict[str, object]] = []
    base_results: list[dict[str, object]] = []
    if response.scope in {SearchScope.WORKSPACE, SearchScope.CHANGED}:
        changed_results = [_format_workspace_hit(result, max_snippet_lines) for result in response.delta_results]
    if response.scope in {SearchScope.WORKSPACE, SearchScope.UNCHANGED}:
        unchanged_results = [_format_workspace_hit(result, max_snippet_lines) for result in response.base_results]
    if response.scope is SearchScope.BASE:
        base_results = [_format_workspace_hit(result, max_snippet_lines) for result in response.base_results]
    return {
        "query": query,
        "index_context": context,
        "changed_results": changed_results,
        "unchanged_results": unchanged_results,
        "base_results": base_results,
    }


def _error_response(action: str, error: Exception, binding: WorkspaceBinding | None) -> str:
    return json.dumps(
        {
            "error": f"Unable to {action} the current semantic workspace.",
            "detail": str(error),
            "index_context": {
                "workspace_root": None if binding is None else str(binding.workspace_root),
                "index_state": "error",
                "read_your_writes": False,
            },
        }
    )


class _WorkspaceBindingState:
    """Refresh one context-bound workspace when its checked-out commit advances."""

    def __init__(
        self,
        workspace_cache: _WorkspaceCache,
        binding: WorkspaceBinding | None,
        binding_error: str | None,
        resolver: Callable[[], WorkspaceBinding] | None,
    ) -> None:
        self.workspace_cache = workspace_cache
        self.binding = binding
        self.binding_error = binding_error
        self.resolver = resolver
        self.lock = asyncio.Lock()

    async def current(self) -> WorkspaceBinding:
        """Return the active binding, replacing it after a committed HEAD change."""
        async with self.lock:
            if self.binding is None:
                if self.resolver is None:
                    raise RuntimeError(self.binding_error or "Semble was not launched inside a Git workspace")
                self.binding = await asyncio.to_thread(self.resolver)
                return self.binding
            if self.resolver is None:
                return self.binding

            revision = await asyncio.to_thread(resolve_revision, self.binding.workspace_root, "HEAD")
            if revision == self.binding.identity.revision:
                return self.binding

            replacement = await asyncio.to_thread(self.resolver)
            if (
                replacement.workspace_root != self.binding.workspace_root
                or replacement.identity.repository != self.binding.identity.repository
            ):
                raise RuntimeError("Semantic workspace identity changed while rebinding committed HEAD")
            await self.workspace_cache.release_workspace(str(self.binding.workspace_root))
            self.binding = replacement
            return self.binding


def _register_semantic_tools(
    server: FastMCP,
    workspace_cache: _WorkspaceCache,
    default_content: Sequence[ContentType],
    binding: WorkspaceBinding | None,
    binding_error: str | None,
    binding_resolver: Callable[[], WorkspaceBinding] | None,
) -> None:
    binding_state = _WorkspaceBindingState(workspace_cache, binding, binding_error, binding_resolver)

    async def current_session(
        content: Sequence[ContentType],
    ) -> tuple[GitWorkspaceSession, float, WorkspaceBinding]:
        started = time.perf_counter()
        resolved_binding = await binding_state.current()
        session = await workspace_cache.get(
            repo=str(resolved_binding.workspace_root),
            baseline_repo=str(resolved_binding.baseline_root),
            identity=resolved_binding.identity,
            content=content,
        )
        return session, (time.perf_counter() - started) * 1000, resolved_binding

    @server.tool()
    async def semantic_search(
        query: Annotated[
            str,
            Field(
                description=(
                    "Focused natural-language, symbol, or code query for the current Paseo workspace. "
                    "Do not provide a repository path; the server is already bound to this agent's worktree."
                )
            ),
        ],
        facet: Annotated[
            SemanticFacet,
            Field(
                description=(
                    "Optional search view. workspace (default) searches the effective current code and returns "
                    "changed_results from the private delta plus unchanged_results from the immutable baseline. "
                    "changed searches only modified/added/renamed content. unchanged searches only untouched "
                    "baseline paths. base searches the complete active committed snapshot, including old "
                    "versions of files later modified, renamed, or deleted."
                )
            ),
        ] = "workspace",
        top_k: Annotated[
            int,
            Field(
                description=(
                    "Maximum results from each independent physical facet. In workspace mode, each of the "
                    "baseline and delta indexes may return up to this many results."
                ),
                ge=1,
            ),
        ] = 5,
        max_snippet_lines: Annotated[
            int | None,
            Field(
                description=(
                    "Source lines per result. 10 gives a compact confirming snippet; 0 returns locations only; "
                    "None returns complete chunks."
                ),
                ge=0,
            ),
        ] = 10,
        content: Annotated[
            ContentSelection | None,
            Field(
                description=(
                    "Optional content class: code, docs, config, or all. Defaults to the server configuration. "
                    "The response states the exact classes and exclusions searched."
                )
            ),
        ] = None,
    ) -> str:
        """Search the current agent workspace's immutable baseline and private changed-file delta.

        Use the default workspace facet for normal investigation: it searches current changed and unchanged
        content through separate BM25/vector indexes and returns separate result sections with provenance.
        Narrow to changed or unchanged only to remove noise. Use base when comparing against the active pinned
        snapshot, including content since replaced or deleted. A clean committed HEAD advance atomically replaces
        that baseline before the next call. Before every response, Semble synchronizes all remaining Git-visible
        changes, so the first search after an edit provides read-your-writes consistency. Large edit batches may
        make that call wait; index_context reports synchronization time and exact coverage.
        """
        selected_content = _resolve_content_selection(content, default_content)
        try:
            session, acquire_ms, session_binding = await current_session(selected_content)
            sync_started = time.perf_counter()
            await session.synchronize()
            synchronization_ms = (time.perf_counter() - sync_started) * 1000
            scope = SearchScope(facet)
            response = await asyncio.to_thread(session.index.search, query, scope=scope, top_k=top_k)
        except Exception as exc:
            return _error_response("search", exc, binding_state.binding)
        context = _index_context(
            session_binding,
            session,
            selected_content,
            scope,
            session_acquire_ms=acquire_ms,
            synchronization_ms=synchronization_ms,
        )
        return json.dumps(_format_semantic_response(query, response, context, max_snippet_lines))

    @server.tool()
    async def semantic_index_status(
        include_changed_paths: Annotated[
            bool,
            Field(
                description=(
                    "Include the complete changed-path list. Leave false for compact coverage and freshness "
                    "metadata; use true only when the agent needs to inspect every modified, added, deleted, "
                    "or renamed path."
                )
            ),
        ] = False,
        content: Annotated[
            ContentSelection | None,
            Field(description="Content-index variant to inspect. Defaults to the server configuration."),
        ] = None,
    ) -> str:
        """Describe exactly what semantic_search can see in this Paseo workspace.

        Reports repository, worktree, active pinned baseline revision, content coverage, exclusions, baseline and
        delta sizes, change counts, publication generation, batching policy, and current freshness. This call also
        refreshes a clean committed HEAD advance and synchronizes pending Git-visible changes. It never accepts or
        indexes an arbitrary external path.
        """
        selected_content = _resolve_content_selection(content, default_content)
        try:
            session, acquire_ms, session_binding = await current_session(selected_content)
            sync_started = time.perf_counter()
            await session.synchronize()
            synchronization_ms = (time.perf_counter() - sync_started) * 1000
        except Exception as exc:
            return _error_response("inspect", exc, binding_state.binding)
        context = _index_context(
            session_binding,
            session,
            selected_content,
            SearchScope.WORKSPACE,
            session_acquire_ms=acquire_ms,
            synchronization_ms=synchronization_ms,
        )
        if include_changed_paths:
            context["changed_paths"] = [
                {"file_path": path, "change": change.value}
                for path, change in sorted(session.index.changed_paths.items())
            ]
        return json.dumps({"index_context": context})


def create_server(
    cache: _IndexCache,
    default_content: Sequence[ContentType] = (ContentType.CODE,),
    *,
    workspace_cache: _WorkspaceCache | None = None,
    binding: WorkspaceBinding | None = None,
    binding_error: str | None = None,
    binding_resolver: Callable[[], WorkspaceBinding] | None = None,
) -> FastMCP:
    """Build a context-bound semantic-search MCP server."""
    workspace_cache = workspace_cache or _WorkspaceCache(cache)
    server = FastMCP(
        "semble",
        instructions=(
            "Use semantic_search for semantic discovery in the current Paseo agent workspace. "
            "The server already knows the worktree, repository, and active committed revision; never guess or "
            "send paths. A clean committed HEAD advance automatically becomes the next immutable baseline. "
            "The default workspace facet is normally correct: it searches the private changed-file delta and "
            "untouched baseline independently, then returns changed_results and unchanged_results as separate "
            "sections. Use changed or unchanged only to reduce noise. Use base to inspect the complete active "
            "committed snapshot, including old versions of modified or deleted files. Every search synchronizes "
            "pending Git-visible edits before reading and reports exact index coverage and freshness. "
            "Use semantic_index_status when you need the full changed-path list or index diagnostics."
        ),
    )
    _register_semantic_tools(server, workspace_cache, default_content, binding, binding_error, binding_resolver)
    return server


async def serve(
    content: Sequence[ContentType] = (ContentType.CODE,),
) -> None:
    """Start a context-bound MCP stdio server for the current workspace."""
    cache = _IndexCache(watch=False)
    workspace_cache = _WorkspaceCache(cache)
    cache_root = resolve_cache_folder()

    async def _load_and_prewarm() -> None:
        try:
            _, cache._model_path = await asyncio.to_thread(load_model)
        except Exception as exc:
            logger.exception("Failed to load embedding model")
            cache._model_error = exc
        finally:
            cache._model_ready.set()

    init_task = asyncio.create_task(_load_and_prewarm())
    binding: WorkspaceBinding | None = None
    binding_error: str | None = None
    try:
        binding = await asyncio.to_thread(resolve_workspace_binding, Path.cwd(), cache_root)
    except Exception as exc:
        binding_error = str(exc)
        logger.warning("Semantic workspace binding is unavailable: %s", exc)
    server = create_server(
        cache,
        default_content=content,
        workspace_cache=workspace_cache,
        binding=binding,
        binding_error=binding_error,
        binding_resolver=lambda: resolve_workspace_binding(Path.cwd(), cache_root),
    )
    try:
        await server.run_stdio_async()
    finally:
        await workspace_cache.close()
        await cache.close()
        if not init_task.done():
            init_task.cancel()
        await asyncio.gather(init_task, return_exceptions=True)


class _IndexCache:
    """Cache indexed repositories and watch local paths for filesystem changes."""

    def __init__(self, *, watch: bool = True) -> None:
        """Initialise an empty cache.

        :param watch: Watch local source paths and invalidate affected indexes.
            Tests that exercise cache mechanics without filesystem events may
            disable watching explicitly.
        """
        self._model_path: str | None = None
        self._model_error: BaseException | None = None
        self._model_ready = asyncio.Event()
        self._tasks: OrderedDict[_CacheKey, asyncio.Task[SembleIndex]] = OrderedDict()
        self._dirty: set[_CacheKey] = set()
        self._watch_enabled = watch
        self._watch_tasks: dict[str, asyncio.Task[None]] = {}
        self._cache_root = resolve_cache_folder().resolve() if watch else None
        self._closed = False

    async def _await_model(self) -> str:
        """Block until the model is installed; re-raise the load error if it failed."""
        await self._model_ready.wait()
        if self._model_error is not None:
            raise self._model_error
        assert self._model_path is not None
        return self._model_path

    def _compute_cache_key(
        self,
        source: str,
        ref: str | None = None,
        content: Sequence[ContentType] = (ContentType.CODE,),
    ) -> _CacheKey:
        """Compute the canonical key for an exact index variant."""
        is_git = is_git_url(source)
        source_key = (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())
        normalized = tuple(content_type for content_type in ContentType if content_type in content)
        return source_key, normalized

    def _build_index(self, source: str, ref: str | None, model_path: str, cache_key: _CacheKey) -> SembleIndex:
        """Build an index for the given source and cache it."""
        source_key, content = cache_key
        index = (
            SembleIndex.from_git(source, ref=ref, model_path=model_path, content=content)
            if is_git_url(source)
            else SembleIndex.from_path(source_key, model_path=model_path, content=content)
        )
        try:
            save_index_to_cache(index, source_key)
        except Exception:
            logger.warning("Failed to save index cache for %r", source_key, exc_info=True)
        return index

    async def _build_tracked(
        self,
        source: str,
        ref: str | None,
        model_path: str,
        cache_key: _CacheKey,
    ) -> SembleIndex:
        """Build an index without blocking the MCP event loop."""
        return await asyncio.to_thread(self._build_index, source, ref, model_path, cache_key)

    @staticmethod
    def _change_affects_content(path: Path, content: Sequence[ContentType]) -> bool:
        """Return whether one changed path can affect an exact content index."""
        return path.name in _INDEX_RULE_FILES or path.suffix.lower() in get_extensions(content)

    def _mark_dirty(self, source_key: str, paths: set[Path] | None = None) -> None:
        """Mark loaded variants for one local source stale."""
        for cache_key in tuple(self._tasks):
            if cache_key[0] != source_key:
                continue
            if paths is None or any(self._change_affects_content(path, cache_key[1]) for path in paths):
                self._dirty.add(cache_key)

    async def _watch_source(self, source_key: str) -> None:
        """Watch one local source path and invalidate affected in-memory indexes."""
        current = asyncio.current_task()
        try:
            async for changes in awatch(source_key, debounce=_WATCH_DEBOUNCE_MS, step=50):
                paths = {
                    Path(path).resolve()
                    for _change, path in changes
                    if self._cache_root is None or not Path(path).resolve().is_relative_to(self._cache_root)
                }
                if paths:
                    self._mark_dirty(source_key, paths)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Filesystem watcher failed for %r; forcing revalidation", source_key, exc_info=True)
            self._mark_dirty(source_key)
        finally:
            if self._watch_tasks.get(source_key) is current:
                self._watch_tasks.pop(source_key, None)

    def _ensure_watcher(self, source_key: str) -> None:
        """Start one shared watcher for a local source path."""
        if not self._watch_enabled or self._closed or source_key in self._watch_tasks:
            return
        self._watch_tasks[source_key] = asyncio.create_task(
            self._watch_source(source_key),
            name=f"semble-watch:{source_key}",
        )

    def _stop_watcher_if_unused(self, source_key: str) -> None:
        """Stop a source watcher after its final exact index variant is evicted."""
        if any(cache_key[0] == source_key for cache_key in self._tasks):
            return
        watcher = self._watch_tasks.pop(source_key, None)
        if watcher is not None:
            watcher.cancel()

    def evict(self, cache_key: _CacheKey, *, keep_watcher: bool = False) -> None:
        """Evict one exact index variant."""
        self._tasks.pop(cache_key, None)
        self._dirty.discard(cache_key)
        if not keep_watcher:
            self._stop_watcher_if_unused(cache_key[0])

    async def close(self) -> None:
        """Stop every local filesystem watcher owned by this cache."""
        self._closed = True
        watchers = list(self._watch_tasks.values())
        self._watch_tasks.clear()
        for watcher in watchers:
            watcher.cancel()
        if watchers:
            await asyncio.gather(*watchers, return_exceptions=True)

    async def _task_for(
        self,
        source: str,
        ref: str | None,
        cache_key: _CacheKey,
    ) -> asyncio.Task[SembleIndex]:
        """Return the shared build task for an exact index variant."""
        if cache_key not in self._tasks:
            model_path = await self._await_model()
            if cache_key not in self._tasks:
                if len(self._tasks) >= _CACHE_MAX_SIZE:
                    self.evict(next(iter(self._tasks)))
                self._dirty.discard(cache_key)
                self._tasks[cache_key] = asyncio.create_task(self._build_tracked(source, ref, model_path, cache_key))
        self._tasks.move_to_end(cache_key)
        return self._tasks[cache_key]

    async def _await_index(
        self,
        cache_key: _CacheKey,
        task: asyncio.Task[SembleIndex],
    ) -> SembleIndex:
        """Await one build and evict failed tasks so a later request can retry."""
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                self.evict(cache_key)
            raise
        except Exception:
            if self._tasks.get(cache_key) is task:
                self.evict(cache_key)
            raise

    async def get(
        self,
        source: str,
        ref: str | None = None,
        content: Sequence[ContentType] = (ContentType.CODE,),
    ) -> SembleIndex:
        """Return an index, incrementally rebuilding dirty local-path variants."""
        cache_key = self._compute_cache_key(source, ref, content)
        local = not is_git_url(source)
        if local:
            self._ensure_watcher(cache_key[0])

        while True:
            if cache_key in self._dirty:
                self.evict(cache_key, keep_watcher=True)
            index = await self._await_index(cache_key, await self._task_for(source, ref, cache_key))
            if not local or cache_key not in self._dirty:
                return index


class _WorkspaceCache:
    """LRU cache of live Git workspace sessions sharing immutable baselines."""

    def __init__(self, index_cache: _IndexCache) -> None:
        """Create an empty workspace cache sharing the MCP model lifecycle."""
        self._index_cache = index_cache
        self._baselines = BaselineRegistry()
        self._tasks: OrderedDict[_WorkspaceKey, asyncio.Task[GitWorkspaceSession]] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(
        repo: str,
        baseline_repo: str,
        identity: BaselineIdentity,
        content: Sequence[ContentType],
    ) -> _WorkspaceKey:
        return (
            str(Path(repo).resolve()),
            str(Path(baseline_repo).resolve()),
            identity,
            tuple(content_type for content_type in ContentType if content_type in content),
        )

    async def _open(
        self,
        key: _WorkspaceKey,
    ) -> GitWorkspaceSession:
        repo, baseline_repo, identity, content = key
        model_path = await self._index_cache._await_model()
        session = await asyncio.to_thread(
            open_git_workspace,
            self._baselines,
            identity,
            baseline_root=baseline_repo,
            workspace_root=repo,
            content=content,
            model_path=model_path,
        )
        session.start()
        return session

    async def get(
        self,
        *,
        repo: str,
        baseline_repo: str,
        identity: BaselineIdentity,
        content: Sequence[ContentType],
    ) -> GitWorkspaceSession:
        """Return one live layered session for an exact workspace contract."""
        key = self._key(repo, baseline_repo, identity, content)
        async with self._lock:
            if key not in self._tasks:
                if len(self._tasks) >= _CACHE_MAX_SIZE:
                    _evicted_key, evicted = self._tasks.popitem(last=False)
                    await self._close_task(evicted)
                self._tasks[key] = asyncio.create_task(self._open(key))
            self._tasks.move_to_end(key)
            task = self._tasks[key]
        try:
            return await asyncio.shield(task)
        except Exception:
            async with self._lock:
                if self._tasks.get(key) is task:
                    self._tasks.pop(key, None)
            raise

    async def release_workspace(self, repo: str) -> int:
        """Release every exact session attached to one canonical worktree path."""
        canonical = str(Path(repo).resolve())
        async with self._lock:
            selected = [key for key in self._tasks if key[0] == canonical]
            tasks = [self._tasks.pop(key) for key in selected]
        for task in tasks:
            await self._close_task(task)
        return len(tasks)

    async def _close_task(self, task: asyncio.Task[GitWorkspaceSession]) -> None:
        try:
            session = await asyncio.shield(task)
        except Exception:
            return
        await session.close()
        self._baselines.evict_unused()

    async def close(self) -> None:
        """Close every workspace overlay and release all baseline references."""
        async with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for task in tasks:
            await self._close_task(task)
