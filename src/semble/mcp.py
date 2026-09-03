from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from watchfiles import awatch

from semble.cache import resolve_cache_folder, save_index_to_cache
from semble.index import SembleIndex
from semble.index.dense import load_model
from semble.index.files import get_extensions
from semble.types import ContentType
from semble.utils import format_results, is_git_url, resolve_chunk

logger = logging.getLogger(__name__)

_REPO_DESCRIPTION = (
    "A local directory path or https:// or http:// git URL (e.g. https://github.com/org/repo) to index and "
    "search. The index is cached after the first call, so repeat queries are fast."
)

_CACHE_MAX_SIZE = 10  # Max number of cached indexes to keep in memory
_WATCH_DEBOUNCE_MS = 200
_INDEX_RULE_FILES = frozenset({".gitignore", ".sembleignore"})
ContentSelection = Literal["code", "docs", "config", "all"]
_CacheKey = tuple[str, tuple[ContentType, ...]]


async def _get_index(repo: str, cache: _IndexCache, content: Sequence[ContentType]) -> SembleIndex:
    """Return a cached index for a repo, rejecting unsafe git transport schemes."""
    if is_git_url(repo) and not repo.startswith(("https://", "http://")):
        raise ValueError(f"Only https://, http://, or local directory paths are accepted as `repo`. Got: {repo!r}")
    try:
        return await cache.get(repo, content=content)
    except Exception as exc:
        raise ValueError(f"Failed to index {repo!r}: {exc}") from exc


def _resolve_content_selection(
    content: ContentSelection | None, default_content: Sequence[ContentType]
) -> tuple[ContentType, ...]:
    """Resolve an MCP content selection to exact index content types."""
    if content is None:
        return tuple(default_content)
    if content == "all":
        return tuple(ContentType)
    return (ContentType(content),)


def create_server(cache: _IndexCache, default_content: Sequence[ContentType] = (ContentType.CODE,)) -> FastMCP:
    """Build and return a configured FastMCP server backed by the given cache."""
    server = FastMCP(
        "semble",
        instructions=(
            "Instant code search for any local or remote git repository. "
            "Call `search` once with a focused query, it returns the file path and exact line. "
            "Navigate directly to that file at the given line; do not grep for the same content. "
            "Use `find_related` to discover similar code elsewhere in the same repo. "
            "When working in a local project, pass the project root as `repo`. "
            "For remote repos, pass an explicit https:// URL. Never guess or infer URLs."
        ),
    )

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code query.")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
        max_snippet_lines: Annotated[
            int | None,
            Field(
                description=(
                    "Lines of source to include per result. "
                    "Default (10): function/class signature + first body lines, enough to confirm the location. "
                    "0: file path and line range only. None: full chunk (~10-20 lines). "
                    "If the snippet does not contain enough context to confirm you have the right location, "
                    "call again with max_snippet_lines=None."
                ),
                ge=0,
            ),
        ] = 10,
        content: Annotated[
            ContentSelection | None,
            Field(description="Content to search. Defaults to the MCP server's configured content."),
        ] = None,
    ) -> str:
        """Search once with a focused query describing what the code does or its name.

        Write queries using function/class names or behavior descriptions, not error messages.
        Returns file paths and line numbers — navigate directly there, do not repeat the search.
        Pass a git URL or local path as `repo`; indexes are cached for the session.
        """
        selected_content = _resolve_content_selection(content, default_content)
        try:
            index = await _get_index(repo, cache, selected_content)
        except ValueError as exc:
            return str(exc)
        results = index.search(query, top_k=top_k, max_snippet_lines=max_snippet_lines)
        if not results:
            return json.dumps({"error": "No results found."})
        return json.dumps(format_results(query, results, max_snippet_lines))

    @server.tool()
    async def find_related(
        file_path: Annotated[
            str,
            Field(description="Path to the file as stored in the index (use file_path from a search result)."),
        ],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
        max_snippet_lines: Annotated[
            int | None,
            Field(
                description=(
                    "Lines of source per result. "
                    "Default 10 = signature + first body lines. 0 = location only. None = full chunk."
                ),
                ge=0,
            ),
        ] = 10,
        content: Annotated[
            ContentSelection | None,
            Field(description="Content containing the related file. Defaults to the MCP server configuration."),
        ] = None,
    ) -> str:
        """Find code similar to a known location.

        Useful for discovering all implementations of an interface, all callers of a function,
        or all tests for a class. Use after `search` when you need related code beyond the primary result.
        Pass `file_path` and `line` from a prior search result.
        """
        selected_content = _resolve_content_selection(content, default_content)
        try:
            index = await _get_index(repo, cache, selected_content)
        except ValueError as exc:
            return str(exc)
        chunk = resolve_chunk(index.chunks, file_path, line)
        if chunk is None:
            return (
                f"No chunk found at {file_path}:{line}. "
                "Make sure the file is indexed and the line number is within a known chunk."
            )
        results = index.find_related(chunk, top_k=top_k, max_snippet_lines=max_snippet_lines)
        if not results:
            return json.dumps({"error": f"No related chunks found for {file_path}:{line}."})
        label = f"Chunks related to {file_path}:{line}"
        return json.dumps(format_results(label, results, max_snippet_lines))

    return server


async def serve(
    content: Sequence[ContentType] = (ContentType.CODE,),
) -> None:
    """Start an MCP stdio server."""
    cache = _IndexCache()

    async def _load_and_prewarm() -> None:
        """Pre-load the embedding model in parallel with starting the server."""
        try:
            _, cache._model_path = await asyncio.to_thread(load_model)
        except Exception as exc:
            logger.exception("Failed to load embedding model")
            cache._model_error = exc
            return
        finally:
            cache._model_ready.set()

    init_task = asyncio.create_task(_load_and_prewarm())
    server = create_server(cache, default_content=content)
    try:
        await server.run_stdio_async()
    finally:
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
                self._tasks[cache_key] = asyncio.create_task(
                    self._build_tracked(source, ref, model_path, cache_key)
                )
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
