import asyncio
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from model2vec import StaticModel
from watchfiles import Change

from semble.mcp import _CACHE_MAX_SIZE, _IndexCache, create_server, serve
from semble.types import Chunk, ContentType, SearchResult
from semble.utils import format_results, is_git_url, resolve_chunk
from tests.conftest import make_chunk


def _tool_text(result: Any) -> str:
    """Extract the text string from a FastMCP call_tool result."""
    return result[0][0].text


async def _call_tool(
    cache: _IndexCache,
    tool: str,
    args: dict[str, Any],
    *,
    index_method: str,
    index_return: list[SearchResult],
    index_chunks: list[Chunk] | None = None,
) -> str:
    """Patch SembleIndex.from_path with a fake index and invoke the tool, returning the text."""
    fake_index = MagicMock()
    getattr(fake_index, index_method).return_value = index_return
    if index_chunks is not None:
        fake_index.chunks = index_chunks
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        server = create_server(cache)
        result = await server.call_tool(tool, args)
    return _tool_text(result)


@pytest.fixture()
def cache() -> _IndexCache:
    """An _IndexCache backed by a stub model."""
    c = _IndexCache(watch=False)
    c._model_path = "/fake/model"
    c._model_ready.set()
    return c


def test_resolve_chunk() -> None:
    """_resolve_chunk returns the correct chunk and handles boundary and miss cases."""
    interior = make_chunk("line1\nline2\nline3", "src/a.py")  # start=1, end=3
    boundary = make_chunk("last line", "src/a.py")  # start=1, end=1 (single-line)

    # Line strictly inside a multi-line chunk hits the early-return path.
    assert resolve_chunk([interior], "src/a.py", 2) is interior

    # Line equal to end_line of a single-line chunk hits the fallback path.
    assert resolve_chunk([boundary], "src/a.py", 1) is boundary

    # Unknown file returns None.
    assert resolve_chunk([interior], "src/other.py", 1) is None

    # Line out of range returns None.
    assert resolve_chunk([interior], "src/a.py", 99) is None

    # Separator mismatch (e.g. backslash-stored path, forward-slash query) still matches.
    backslash_chunk = make_chunk("line1\nline2\nline3", "src\\a.py")
    assert resolve_chunk([backslash_chunk], "src/a.py", 2) is backslash_chunk


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("https://github.com/org/repo", True),
        ("http://github.com/org/repo", True),
        ("git://github.com/org/repo", True),
        ("ssh://git@github.com/org/repo", True),
        ("git+ssh://git@github.com/org/repo", True),
        ("file:///tmp/repo", True),
        ("git@github.com:org/repo", True),  # scp-like
        ("/local/path/to/repo", False),
        ("./relative/path", False),
        ("repo_name", False),
    ],
)
def test_is_git_url(path: str, expected: bool) -> None:
    """Remote git URLs are detected; local paths are not."""
    assert is_git_url(path) is expected


@pytest.mark.parametrize(
    ("max_snippet_lines", "has_content", "content_key"),
    [
        (None, True, "content"),
        (3, True, "content"),
        (0, False, None),
    ],
    ids=["full", "truncated", "location_only"],
)
def test_format_results(max_snippet_lines: int | None, has_content: bool, content_key: str | None) -> None:
    """format_results: consistent flat schema regardless of max_snippet_lines."""
    empty_out = format_results("query", [], max_snippet_lines)
    assert empty_out == {"query": "query", "results": []}

    chunks = [make_chunk(f"line1\nline2\nline3\nline4\ndef fn_{i}(): pass", f"f{i}.py") for i in range(3)]
    results = [SearchResult(chunk=c, score=round(0.1 * (i + 1), 3)) for i, c in enumerate(chunks)]
    out = format_results("foo", results, max_snippet_lines)
    assert out["query"] == "foo"
    for entry in out["results"]:
        assert "file_path" in entry
        assert "start_line" in entry
        assert "end_line" in entry
        assert "score" in entry
        assert "chunk" not in entry
        if has_content:
            assert content_key in entry
            if max_snippet_lines is not None:
                assert entry[content_key].count("\n") < max_snippet_lines
        else:
            assert "content" not in entry


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("source", "patch_target"),
    [
        ("local_tmp_path", "from_path"),
        ("https://github.com/org/repo", "from_git"),
    ],
    ids=["local_path", "git_url"],
)
async def test_index_cache_builds_and_caches(
    cache: _IndexCache, tmp_path: Path, source: str, patch_target: str
) -> None:
    """_IndexCache.get() builds via the correct SembleIndex.* entrypoint and caches subsequent calls."""
    resolved_source = str(tmp_path) if source == "local_tmp_path" else source
    fake_index = MagicMock()
    with (
        patch(f"semble.mcp.SembleIndex.{patch_target}", return_value=fake_index) as mock_build,
        patch("semble.mcp.save_index_to_cache") as mock_save,
    ):
        first = await cache.get(resolved_source)
        second = await cache.get(resolved_source)
        docs_first = await cache.get(resolved_source, content=(ContentType.DOCS,))
        docs_second = await cache.get(resolved_source, content=(ContentType.DOCS,))
    assert first is fake_index
    assert second is fake_index
    assert docs_first is fake_index
    assert docs_second is fake_index
    assert [call.kwargs["content"] for call in mock_build.call_args_list] == [
        (ContentType.CODE,),
        (ContentType.DOCS,),
    ]
    assert mock_save.call_count == 2


@pytest.mark.anyio
async def test_index_cache_rebuilds_dirty_local_path(cache: _IndexCache, tmp_path: Path) -> None:
    """A relevant local file event invalidates the resident index immediately."""
    first_index = MagicMock()
    second_index = MagicMock()
    source = str(tmp_path.resolve())
    with (
        patch("semble.mcp.SembleIndex.from_path", side_effect=[first_index, second_index]) as mock_build,
        patch("semble.mcp.save_index_to_cache") as mock_save,
    ):
        assert await cache.get(source) is first_index
        cache._mark_dirty(source, {tmp_path / "changed.py"})
        assert await cache.get(source) is second_index
    assert mock_build.call_count == 2
    assert mock_save.call_count == 2


@pytest.mark.anyio
async def test_index_cache_scopes_file_events_to_affected_content(cache: _IndexCache, tmp_path: Path) -> None:
    """Source extensions invalidate matching variants; ignore-rule changes invalidate all variants."""
    source = str(tmp_path.resolve())
    with (
        patch("semble.mcp.SembleIndex.from_path", return_value=MagicMock()),
        patch("semble.mcp.save_index_to_cache"),
    ):
        await cache.get(source)
        await cache.get(source, content=(ContentType.DOCS,))

    code_key = cache._compute_cache_key(source)
    docs_key = cache._compute_cache_key(source, content=(ContentType.DOCS,))
    cache._mark_dirty(source, {tmp_path / "changed.py"})
    assert code_key in cache._dirty
    assert docs_key not in cache._dirty

    cache._mark_dirty(source, {tmp_path / ".gitignore"})
    assert code_key in cache._dirty
    assert docs_key in cache._dirty


@pytest.mark.anyio
async def test_index_cache_reconciles_change_during_build(cache: _IndexCache, tmp_path: Path) -> None:
    """A file event arriving during a build forces one shared follow-up reconciliation."""
    build_started = threading.Event()
    release_build = threading.Event()
    first_index = MagicMock()
    second_index = MagicMock()
    calls = 0

    def build(_path: str, **_kwargs: object) -> MagicMock:
        nonlocal calls
        calls += 1
        if calls == 1:
            build_started.set()
            assert release_build.wait(timeout=2.0)
            return first_index
        return second_index

    source = str(tmp_path.resolve())
    with (
        patch("semble.mcp.SembleIndex.from_path", side_effect=build),
        patch("semble.mcp.save_index_to_cache"),
    ):
        request = asyncio.create_task(cache.get(source))
        assert await asyncio.to_thread(build_started.wait, 1.0)
        cache._mark_dirty(source, {tmp_path / "changed.py"})
        release_build.set()
        assert await asyncio.wait_for(request, timeout=2.0) is second_index
    assert calls == 2


@pytest.mark.parametrize("change", list(Change))
@pytest.mark.anyio
async def test_index_cache_watcher_marks_changes_and_closes(change: Change, tmp_path: Path) -> None:
    """The local watcher marks relevant indexes dirty and is cancelled during cache shutdown."""
    source = str(tmp_path.resolve())
    watcher_started = asyncio.Event()
    events: asyncio.Queue[set[tuple[Change, str]]] = asyncio.Queue()

    async def fake_awatch(*_paths: str, **_kwargs: object):  # type: ignore[no-untyped-def]
        watcher_started.set()
        yield await events.get()
        await asyncio.Future()

    async def wait_until_dirty(cache: _IndexCache, key: object) -> None:
        while key not in cache._dirty:
            await asyncio.sleep(0)

    with (
        patch("semble.mcp.resolve_cache_folder", return_value=tmp_path / "cache"),
        patch("semble.mcp.awatch", new=fake_awatch),
        patch("semble.mcp.SembleIndex.from_path", return_value=MagicMock()),
        patch("semble.mcp.save_index_to_cache"),
    ):
        cache = _IndexCache()
        cache._model_path = "/fake/model"
        cache._model_ready.set()
        await cache.get(source)
        await asyncio.wait_for(watcher_started.wait(), timeout=1.0)
        key = cache._compute_cache_key(source)
        watcher = cache._watch_tasks[source]
        await events.put({(change, str(tmp_path / "changed.py"))})
        await asyncio.wait_for(wait_until_dirty(cache, key), timeout=1.0)
        await cache.close()

    assert not cache._watch_tasks
    assert watcher.cancelled()

@pytest.mark.anyio
async def test_index_cache_evicts_on_failure(cache: _IndexCache, tmp_path: Path) -> None:
    """A failed build evicts the entry so the next call can retry."""
    call_count = 0

    def _failing_then_ok(path: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("build failed")
        return MagicMock()

    with patch("semble.mcp.SembleIndex.from_path", side_effect=_failing_then_ok):
        with pytest.raises(RuntimeError, match="build failed"):
            await cache.get(str(tmp_path))
        result = await cache.get(str(tmp_path))
    assert result is not None
    assert call_count == 2


@pytest.mark.anyio
async def test_index_cache_ignores_cache_save_failure(cache: _IndexCache, tmp_path: Path) -> None:
    """A cache save failure must not fail the MCP request."""
    fake_index = MagicMock()
    with (
        patch("semble.mcp.SembleIndex.from_path", return_value=fake_index),
        patch("semble.mcp.save_index_to_cache", side_effect=RuntimeError("save failed")),
    ):
        assert await cache.get(str(tmp_path)) is fake_index


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("search", {"query": "foo", "repo": "https://github.com/x/y"}),
        ("find_related", {"file_path": "src/foo.py", "line": 1, "repo": "https://github.com/x/y"}),
    ],
)
async def test_tool_index_failure(cache: _IndexCache, tool: str, args: dict[str, object]) -> None:
    """Both tools return a friendly error message when indexing fails."""
    with patch("semble.mcp.SembleIndex.from_git", side_effect=RuntimeError("clone failed")):
        server = create_server(cache)
        result = await server.call_tool(tool, args)
    text = _tool_text(result)
    assert "Failed to index" in text
    assert "clone failed" in text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args", "method", "results", "chunks", "expected_substrings"),
    [
        pytest.param(
            "search",
            {"query": "bar", "repo": "/some/path"},
            "search",
            [SearchResult(chunk=make_chunk("def bar(): pass", "src/bar.py"), score=0.9)],
            None,
            ["bar", "0.9"],
            id="search_with_results",
        ),
        pytest.param(
            "search",
            {"query": "nothing", "repo": "/some/path"},
            "search",
            [],
            None,
            ["No results found"],
            id="search_no_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/foo.py", "line": 1, "repo": "/some/path"},
            "find_related",
            [SearchResult(chunk=make_chunk("class Foo: pass", "src/foo.py"), score=0.8)],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["src/foo.py:1", "0.8"],
            id="find_related_with_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/foo.py", "line": 1, "repo": "/some/path"},
            "find_related",
            [],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["No related chunks found"],
            id="find_related_no_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/unknown.py", "line": 1, "repo": "/some/path"},
            "find_related",
            [],
            [],
            ["No chunk found"],
            id="find_related_unknown_file",
        ),
    ],
)
async def test_tool_output(
    cache: _IndexCache,
    tool: str,
    args: dict[str, Any],
    method: str,
    results: list[SearchResult],
    chunks: list[Chunk] | None,
    expected_substrings: list[str],
) -> None:
    """Search and find_related format results (or an empty-state message) through the server."""
    text = await _call_tool(cache, tool, args, index_method=method, index_return=results, index_chunks=chunks)
    for substring in expected_substrings:
        assert substring in text


@pytest.mark.anyio
async def test_search_builds_exact_content_indexes(
    cache: _IndexCache,
    mock_model: StaticModel,
    tmp_project: Path,
) -> None:
    """MCP search lazily builds the exact requested content index."""
    (tmp_project / "settings.toml").write_text("project = 'semble'\n")
    expected = [
        (None, {".py"}),
        ("docs", {".md"}),
        ("config", {".toml"}),
        ("all", {".md", ".py", ".toml"}),
    ]

    with (
        patch("semble.index.index.load_model", return_value=(mock_model, "/fake/model")),
        patch("semble.mcp.save_index_to_cache"),
    ):
        server = create_server(cache)
        for content, expected_suffixes in expected:
            args = {"query": "project", "repo": str(tmp_project), "top_k": 20}
            if content is not None:
                args["content"] = content
            result = await server.call_tool("search", args)
            payload = json.loads(_tool_text(result))
            assert {Path(item["file_path"]).suffix for item in payload["results"]} == expected_suffixes


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("load_err", "stdio_yields"),
    [
        (None, True),
        (RuntimeError("boom"), True),
        (None, False),
    ],
    ids=["model_loads", "model_load_fails", "cancel_pending_init"],
)
async def test_serve_runs_stdio(
    load_err: Exception | None,
    stdio_yields: bool,
) -> None:
    """serve() runs stdio and handles all background init outcomes without raising."""

    async def fake_stdio() -> None:
        if stdio_yields:
            await asyncio.sleep(0.05)  # let the background init task run

    load_kwargs = (
        {"side_effect": load_err} if load_err else {"return_value": (MagicMock(spec=StaticModel), "/fake/model")}
    )
    with (
        patch("semble.mcp.load_model", **load_kwargs),
        patch("mcp.server.fastmcp.FastMCP.run_stdio_async", side_effect=fake_stdio) as mock_run,
    ):
        await serve()

    mock_run.assert_called_once()


@pytest.mark.anyio
async def test_serve_opens_stdio_before_model_loads() -> None:
    """Stdio must open before load_model() finishes."""
    stdio_opened = threading.Event()

    def blocking_load_model() -> StaticModel:
        assert stdio_opened.wait(timeout=1.0), "stdio did not open"
        return MagicMock(spec=StaticModel)

    async def fake_run_stdio() -> None:
        stdio_opened.set()
        await asyncio.sleep(0.05)

    with (
        patch("semble.mcp.load_model", side_effect=blocking_load_model),
        patch("mcp.server.fastmcp.FastMCP.run_stdio_async", side_effect=fake_run_stdio),
    ):
        await serve()


@pytest.mark.anyio
async def test_index_cache_awaits_model(tmp_path: Path) -> None:
    """get() blocks until the model is installed, then proceeds."""
    cache = _IndexCache()  # no model yet
    fake_index = MagicMock()
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        get_task = asyncio.create_task(cache.get(str(tmp_path)))
        await asyncio.sleep(0.01)
        assert not get_task.done(), "get() must block until the model is installed"
        cache._model_path = "/fake/model"
        cache._model_ready.set()
        result = await asyncio.wait_for(get_task, timeout=1.0)
    assert result is fake_index


@pytest.mark.anyio
async def test_index_cache_propagates_model_error(tmp_path: Path) -> None:
    """If model load fails, awaiting tool calls re-raise the original exception."""
    cache = _IndexCache()
    get_task = asyncio.create_task(cache.get(str(tmp_path)))
    await asyncio.sleep(0.01)
    assert not get_task.done()
    cache._model_error = RuntimeError("HF download failed")
    cache._model_ready.set()
    with pytest.raises(RuntimeError, match="HF download failed"):
        await asyncio.wait_for(get_task, timeout=1.0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("repo", "tool", "extra_args"),
    [
        ("file:///home/user/secret", "search", {"query": "foo"}),
        ("ssh://internal-host/repo", "search", {"query": "foo"}),
        ("git@github.com:org/repo", "search", {"query": "foo"}),
        ("file:///home/user/secret", "find_related", {"file_path": "src/foo.py", "line": 1}),
        ("ssh://internal-host/repo", "find_related", {"file_path": "src/foo.py", "line": 1}),
    ],
    ids=["file_search", "ssh_search", "scp_search", "file_find_related", "ssh_find_related"],
)
async def test_tool_rejects_unsafe_repo(
    cache: _IndexCache, repo: str, tool: str, extra_args: dict[str, object]
) -> None:
    """Both tools reject unsafe git transport schemes (ssh://, file://, SCP-form) supplied as repo."""
    server = create_server(cache)
    result = await server.call_tool(tool, {**extra_args, "repo": repo})
    assert "Only https://" in _tool_text(result)


@pytest.mark.anyio
async def test_index_cache_lru_eviction(cache: _IndexCache, tmp_path: Path) -> None:
    """_IndexCache evicts the least-recently-used entry when the cache is full."""
    dirs = [tmp_path / str(i) for i in range(_CACHE_MAX_SIZE + 1)]
    for d in dirs:
        d.mkdir()
    with patch("semble.mcp.SembleIndex.from_path", return_value=MagicMock()):
        for d in dirs[:_CACHE_MAX_SIZE]:
            await cache.get(str(d))
        first_key = cache._compute_cache_key(str(dirs[0]))
        assert first_key in cache._tasks
        await cache.get(str(dirs[_CACHE_MAX_SIZE]))
    assert first_key not in cache._tasks
    assert len(cache._tasks) == _CACHE_MAX_SIZE


def test_cache_evict(cache: _IndexCache, tmp_path: Path) -> None:
    """evict() removes an existing exact cache entry."""
    key = cache._compute_cache_key(str(tmp_path))
    cache._tasks[key] = MagicMock()
    cache.evict(key)
    assert key not in cache._tasks


def test_cache_evict_missing(cache: _IndexCache, tmp_path: Path) -> None:
    """evict() on an unknown key is a no-op."""
    cache.evict(cache._compute_cache_key(str(tmp_path)))  # should not raise
