import asyncio
import json
import subprocess
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from model2vec import StaticModel
from watchfiles import Change

from semble.git_workspace import WorkspaceBinding
from semble.mcp import _CACHE_MAX_SIZE, _get_index, _IndexCache, _WorkspaceCache, create_server, serve
from semble.types import Chunk, ContentType, SearchResult
from semble.utils import format_results, is_git_url, resolve_chunk
from semble.workspace import (
    BaselineIdentity,
    ChangeKind,
    SearchOrigin,
    SearchScope,
    WorkspaceSearchHit,
    WorkspaceSearchResponse,
)
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


def _semantic_server(
    cache: _IndexCache,
    tmp_path: Path,
    *,
    scope: SearchScope = SearchScope.WORKSPACE,
) -> tuple[Any, MagicMock, MagicMock, WorkspaceBinding]:
    identity = BaselineIdentity("https://github.com/example/project.git", "a" * 40)
    binding = WorkspaceBinding(
        workspace_root=tmp_path / "worktree",
        baseline_root=tmp_path / "baseline",
        identity=identity,
        branch="agent-task",
        baseline_source="primary_checkout",
    )
    response = WorkspaceSearchResponse(
        baseline=identity,
        scope=scope,
        changed_file_count=1,
        base_results=(
            WorkspaceSearchHit(
                SearchResult(make_chunk("def stable(): pass", "stable.py"), 0.7),
                SearchOrigin.BASE,
                ChangeKind.UNCHANGED,
            ),
        ),
        delta_results=(
            WorkspaceSearchHit(
                SearchResult(make_chunk("def changed(): pass", "changed.py"), 0.9),
                SearchOrigin.DELTA,
                ChangeKind.MODIFIED,
            ),
        ),
    )
    session = MagicMock()
    session.synchronize = AsyncMock()
    session.index.search.return_value = response
    session.index.changed_paths = {"changed.py": ChangeKind.MODIFIED}
    session.index.generation = 4
    session.index.baseline_file_count = 120
    session.index.baseline_chunk_count = 840
    session.index.delta_file_count = 1
    session.index.delta_chunk_count = 3
    session.index.shadowed_file_count = 1
    workspace_cache = MagicMock()
    workspace_cache.get = AsyncMock(return_value=session)
    server = create_server(cache, workspace_cache=workspace_cache, binding=binding)
    return server, workspace_cache, session, binding


@pytest.mark.anyio
async def test_semantic_search_defaults_to_current_workspace_with_separate_facets(
    cache: _IndexCache,
    tmp_path: Path,
) -> None:
    """The primary tool is path-free, current, and provenance-separated by default."""
    server, workspace_cache, session, binding = _semantic_server(cache, tmp_path)

    result = await server.call_tool("semantic_search", {"query": "changed behavior"})
    payload = json.loads(_tool_text(result))

    workspace_cache.get.assert_awaited_once_with(
        repo=str(binding.workspace_root),
        baseline_repo=str(binding.baseline_root),
        identity=binding.identity,
        content=(ContentType.CODE,),
    )
    session.synchronize.assert_awaited_once()
    session.index.search.assert_called_once_with(
        "changed behavior",
        scope=SearchScope.WORKSPACE,
        top_k=5,
    )
    assert payload["changed_results"][0]["file_path"] == "changed.py"
    assert payload["changed_results"][0]["origin"] == "delta"
    assert payload["unchanged_results"][0]["file_path"] == "stable.py"
    assert payload["unchanged_results"][0]["origin"] == "base"
    assert not payload["base_results"]
    context = payload["index_context"]
    assert context["facet"] == "workspace"
    assert context["baseline_revision"] == "a" * 40
    assert context["index_state"] == "current"
    assert context["read_your_writes"] is True
    assert context["generation"] == 4
    assert context["baseline_index"] == {"immutable": True, "files": 120, "chunks": 840}
    assert context["delta_index"]["changed_paths"] == 1
    assert context["watcher"]["quiet_window_ms"] == 50
    assert context["watcher"]["maximum_batch_window_ms"] == 200


@pytest.mark.anyio
async def test_semantic_search_immediate_first_call_reads_new_workspace_bytes(
    cache: _IndexCache,
    tmp_path: Path,
    mock_model: StaticModel,
) -> None:
    """A post-write search synchronizes the delta and preserves the immutable base facet."""
    baseline = tmp_path / "baseline"
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "init", "-b", "main", str(baseline)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(baseline), "config", "user.name", "Semble Test"], check=True)
    subprocess.run(["git", "-C", str(baseline), "config", "user.email", "semble@example.invalid"], check=True)
    (baseline / "auth.py").write_text("def authenticate():\n    return 'original amber credential'\n")
    subprocess.run(["git", "-C", str(baseline), "add", "."], check=True)
    subprocess.run(["git", "-C", str(baseline), "commit", "-m", "base"], check=True, capture_output=True)
    revision = subprocess.run(
        ["git", "-C", str(baseline), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "clone", str(baseline), str(worktree)], check=True, capture_output=True)
    binding = WorkspaceBinding(
        workspace_root=worktree,
        baseline_root=baseline,
        identity=BaselineIdentity("example/project", revision),
        branch="main",
        baseline_source="primary_checkout",
    )
    workspace_cache = _WorkspaceCache(cache)
    server = create_server(cache, workspace_cache=workspace_cache, binding=binding)

    with (
        patch("semble.index.index.load_model", return_value=(mock_model, "/fake/model")),
        patch("semble.cache.save_index_to_cache"),
    ):
        initial = json.loads(
            _tool_text(await server.call_tool("semantic_search", {"query": "original amber credential"}))
        )
        assert initial["unchanged_results"][0]["file_path"] == "auth.py"

        (worktree / "auth.py").write_text("def authenticate():\n    return 'current violet credential'\n")
        current = json.loads(
            _tool_text(await server.call_tool("semantic_search", {"query": "current violet credential"}))
        )
        assert current["changed_results"][0]["file_path"] == "auth.py"
        assert current["changed_results"][0]["origin"] == "delta"
        assert not current["unchanged_results"]
        assert current["index_context"]["generation"] == 1
        assert current["index_context"]["read_your_writes"] is True

        original = json.loads(
            _tool_text(
                await server.call_tool(
                    "semantic_search",
                    {"query": "original amber credential", "facet": "base"},
                )
            )
        )
        assert original["base_results"][0]["file_path"] == "auth.py"
        assert original["base_results"][0]["change"] == "modified"

    await workspace_cache.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("facet", "scope", "populated"),
    [
        ("changed", SearchScope.CHANGED, "changed_results"),
        ("unchanged", SearchScope.UNCHANGED, "unchanged_results"),
        ("base", SearchScope.BASE, "base_results"),
    ],
)
async def test_semantic_search_facets_remove_noise_without_model_paths(
    cache: _IndexCache,
    tmp_path: Path,
    facet: str,
    scope: SearchScope,
    populated: str,
) -> None:
    """Narrow facets retain one stable response schema and never accept repository context."""
    server, _workspace_cache, session, _binding = _semantic_server(cache, tmp_path, scope=scope)

    result = await server.call_tool(
        "semantic_search",
        {"query": "focused behavior", "facet": facet, "max_snippet_lines": 0},
    )
    payload = json.loads(_tool_text(result))

    session.index.search.assert_called_once_with("focused behavior", scope=scope, top_k=5)
    assert payload["index_context"]["facet"] == facet
    assert payload[populated]
    for section in {"changed_results", "unchanged_results", "base_results"} - {populated}:
        assert not payload[section]


@pytest.mark.anyio
async def test_semantic_index_status_explains_coverage_and_changed_paths(
    cache: _IndexCache,
    tmp_path: Path,
) -> None:
    """Status makes current scope, exclusions, physical indexes, and freshness explicit."""
    server, _workspace_cache, session, binding = _semantic_server(cache, tmp_path)

    result = await server.call_tool("semantic_index_status", {"include_changed_paths": True})
    context = json.loads(_tool_text(result))["index_context"]

    session.synchronize.assert_awaited_once()
    assert context["repository"] == binding.identity.repository
    assert context["workspace_root"] == str(binding.workspace_root)
    assert context["content"] == ["code"]
    assert context["delta_index"]["changes_by_kind"]["modified"] == 1
    assert context["changed_paths"] == [{"file_path": "changed.py", "change": "modified"}]
    assert "paths outside the current workspace" in context["coverage"]["excluded"]


@pytest.mark.anyio
async def test_semantic_tool_catalog_is_unambiguous_and_context_bound(
    cache: _IndexCache,
    tmp_path: Path,
) -> None:
    """Agents see only descriptive semantic tools and cannot provide index identity."""
    server, _workspace_cache, _session, _binding = _semantic_server(cache, tmp_path)

    tools = {tool.name: tool for tool in await server.list_tools()}

    assert set(tools) == {"semantic_search", "semantic_index_status"}
    search = tools["semantic_search"]
    assert "immutable baseline" in search.description
    assert "private changed-file delta" in search.description
    assert "read-your-writes" in search.description
    assert set(search.inputSchema["properties"]) == {
        "query",
        "facet",
        "top_k",
        "max_snippet_lines",
        "content",
    }
    assert search.inputSchema["properties"]["facet"]["default"] == "workspace"
    assert "old versions" in search.inputSchema["properties"]["facet"]["description"]


@pytest.mark.anyio
async def test_semantic_search_reports_binding_failure_instead_of_guessing(
    cache: _IndexCache,
) -> None:
    """A server outside a bound Git workspace never asks the model for a path."""
    server = create_server(cache, binding_error="not inside a Git workspace")

    result = await server.call_tool("semantic_search", {"query": "anything"})
    payload = json.loads(_tool_text(result))

    assert payload["index_context"]["index_state"] == "error"
    assert payload["index_context"]["workspace_root"] is None
    assert "not inside a Git workspace" in payload["detail"]


@pytest.mark.anyio
async def test_workspace_cache_reuses_and_releases_exact_session(cache: _IndexCache, tmp_path: Path) -> None:
    """An exact workspace contract opens once and release closes its layered session."""
    workspace_cache = _WorkspaceCache(cache)
    session = MagicMock()
    session.close = AsyncMock()
    identity = BaselineIdentity("example/repository", "b" * 40)
    repo = str(tmp_path / "worktree")
    baseline = str(tmp_path / "baseline")
    with patch("semble.mcp.open_git_workspace", return_value=session) as mock_open:
        first = await workspace_cache.get(
            repo=repo,
            baseline_repo=baseline,
            identity=identity,
            content=(ContentType.CODE,),
        )
        second = await workspace_cache.get(
            repo=repo,
            baseline_repo=baseline,
            identity=identity,
            content=(ContentType.CODE,),
        )
    assert first is second is session
    mock_open.assert_called_once()
    session.start.assert_called_once()
    assert await workspace_cache.release_workspace(repo) == 1
    session.close.assert_awaited_once()
    assert await workspace_cache.release_workspace(repo) == 0


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
        patch("semble.mcp.resolve_workspace_binding", side_effect=RuntimeError("not bound")),
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
        patch("semble.mcp.resolve_workspace_binding", side_effect=RuntimeError("not bound")),
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
    "repo",
    [
        "file:///home/user/secret",
        "ssh://internal-host/repo",
        "git@github.com:org/repo",
    ],
)
async def test_explicit_index_cache_rejects_unsafe_repo(cache: _IndexCache, repo: str) -> None:
    """The retained non-agent cache helper still rejects unsafe Git transports."""
    with pytest.raises(ValueError, match="Only https://"):
        await _get_index(repo, cache, (ContentType.CODE,))


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
