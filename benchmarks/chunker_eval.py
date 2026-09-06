"""Evaluate Semble's current chunker against the native legacy cAST chunker.

This is an evaluation-only integration. It replaces the chunking seam used by
index creation without changing Semble's production indexing, ranking, model,
or cache formats.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from benchmarks.data import (
    BENCHMARKS_DIR,
    RepoSpec,
    Task,
    add_filter_args,
    available_repo_specs,
    current_sha,
    grouped_tasks,
    load_filtered_tasks,
)
from benchmarks.metrics import ndcg_at_k, target_rank
from semble import SembleIndex
from semble.index import create as create_module
from semble.index.file_walker import walk_files
from semble.index.files import FileStatus, get_extensions, get_file_status
from semble.types import Chunk, ContentType, SearchResult
from semble.utils import DEFAULT_MODEL_NAME

_LATENCY_RUNS = 5
_TOP_K = 10
_NATIVE_MODULE = "code_chunker_native"
_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ChunkerEvalError(RuntimeError):
    """A configuration or native-boundary failure that invalidates an evaluation."""


@dataclass
class ChunkSourceCounters:
    """Aggregate observations at Semble's chunk-source boundary."""

    calls: int = 0
    wall_ns: int = 0
    native_files: int = 0
    native_chunks: int = 0
    fallback_files: int = 0
    fallback_chunks: int = 0
    native_errors: int = 0

    def mark(self) -> tuple[int, ...]:
        """Return an immutable counter snapshot."""
        return (
            self.calls,
            self.wall_ns,
            self.native_files,
            self.native_chunks,
            self.fallback_files,
            self.fallback_chunks,
            self.native_errors,
        )

    @staticmethod
    def delta(before: tuple[int, ...], after: tuple[int, ...]) -> dict[str, int | float]:
        """Return a JSON-compatible difference between two snapshots."""
        calls, wall_ns, native_files, native_chunks, fallback_files, fallback_chunks, native_errors = (
            end - start for start, end in zip(before, after, strict=True)
        )
        return {
            "calls": calls,
            "wall_ms": wall_ns / 1_000_000,
            "mean_ms": wall_ns / calls / 1_000_000 if calls else 0.0,
            "native_files": native_files,
            "native_chunks": native_chunks,
            "fallback_files": fallback_files,
            "fallback_chunks": fallback_chunks,
            "native_errors": native_errors,
        }

    def as_dict(self) -> dict[str, int | float]:
        """Return all counters as JSON-compatible values."""
        zero = (0,) * len(self.mark())
        return self.delta(zero, self.mark())


class VariantChunker:
    """Dispatch Semble chunk-source calls to the selected implementation."""

    def __init__(
        self,
        native: Any | None,
        max_size: int,
        min_size: int,
        counters: ChunkSourceCounters,
    ) -> None:
        """Capture the native module, size configuration, and shared counters."""
        self._native = native
        self._max_size = max_size
        self._min_size = min_size
        self._counters = counters
        self._current = create_module.chunk_source

    def __call__(self, source: str, file_path: str, language: str | None) -> list[Chunk]:
        """Chunk one source with native dispatch only when its path is supported."""
        started = time.perf_counter_ns()
        try:
            if self._native is None:
                chunks = self._current(source, file_path, language)
                self._counters.fallback_files += 1
                self._counters.fallback_chunks += len(chunks)
                return chunks

            try:
                using_native = bool(self._native.supports_path(file_path))
            except Exception as exc:
                self._counters.native_errors += 1
                raise ChunkerEvalError(f"native support check failed for {file_path!r}: {exc}") from exc

            if not using_native:
                chunks = self._current(source, file_path, language)
                self._counters.fallback_files += 1
                self._counters.fallback_chunks += len(chunks)
                return chunks

            try:
                native_chunks = self._native.chunk_source(
                    source,
                    file_path,
                    self._max_size,
                    self._min_size,
                )
                chunks = [
                    Chunk(
                        content=native_chunk.content,
                        file_path=file_path,
                        start_line=native_chunk.start_line,
                        end_line=native_chunk.end_line,
                        language=language,
                    )
                    for native_chunk in native_chunks
                ]
            except Exception as exc:
                self._counters.native_errors += 1
                raise ChunkerEvalError(f"native chunking failed for {file_path!r}: {exc}") from exc

            self._counters.native_files += 1
            self._counters.native_chunks += len(chunks)
            return chunks
        finally:
            self._counters.calls += 1
            self._counters.wall_ns += time.perf_counter_ns() - started


@dataclass(frozen=True)
class RepoEvaluation:
    """Quality, latency, and indexing observations for one repository."""

    repo: str
    language: str
    mode: str
    files: int
    chunks: int
    tokens: int
    ndcg5: float
    ndcg10: float
    recall10: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    index_ms: float
    chunk_source: dict[str, int | float]
    by_category: dict[str, float]


def _load_native() -> Any:
    """Import and validate the fixed Python boundary for the native chunker."""
    try:
        native = importlib.import_module(_NATIVE_MODULE)
    except ImportError as exc:
        raise ChunkerEvalError(
            f"{_NATIVE_MODULE!r} is not installed for {sys.executable}. Build the "
            "code-chunker-python crate with maturin and install its wheel into this environment."
        ) from exc

    required = ("supports_path", "chunk_source", "chunk_files")
    missing = [name for name in required if not callable(getattr(native, name, None))]
    if missing:
        raise ChunkerEvalError(f"{_NATIVE_MODULE!r} is missing API members: {', '.join(missing)}")
    return native


@contextlib.contextmanager
def _isolated_cache() -> Iterator[Path]:
    """Use a fresh cache location so every variant receives a cold index build."""
    cache_dir = Path(tempfile.mkdtemp(prefix="semble-chunker-eval-"))
    previous = os.environ.get("SEMBLE_CACHE_LOCATION")
    os.environ["SEMBLE_CACHE_LOCATION"] = str(cache_dir)
    try:
        yield cache_dir
    finally:
        if previous is None:
            os.environ.pop("SEMBLE_CACHE_LOCATION", None)
        else:
            os.environ["SEMBLE_CACHE_LOCATION"] = previous
        shutil.rmtree(cache_dir, ignore_errors=True)


def _evaluate_queries(
    index: SembleIndex,
    tasks: list[Task],
    *,
    latency_runs: int,
    verbose: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mirror the hybrid benchmark and retain exact per-query outcomes."""
    ndcg5_values: list[float] = []
    ndcg10_values: list[float] = []
    recall_values: list[float] = []
    latencies: list[float] = []
    tokens: list[int] = []
    categories: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for ordinal, task in enumerate(tasks):
        query_latencies: list[float] = []
        results: list[SearchResult] = []
        for _ in range(latency_runs):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_TOP_K, alpha=None, rerank=True)
            query_latencies.append((time.perf_counter() - started) * 1000)

        target_ranks = [target_rank(results, target) for target in task.all_relevant]
        relevant_ranks = [rank for rank in target_ranks if rank is not None]
        n_relevant = len(task.all_relevant)
        q_ndcg5 = ndcg_at_k(relevant_ranks, n_relevant, 5)
        q_ndcg10 = ndcg_at_k(relevant_ranks, n_relevant, _TOP_K)
        q_recall10 = len(relevant_ranks) / n_relevant if n_relevant else 0.0
        q_latency = float(np.median(query_latencies))
        q_tokens = sum(len(result.chunk.content) // 4 for result in results)
        category = task.category or "unknown"

        ndcg5_values.append(q_ndcg5)
        ndcg10_values.append(q_ndcg10)
        recall_values.append(q_recall10)
        latencies.append(q_latency)
        tokens.append(q_tokens)
        categories[category].append(q_ndcg10)
        rows.append(
            {
                "repo": task.repo,
                "ordinal": ordinal,
                "language": task.language,
                "category": category,
                "query": task.query,
                "targets": [asdict(target) for target in task.all_relevant],
                "target_ranks": target_ranks,
                "ndcg5": q_ndcg5,
                "ndcg10": q_ndcg10,
                "recall10": q_recall10,
                "p50_ms": q_latency,
                "tokens": q_tokens,
                "top": [
                    {
                        "file": result.chunk.file_path,
                        "start_line": result.chunk.start_line,
                        "end_line": result.chunk.end_line,
                        "score": result.score,
                    }
                    for result in results
                ],
            }
        )

        if verbose:
            print(
                f"  [{category:<12}] ndcg@10={q_ndcg10:.3f} ranks={relevant_ranks} q={task.query!r}",
                file=sys.stderr,
            )

    p50, p90, p95, p99 = np.percentile(latencies, [50, 90, 95, 99]).tolist()
    return (
        {
            "ndcg5": statistics.fmean(ndcg5_values),
            "ndcg10": statistics.fmean(ndcg10_values),
            "recall10": statistics.fmean(recall_values),
            "p50_ms": p50,
            "p90_ms": p90,
            "p95_ms": p95,
            "p99_ms": p99,
            "tokens": sum(tokens) // len(tokens),
            "by_category": {
                category: statistics.fmean(values) for category, values in sorted(categories.items())
            },
        },
        rows,
    )


def _mean(group: list[RepoEvaluation], field: str) -> float:
    """Return the arithmetic mean of one numeric RepoEvaluation field."""
    return statistics.fmean(float(getattr(row, field)) for row in group)


def _aggregate(repos: list[RepoEvaluation], queries: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build baseline-compatible macro summaries plus query-level totals."""
    by_language_rows: dict[str, list[RepoEvaluation]] = defaultdict(list)
    for row in repos:
        by_language_rows[row.language].append(row)

    by_language = {
        language: {
            "repos": len(group),
            "files": sum(row.files for row in group),
            "chunks": sum(row.chunks for row in group),
            "tokens": round(_mean(group, "tokens"), 0),
            "ndcg10": round(_mean(group, "ndcg10"), 4),
            "recall10": round(_mean(group, "recall10"), 4),
            "p50_ms": round(_mean(group, "p50_ms"), 3),
            "p90_ms": round(_mean(group, "p90_ms"), 3),
            "p95_ms": round(_mean(group, "p95_ms"), 3),
            "p99_ms": round(_mean(group, "p99_ms"), 3),
            "index_ms": round(_mean(group, "index_ms"), 1),
            "index_total_ms": round(sum(row.index_ms for row in group), 1),
        }
        for language, group in sorted(by_language_rows.items())
    }

    categories = sorted({category for row in repos for category in row.by_category})
    by_category = {
        category: round(
            statistics.fmean(row.by_category[category] for row in repos if category in row.by_category),
            4,
        )
        for category in categories
    }
    return (
        {
            "repos": len(repos),
            "queries": len(queries),
            "files": sum(row.files for row in repos),
            "chunks": sum(row.chunks for row in repos),
            "ndcg5": round(_mean(repos, "ndcg5"), 4),
            "ndcg10": round(_mean(repos, "ndcg10"), 4),
            "query_ndcg10": round(statistics.fmean(float(row["ndcg10"]) for row in queries), 4),
            "recall10": round(_mean(repos, "recall10"), 4),
            "query_recall10": round(statistics.fmean(float(row["recall10"]) for row in queries), 4),
            "tokens": round(_mean(repos, "tokens"), 0),
            "p50_ms": round(_mean(repos, "p50_ms"), 3),
            "p90_ms": round(_mean(repos, "p90_ms"), 3),
            "p95_ms": round(_mean(repos, "p95_ms"), 3),
            "p99_ms": round(_mean(repos, "p99_ms"), 3),
            "index_ms": round(_mean(repos, "index_ms"), 1),
            "index_total_ms": round(sum(row.index_ms for row in repos), 1),
            "by_category": by_category,
        },
        by_language,
    )


def _result_path(label: str) -> Path:
    """Return a stable, caller-labelled result path."""
    results_dir = BENCHMARKS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir / f"chunker-eval-{label}.json"


def _write_result(label: str, payload: dict[str, Any]) -> Path:
    """Persist a benchmark payload."""
    path = _result_path(label)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {path}", file=sys.stderr)
    return path


def _run_retrieval(args: argparse.Namespace, native: Any | None) -> None:
    """Build and query each selected repository using one chunker variant."""
    specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)
    repo_tasks = grouped_tasks(tasks)
    counters = ChunkSourceCounters()
    variant = VariantChunker(native, args.max_size, args.min_size, counters)
    repo_rows: list[RepoEvaluation] = []
    query_rows: list[dict[str, Any]] = []

    print(
        f"{'Repo':<14} {'Language':<12} {'Files':>6} {'Chunks':>7} {'Index':>9} {'NDCG@10':>8} {'Recall':>8}",
        file=sys.stderr,
    )
    with _isolated_cache():
        with patch.object(create_module, "chunk_source", variant):
            for repo, selected_tasks in sorted(repo_tasks.items()):
                spec = specs[repo]
                before = counters.mark()
                started = time.perf_counter()
                index = SembleIndex.from_path(spec.benchmark_dir)
                index_ms = (time.perf_counter() - started) * 1000
                metrics, rows = _evaluate_queries(
                    index,
                    selected_tasks,
                    latency_runs=args.latency_runs,
                    verbose=args.verbose,
                )
                chunk_stats = ChunkSourceCounters.delta(before, counters.mark())
                stats = index.stats
                repo_result = RepoEvaluation(
                    repo=repo,
                    language=spec.language,
                    mode=args.mode,
                    files=stats.indexed_files,
                    chunks=len(index.chunks),
                    tokens=metrics["tokens"],
                    ndcg5=metrics["ndcg5"],
                    ndcg10=metrics["ndcg10"],
                    recall10=metrics["recall10"],
                    p50_ms=metrics["p50_ms"],
                    p90_ms=metrics["p90_ms"],
                    p95_ms=metrics["p95_ms"],
                    p99_ms=metrics["p99_ms"],
                    index_ms=index_ms,
                    chunk_source=chunk_stats,
                    by_category=metrics["by_category"],
                )
                repo_rows.append(repo_result)
                query_rows.extend(rows)
                print(
                    f"{repo:<14} {spec.language:<12} {stats.indexed_files:>6} {len(index.chunks):>7} "
                    f"{index_ms:>8.1f}ms {metrics['ndcg10']:>8.3f} {metrics['recall10']:>8.3f}",
                    file=sys.stderr,
                )
                del index

    summary, by_language = _aggregate(repo_rows, query_rows)
    payload = {
        "tool": "chunker-eval",
        "mode": args.mode,
        "label": args.label,
        "model": DEFAULT_MODEL_NAME,
        "sha": current_sha(),
        "config": {
            "repos": args.repo or None,
            "languages": args.language or None,
            "latency_runs": args.latency_runs,
            "top_k": _TOP_K,
            "native_max_size": args.max_size if native is not None else None,
            "native_min_size": args.min_size if native is not None else None,
            "persistent_index_cache": "bypassed with a fresh temporary directory",
        },
        "native_module": None
        if native is None
        else {"name": _NATIVE_MODULE, "path": getattr(native, "__file__", None)},
        "chunk_source": counters.as_dict(),
        "summary": summary,
        "by_language": by_language,
        "repos": [asdict(row) for row in repo_rows],
        "queries": query_rows,
    }
    print(json.dumps(summary, indent=2), file=sys.stderr)
    _write_result(args.label, payload)


def _selected_specs(args: argparse.Namespace) -> list[RepoSpec]:
    """Return locally available repository specs selected for timing."""
    specs = available_repo_specs().values()
    selected = [
        spec
        for spec in specs
        if (not args.repo or spec.name in args.repo) and (not args.language or spec.language in args.language)
    ]
    if not selected:
        raise ChunkerEvalError("No available benchmark repositories matched the timing filters.")
    return sorted(selected, key=lambda spec: spec.name)


def _scan_native_files(native: Any, specs: list[RepoSpec]) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    """Select files exactly as Semble does, then retain native-supported paths."""
    extensions = get_extensions((ContentType.CODE,))
    paths: list[str] = []
    repos: list[dict[str, Any]] = []
    totals = {"walked": 0, "valid": 0, "unsupported": 0, "bytes": 0, "scan_errors": 0}

    for spec in specs:
        repo_files = 0
        repo_bytes = 0
        for path in walk_files(spec.benchmark_dir, extensions):
            totals["walked"] += 1
            try:
                if get_file_status(path, None) is not FileStatus.VALID:
                    continue
                totals["valid"] += 1
                if not native.supports_path(str(path)):
                    totals["unsupported"] += 1
                    continue
                size = path.stat().st_size
            except OSError:
                totals["scan_errors"] += 1
                continue
            paths.append(str(path))
            repo_files += 1
            repo_bytes += size
            totals["bytes"] += size
        repos.append({"repo": spec.name, "language": spec.language, "files": repo_files, "bytes": repo_bytes})
    return paths, repos, totals


def _summarize_native_results(results: Any, expected_paths: list[str]) -> dict[str, Any]:
    """Count ordered file results, rejecting missing, reordered, or hidden failures."""
    if not isinstance(results, list):
        raise ChunkerEvalError(f"native chunk_files returned {type(results).__name__}, expected list")
    if len(results) != len(expected_paths):
        raise ChunkerEvalError(
            f"native chunk_files returned {len(results)} records for {len(expected_paths)} paths"
        )

    chunks = 0
    error_count = 0
    error_samples: list[dict[str, str]] = []
    for expected_path, result in zip(expected_paths, results, strict=True):
        if result.path != expected_path:
            raise ChunkerEvalError(
                f"native chunk_files reordered results: expected {expected_path!r}, got {result.path!r}"
            )
        error = result.error
        if error is not None:
            error_count += 1
            if len(error_samples) < 20:
                error_samples.append({"path": result.path, "error": str(error)})
            continue
        chunks += len(result.chunks)
    return {
        "files": len(results),
        "chunks": chunks,
        "errors": error_count,
        "error_samples": error_samples,
    }


def _run_chunk_timing(args: argparse.Namespace, native: Any) -> None:
    """Measure the native ordered batch API independently from retrieval."""
    specs = _selected_specs(args)
    paths, repo_rows, scan = _scan_native_files(native, specs)
    if not paths:
        raise ChunkerEvalError("No selected files are supported by the native chunker.")

    thread_limit = max(1, min(os.cpu_count() or 1, 32))
    threads = min(args.threads, thread_limit)
    records: list[dict[str, Any]] = []
    expected_shape: tuple[int, int, int] | None = None
    for repetition in range(1, args.repeats + 1):
        started = time.perf_counter()
        results = native.chunk_files(paths, args.max_size, args.min_size, threads)
        wall_ms = (time.perf_counter() - started) * 1000
        counts = _summarize_native_results(results, paths)
        shape = (counts["files"], counts["chunks"], counts["errors"])
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ChunkerEvalError(f"native batch output changed across repetitions: {expected_shape} != {shape}")
        records.append(
            {
                "repetition": repetition,
                "wall_ms": wall_ms,
                "mib_per_second": scan["bytes"] / (1024 * 1024) / (wall_ms / 1000),
                **counts,
            }
        )
        del results

    wall_values = [record["wall_ms"] for record in records]
    throughput_values = [record["mib_per_second"] for record in records]
    payload = {
        "tool": "chunker-eval",
        "mode": "chunk-timing",
        "label": args.label,
        "sha": current_sha(),
        "native_module": {"name": _NATIVE_MODULE, "path": getattr(native, "__file__", None)},
        "config": {
            "repos": args.repo or None,
            "languages": args.language or None,
            "max_size": args.max_size,
            "min_size": args.min_size,
            "threads_requested": args.threads,
            "threads": threads,
            "thread_limit": thread_limit,
            "repeats": args.repeats,
        },
        "corpus": {**scan, "supported": len(paths), "repos": repo_rows},
        "summary": {
            "wall_median_ms": statistics.median(wall_values),
            "wall_min_ms": min(wall_values),
            "wall_max_ms": max(wall_values),
            "mib_per_second_median": statistics.median(throughput_values),
            "files": expected_shape[0] if expected_shape else 0,
            "chunks": expected_shape[1] if expected_shape else 0,
            "errors": expected_shape[2] if expected_shape else 0,
        },
        "runs": records,
    }
    print(json.dumps(payload["summary"], indent=2), file=sys.stderr)
    _write_result(args.label, payload)


def _parse_args() -> argparse.Namespace:
    """Parse and validate evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("current", "legacy-cast", "chunk-timing"), default="current")
    parser.add_argument("--label", help="Stable result label (letters, digits, dot, underscore, hyphen).")
    parser.add_argument("--max-size", type=int, default=1500, help="Native non-whitespace target size.")
    parser.add_argument("--min-size", type=int, default=50, help="Native minimum merge size.")
    parser.add_argument("--threads", type=int, default=1, help="Native batch threads for chunk-timing.")
    parser.add_argument("--repeats", type=int, default=5, help="Repetitions for chunk-timing.")
    parser.add_argument("--latency-runs", type=int, default=_LATENCY_RUNS, help="Search repetitions per query.")
    add_filter_args(parser, verbose=True)
    args = parser.parse_args()

    if args.max_size <= 0:
        parser.error("--max-size must be positive")
    if args.min_size < 0 or args.min_size > args.max_size:
        parser.error("--min-size must be between zero and --max-size")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.latency_runs <= 0:
        parser.error("--latency-runs must be positive")
    if args.label is None:
        args.label = f"{args.mode}-{args.max_size}-{args.min_size}"
    if _LABEL_RE.fullmatch(args.label) is None:
        parser.error("--label may contain only letters, digits, dot, underscore, and hyphen")
    return args


def main() -> int:
    """Run the selected evaluation and return a process exit status."""
    args = _parse_args()
    try:
        native = _load_native() if args.mode != "current" else None
        if args.mode == "chunk-timing":
            _run_chunk_timing(args, native)
        else:
            _run_retrieval(args, native)
    except ChunkerEvalError as exc:
        print(f"chunker evaluation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
