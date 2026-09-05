from __future__ import annotations

import argparse
import contextlib
import gc
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

import semble.chunking.chunking as chunking_module
import semble.index.create as create_module
from semble.index.dense import load_model

_PHASE_NAMES = (
    "file_walk_iterator",
    "file_status_checks",
    "language_detection",
    "source_reads",
    "chunk_source",
    "tree_sitter_chunking",
    "fallback_line_chunking",
    "bm25_replace_add_total",
    "bm25_tokenization",
    "embed_chunks",
    "static_model_encode",
    "model2vec_tokenization",
)


@dataclass
class _PhaseTotals:
    """Aggregate measurements for one instrumented boundary."""

    wall_ns: int = 0
    exclusive_wall_ns: int = 0
    calls: int = 0
    items: int = 0
    call_sizes: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible phase record."""
        result: dict[str, Any] = {
            "wall_ns": self.wall_ns,
            "exclusive_wall_ns": self.exclusive_wall_ns,
            "calls": self.calls,
            "items": self.items,
        }
        if self.call_sizes:
            result["call_sizes"] = list(self.call_sizes)
            result["call_size_distribution"] = _number_summary(self.call_sizes)
        return result


@dataclass
class _ActivePhase:
    """One live timing frame used to remove nested intervals."""

    name: str
    started_ns: int
    child_wall_ns: int = 0


class _PhaseRecorder:
    """Collect single-threaded inclusive boundaries and exclusive-time accounting."""

    def __init__(self, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        """Create a recorder, optionally with a deterministic test clock."""
        self._clock = clock
        self._phases = {name: _PhaseTotals() for name in ("instrumented_index", *_PHASE_NAMES)}
        self._stack: list[_ActivePhase] = []
        self.counters: dict[str, int] = {}
        self.nested_wall_ns: dict[str, int] = {}

    @contextlib.contextmanager
    def measure(self, name: str, *, items: int = 0, call_size: int | None = None) -> Iterator[None]:
        """Measure one call and subtract nested instrumented boundaries from its exclusive time."""
        phase = self._phases.setdefault(name, _PhaseTotals())
        phase.calls += 1
        phase.items += items
        if call_size is not None:
            phase.call_sizes.append(call_size)
        frame = _ActivePhase(name=name, started_ns=self._clock())
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed_ns = self._clock() - frame.started_ns
            if self._stack.pop() is not frame:
                raise RuntimeError("Profiler phase stack became unbalanced")
            phase.wall_ns += elapsed_ns
            phase.exclusive_wall_ns += elapsed_ns - frame.child_wall_ns
            if self._stack:
                parent = self._stack[-1]
                parent.child_wall_ns += elapsed_ns
                edge = f"{parent.name}>{name}"
                self.nested_wall_ns[edge] = self.nested_wall_ns.get(edge, 0) + elapsed_ns

    def add_counter(self, name: str, value: int = 1) -> None:
        """Add to a raw event counter."""
        self.counters[name] = self.counters.get(name, 0) + value

    def add_items(self, phase: str, value: int) -> None:
        """Add output items whose count is known only after a boundary returns."""
        self._phases[phase].items += value

    def phase(self, name: str) -> _PhaseTotals:
        """Return aggregate totals for a named phase."""
        return self._phases[name]

    def phase_records(self) -> dict[str, dict[str, Any]]:
        """Return all non-root phase records in stable order."""
        return {name: self._phases[name].as_dict() for name in _PHASE_NAMES}


def _number_summary(values: Sequence[int | float]) -> dict[str, int | float]:
    """Return count, total, median, minimum, and maximum for numeric values."""
    return {
        "count": len(values),
        "total": sum(values),
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
    }


def _summarize_mappings(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recursively summarize numeric leaves shared by every mapping."""
    if not records:
        return {}
    shared_keys = set(records[0])
    for record in records[1:]:
        shared_keys.intersection_update(record)

    result: dict[str, Any] = {}
    for key in sorted(shared_keys):
        values = [record[key] for record in records]
        if all(isinstance(value, Mapping) for value in values):
            nested = _summarize_mappings([value for value in values if isinstance(value, Mapping)])
            if nested:
                result[key] = nested
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            result[key] = {
                "median": float(statistics.median(values)),
                "min": min(values),
                "max": max(values),
            }
    return result


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build numeric summaries plus duplicate-pass verdicts across all repetitions."""
    stripped = [{key: value for key, value in record.items() if key != "repetition"} for record in records]
    summary = _summarize_mappings(stripped)
    invariants = _summarize_embedding_invariants(records)
    if invariants:
        summary["embedding_invariants"] = invariants
    return summary


def _summarize_embedding_invariants(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int | bool]]:
    """Report whether each duplicate-pass invariant held in every repetition."""
    names = (
        "embedded_chunks_equal_produced_chunks",
        "encoded_texts_equal_produced_chunks",
        "encoded_texts_equal_unique_chunks",
    )
    if not records or any(
        not isinstance(record.get("counts"), Mapping) or any(name not in record["counts"] for name in names)
        for record in records
    ):
        return {}

    result: dict[str, dict[str, int | bool]] = {}
    for name in names:
        matches = sum(record["counts"][name] is True for record in records)
        result[name] = {
            "all_repetitions": matches == len(records),
            "matching_repetitions": matches,
            "repetitions": len(records),
        }
    return result


def _sequence_size(value: Sequence[str] | str) -> int:
    """Return the number of texts represented by a model API argument."""
    return 1 if isinstance(value, str) else len(value)


class _ColdPathWrappers:
    """Single-threaded benchmark wrappers for cold-index boundaries."""

    def __init__(self, recorder: _PhaseRecorder, model: Any) -> None:
        """Capture original callables before ExitStack applies replacements."""
        self._recorder = recorder
        self._walk_files = create_module.walk_files
        self._get_file_status = create_module.get_file_status
        self._detect_language = create_module.detect_language
        self._read_file_text = create_module.read_file_text
        self._chunk_source = create_module.chunk_source
        self._tree_sitter_chunk = chunking_module.chunk
        self._fallback_chunk = chunking_module.chunk_lines
        self._reindex_file = create_module._reindex_file
        self._bm25_tokenize = create_module.tokenize
        self._embed_chunks = create_module.embed_chunks
        self._model_encode = model.encode
        self._model_tokenize = model.tokenize

    def walk_files(self, *args: Any, **kwargs: Any) -> Iterator[Path]:
        """Measure only time spent advancing the lazy file iterator."""
        self._recorder.add_counter("file_walk_iterators")
        iterator = iter(self._walk_files(*args, **kwargs))
        while True:
            try:
                with self._recorder.measure("file_walk_iterator"):
                    item = next(iterator)
            except StopIteration:
                return
            self._recorder.add_items("file_walk_iterator", 1)
            self._recorder.add_counter("walked_files")
            yield item

    def get_file_status(self, *args: Any, **kwargs: Any) -> Any:
        """Measure eligibility checks, including their stats and small-file emptiness probes."""
        with self._recorder.measure("file_status_checks", items=1):
            result = self._get_file_status(*args, **kwargs)
        self._recorder.add_counter(f"file_status_{result.value}")
        return result

    def detect_language(self, *args: Any, **kwargs: Any) -> Any:
        """Measure extension-based language detection."""
        with self._recorder.measure("language_detection", items=1):
            result = self._detect_language(*args, **kwargs)
        if result is not None:
            self._recorder.add_counter("languages_detected")
        return result

    def read_file_text(self, file_path: Path) -> str:
        """Measure the production source read and count returned UTF-8 bytes afterward."""
        with self._recorder.measure("source_reads", items=1):
            text = self._read_file_text(file_path)
        self._recorder.add_counter("source_read_bytes", len(text.encode("utf-8")))
        return text

    def chunk_source(self, *args: Any, **kwargs: Any) -> Any:
        """Measure the full chunk-source boundary and its produced chunks."""
        with self._recorder.measure("chunk_source"):
            result = self._chunk_source(*args, **kwargs)
        self._recorder.add_items("chunk_source", len(result))
        self._recorder.add_counter("chunk_source_calls")
        return result

    def tree_sitter_chunk(self, *args: Any, **kwargs: Any) -> Any:
        """Measure tree-sitter chunking attempts before any fallback."""
        with self._recorder.measure("tree_sitter_chunking"):
            result = self._tree_sitter_chunk(*args, **kwargs)
        if result is not None:
            self._recorder.add_items("tree_sitter_chunking", len(result))
            self._recorder.add_counter("tree_sitter_successes")
        return result

    def fallback_chunk(self, *args: Any, **kwargs: Any) -> Any:
        """Measure line-based fallback chunking."""
        with self._recorder.measure("fallback_line_chunking"):
            result = self._fallback_chunk(*args, **kwargs)
        self._recorder.add_items("fallback_line_chunking", len(result))
        return result

    def reindex_file(
        self,
        bm25_index: Any,
        indexed_path: str,
        file_chunks: list[Any],
        previous_entry: Any,
    ) -> None:
        """Measure BM25 replacement/addition around its nested tokenization."""
        size = len(file_chunks)
        with self._recorder.measure("bm25_replace_add_total", items=size, call_size=size):
            self._reindex_file(bm25_index, indexed_path, file_chunks, previous_entry)
        self._recorder.add_counter("bm25_documents_added", size)
        if previous_entry is not None:
            self._recorder.add_counter("bm25_document_slots_removed", previous_entry.count)

    def bm25_tokenize(self, text: str) -> list[str]:
        """Measure BM25 tokenization nested in replacement/addition."""
        with self._recorder.measure("bm25_tokenization", items=1):
            result = self._bm25_tokenize(text)
        self._recorder.add_counter("bm25_tokens", len(result))
        return result

    def embed_chunks(self, profiled_model: Any, chunks: list[Any]) -> Any:
        """Measure the exact production embed_chunks boundary."""
        size = len(chunks)
        with self._recorder.measure("embed_chunks", items=size, call_size=size):
            return self._embed_chunks(profiled_model, chunks)

    def model_encode(self, sentences: Sequence[str] | str, *args: Any, **kwargs: Any) -> Any:
        """Measure the exact StaticModel.encode API boundary."""
        size = _sequence_size(sentences)
        with self._recorder.measure("static_model_encode", items=size, call_size=size):
            return self._model_encode(sentences, *args, **kwargs)

    def model_tokenize(self, sentences: Sequence[str] | str, *args: Any, **kwargs: Any) -> Any:
        """Measure Model2Vec tokenization nested inside StaticModel.encode."""
        size = _sequence_size(sentences)
        with self._recorder.measure("model2vec_tokenization", items=size, call_size=size):
            return self._model_tokenize(sentences, *args, **kwargs)


@contextlib.contextmanager
def _instrument_cold_path(recorder: _PhaseRecorder, model: Any) -> Iterator[None]:
    """Install and exception-safely restore benchmark-local cold-path wrappers."""
    wrappers = _ColdPathWrappers(recorder, model)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(create_module, "walk_files", wrappers.walk_files))
        stack.enter_context(patch.object(create_module, "get_file_status", wrappers.get_file_status))
        stack.enter_context(patch.object(create_module, "detect_language", wrappers.detect_language))
        stack.enter_context(patch.object(create_module, "read_file_text", wrappers.read_file_text))
        stack.enter_context(patch.object(create_module, "chunk_source", wrappers.chunk_source))
        stack.enter_context(patch.object(chunking_module, "chunk", wrappers.tree_sitter_chunk))
        stack.enter_context(patch.object(chunking_module, "chunk_lines", wrappers.fallback_chunk))
        stack.enter_context(patch.object(create_module, "_reindex_file", wrappers.reindex_file))
        stack.enter_context(patch.object(create_module, "tokenize", wrappers.bm25_tokenize))
        stack.enter_context(patch.object(create_module, "embed_chunks", wrappers.embed_chunks))
        stack.enter_context(patch.object(model, "encode", wrappers.model_encode))
        stack.enter_context(patch.object(model, "tokenize", wrappers.model_tokenize))
        yield


def _peak_rss_bytes() -> int | None:
    """Return the process high-water RSS in bytes when the platform exposes it."""
    try:
        resource_module: Any = import_module("resource")
    except ModuleNotFoundError:  # pragma: no cover - resource is available on supported Unix platforms
        return None
    peak = int(resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _derived_timings(recorder: _PhaseRecorder) -> dict[str, dict[str, Any]]:
    """Build explicitly labeled residuals from exclusive boundary accounting."""
    return {
        "chunk_object_assembly_residual": {
            "wall_ns": recorder.phase("chunk_source").exclusive_wall_ns,
            "derived_from": "chunk_source inclusive minus nested tree-sitter/fallback boundaries",
        },
        "bm25_postings_residual": {
            "wall_ns": recorder.phase("bm25_replace_add_total").exclusive_wall_ns,
            "derived_from": "BM25 replacement/addition inclusive minus nested BM25 tokenization",
        },
        "embed_array_conversion_residual": {
            "wall_ns": recorder.phase("embed_chunks").exclusive_wall_ns,
            "derived_from": "embed_chunks inclusive minus nested StaticModel.encode",
        },
        "model2vec_lookup_mean_stack_normalization": {
            "wall_ns": recorder.phase("static_model_encode").exclusive_wall_ns,
            "derived_from": "StaticModel.encode inclusive minus nested Model2Vec tokenization",
            "native_only": False,
        },
        "unattributed_including_profiler_overhead": {
            "wall_ns": recorder.phase("instrumented_index").exclusive_wall_ns,
            "derived_from": "instrumented index wall minus directly nested measured boundaries",
            "includes": (
                "unwrapped index orchestration, vector/BM25 finalization, manifest mtime stats, "
                "and profiler bookkeeping"
            ),
        },
    }


def _embedding_counters(
    produced_chunks: int,
    unique_chunk_ids: int,
    embedded_chunks: int,
    encoded_texts: int,
) -> dict[str, int | bool]:
    """Return raw embedding totals and explicit fresh-index equality evidence."""
    return {
        "chunks": produced_chunks,
        "unique_chunk_ids": unique_chunk_ids,
        "embedded_chunks": embedded_chunks,
        "encoded_texts": encoded_texts,
        "embedded_chunks_equal_produced_chunks": embedded_chunks == produced_chunks,
        "encoded_texts_equal_produced_chunks": encoded_texts == produced_chunks,
        "encoded_texts_equal_unique_chunks": encoded_texts == unique_chunk_ids,
    }


def _profile_once(corpus_path: Path, model: Any, repetition: int) -> dict[str, Any]:
    """Profile one fresh create_index_from_path call with no previous index."""
    recorder = _PhaseRecorder()
    with _instrument_cold_path(recorder, model):
        with recorder.measure("instrumented_index"):
            cpu_started_ns = time.process_time_ns()
            bm25_index, semantic_index, chunks, manifest = create_module.create_index_from_path(
                corpus_path,
                model,
                display_root=corpus_path,
                previous=None,
            )
            instrumented_process_cpu_ns = time.process_time_ns() - cpu_started_ns
    peak_rss_bytes = _peak_rss_bytes()

    produced_chunks = len(chunks)
    unique_chunk_ids = len(set(bm25_index.doc_order))
    embedded_chunks = recorder.phase("embed_chunks").items
    encoded_texts = recorder.phase("static_model_encode").items
    counters: dict[str, int | bool] = dict(sorted(recorder.counters.items()))
    counters["files"] = len(manifest)
    counters.update(_embedding_counters(produced_chunks, unique_chunk_ids, embedded_chunks, encoded_texts))

    phases = recorder.phase_records()
    exclusive_phase_wall_ns = {name: phase["exclusive_wall_ns"] for name, phase in phases.items()}
    instrumented_wall_ns = recorder.phase("instrumented_index").wall_ns

    memory: dict[str, int | None] = {
        "process_peak_rss_bytes": peak_rss_bytes,
    }
    vectors = semantic_index.vectors
    return {
        "repetition": repetition,
        "total": {
            "instrumented_wall_ns": instrumented_wall_ns,
            "instrumented_process_cpu_ns": instrumented_process_cpu_ns,
        },
        "phases": phases,
        "derived": _derived_timings(recorder),
        "counts": counters,
        "index": {
            "files": len(manifest),
            "chunks": produced_chunks,
            "dimensions": int(vectors.shape[1]),
            "vector_bytes": int(vectors.nbytes),
        },
        "memory": memory,
        "overlap": {
            "nested_wall_ns": dict(sorted(recorder.nested_wall_ns.items())),
            "exclusive_phase_wall_ns": exclusive_phase_wall_ns,
        },
    }


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse cold-index profiler arguments."""
    parser = argparse.ArgumentParser(description="Profile Semble create_index_from_path phase boundaries.")
    parser.add_argument("--corpus-path", type=Path, required=True, help="Source tree to index.")
    parser.add_argument("--model-path", required=True, help="Local model path or Model2Vec model identifier.")
    parser.add_argument("--repetitions", type=_positive_int, default=5, help="Number of fresh index builds.")
    parser.add_argument("--label", required=True, help="Free-form run label stored as metadata.")
    parser.add_argument("--revision", required=True, help="Corpus or experiment revision stored as metadata.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON path.")
    args = parser.parse_args(argv)
    args.corpus_path = args.corpus_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.corpus_path.is_dir():
        parser.error(f"corpus path is not a directory: {args.corpus_path}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run repeated fresh-index profiles and write raw records plus summaries."""
    args = _parse_args(argv)
    model = load_model(args.model_path)[0]
    records: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        records.append(_profile_once(args.corpus_path, model, repetition))
        gc.collect()

    payload = {
        "schema_version": 2,
        "benchmark": "semble-create-index-cold-profile",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "label": args.label,
            "revision": args.revision,
            "corpus_path": str(args.corpus_path),
            "model_path": args.model_path,
            "repetitions": args.repetitions,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "model2vec_version": version("model2vec"),
        },
        "timing_semantics": {
            "wall_unit": "nanoseconds",
            "inclusive": "wall_ns includes nested instrumented boundaries and must not be summed with them",
            "exclusive": "exclusive_wall_ns removes directly nested instrumented intervals and is non-overlapping",
            "instrumented_total": (
                "total instrumented fields include wrapper/recorder overhead and are not an uninstrumented benchmark"
            ),
            "static_model_encode": "exact StaticModel.encode API wall boundary, not pure native model time",
            "model2vec_residual": (
                "derived encode remainder after tokenization; includes lookup/mean/stack/"
                "normalization and Python overhead"
            ),
            "unattributed": (
                "derived root remainder includes unwrapped index orchestration, vector/BM25 finalization, "
                "manifest mtime stats, and profiler bookkeeping"
            ),
            "cold_index": (
                "every repetition passes previous=None; model loading is outside the boundary, while process, parser, "
                "and filesystem caches may be warm"
            ),
            "process_peak_rss": (
                "monotonic process-lifetime high-water RSS; reflects earlier peaks including model load and repetitions"
            ),
            "single_thread_only": (
                "recorder state is unsynchronized; a parallel indexer requires thread-local or "
                "synchronized instrumentation"
            ),
        },
        "records": records,
        "summary": _summarize_records(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
