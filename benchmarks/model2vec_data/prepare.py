"""Reproducible sample preparation CLI for Semble's Model2Vec experiment.

Reads local Parquet/JSONL exports of the pinned datasets in ``sources.py``,
normalizes rows through thin adapters, deterministically samples within
configured caps, splits evaluation groups by crate/repository identity (groups
in upstream eval inputs are reserved for eval), and removes exact duplicates
across splits and output kinds with preference for held-out evaluation content,
then writes JSONL outputs plus a manifest.

Default preparation is Rust-only and capped at a plumbing smoke sample; the
caps are configurable defaults, not a claim of training sufficiency. Dataset
strings are never executed (see ``adapters.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.model2vec_data.adapters import (
    ADAPTERS,
    SKIP_INVALID_ROW,
    SKIP_UNSUPPORTED_CATEGORY,
    Adapted,
    CorpusRecord,
    PairRecord,
    Skip,
)
from benchmarks.model2vec_data.readers import iter_input_rows, sha256_file
from benchmarks.model2vec_data.sources import (
    KIND_CORPUS,
    KIND_PAIRS,
    ROLE_EVAL,
    ROLE_TRAIN,
    SourcePin,
    load_pins,
)

SOURCE_ORDER: tuple[str, ...] = ("rust", "typescript", "swift", "kotlin")
SPLITS: tuple[str, ...] = (ROLE_TRAIN, ROLE_EVAL)
INPUT_SUFFIXES = (".parquet", ".jsonl", ".ndjson")

SCHEMA_VERSION = 1
DEFAULT_SEED = 0
DEFAULT_EVAL_FRACTION = 0.1
DEFAULT_MAX_FIELD_BYTES = 1_048_576
DEFAULT_MAX_SCAN_ROWS = 100_000
DEFAULT_KOTLIN_MAX_SHARDS = 1
DEFAULT_CAPS: dict[str, int] = {"rust": 1000, "typescript": 100, "swift": 100, "kotlin": 100}
DEFAULT_INPUT_ROOT = Path.home() / ".cache" / "semble-model2vec-data"

_MANIFEST_NOTES: tuple[str, ...] = (
    "Inputs are recorded by local path and SHA-256 only; this manifest does not attest that a "
    "local file matches its pinned Hub revision. Verify with `hf cache verify` when needed.",
    "Caps are reproducible plumbing smoke-sample sizes, not claims of training sufficiency.",
    "Upstream validation/test inputs are eval-only and reserve their groups. Exact duplicate "
    "content is removed with preference for held-out evaluation records: training records "
    "conflicting with evaluation code/text are excluded, so pairs.train and corpus.train share "
    "no exact code/text with pairs.eval and corpus.eval.",
)


class ConfigError(ValueError):
    """Raised when the CLI configuration or requested inputs are invalid."""


class PreparationError(RuntimeError):
    """Raised when a required source cannot produce usable rows."""


@dataclass(frozen=True)
class PrepareConfig:
    """Validated preparation settings; these fully determine selection behaviour."""

    seed: int
    eval_fraction: float
    caps: Mapping[str, int]
    max_field_bytes: int
    max_scan_rows: int
    kotlin_max_shards: int


@dataclass(frozen=True)
class ResolvedFile:
    """A concrete local input file standing in for a pinned file."""

    path: Path
    pinned_name: str
    role: str
    resolution: str  # "input_root" or "explicit"


@dataclass(frozen=True)
class SelectedRow:
    """An eligible row selected for a split, ordered by (key, row_id)."""

    key: int
    row_id: str
    file: ResolvedFile
    row_index: int
    record: PairRecord | CorpusRecord


@dataclass
class FileScan:
    """Per-input scan bookkeeping."""

    path: str
    pinned_name: str
    role: str
    resolution: str
    sha256: str
    rows_scanned: int = 0
    hit_scan_bound: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the manifest view of this input file scan."""
        return {
            "path": self.path,
            "pinned_name": self.pinned_name,
            "role": self.role,
            "upstream_split": upstream_split(self.pinned_name),
            "resolution": self.resolution,
            "sha256": self.sha256,
            "rows_scanned": self.rows_scanned,
            "hit_scan_bound": self.hit_scan_bound,
        }


@dataclass
class SourceStats:
    """Scan and skip counters for one source."""

    scanned: int = 0
    eligible: int = 0
    reserved_eval_groups: int = 0
    train_rows_routed_to_reserved_groups: int = 0
    skipped: Counter = field(default_factory=Counter)
    unsupported_detail: Counter = field(default_factory=Counter)
    scanned_by_category: Counter = field(default_factory=Counter)
    eligible_by_category: Counter = field(default_factory=Counter)
    inputs: list[FileScan] = field(default_factory=list)


def upstream_split(pinned_name: str) -> str:
    """Return the upstream split label (train/validation/test) encoded in a file name."""
    if "validation" in pinned_name:
        return "validation"
    if "test" in pinned_name:
        return "test"
    return "train"


def row_key(seed: int, row_id: str) -> int:
    """Return a stable selection hash for a row identity."""
    digest = hashlib.sha256(f"{seed}:select:{row_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def split_for_group(seed: int, group_id: str, eval_fraction: float) -> str:
    """Deterministically assign a group identity to the train or eval split."""
    digest = hashlib.sha256(f"{seed}:split:{group_id}".encode("utf-8")).digest()
    threshold = int(eval_fraction * 10_000)
    return ROLE_EVAL if int.from_bytes(digest[:16], "big") % 10_000 < threshold else ROLE_TRAIN


class _WorstEntry:
    """Heap wrapper that inverts (key, row_id) order so heapq surfaces the worst row."""

    __slots__ = ("row",)

    def __init__(self, row: SelectedRow) -> None:
        """Store the wrapped row."""
        self.row = row

    def __lt__(self, other: "_WorstEntry") -> bool:
        """Compare reversed so the max-heap root is the worst selected row."""
        return (self.row.key, self.row.row_id) > (other.row.key, other.row.row_id)


class TopK:
    """Bounded collector keeping the k rows with the smallest (key, row_id)."""

    def __init__(self, k: int) -> None:
        """Prepare a collector with capacity k."""
        self._k = k
        self._heap: list[_WorstEntry] = []

    def add(self, row: SelectedRow) -> None:
        """Consider a row for selection, evicting the worst row when full."""
        if self._k <= 0:
            return
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, _WorstEntry(row))
            return
        worst = self._heap[0].row
        if (row.key, row.row_id) < (worst.key, worst.row_id):
            heapq.heapreplace(self._heap, _WorstEntry(row))

    def ordered(self) -> list[SelectedRow]:
        """Return the selected rows ordered by (key, row_id)."""
        return sorted((entry.row for entry in self._heap), key=lambda row: (row.key, row.row_id))


def _budgets(config: PrepareConfig, key: str) -> dict[str, int]:
    """Split a source cap into train/eval selection budgets."""
    cap = config.caps[key]
    eval_budget = int(cap * config.eval_fraction + 0.5)
    return {ROLE_TRAIN: cap - eval_budget, ROLE_EVAL: eval_budget}


def _category_label(value: object) -> str:
    """Return a stable category label for manifest counting."""
    return value if isinstance(value, str) and value else "<missing>"


def _pair_category(row: SelectedRow) -> str | None:
    """Return the category of a selected pair row, if it is a pair."""
    return row.record.category if isinstance(row.record, PairRecord) else None


def _route_split(
    resolved: ResolvedFile,
    group_id: str,
    reserved_groups: set[str],
    config: PrepareConfig,
    stats: SourceStats,
) -> str:
    """Return the target split for an eligible row.

    Upstream eval-role rows are eval-only and reserve their group identity.
    Train-role rows in a reserved group follow their group to eval so one group
    never straddles splits; remaining train rows follow the deterministic group
    hash. The guarantee holds over the scanned rows only.
    """
    if resolved.role == ROLE_EVAL:
        reserved_groups.add(group_id)
        return ROLE_EVAL
    if group_id in reserved_groups:
        stats.train_rows_routed_to_reserved_groups += 1
        return ROLE_EVAL
    return split_for_group(config.seed, group_id, config.eval_fraction)


def scan_source(
    pin: SourcePin, files: Sequence[ResolvedFile], config: PrepareConfig
) -> tuple[dict[str, list[SelectedRow]], SourceStats]:
    """Stream a source's inputs, adapt rows, and select the smallest-hash sample per split.

    Upstream eval inputs are scanned first so their group identities are
    reserved before train selection; train rows in a reserved group are routed
    to eval (consistent group ownership). Scanning stops at
    ``config.max_scan_rows`` rows per file and reports the truncation.
    """
    adapter = ADAPTERS[pin.adapter]
    budgets = _budgets(config, pin.key)
    pools = {split: TopK(budgets[split]) for split in SPLITS}
    stats = SourceStats()
    track_categories = pin.key == "rust"
    reserved_groups: set[str] = set()
    ordered_files = sorted(files, key=lambda resolved: resolved.role != ROLE_EVAL)
    for resolved in ordered_files:
        file_scan = FileScan(
            path=str(resolved.path),
            pinned_name=resolved.pinned_name,
            role=resolved.role,
            resolution=resolved.resolution,
            sha256=sha256_file(resolved.path),
        )
        for raw in iter_input_rows(resolved.path):
            if file_scan.rows_scanned >= config.max_scan_rows:
                file_scan.hit_scan_bound = True
                break
            file_scan.rows_scanned += 1
            stats.scanned += 1
            if track_categories:
                raw_category = raw.get("task_category") if isinstance(raw, Mapping) else None
                stats.scanned_by_category[_category_label(raw_category)] += 1
            if not isinstance(raw, Mapping):
                stats.skipped[SKIP_INVALID_ROW] += 1
                continue
            row_id = f"{pin.key}:{resolved.pinned_name}:{file_scan.rows_scanned - 1}"
            outcome: Adapted = adapter(raw, row_id=row_id, max_field_bytes=config.max_field_bytes)
            if isinstance(outcome, Skip):
                stats.skipped[outcome.reason] += 1
                if outcome.reason == SKIP_UNSUPPORTED_CATEGORY:
                    stats.unsupported_detail[outcome.detail] += 1
                continue
            stats.eligible += 1
            if track_categories and isinstance(outcome, PairRecord) and outcome.category is not None:
                stats.eligible_by_category[outcome.category] += 1
            split = _route_split(resolved, outcome.group_id, reserved_groups, config, stats)
            pools[split].add(
                SelectedRow(row_key(config.seed, row_id), row_id, resolved, file_scan.rows_scanned - 1, outcome)
            )
        stats.inputs.append(file_scan)
    stats.reserved_eval_groups = len(reserved_groups)
    return {split: pools[split].ordered() for split in SPLITS}, stats


def _text_digest(text: str) -> str:
    """Return the SHA-256 digest of an exact text, the deduplication identity."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finalize_pairs(
    selections: Mapping[str, Mapping[str, Sequence[SelectedRow]]], pins: Mapping[str, SourcePin]
) -> tuple[dict[str, dict[str, list[SelectedRow]]], dict[str, Counter], dict[str, str]]:
    """Deduplicate pairs exactly, reserving held-out eval content before train rows.

    Eval rows are registered first and identical eval code collapses to the
    first occurrence. A train row whose exact code is already owned by eval is
    excluded (the held-out copy wins); identical train rows collapse to the
    first train occurrence. Returns the final pairs, per-source duplicate
    counters, and the eval code digests (digest -> "pair") for cross-kind
    enforcement in ``build_corpus``.
    """
    final: dict[str, dict[str, list[SelectedRow]]] = {}
    seen: dict[str, tuple[str, str]] = {}  # code digest -> (query digest, owning split)
    duplicates: dict[str, Counter] = {}
    for split in (ROLE_EVAL, ROLE_TRAIN):
        for key in SOURCE_ORDER:
            if pins[key].kind != KIND_PAIRS:
                continue
            kept: list[SelectedRow] = []
            for row in selections.get(key, {}).get(split, []):
                rec = row.record
                if not isinstance(rec, PairRecord):
                    continue
                code_digest = _text_digest(rec.code)
                query_digest = _text_digest(rec.query)
                first = seen.get(code_digest)
                if first is not None:
                    first_query, first_split = first
                    counter = duplicates.setdefault(key, Counter())
                    if first_split != split:
                        counter["cross_split_duplicate"] += 1
                    if first_query == query_digest:
                        counter["duplicate_pair"] += 1
                    else:
                        counter["duplicate_code"] += 1
                    continue
                seen[code_digest] = (query_digest, split)
                kept.append(row)
            final.setdefault(key, {})[split] = kept
    eval_digests = {digest: "pair" for digest, (_, split) in seen.items() if split == ROLE_EVAL}
    return final, duplicates, eval_digests


def _corpus_candidates(
    final_pairs: Mapping[str, Mapping[str, Sequence[SelectedRow]]],
    selections: Mapping[str, Mapping[str, Sequence[SelectedRow]]],
    pins: Mapping[str, SourcePin],
    split: str,
) -> dict[str, list[SelectedRow]]:
    """Build corpus candidates for one split: derived pair code plus corpus-only rows."""
    candidates: dict[str, list[SelectedRow]] = {}
    for key in SOURCE_ORDER:
        rows: list[SelectedRow] = []
        if pins[key].kind == KIND_PAIRS:
            for row in final_pairs.get(key, {}).get(split, []):
                rec = row.record
                if not isinstance(rec, PairRecord):
                    continue
                derived = CorpusRecord(
                    text=rec.code,
                    group_id=rec.group_id,
                    group_strength=rec.group_strength,
                    provenance=rec.provenance,
                )
                rows.append(SelectedRow(row.key, row.row_id, row.file, row.row_index, derived))
        else:
            rows.extend(selections.get(key, {}).get(split, []))
        candidates[key] = rows
    return candidates


def _dedup_corpus_phase(
    candidates: Mapping[str, Sequence[SelectedRow]],
    blocked: Mapping[str, str],
    split: str,
) -> tuple[dict[str, list[SelectedRow]], dict[str, str], dict[str, Counter]]:
    """Keep corpus rows whose text is not eval-owned, collapsing same-split duplicates.

    ``blocked`` maps eval-owned content digests to the owning kind ("pair" or
    "corpus"); rows hitting it are excluded so held-out evaluation content is
    preserved. Same-split duplicates collapse to the first occurrence in source
    order. Returns kept rows per source, the kept digests, and duplicate counters.
    """
    kept: dict[str, list[SelectedRow]] = {key: [] for key in SOURCE_ORDER}
    seen: set[str] = set()
    digests: dict[str, str] = {}
    duplicates: dict[str, Counter] = {}
    for key in SOURCE_ORDER:
        for row in candidates[key]:
            rec = row.record
            if not isinstance(rec, CorpusRecord):
                continue
            text_digest = _text_digest(rec.text)
            owner = blocked.get(text_digest)
            if owner is not None:
                counter = duplicates.setdefault(key, Counter())
                if owner == "corpus":
                    counter["cross_split_duplicate_text"] += 1
                else:
                    counter["cross_kind_eval_conflict"] += 1
                continue
            if text_digest in seen:
                duplicates.setdefault(key, Counter())["duplicate_text"] += 1
                continue
            seen.add(text_digest)
            digests[text_digest] = "corpus"
            kept[key].append(row)
    return kept, digests, duplicates


def _drop_train_pairs_owned_by_eval(
    final_pairs: dict[str, dict[str, list[SelectedRow]]], blocked: Mapping[str, str]
) -> dict[str, Counter]:
    """Exclude, in place, train pairs whose exact code is owned by eval-side content."""
    duplicates: dict[str, Counter] = {}
    for key in SOURCE_ORDER:
        kept: list[SelectedRow] = []
        for row in final_pairs.get(key, {}).get(ROLE_TRAIN, []):
            rec = row.record
            if not isinstance(rec, PairRecord):
                continue
            owner = blocked.get(_text_digest(rec.code))
            if owner is not None:
                counter = duplicates.setdefault(key, Counter())
                if owner == "pair":
                    counter["cross_split_duplicate"] += 1
                else:
                    counter["cross_kind_eval_conflict"] += 1
                continue
            kept.append(row)
        final_pairs.setdefault(key, {})[ROLE_TRAIN] = kept
    return duplicates


def build_corpus(
    final_pairs: dict[str, dict[str, list[SelectedRow]]],
    selections: Mapping[str, Mapping[str, Sequence[SelectedRow]]],
    pins: Mapping[str, SourcePin],
    eval_pair_digests: Mapping[str, str],
) -> tuple[dict[str, dict[str, list[SelectedRow]]], dict[str, Counter]]:
    """Assemble corpus rows so train and eval outputs share no exact code/text.

    Eval-side corpus rows (derived from eval pairs plus corpus-only eval
    selections) are chosen first; their digests, together with eval pair code
    digests, form the eval-owned content set. Train pairs conflicting with that
    set are excluded before their derived corpus rows exist, and train corpus
    rows hitting it are dropped, so held-out evaluation content is preserved.
    """
    eval_candidates = _corpus_candidates(final_pairs, selections, pins, ROLE_EVAL)
    eval_kept, eval_digests, duplicates = _dedup_corpus_phase(eval_candidates, blocked={}, split=ROLE_EVAL)
    blocked = {**eval_digests, **dict(eval_pair_digests)}
    for key, counter in _drop_train_pairs_owned_by_eval(final_pairs, blocked).items():
        duplicates.setdefault(key, Counter()).update(counter)
    train_candidates = _corpus_candidates(final_pairs, selections, pins, ROLE_TRAIN)
    train_kept, _, train_duplicates = _dedup_corpus_phase(train_candidates, blocked=blocked, split=ROLE_TRAIN)
    for key, counter in train_duplicates.items():
        duplicates.setdefault(key, Counter()).update(counter)
    final = {key: {ROLE_EVAL: eval_kept[key], ROLE_TRAIN: train_kept[key]} for key in SOURCE_ORDER}
    return final, duplicates


def _source_block(pin: SourcePin, row: SelectedRow, *, derived_from_pair: bool) -> dict[str, Any]:
    """Return the source/provenance block shared by pair and corpus outputs."""
    return {
        "dataset": pin.dataset,
        "revision": pin.revision,
        "input_role": row.file.role,
        "upstream_split": upstream_split(row.file.pinned_name),
        "file": row.file.pinned_name,
        "row_index": row.row_index,
        "resolution": row.file.resolution,
        "derived_from_pair": derived_from_pair,
    }


def _pair_output(pin: SourcePin, row: SelectedRow) -> dict[str, Any] | None:
    """Return the output view of a selected pair row."""
    rec = row.record
    if not isinstance(rec, PairRecord):
        return None
    return {
        "id": row.row_id,
        "language": pin.language,
        "category": rec.category,
        "query": rec.query,
        "code": rec.code,
        "group": {"id": rec.group_id, "strength": rec.group_strength},
        "source": _source_block(pin, row, derived_from_pair=False),
        "provenance": dict(rec.provenance),
    }


def _corpus_output(pin: SourcePin, row: SelectedRow) -> dict[str, Any] | None:
    """Return the output view of a corpus row."""
    rec = row.record
    if not isinstance(rec, CorpusRecord):
        return None
    return {
        "id": row.row_id,
        "language": pin.language,
        "text": rec.text,
        "group": {"id": rec.group_id, "strength": rec.group_strength},
        "source": _source_block(pin, row, derived_from_pair=pins_kind_is_pairs(pin)),
        "provenance": dict(rec.provenance),
    }


def pins_kind_is_pairs(pin: SourcePin) -> bool:
    """Return True when a source contributes query/code pairs rather than corpus-only text."""
    return pin.kind == KIND_PAIRS


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write canonical UTF-8 JSONL and return row count, byte size, and SHA-256."""
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
    return {"rows": len(records), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare_sample(
    config: PrepareConfig,
    files: Mapping[str, Sequence[ResolvedFile]],
    notes: Sequence[str],
    output_dir: Path,
    pins: Mapping[str, SourcePin] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline into output_dir and return the manifest dict.

    Raises ``ConfigError`` when output_dir exists and is not an empty
    directory and ``PreparationError`` when the required Rust source yields no
    usable pairs. Output ordering, selection, and file hashes are fully
    determined by the config and input contents.
    """
    pins = pins if pins is not None else load_pins()
    check_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selections: dict[str, dict[str, list[SelectedRow]]] = {}
    stats_by_source: dict[str, SourceStats] = {}
    budgets_by_source: dict[str, dict[str, int]] = {}
    for key in SOURCE_ORDER:
        source_files = list(files.get(key, []))
        if not source_files:
            continue
        selections[key], stats_by_source[key] = scan_source(pins[key], source_files, config)
        budgets_by_source[key] = _budgets(config, key)
    rust_stats = stats_by_source.get("rust")
    if rust_stats is None:
        raise PreparationError("required Rust source has no input files")
    if rust_stats.eligible == 0:
        detail = ", ".join(f"{reason}={count}" for reason, count in sorted(rust_stats.skipped.items()))
        raise PreparationError(
            f"required Rust input is unusable: scanned {rust_stats.scanned} rows, 0 eligible "
            f"(skip reasons: {detail or 'none'})"
        )
    final_pairs, pair_duplicates, eval_pair_digests = finalize_pairs(selections, pins)
    final_corpus, corpus_duplicates = build_corpus(final_pairs, selections, pins, eval_pair_digests)
    outputs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        pair_rows = [
            output
            for key in SOURCE_ORDER
            if pins[key].kind == KIND_PAIRS
            for row in final_pairs.get(key, {}).get(split, [])
            if (output := _pair_output(pins[key], row)) is not None
        ]
        corpus_rows = [
            output
            for key in SOURCE_ORDER
            for row in final_corpus.get(key, {}).get(split, [])
            if (output := _corpus_output(pins[key], row)) is not None
        ]
        outputs[f"pairs.{split}.jsonl"] = _write_jsonl(output_dir / f"pairs.{split}.jsonl", pair_rows)
        outputs[f"corpus.{split}.jsonl"] = _write_jsonl(output_dir / f"corpus.{split}.jsonl", corpus_rows)
    manifest = _build_manifest(
        config=config,
        pins=pins,
        stats_by_source=stats_by_source,
        budgets_by_source=budgets_by_source,
        final_pairs=final_pairs,
        final_corpus=final_corpus,
        pair_duplicates=pair_duplicates,
        corpus_duplicates=corpus_duplicates,
        outputs=outputs,
        extra_notes=notes,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_manifest(
    *,
    config: PrepareConfig,
    pins: Mapping[str, SourcePin],
    stats_by_source: Mapping[str, SourceStats],
    budgets_by_source: Mapping[str, Mapping[str, int]],
    final_pairs: Mapping[str, Mapping[str, Sequence[SelectedRow]]],
    final_corpus: Mapping[str, Mapping[str, Sequence[SelectedRow]]],
    pair_duplicates: Mapping[str, Mapping[str, int]],
    corpus_duplicates: Mapping[str, Mapping[str, int]],
    outputs: Mapping[str, Mapping[str, Any]],
    extra_notes: Sequence[str],
) -> dict[str, Any]:
    """Assemble the preparation manifest dict."""
    sources_manifest: dict[str, Any] = {}
    for key in SOURCE_ORDER:
        stats = stats_by_source.get(key)
        if stats is None:
            continue
        pin = pins[key]
        final = final_pairs if pin.kind == KIND_PAIRS else final_corpus
        duplicates = dict(pair_duplicates.get(key, {}))
        for reason, count in corpus_duplicates.get(key, {}).items():
            duplicates[reason] = duplicates.get(reason, 0) + count
        selected_rows = [row for split in SPLITS for row in final.get(key, {}).get(split, [])]
        strengths = Counter(row.record.group_strength for row in selected_rows)
        entry: dict[str, Any] = {
            "pin": {
                "dataset": pin.dataset,
                "revision": pin.revision,
                "url": pin.url,
                "language": pin.language,
                "kind": pin.kind,
                "adapter": pin.adapter,
                "pinned_files": [fp.name for fp in pin.files],
            },
            "inputs": [file_scan.as_dict() for file_scan in stats.inputs],
            "coverage": {
                "pinned_files": len(pin.files),
                "used": len(stats.inputs),
                "missing_pinned": [
                    fp.name for fp in pin.files if fp.name not in {fs.pinned_name for fs in stats.inputs}
                ],
            },
            "counts": {
                "scanned": stats.scanned,
                "eligible": stats.eligible,
                "skipped": dict(stats.skipped),
                "selected": {split: len(final.get(key, {}).get(split, [])) for split in SPLITS},
                "requested": dict(budgets_by_source.get(key, {})),
            },
            "duplicates": duplicates,
            "grouping": {
                "note": pin.grouping_note,
                "strengths_present": dict(strengths),
                "reserved_eval_groups": stats.reserved_eval_groups,
                "train_rows_routed_to_reserved_groups": stats.train_rows_routed_to_reserved_groups,
                "split_rule": (
                    "upstream eval inputs are eval-only and reserve their group identities; "
                    "train rows in a reserved group follow it to eval; remaining groups are "
                    "assigned by sha256(seed:split:<group_id>) < eval_fraction. The disjointness "
                    "guarantee covers the scanned rows only."
                ),
            },
            "shortfalls": {},
        }
        budgets = budgets_by_source.get(key, {})
        for split in SPLITS:
            selected = entry["counts"]["selected"][split]
            if selected < budgets.get(split, selected):
                entry["shortfalls"][split] = budgets[split] - selected
        if key == "rust":
            categories = sorted(set(stats.scanned_by_category) | set(stats.eligible_by_category))
            entry["counts"]["by_category"] = {
                category: {
                    "scanned": stats.scanned_by_category.get(category, 0),
                    "eligible": stats.eligible_by_category.get(category, 0),
                    "selected_train": sum(
                        1 for row in final.get("rust", {}).get(ROLE_TRAIN, []) if _pair_category(row) == category
                    ),
                    "selected_eval": sum(
                        1 for row in final.get("rust", {}).get(ROLE_EVAL, []) if _pair_category(row) == category
                    ),
                }
                for category in categories
            }
            entry["counts"]["unsupported_categories"] = dict(stats.unsupported_detail)
        sources_manifest[key] = entry
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "benchmarks.model2vec_data.prepare",
        "config": {
            "seed": config.seed,
            "eval_fraction": config.eval_fraction,
            "caps": dict(config.caps),
            "max_field_bytes": config.max_field_bytes,
            "max_scan_rows": config.max_scan_rows,
            "kotlin_max_shards": config.kotlin_max_shards,
        },
        "disabled_sources": [key for key in SOURCE_ORDER if key not in stats_by_source],
        "sources": sources_manifest,
        "outputs": dict(outputs),
        "notes": list(_MANIFEST_NOTES) + list(extra_notes),
    }


def _parse_file_spec(key: str, spec: str, pin: SourcePin) -> tuple[str, Path]:
    """Parse and validate a ROLE:PATH file override specification."""
    role_str, separator, path_str = spec.partition(":")
    if not separator or not path_str:
        raise ConfigError(f"invalid {key} file spec {spec!r}; expected ROLE:PATH with ROLE in (train, eval)")
    if role_str not in SPLITS:
        raise ConfigError(f"invalid role {role_str!r} for {key}; expected train or eval")
    if pin.kind == KIND_CORPUS and role_str != ROLE_TRAIN:
        raise ConfigError(f"{key} is corpus-only and accepts only train-role inputs")
    path = Path(path_str)
    if path.suffix.lower() not in INPUT_SUFFIXES:
        raise ConfigError(f"unsupported input format for {path} (expected one of {', '.join(INPUT_SUFFIXES)})")
    if not path.is_file():
        raise ConfigError(f"input file not found: {path}")
    return role_str, path


def _missing_input_error(pin: SourcePin, input_root: Path) -> str:
    """Return a configuration error message including the pinned download command."""
    local = input_root / pin.slug
    return (
        f"{pin.key} source enabled but no usable inputs found under {local}. "
        f"Fetch the pinned revision first:\n  {pin.download_command(str(local))}"
    )


def _resolve_explicit(key: str, specs: Sequence[str], pin: SourcePin) -> list[ResolvedFile]:
    """Resolve explicit ROLE:PATH override specifications for one source."""
    return [
        ResolvedFile(path=path, pinned_name=path.name, role=role, resolution="explicit")
        for role, path in (_parse_file_spec(key, spec, pin) for spec in specs)
    ]


def _resolve_from_root(
    key: str, pin: SourcePin, input_root: Path, kotlin_max_shards: int, notes: list[str]
) -> list[ResolvedFile]:
    """Auto-derive inputs from <input_root>/<slug>/<pinned path>, recording misses as notes."""
    candidates = pin.files if key != "kotlin" else pin.files[:kotlin_max_shards]
    if key == "kotlin" and kotlin_max_shards < len(pin.files):
        notes.append(
            f"kotlin: considering only {kotlin_max_shards} of {len(pin.files)} pinned shards "
            f"(kotlin-max-shards); partial-source coverage is reported in the manifest"
        )
    resolved: list[ResolvedFile] = []
    for file_pin in candidates:
        path = input_root / pin.slug / file_pin.name
        if path.is_file():
            resolved.append(
                ResolvedFile(path=path, pinned_name=file_pin.name, role=file_pin.role, resolution="input_root")
            )
        else:
            notes.append(f"{key}: missing pinned input {file_pin.name} (role {file_pin.role})")
    return resolved


def _resolve_source_files(
    key: str,
    pin: SourcePin,
    input_root: Path,
    explicit: Sequence[str],
    kotlin_max_shards: int,
    notes: list[str],
) -> list[ResolvedFile]:
    """Resolve one source's inputs: explicit overrides, else the pinned layout under input_root.

    Optional sources with no local inputs append a skip note and return an
    empty list; invalid explicit ROLE:PATH specifications raise, as does an
    unresolvable required Rust source.
    """
    if explicit:
        return _resolve_explicit(key, explicit, pin)
    resolved = _resolve_from_root(key, pin, input_root, kotlin_max_shards, notes)
    if resolved or key == "rust":
        return resolved
    notes.append(f"{key}: no usable inputs found under {input_root / pin.slug}; source skipped")
    return resolved


def resolve_inputs(
    pins: Mapping[str, SourcePin],
    *,
    input_root: Path,
    enabled: Mapping[str, bool],
    overrides: Mapping[str, Sequence[str]] | None = None,
    kotlin_max_shards: int = DEFAULT_KOTLIN_MAX_SHARDS,
) -> tuple[dict[str, list[ResolvedFile]], list[str]]:
    """Resolve concrete input files for enabled sources.

    Explicit ``ROLE:PATH`` overrides win; otherwise pinned file names are looked
    up under ``<input_root>/<slug>/<pinned path>``. The Rust source is required
    and raises when unresolvable; an enabled optional source with no local
    inputs is skipped with a note instead of failing, while invalid explicit
    specifications always raise.
    """
    overrides = overrides or {}
    files: dict[str, list[ResolvedFile]] = {}
    notes: list[str] = []
    for key in SOURCE_ORDER:
        pin = pins[key]
        if not enabled.get(key, key == "rust"):
            continue
        resolved = _resolve_source_files(
            key, pin, input_root, overrides.get(key, ()), kotlin_max_shards, notes
        )
        if not resolved:
            if key == "rust":
                raise ConfigError(_missing_input_error(pin, input_root))
            continue
        if key == "rust" and not any(item.role == ROLE_TRAIN for item in resolved):
            raise ConfigError(
                f"rust source has no train-role input; pass --rust-file train:<path> or place "
                f"{pin.files[0].name} under {input_root / pin.slug}"
            )
        files[key] = resolved
    if "rust" not in files:
        raise ConfigError(
            "the Rust source is required; point --input-root at the pinned download layout or use "
            "--rust-file (see --list-sources for the exact hf download commands)"
        )
    return files, notes


def check_output_dir(path: Path) -> None:
    """Refuse unsafe output destinations without creating anything."""
    if path.exists():
        if not path.is_dir():
            raise ConfigError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ConfigError(f"output directory is not empty (refusing to overwrite existing data): {path}")


def _config_from_args(args: argparse.Namespace) -> PrepareConfig:
    """Validate CLI numeric settings and build the preparation config."""
    caps = {
        "rust": args.max_rust_pairs,
        "typescript": args.max_typescript_pairs,
        "swift": args.max_swift_pairs,
        "kotlin": args.max_kotlin_docs,
    }
    problems: list[str] = []
    if not 0.0 <= args.eval_fraction <= 1.0:
        problems.append("--eval-fraction must be within [0, 1]")
    for key, value in caps.items():
        if value < 1:
            problems.append(f"cap for {key} must be >= 1 (got {value})")
    if args.max_field_bytes < 1:
        problems.append("--max-field-bytes must be >= 1")
    if args.max_scan_rows < 1:
        problems.append("--max-scan-rows must be >= 1")
    if args.kotlin_max_shards < 1:
        problems.append("--kotlin-max-shards must be >= 1")
    if problems:
        raise ConfigError("; ".join(problems))
    return PrepareConfig(
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        caps=caps,
        max_field_bytes=args.max_field_bytes,
        max_scan_rows=args.max_scan_rows,
        kotlin_max_shards=args.kotlin_max_shards,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.model2vec_data.prepare",
        description=(
            "Prepare reproducible Model2Vec training samples from pinned datasets. "
            "Default is Rust-only; optional sources activate explicitly."
        ),
    )
    parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="root of the pinned download layout"
    )
    parser.add_argument("--output", type=Path, help="new or empty output directory for the prepared sample")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="fixed sampling/split seed")
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION, help="eval share of each cap")
    parser.add_argument("--max-rust-pairs", type=int, default=DEFAULT_CAPS["rust"], help="cap on Rust pairs")
    parser.add_argument(
        "--max-typescript-pairs", type=int, default=DEFAULT_CAPS["typescript"], help="cap on TypeScript pairs"
    )
    parser.add_argument("--max-swift-pairs", type=int, default=DEFAULT_CAPS["swift"], help="cap on Swift pairs")
    parser.add_argument("--max-kotlin-docs", type=int, default=DEFAULT_CAPS["kotlin"], help="cap on Kotlin corpus docs")
    parser.add_argument(
        "--max-field-bytes", type=int, default=DEFAULT_MAX_FIELD_BYTES, help="parser safety limit per field"
    )
    parser.add_argument("--max-scan-rows", type=int, default=DEFAULT_MAX_SCAN_ROWS, help="scan bound per input file")
    parser.add_argument(
        "--kotlin-max-shards", type=int, default=DEFAULT_KOTLIN_MAX_SHARDS, help="KStack shards to consider"
    )
    parser.add_argument("--with-typescript", action="store_true", help="enable the optional TypeScript source")
    parser.add_argument("--with-swift", action="store_true", help="enable the optional Swift source")
    parser.add_argument("--with-kotlin", action="store_true", help="enable the optional Kotlin corpus source")
    parser.add_argument(
        "--rust-file", action="append", default=[], metavar="ROLE:PATH", help="explicit Rust input (repeatable)"
    )
    parser.add_argument(
        "--typescript-file", action="append", default=[], metavar="ROLE:PATH", help="explicit TypeScript input"
    )
    parser.add_argument("--swift-file", action="append", default=[], metavar="ROLE:PATH", help="explicit Swift input")
    parser.add_argument(
        "--kotlin-file", action="append", default=[], metavar="ROLE:PATH", help="explicit Kotlin corpus input"
    )
    parser.add_argument("--list-sources", action="store_true", help="print pinned sources and hf commands, then exit")
    return parser.parse_args(argv)


def _print_summary(manifest: Mapping[str, Any], output_dir: Path) -> None:
    """Print a short human-readable preparation summary."""
    for key, entry in manifest["sources"].items():
        counts = entry["counts"]
        skipped = dict(counts["skipped"])
        print(
            f"{key}: scanned={counts['scanned']} eligible={counts['eligible']} "
            f"selected train={counts['selected']['train']} eval={counts['selected']['eval']} "
            f"skipped={skipped or '{}'}"
        )
    for name, info in manifest["outputs"].items():
        print(f"wrote {name}: {info['rows']} rows (sha256 {info['sha256'][:12]})")
    print(f"manifest: {output_dir / 'manifest.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parse_args(argv)
    pins = load_pins()
    if args.list_sources:
        for key in SOURCE_ORDER:
            pin = pins[key]
            print(f"{key}: {pin.dataset} @ {pin.revision} ({pin.kind}, adapter {pin.adapter})")
            print(f"  {pin.download_command(str(args.input_root / pin.slug))}")
        return 0
    if args.output is None:
        print("error: --output is required", file=sys.stderr)
        return 2
    try:
        config = _config_from_args(args)
        check_output_dir(args.output)
        enabled = {
            "rust": True,
            "typescript": args.with_typescript,
            "swift": args.with_swift,
            "kotlin": args.with_kotlin,
        }
        overrides = {
            "rust": args.rust_file,
            "typescript": args.typescript_file,
            "swift": args.swift_file,
            "kotlin": args.kotlin_file,
        }
        files, notes = resolve_inputs(
            pins,
            input_root=args.input_root,
            enabled=enabled,
            overrides=overrides,
            kotlin_max_shards=args.kotlin_max_shards,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for note in notes:
        print(f"note: {note}")
    try:
        manifest = prepare_sample(config, files, notes, args.output, pins)
    except PreparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_summary(manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
