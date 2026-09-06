"""Build a balanced Rust-plus-CornStack contrastive training file."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.model2vec_training.contrastive_sources import (
    MIN_TEXT_CHARACTERS,
    REPLAY_BUFFER_SIZE,
    REPLAY_SEED,
    REPLAY_SOURCES,
    RUST_DATASET_REVISION,
    RUST_EVAL_ROWS,
    RUST_EVAL_SHA256,
    RUST_MANIFEST_SHA256,
    RUST_TRAIN_ROWS,
    RUST_TRAIN_SHA256,
    ReplaySource,
)
from benchmarks.model2vec_training.core import read_pairs, sha256_file

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ContrastivePair:
    """A selected query/document pair plus non-training provenance."""

    anchor: str
    positive: str
    language: str
    source_id: str
    source_row: int


@dataclass(slots=True)
class SelectionStats:
    """Replay selection counters."""

    scanned: int = 0
    selected: int = 0
    invalid_type: int = 0
    too_short: int = 0
    duplicate_query: int = 0
    duplicate_document: int = 0
    held_out_overlap: int = 0


def _parse_args() -> argparse.Namespace:
    """Parse preparation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-train", type=Path, required=True)
    parser.add_argument("--rust-eval", type=Path, required=True)
    parser.add_argument("--rust-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-language", type=int, default=RUST_TRAIN_ROWS)
    parser.add_argument("--seed", type=int, default=REPLAY_SEED)
    parser.add_argument("--buffer-size", type=int, default=REPLAY_BUFFER_SIZE)
    return parser.parse_args()


def _prepare_output(path: Path) -> None:
    """Create a new output directory without replacing prior evidence."""
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _digest(text: str) -> str:
    """Return a stable content digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_rust_inputs(train_path: Path, eval_path: Path, manifest_path: Path) -> tuple[list, list]:
    """Require the complete, pinned, crate-disjoint Rust extraction."""
    identities = {
        "train": (train_path, RUST_TRAIN_SHA256, RUST_TRAIN_ROWS),
        "eval": (eval_path, RUST_EVAL_SHA256, RUST_EVAL_ROWS),
        "manifest": (manifest_path, RUST_MANIFEST_SHA256, None),
    }
    for label, (path, expected_sha, _rows) in identities.items():
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"Rust {label} SHA-256 is {actual_sha}, expected {expected_sha}")
    train_pairs = read_pairs(train_path)
    eval_pairs = read_pairs(eval_path)
    if len(train_pairs) != RUST_TRAIN_ROWS or len(eval_pairs) != RUST_EVAL_ROWS:
        raise ValueError(
            f"Rust row counts are train={len(train_pairs)} eval={len(eval_pairs)}, "
            f"expected train={RUST_TRAIN_ROWS} eval={RUST_EVAL_ROWS}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("sources", {}).get("rust", {})
    if source.get("pin", {}).get("revision") != RUST_DATASET_REVISION:
        raise ValueError("Rust manifest does not reference the pinned Strandset revision")
    if source.get("counts", {}).get("scanned") != 191_233:
        raise ValueError("Rust manifest does not cover all 191,233 pinned rows")
    if source.get("coverage", {}).get("used") != source.get("coverage", {}).get("pinned_files"):
        raise ValueError("Rust manifest did not use every pinned input")
    train_groups = {pair.group_id for pair in train_pairs}
    eval_groups = {pair.group_id for pair in eval_pairs}
    if not train_groups.isdisjoint(eval_groups):
        raise ValueError("Rust train and evaluation groups overlap")
    return train_pairs, eval_pairs


def select_replay_rows(
    rows: Iterable[Mapping[str, Any]],
    source: ReplaySource,
    limit: int,
    held_out_query_digests: set[str],
    held_out_document_digests: set[str],
) -> tuple[list[ContrastivePair], SelectionStats]:
    """Apply the published v1 filters and select one balanced replay language."""
    selected: list[ContrastivePair] = []
    stats = SelectionStats()
    seen_queries: set[str] = set()
    seen_documents: set[str] = set()
    for row_index, row in enumerate(rows):
        stats.scanned += 1
        query = row.get("query")
        document = row.get("document")
        if not isinstance(query, str) or not isinstance(document, str):
            stats.invalid_type += 1
            continue
        if len(query) < MIN_TEXT_CHARACTERS or len(document) < MIN_TEXT_CHARACTERS:
            stats.too_short += 1
            continue
        query_digest = _digest(query)
        document_digest = _digest(document)
        if query_digest in held_out_query_digests or document_digest in held_out_document_digests:
            stats.held_out_overlap += 1
            continue
        if query_digest in seen_queries:
            stats.duplicate_query += 1
            continue
        if document_digest in seen_documents:
            stats.duplicate_document += 1
            continue
        seen_queries.add(query_digest)
        seen_documents.add(document_digest)
        selected.append(
            ContrastivePair(
                anchor=query,
                positive=document,
                language=source.language,
                source_id=source.dataset,
                source_row=row_index,
            )
        )
        if len(selected) == limit:
            break
    stats.selected = len(selected)
    if len(selected) != limit:
        raise ValueError(f"{source.dataset} produced {len(selected)} usable rows, expected {limit}")
    return selected, stats


def _iter_hub_source(source: ReplaySource, seed: int, buffer_size: int) -> Iterator[Mapping[str, Any]]:
    """Stream one immutable CornStack dataset in the published shuffled-selection form."""
    from datasets import load_dataset

    dataset = load_dataset(source.dataset, split="train", streaming=True, revision=source.revision)
    return iter(dataset.shuffle(seed=seed, buffer_size=buffer_size))


def _order_key(pair: ContrastivePair, seed: int) -> bytes:
    """Return a deterministic whole-dataset shuffle key."""
    identity = f"{seed}:{pair.language}:{pair.source_id}:{pair.source_row}:{_digest(pair.anchor)}"
    return hashlib.sha256(identity.encode()).digest()


def _write_outputs(output: Path, pairs: list[ContrastivePair], seed: int) -> dict[str, dict[str, int | str]]:
    """Write training-only columns and a parallel provenance stream."""
    ordered = sorted(pairs, key=lambda pair: _order_key(pair, seed))
    train_path = output / "train.jsonl"
    provenance_path = output / "provenance.jsonl"
    with train_path.open("w", encoding="utf-8") as train_handle, provenance_path.open(
        "w", encoding="utf-8"
    ) as provenance_handle:
        for pair in ordered:
            train_handle.write(json.dumps({"anchor": pair.anchor, "positive": pair.positive}, sort_keys=True) + "\n")
            provenance_handle.write(
                json.dumps(
                    {
                        "language": pair.language,
                        "source_id": pair.source_id,
                        "source_row": pair.source_row,
                        "anchor_sha256": _digest(pair.anchor),
                        "positive_sha256": _digest(pair.positive),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        path.name: {"rows": len(ordered), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (train_path, provenance_path)
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Build and attest the complete balanced training set."""
    if args.per_language != RUST_TRAIN_ROWS:
        raise ValueError(f"the full experiment requires per_language={RUST_TRAIN_ROWS}")
    _prepare_output(args.output)
    rust_train, rust_eval = validate_rust_inputs(args.rust_train, args.rust_eval, args.rust_manifest)
    held_out_query_digests = {_digest(pair.query) for pair in rust_eval}
    held_out_document_digests = {_digest(pair.code) for pair in rust_eval}
    all_pairs = [
        ContrastivePair(pair.query, pair.code, "rust", RUST_DATASET_REVISION, row_index)
        for row_index, pair in enumerate(rust_train)
    ]
    selection: dict[str, dict[str, Any]] = {}
    for source in REPLAY_SOURCES:
        LOGGER.info("Selecting %d rows from %s", args.per_language, source.dataset)
        replay, stats = select_replay_rows(
            _iter_hub_source(source, args.seed, args.buffer_size),
            source,
            args.per_language,
            held_out_query_digests,
            held_out_document_digests,
        )
        all_pairs.extend(replay)
        selection[source.language] = {"source": asdict(source), "stats": asdict(stats)}
    outputs = _write_outputs(args.output, all_pairs, args.seed)
    expected_total = RUST_TRAIN_ROWS * (len(REPLAY_SOURCES) + 1)
    if outputs["train.jsonl"]["rows"] != expected_total:
        raise ValueError("balanced output row count is incorrect")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "benchmarks.model2vec_training.prepare_contrastive",
        "seed": args.seed,
        "shuffle_buffer_size": args.buffer_size,
        "per_language": args.per_language,
        "languages": ["rust", *[source.language for source in REPLAY_SOURCES]],
        "total_pairs": expected_total,
        "rust": {
            "dataset_revision": RUST_DATASET_REVISION,
            "train_rows": len(rust_train),
            "eval_rows_excluded": len(rust_eval),
            "train_sha256": sha256_file(args.rust_train),
            "eval_sha256": sha256_file(args.rust_eval),
            "manifest_sha256": sha256_file(args.rust_manifest),
        },
        "replay": selection,
        "outputs": outputs,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Prepared %d balanced pairs at %s", expected_total, args.output)
    return manifest


def main() -> None:
    """Run contrastive data preparation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare(_parse_args())


if __name__ == "__main__":
    main()
