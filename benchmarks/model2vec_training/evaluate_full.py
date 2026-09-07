"""Compare baseline and contrastive candidates on full Rust and real repositories."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from model2vec import StaticModel
from scipy.stats import binomtest
from vicinity.backends.basic import BasicArgs

from benchmarks.data import RepoSpec, grouped_tasks, load_repo_specs, load_tasks
from benchmarks.metrics import ndcg_at_k, target_rank
from benchmarks.model2vec_training.contrastive_sources import (
    RUST_EVAL_ROWS,
    RUST_EVAL_SHA256,
    RUST_TRAIN_ROWS,
    RUST_TRAIN_SHA256,
)
from benchmarks.model2vec_training.core import Pair, read_pairs, sha256_file
from semble.index.bm25 import BM25
from semble.index.create import create_index_from_path
from semble.index.dense import SelectableBasicBackend, embed_chunks
from semble.index.sparse import enrich_for_bm25
from semble.search import search
from semble.tokens import tokenize
from semble.types import Chunk

LOGGER = logging.getLogger(__name__)
ENCODE_BATCH_SIZE = 1_024
REQUIRED_MODEL_LABELS = frozenset({"potion-v1", "potion-v2", "candidate-raw", "candidate-post-sif"})
REQUIRED_BASELINE_LABEL = "potion-v2"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A named local Model2Vec directory."""

    label: str
    path: Path


@dataclass(frozen=True, slots=True)
class PairCorpus:
    """One Rust document scope and each held-out positive's row position."""

    documents: list[str]
    positive_indices: np.ndarray


def _parse_args() -> argparse.Namespace:
    """Parse complete evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="LABEL=LOCAL_MODEL_PATH")
    parser.add_argument("--baseline-label", default="potion-v2")
    parser.add_argument("--rust-train", type=Path, required=True)
    parser.add_argument("--rust-eval", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--repo-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--hybrid-rank-depth", type=int, default=100)
    return parser.parse_args()


def parse_model_specs(values: list[str]) -> list[ModelSpec]:
    """Parse unique LABEL=PATH model specifications."""
    specs: list[ModelSpec] = []
    labels: set[str] = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"invalid model specification: {value!r}")
        if label in labels:
            raise ValueError(f"duplicate model label: {label}")
        path = Path(raw_path)
        if not (path / "config.json").is_file():
            raise ValueError(f"model {label} is incomplete: {path}")
        labels.add(label)
        specs.append(ModelSpec(label, path))
    return specs


def validate_comparison_specs(specs: list[ModelSpec], baseline_label: str) -> None:
    """Require the complete fixed comparison used for decision evidence."""
    labels = {spec.label for spec in specs}
    if labels != REQUIRED_MODEL_LABELS:
        raise ValueError(
            f"model labels must be exactly {sorted(REQUIRED_MODEL_LABELS)}, received {sorted(labels)}"
        )
    if baseline_label != REQUIRED_BASELINE_LABEL:
        raise ValueError(f"baseline label must be {REQUIRED_BASELINE_LABEL!r}")


def _normalize(values: np.ndarray) -> np.ndarray:
    """Normalize finite vector rows."""
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("model produced non-finite vectors")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("model produced zero vectors")
    return array / norms


def dense_ranks_against_corpus(
    model: StaticModel,
    pairs: list[Pair],
    documents_text: list[str],
    positive_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Rank held-out positives at known positions in a supplied Rust corpus."""
    if positive_indices.shape != (len(pairs),):
        raise ValueError("positive indices must align with held-out queries")
    if np.any(positive_indices < 0) or np.any(positive_indices >= len(documents_text)):
        raise ValueError("positive index lies outside the document corpus")
    documents = _normalize(
        np.asarray(
            model.encode(documents_text, batch_size=ENCODE_BATCH_SIZE, use_multiprocessing=False),
            dtype=np.float32,
        )
    )
    queries = _normalize(
        np.asarray(
            model.encode([pair.query for pair in pairs], batch_size=ENCODE_BATCH_SIZE, use_multiprocessing=False),
            dtype=np.float32,
        )
    )
    scores = queries @ documents.T
    row_indices = np.arange(len(pairs))
    positives = scores[row_indices, positive_indices].copy()
    column_indices = np.arange(len(documents_text))[None, :]
    greater = (scores > positives[:, None]).sum(axis=1)
    earlier_ties = ((scores == positives[:, None]) & (column_indices < positive_indices[:, None])).sum(axis=1)
    ranks = (1 + greater + earlier_ties).astype(np.int32)
    negative_scores = scores.copy()
    negative_scores[row_indices, positive_indices] = -np.inf
    hard_negatives = negative_scores.max(axis=1)
    margin = positives - hard_negatives
    return ranks, {
        "mean_positive_cosine": float(positives.mean()),
        "mean_hard_negative_cosine": float(hard_negatives.mean()),
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "negative_margin_count": int(np.sum(margin < 0)),
    }


def dense_ranks(model: StaticModel, pairs: list[Pair]) -> tuple[np.ndarray, dict[str, float]]:
    """Rank aligned positives within only the held-out Rust documents."""
    return dense_ranks_against_corpus(
        model,
        pairs,
        [pair.code for pair in pairs],
        np.arange(len(pairs), dtype=np.int32),
    )


def build_pair_bm25(documents: list[str]) -> tuple[BM25, list[Chunk]]:
    """Build the exact Semble sparse index over a Rust document corpus."""
    bm25 = BM25()
    chunks: list[Chunk] = []
    ids: list[str] = []
    for index, document in enumerate(documents):
        chunk = Chunk(document, f"rust-corpus/{index:05d}.rs", 1, 1, "rust")
        chunk_id = str(index)
        chunks.append(chunk)
        ids.append(chunk_id)
        bm25.add_document(chunk_id, tokenize(enrich_for_bm25(chunk)))
    bm25.set_doc_order(ids)
    return bm25, chunks


def hybrid_ranks(
    model: StaticModel,
    pairs: list[Pair],
    bm25: BM25,
    chunks: list[Chunk],
    positive_indices: np.ndarray,
    rank_depth: int,
) -> np.ndarray:
    """Run Semble's normal hybrid path for every held-out Rust query."""
    semantic = SelectableBasicBackend(embed_chunks(model, chunks), BasicArgs())
    ranks = np.zeros(len(pairs), dtype=np.int32)
    for index, pair in enumerate(pairs):
        results = search(
            pair.query,
            model,
            semantic,
            bm25,
            chunks,
            top_k=min(rank_depth, len(chunks)),
            alpha=None,
            rerank=True,
        )
        positive = chunks[int(positive_indices[index])]
        ranks[index] = next((rank for rank, result in enumerate(results, 1) if result.chunk == positive), 0)
    return ranks


def rank_summary(ranks: np.ndarray) -> dict[str, float]:
    """Summarize ranks, treating zero as absent from the measured depth."""
    reciprocal = np.where(ranks > 0, 1 / np.maximum(ranks, 1), 0)
    return {
        "recall_at_1": float(np.mean((ranks > 0) & (ranks <= 1))),
        "recall_at_5": float(np.mean((ranks > 0) & (ranks <= 5))),
        "recall_at_10": float(np.mean((ranks > 0) & (ranks <= 10))),
        "mrr": float(reciprocal.mean()),
        "mean_rank_found": float(ranks[ranks > 0].mean()) if np.any(ranks > 0) else 0.0,
        "not_found": int(np.sum(ranks == 0)),
    }


def bootstrap_mean_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Return paired mean difference with a deterministic percentile interval."""
    if baseline.shape != candidate.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("bootstrap vectors must be same-length non-empty rows")
    differences = candidate.astype(np.float64) - baseline.astype(np.float64)
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    batch_size = 250
    for start in range(0, resamples, batch_size):
        size = min(batch_size, resamples - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        means.append(differences[indices].mean(axis=1))
    samples = np.concatenate(means)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"delta": float(differences.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def paired_rank_comparison(
    baseline: np.ndarray,
    candidate: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare paired top-1 and reciprocal-rank outcomes."""
    baseline_top1 = baseline == 1
    candidate_top1 = candidate == 1
    gained = int(np.sum(~baseline_top1 & candidate_top1))
    lost = int(np.sum(baseline_top1 & ~candidate_top1))
    discordant = gained + lost
    p_value = float(binomtest(min(gained, lost), discordant, 0.5).pvalue) if discordant else 1.0
    baseline_rr = np.where(baseline > 0, 1 / np.maximum(baseline, 1), 0)
    candidate_rr = np.where(candidate > 0, 1 / np.maximum(candidate, 1), 0)
    return {
        "top1": {
            **bootstrap_mean_delta(
                baseline_top1.astype(np.float64), candidate_top1.astype(np.float64), resamples, seed
            ),
            "gained": gained,
            "lost": lost,
            "mcnemar_exact_p": p_value,
        },
        "reciprocal_rank": bootstrap_mean_delta(baseline_rr, candidate_rr, resamples, seed + 1),
        "rank_movements": {
            "improved": int(np.sum((candidate > 0) & ((baseline == 0) | (candidate < baseline)))),
            "unchanged": int(np.sum(candidate == baseline)),
            "worsened": int(np.sum((baseline > 0) & ((candidate == 0) | (candidate > baseline)))),
        },
    }


def _category_summaries(ranks: np.ndarray, categories: list[str]) -> dict[str, dict[str, float]]:
    """Summarize held-out ranks within each prepared task category."""
    output: dict[str, dict[str, float]] = {}
    category_array = np.asarray(categories)
    for category in sorted(set(categories)):
        selected = ranks[category_array == category]
        output[category] = {"queries": int(len(selected)), **rank_summary(selected)}
    return output


def evaluate_pairs(
    models: dict[str, StaticModel],
    rust_train_path: Path,
    rust_eval_path: Path,
    baseline_label: str,
    rank_depth: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate all models on every held-out Strandset Rust pair."""
    if sha256_file(rust_train_path) != RUST_TRAIN_SHA256:
        raise ValueError("training Rust JSONL does not match the complete pinned extraction")
    if sha256_file(rust_eval_path) != RUST_EVAL_SHA256:
        raise ValueError("held-out Rust JSONL does not match the complete pinned extraction")
    train_pairs = read_pairs(rust_train_path)
    pairs = read_pairs(rust_eval_path)
    if len(train_pairs) != RUST_TRAIN_ROWS or len(pairs) != RUST_EVAL_ROWS:
        raise ValueError(
            f"Rust counts are train={len(train_pairs)} eval={len(pairs)}, "
            f"expected train={RUST_TRAIN_ROWS} eval={RUST_EVAL_ROWS}"
        )
    raw_rows = [json.loads(line) for line in rust_eval_path.read_text(encoding="utf-8").splitlines()]
    categories = [row["category"] for row in raw_rows]
    scopes: dict[str, PairCorpus] = {
        "held_out_only": PairCorpus(
            documents=[pair.code for pair in pairs],
            positive_indices=np.arange(len(pairs), dtype=np.int32),
        ),
        "full_corpus": PairCorpus(
            documents=[pair.code for pair in train_pairs] + [pair.code for pair in pairs],
            positive_indices=np.arange(len(pairs), dtype=np.int32) + len(train_pairs),
        ),
    }
    indexes = {name: build_pair_bm25(scope.documents) for name, scope in scopes.items()}
    rank_sets: dict[str, dict[str, dict[str, np.ndarray]]] = {name: {} for name in scopes}
    results: dict[str, Any] = {}
    for label, model in models.items():
        started = time.perf_counter()
        results[label] = {"scopes": {}}
        for scope_name, scope in scopes.items():
            documents = scope.documents
            positive_indices = scope.positive_indices
            bm25, chunks = indexes[scope_name]
            dense, geometry = dense_ranks_against_corpus(model, pairs, documents, positive_indices)
            hybrid = hybrid_ranks(model, pairs, bm25, chunks, positive_indices, rank_depth)
            rank_sets[scope_name][label] = {"dense": dense, "hybrid": hybrid}
            results[label]["scopes"][scope_name] = {
                "corpus_rows": len(documents),
                "dense": {
                    **rank_summary(dense),
                    **geometry,
                    "by_category": _category_summaries(dense, categories),
                },
                "hybrid": {
                    **rank_summary(hybrid),
                    "rank_depth": rank_depth,
                    "by_category": _category_summaries(hybrid, categories),
                },
                "per_query": [
                    {
                        "id": pair.id,
                        "category": categories[index],
                        "dense_rank": int(dense[index]),
                        "hybrid_rank": int(hybrid[index]),
                    }
                    for index, pair in enumerate(pairs)
                ],
            }
        results[label]["duration_seconds"] = time.perf_counter() - started
    comparisons: dict[str, Any] = {}
    for label in models:
        if label == baseline_label:
            continue
        comparisons[label] = {
            scope_name: {
                mode: paired_rank_comparison(
                    rank_sets[scope_name][baseline_label][mode],
                    rank_sets[scope_name][label][mode],
                    resamples,
                    seed,
                )
                for mode in ("dense", "hybrid")
            }
            for scope_name in scopes
        }
    return {
        "eval_rows": len(pairs),
        "train_corpus_rows": len(train_pairs),
        "full_corpus_rows": len(train_pairs) + len(pairs),
        "models": results,
        "versus_baseline": comparisons,
    }


def _load_repo_manifest(path: Path, root: Path) -> dict[str, dict[str, Any]]:
    """Verify the synchronization receipt and return records by name."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {record["name"]: record for record in payload["repositories"]}
    specs = load_repo_specs()
    if set(records) != set(specs):
        raise ValueError("repository receipt does not cover the complete Semble suite")
    for name, spec in specs.items():
        record = records[name]
        if record["revision"] != spec.revision or Path(record["path"]) != root / name:
            raise ValueError(f"repository receipt mismatch for {name}")
    return records


def _aggregate_repo_queries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate real-repository NDCG values overall and by language."""
    output: dict[str, Any] = {}
    for mode in ("dense", "hybrid"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        by_language: dict[str, list[float]] = defaultdict(list)
        for row in mode_rows:
            by_language[row["language"]].append(row["ndcg_at_10"])
        output[mode] = {
            "queries": len(mode_rows),
            "mean_ndcg_at_10": float(np.mean([row["ndcg_at_10"] for row in mode_rows])),
            "by_language": {
                language: {"queries": len(values), "mean_ndcg_at_10": float(np.mean(values))}
                for language, values in sorted(by_language.items())
            },
        }
    return output


def evaluate_repositories(
    models: dict[str, StaticModel],
    repo_root: Path,
    repo_manifest: Path,
    baseline_label: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate every annotated Semble repository with shared chunks and BM25."""
    _load_repo_manifest(repo_manifest, repo_root)
    specs = load_repo_specs()
    tasks = grouped_tasks(load_tasks(specs))
    if set(tasks) != set(specs):
        raise ValueError("every pinned repository must have evaluation tasks")
    model_rows: dict[str, list[dict[str, Any]]] = {label: [] for label in models}
    repo_records: list[dict[str, Any]] = []
    baseline_model = models[baseline_label]
    for repo_name, repo_tasks in sorted(tasks.items()):
        spec: RepoSpec = specs[repo_name]
        checkout = repo_root / repo_name
        benchmark_root = checkout if spec.benchmark_root is None else checkout / spec.benchmark_root
        started = time.perf_counter()
        bm25, baseline_semantic, chunks, _manifest = create_index_from_path(benchmark_root, baseline_model)
        repo_record: dict[str, Any] = {
            "repo": repo_name,
            "language": spec.language,
            "revision": spec.revision,
            "chunks": len(chunks),
            "preparation_seconds": time.perf_counter() - started,
            "models": {},
        }
        for label, model in models.items():
            embed_started = time.perf_counter()
            semantic = (
                baseline_semantic
                if label == baseline_label
                else SelectableBasicBackend(embed_chunks(model, chunks), BasicArgs())
            )
            embedding_seconds = time.perf_counter() - embed_started
            query_started = time.perf_counter()
            for task_index, task in enumerate(repo_tasks):
                for mode, alpha, rerank in (("dense", 1.0, False), ("hybrid", None, True)):
                    results = search(
                        task.query,
                        model,
                        semantic,
                        bm25,
                        chunks,
                        top_k=10,
                        alpha=alpha,
                        rerank=rerank,
                    )
                    relevant_ranks = [
                        rank for target in task.all_relevant if (rank := target_rank(results, target)) is not None
                    ]
                    model_rows[label].append(
                        {
                            "repo": repo_name,
                            "language": spec.language,
                            "task_index": task_index,
                            "category": task.category,
                            "mode": mode,
                            "ndcg_at_10": ndcg_at_k(relevant_ranks, len(task.all_relevant), 10),
                        }
                    )
            repo_record["models"][label] = {
                "embedding_seconds": embedding_seconds,
                "query_seconds": time.perf_counter() - query_started,
            }
        repo_records.append(repo_record)
        LOGGER.info("Evaluated %s (%s, %d chunks)", repo_name, spec.language, len(chunks))
    aggregates = {label: _aggregate_repo_queries(rows) for label, rows in model_rows.items()}
    comparisons: dict[str, Any] = {}
    for label, rows in model_rows.items():
        if label == baseline_label:
            continue
        comparisons[label] = {}
        for mode in ("dense", "hybrid"):
            base_rows = [row for row in model_rows[baseline_label] if row["mode"] == mode]
            candidate_rows = [row for row in rows if row["mode"] == mode]
            identities = [(row["repo"], row["task_index"]) for row in base_rows]
            if identities != [(row["repo"], row["task_index"]) for row in candidate_rows]:
                raise ValueError(f"repository query order mismatch for {label}/{mode}")
            base_values = np.asarray([row["ndcg_at_10"] for row in base_rows])
            candidate_values = np.asarray([row["ndcg_at_10"] for row in candidate_rows])
            rust_mask = np.asarray([row["language"] == "rust" for row in base_rows])
            comparisons[label][mode] = {
                "all_languages": bootstrap_mean_delta(base_values, candidate_values, resamples, seed),
                "rust": bootstrap_mean_delta(base_values[rust_mask], candidate_values[rust_mask], resamples, seed + 7),
                "non_rust": bootstrap_mean_delta(
                    base_values[~rust_mask], candidate_values[~rust_mask], resamples, seed + 13
                ),
            }
    return {
        "repositories": len(repo_records),
        "queries_per_mode": len(next(iter(model_rows.values()))) // 2,
        "repo_records": repo_records,
        "aggregates": aggregates,
        "versus_baseline": comparisons,
        "per_query": model_rows,
    }


def _quality_verdict(pair_results: dict[str, Any], repo_results: dict[str, Any], label: str) -> str:
    """Classify evidence without turning training completion into promotion."""
    pair_dense = pair_results["versus_baseline"][label]["full_corpus"]["dense"]["top1"]["delta"]
    pair_hybrid = pair_results["versus_baseline"][label]["full_corpus"]["hybrid"]["top1"]["delta"]
    repo_rust = repo_results["versus_baseline"][label]["hybrid"]["rust"]["delta"]
    repo_non_rust = repo_results["versus_baseline"][label]["hybrid"]["non_rust"]["delta"]
    if pair_dense > 0 and pair_hybrid >= 0 and repo_rust > 0 and repo_non_rust >= -0.005:
        return "promising-not-deployment-qualified"
    if pair_dense < 0 and repo_rust < 0:
        return "rust-regressed"
    return "mixed"


def _model_files(path: Path) -> dict[str, dict[str, int | str]]:
    """Hash every persisted model file used by evaluation."""
    return {
        str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _decision_summary(
    pair_results: dict[str, Any], repo_results: dict[str, Any], verdicts: dict[str, str]
) -> dict[str, Any]:
    """Return the high-signal metrics without dropping the full query evidence."""
    return {
        "quality_verdicts": verdicts,
        "rust_pairs": {
            label: {
                scope_name: {
                    "dense": {key: value for key, value in scope["dense"].items() if key != "by_category"},
                    "hybrid": {key: value for key, value in scope["hybrid"].items() if key != "by_category"},
                }
                for scope_name, scope in result["scopes"].items()
            }
            for label, result in pair_results["models"].items()
        },
        "real_repositories": {
            label: {
                mode: {
                    "all_languages": values["mean_ndcg_at_10"],
                    "rust": values["by_language"]["rust"]["mean_ndcg_at_10"],
                }
                for mode, values in aggregate.items()
            }
            for label, aggregate in repo_results["aggregates"].items()
        },
        "pairwise_comparisons": pair_results["versus_baseline"],
        "repository_comparisons": repo_results["versus_baseline"],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run both full evaluation surfaces and persist one receipt."""
    started = time.perf_counter()
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    specs = parse_model_specs(args.model)
    validate_comparison_specs(specs, args.baseline_label)
    args.output.mkdir(parents=True, exist_ok=False)
    models = {spec.label: StaticModel.from_pretrained(spec.path, force_download=False) for spec in specs}
    model_gates = {
        label: {"dimension_256": model.dim == 256, "vectors_finite": bool(np.isfinite(model.embedding).all())}
        for label, model in models.items()
    }
    pair_results = evaluate_pairs(
        models,
        args.rust_train,
        args.rust_eval,
        args.baseline_label,
        args.hybrid_rank_depth,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    repo_results = evaluate_repositories(
        models,
        args.repo_root,
        args.repo_manifest,
        args.baseline_label,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    verdicts = {
        spec.label: _quality_verdict(pair_results, repo_results, spec.label)
        for spec in specs
        if spec.label != args.baseline_label
    }
    gates = {
        "models_valid": all(all(values.values()) for values in model_gates.values()),
        "full_rust_eval": pair_results["eval_rows"] == RUST_EVAL_ROWS,
        "full_rust_corpus": pair_results["full_corpus_rows"] == RUST_TRAIN_ROWS + RUST_EVAL_ROWS,
        "complete_repository_suite": repo_results["repositories"] == len(load_repo_specs()),
    }
    receipt = {
        "schema_version": 1,
        "tool": "benchmarks.model2vec_training.evaluate_full",
        "status": "passed" if all(gates.values()) else "failed",
        "run_id": args.run_id,
        "source_revision": args.source_revision,
        "baseline_label": args.baseline_label,
        "models": {
            spec.label: {"path": str(spec.path), "files": _model_files(spec.path)}
            for spec in specs
        },
        "model_gates": model_gates,
        "rust_pairs": pair_results,
        "real_repositories": repo_results,
        "quality_verdicts": verdicts,
        "duration_seconds": time.perf_counter() - started,
        "gates": gates,
        "quality_boundary": (
            "The complete evaluation is decision evidence, but no candidate is promoted or deployed by this command."
        ),
    }
    (args.output / "evaluation.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(_decision_summary(pair_results, repo_results, verdicts), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    """Run evaluation and return nonzero only for incomplete mechanics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    receipt = evaluate(_parse_args())
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
