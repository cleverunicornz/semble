"""Run a pinned Qwen-to-Model2Vec Rust training smoke experiment."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer

from benchmarks.model2vec_training.core import (
    StaticStudent,
    TrainConfig,
    export_student,
    initialize_ridge_projection,
    minimum_row_cosine,
    pair_targets,
    pair_texts,
    probable_pad_id,
    read_pairs,
    retrieval_metrics,
    sha256_file,
    stable_group_split,
    static_batch_invariance,
    static_retrieval_vectors,
    tokenize_texts,
    train_student,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TEACHER_ID = "Qwen/Qwen3-Embedding-8B"
DEFAULT_TEACHER_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
DEFAULT_STUDENT_ID = "minishlab/potion-code-16M-v2"
DEFAULT_STUDENT_REVISION = "e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b"
DEFAULT_DATASET_REVISION = "0a8d223302712a2b34a6ad4ce1fd679031894b3d"
EXPECTED_TRAIN_SHA256 = "3f6e2e4e657c899fac48ad7af14c2f2c74e44beb0b67418e3c50b4619634dd68"
EXPECTED_EVAL_SHA256 = "a605d01165eab7fb81d066b8992953c6a7a59bf2c3ad0df320a5ac9e1ee51db8"
EXPECTED_TRAIN_ROWS = 894
EXPECTED_EVAL_ROWS = 100
DEFAULT_QUERY_PROMPT = "Instruct: Given a natural-language code search query, retrieve relevant Rust code\nQuery:"
SCHEMA_VERSION = 1


def _parse_args() -> argparse.Namespace:
    """Parse the bounded smoke-run arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--eval-pairs", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True, help="Pinned local teacher snapshot")
    parser.add_argument("--student-model", type=Path, required=True, help="Pinned local Model2Vec snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-id", default=DEFAULT_TEACHER_ID)
    parser.add_argument("--teacher-revision", default=DEFAULT_TEACHER_REVISION)
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument("--student-revision", default=DEFAULT_STUDENT_REVISION)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--query-prompt", default=DEFAULT_QUERY_PROMPT)
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--student-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    return parser.parse_args()


def _prepare_output(path: Path) -> None:
    """Create a new output directory without overwriting prior evidence."""
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _package_version(name: str) -> str:
    """Return an installed distribution version."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _environment() -> dict[str, Any]:
    """Capture the training process and accelerator environment."""
    cuda = torch.cuda.is_available()
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("model2vec", "numpy", "sentence-transformers", "torch", "transformers")
        },
        "torch_cuda_available": cuda,
        "torch_cuda_version": torch.version.cuda,
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        environment["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
        }
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        environment["nvidia_driver"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    return environment


def _validate_local_snapshot(path: Path, label: str) -> None:
    """Require a concrete local model snapshot."""
    if not path.is_dir():
        raise ValueError(f"{label} model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise ValueError(f"{label} snapshot has no config.json: {path}")


def _validate_sample_manifest(args: argparse.Namespace, train_rows: int, eval_rows: int) -> dict[str, Any]:
    """Require the exact sample qualified before this smoke experiment."""
    manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    train_sha = sha256_file(args.train_pairs)
    eval_sha = sha256_file(args.eval_pairs)
    expected = {
        "train": (EXPECTED_TRAIN_SHA256, EXPECTED_TRAIN_ROWS, "pairs.train.jsonl"),
        "eval": (EXPECTED_EVAL_SHA256, EXPECTED_EVAL_ROWS, "pairs.eval.jsonl"),
    }
    actual = {"train": (train_sha, train_rows), "eval": (eval_sha, eval_rows)}
    for split, (expected_sha, expected_rows, filename) in expected.items():
        actual_sha, actual_rows = actual[split]
        if (actual_sha, actual_rows) != (expected_sha, expected_rows):
            raise ValueError(
                f"{split} sample identity mismatch: got rows={actual_rows} sha256={actual_sha}, "
                f"expected rows={expected_rows} sha256={expected_sha}"
            )
        manifest_output = manifest.get("outputs", {}).get(filename, {})
        if (manifest_output.get("sha256"), manifest_output.get("rows")) != (expected_sha, expected_rows):
            raise ValueError(f"sample manifest does not attest the qualified {filename}")
    revision = manifest.get("sources", {}).get("rust", {}).get("pin", {}).get("revision")
    if revision != DEFAULT_DATASET_REVISION:
        raise ValueError(f"sample manifest Rust revision is {revision!r}, expected {DEFAULT_DATASET_REVISION}")
    return manifest


def _encode_teacher(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    target_dim: int,
    prompt: str | None,
) -> np.ndarray:
    """Encode and validate teacher sentence embeddings."""
    vectors = model.encode(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        truncate_dim=target_dim,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (len(texts), target_dim):
        raise ValueError(f"teacher returned shape {array.shape}, expected {(len(texts), target_dim)}")
    if not np.isfinite(array).all():
        raise ValueError("teacher returned non-finite embeddings")
    return array


def _encode_pair_partitions(
    teacher: SentenceTransformer,
    partitions: dict[str, list],
    *,
    batch_size: int,
    target_dim: int,
    query_prompt: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Encode query and code sides for every partition."""
    targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    durations: dict[str, float] = {}
    for name, pairs in partitions.items():
        started = time.perf_counter()
        queries = _encode_teacher(
            teacher,
            [pair.query for pair in pairs],
            batch_size=batch_size,
            target_dim=target_dim,
            prompt=query_prompt,
        )
        documents = _encode_teacher(
            teacher,
            [pair.code for pair in pairs],
            batch_size=batch_size,
            target_dim=target_dim,
            prompt=None,
        )
        targets[name] = (queries, documents)
        durations[f"teacher_encode_{name}"] = time.perf_counter() - started
    return targets, durations


def _save_teacher_targets(
    path: Path,
    partitions: dict[str, list],
    targets: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Save teacher targets and stable pair IDs for post-run inspection."""
    arrays: dict[str, Any] = {}
    for name, pairs in partitions.items():
        queries, documents = targets[name]
        arrays[f"{name}_query"] = queries
        arrays[f"{name}_code"] = documents
        arrays[f"{name}_id"] = np.asarray([pair.id for pair in pairs], dtype=np.str_)
    np.savez_compressed(path, **arrays)


def _static_metrics(model: StaticModel, pairs: list) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray]]:
    """Encode held-out pairs with Semble-like static call topology and score them."""
    queries, documents = static_retrieval_vectors(model, pairs)
    return retrieval_metrics(queries, documents), (queries, documents)


def _model_files(path: Path) -> dict[str, dict[str, int | str]]:
    """Hash every regular file in a saved model directory."""
    return {
        str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the complete teacher, student, export, and smoke-gate sequence."""
    started_at = datetime.now(timezone.utc)
    overall_started = time.perf_counter()
    _prepare_output(args.output)
    _validate_local_snapshot(args.teacher_model, "teacher")
    _validate_local_snapshot(args.student_model, "student")
    environment = _environment()
    if args.device.startswith("cuda") and not environment["torch_cuda_available"]:
        raise RuntimeError("CUDA was requested but torch cannot see a CUDA device")

    train_pairs_all = read_pairs(args.train_pairs)
    eval_pairs = read_pairs(args.eval_pairs)
    _validate_sample_manifest(args, len(train_pairs_all), len(eval_pairs))
    train_pairs, validation_pairs = stable_group_split(
        train_pairs_all, args.validation_fraction, args.seed
    )
    partitions = {"train": train_pairs, "validation": validation_pairs, "eval": eval_pairs}

    stage_durations: dict[str, float] = {}
    baseline_started = time.perf_counter()
    base_model = StaticModel.from_pretrained(str(args.student_model), force_download=False)
    if base_model.dim != args.target_dim:
        raise ValueError(f"base student dimension is {base_model.dim}, expected {args.target_dim}")
    baseline_metrics, _ = _static_metrics(base_model, eval_pairs)
    stage_durations["baseline_evaluation"] = time.perf_counter() - baseline_started

    teacher_started = time.perf_counter()
    dtype = getattr(torch, args.teacher_dtype)
    teacher = SentenceTransformer(
        str(args.teacher_model),
        device=args.device,
        local_files_only=True,
        truncate_dim=args.target_dim,
        model_kwargs={"torch_dtype": dtype, "attn_implementation": args.attention},
    )
    teacher.tokenizer.padding_side = "left"
    teacher.max_seq_length = args.max_length
    stage_durations["teacher_load"] = time.perf_counter() - teacher_started
    targets, encode_durations = _encode_pair_partitions(
        teacher,
        partitions,
        batch_size=args.teacher_batch_size,
        target_dim=args.target_dim,
        query_prompt=args.query_prompt,
    )
    stage_durations.update(encode_durations)
    teacher_metrics = retrieval_metrics(*targets["eval"])
    target_path = args.output / "teacher-targets.npz"
    _save_teacher_targets(target_path, partitions, targets)
    gpu_peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    student_started = time.perf_counter()
    pad_id = probable_pad_id(base_model)
    student = StaticStudent(base_model.embedding, pad_id=pad_id, output_dim=args.target_dim)
    train_text = pair_texts(train_pairs)
    validation_text = pair_texts(validation_pairs)
    train_sequences = tokenize_texts(base_model, train_text, args.max_length)
    validation_sequences = tokenize_texts(base_model, validation_text, args.max_length)
    train_target = pair_targets(*targets["train"])
    validation_target = pair_targets(*targets["validation"])
    initialize_ridge_projection(student, train_sequences, train_target)
    training = train_student(
        student,
        train_sequences,
        train_target,
        validation_sequences,
        validation_target,
        TrainConfig(
            learning_rate=args.learning_rate,
            batch_size=args.student_batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        ),
        args.device,
    )
    stage_durations["student_training"] = time.perf_counter() - student_started

    model_path = args.output / "model"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "teacher": {"id": args.teacher_id, "revision": args.teacher_revision},
        "student_initialization": {"id": args.student_id, "revision": args.student_revision},
        "source_revision": args.source_revision,
        "run_id": args.run_id,
        "target_dim": args.target_dim,
        "max_length": args.max_length,
        "query_prompt": args.query_prompt,
    }
    export_student(student, base_model, model_path, metadata)
    candidate = StaticModel.from_pretrained(str(model_path), force_download=False)
    candidate_metrics, candidate_vectors = _static_metrics(candidate, eval_pairs)
    teacher_queries, teacher_documents = targets["eval"]
    candidate_teacher_cosine = {
        "queries": minimum_row_cosine(candidate_vectors[0], teacher_queries),
        "documents": minimum_row_cosine(candidate_vectors[1], teacher_documents),
    }
    candidate_batch_cosine = static_batch_invariance(
        candidate, [pair.query for pair in eval_pairs] + [pair.code for pair in eval_pairs]
    )
    pad_norm = float(np.linalg.norm(candidate.embedding[pad_id].astype(np.float32)))
    gates = {
        "candidate_loads": candidate.dim == args.target_dim,
        "candidate_vectors_finite": bool(np.isfinite(candidate.embedding).all()),
        "candidate_dimension_256": candidate.dim == 256,
        "candidate_pad_vector_zero": pad_norm == 0,
        "candidate_batch_invariant": candidate_batch_cosine >= 0.99999,
        "training_validation_finite": bool(np.isfinite(training.best_validation_loss)),
        "teacher_metrics_finite": all(np.isfinite(value) for value in teacher_metrics.values()),
    }
    status = "passed" if all(gates.values()) else "failed"
    completed_at = datetime.now(timezone.utc)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "benchmarks.model2vec_training.train_smoke",
        "status": status,
        "run_id": args.run_id,
        "source_revision": args.source_revision,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": time.perf_counter() - overall_started,
        "inputs": {
            "train_pairs": {"path": str(args.train_pairs), "sha256": sha256_file(args.train_pairs)},
            "eval_pairs": {"path": str(args.eval_pairs), "sha256": sha256_file(args.eval_pairs)},
            "sample_manifest": {
                "path": str(args.sample_manifest),
                "sha256": sha256_file(args.sample_manifest),
                "dataset_revision": DEFAULT_DATASET_REVISION,
            },
        },
        "models": {
            "teacher": {"id": args.teacher_id, "revision": args.teacher_revision},
            "student_initialization": {"id": args.student_id, "revision": args.student_revision},
        },
        "config": {
            "query_prompt": args.query_prompt,
            "target_dim": args.target_dim,
            "max_length": args.max_length,
            "teacher_batch_size": args.teacher_batch_size,
            "student_batch_size": args.student_batch_size,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "teacher_dtype": args.teacher_dtype,
            "attention": args.attention,
        },
        "counts": {
            "prepared_train_pairs": len(train_pairs_all),
            "student_train_pairs": len(train_pairs),
            "student_validation_pairs": len(validation_pairs),
            "held_out_eval_pairs": len(eval_pairs),
            "student_train_tokens": sum(map(len, train_sequences)),
            "student_validation_tokens": sum(map(len, validation_sequences)),
            "student_vocabulary_rows": len(base_model.embedding),
            "student_train_unique_token_ids": len({token for sequence in train_sequences for token in sequence}),
            "student_train_groups": len({pair.group_id for pair in train_pairs}),
            "student_validation_groups": len({pair.group_id for pair in validation_pairs}),
        },
        "training": {
            "initial_validation_loss": training.initial_validation_loss,
            "best_validation_loss": training.best_validation_loss,
            "best_epoch": training.best_epoch,
            "epochs_completed": training.epochs_completed,
            "history": list(training.history),
        },
        "metrics": {
            "teacher": teacher_metrics,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_baseline": {
                key: candidate_metrics[key] - baseline_metrics[key] for key in candidate_metrics
            },
            "candidate_min_cosine_to_teacher": candidate_teacher_cosine,
            "candidate_min_batch_invariance_cosine": candidate_batch_cosine,
            "candidate_pad_vector_norm": pad_norm,
        },
        "stage_durations_seconds": stage_durations,
        "environment": environment,
        "gpu_peak_allocated_bytes": gpu_peak_bytes,
        "artifacts": {
            "teacher_targets": {
                "path": target_path.name,
                "bytes": target_path.stat().st_size,
                "sha256": sha256_file(target_path),
            },
            "model": _model_files(model_path),
        },
        "gates": gates,
        "quality_boundary": (
            "Held-out pair metrics are a smoke signal on 100 prepared Rust examples; "
            "they are not a production retrieval benchmark or deployment decision."
        ),
    }
    _json_dump(args.output / "receipt.json", receipt)
    LOGGER.info("Smoke status: %s", status)
    LOGGER.info("Receipt: %s", args.output / "receipt.json")
    return receipt


def main() -> None:
    """Run the CLI and return nonzero when a smoke gate fails."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    receipt = run(_parse_args())
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
