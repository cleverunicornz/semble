"""Run the full Rust-plus-replay Model2Vec contrastive training stage."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from tokenizers import Tokenizer

from benchmarks.model2vec_training.contrastive_sources import (
    POTION_V2_ID,
    POTION_V2_REVISION,
    REPLAY_SOURCES,
    RUST_TRAIN_ROWS,
)
from benchmarks.model2vec_training.core import sha256_file

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
FULL_TRAINING_ROWS = RUST_TRAIN_ROWS * (len(REPLAY_SOURCES) + 1)


def _parse_args() -> argparse.Namespace:
    """Parse bounded smoke or full-training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=28)
    parser.add_argument("--smoke-pairs", type=int, default=2_048)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prepare_output(path: Path) -> None:
    """Create a new evidence directory."""
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _version(name: str) -> str:
    """Return an installed package version."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _load_training_dataset(
    path: Path, manifest_path: Path, mode: str, smoke_pairs: int
) -> tuple[Any, dict[str, Any]]:
    """Load the attested two-column training dataset."""
    from datasets import load_dataset

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest.get("outputs", {}).get("train.jsonl", {})
    actual_sha = sha256_file(path)
    if output.get("sha256") != actual_sha or output.get("rows") != FULL_TRAINING_ROWS:
        raise ValueError("training JSONL does not match its full balanced-data manifest")
    dataset = load_dataset("json", data_files=str(path), split="train")
    if len(dataset) != FULL_TRAINING_ROWS:
        raise ValueError(f"loaded {len(dataset)} training rows, expected {FULL_TRAINING_ROWS}")
    if dataset.column_names != ["anchor", "positive"]:
        raise ValueError(f"unexpected training columns: {dataset.column_names}")
    if mode == "smoke":
        if not 1 <= smoke_pairs < len(dataset):
            raise ValueError("smoke_pairs must be smaller than the full dataset")
        dataset = dataset.select(range(smoke_pairs))
    return dataset, manifest


def _training_environment(torch_threads: int) -> dict[str, Any]:
    """Capture CPU and package identity."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch_threads,
        "packages": {
            name: _version(name)
            for name in ("accelerate", "datasets", "model2vec", "numpy", "sentence-transformers", "torch")
        },
        "cuda_available": torch.cuda.is_available(),
    }


def _model_file_manifest(path: Path) -> dict[str, dict[str, int | str]]:
    """Return hashes and sizes for a saved model directory."""
    return {
        str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _build_model(base: StaticModel) -> tuple[SentenceTransformer, StaticEmbedding]:
    """Construct a float32 trainable StaticEmbedding without mutating the base tokenizer."""
    if base.weights is not None or base.token_mapping is not None:
        raise ValueError("the v2 base must have a direct, unquantized vocabulary")
    tokenizer = Tokenizer.from_str(base.tokenizer.to_str())
    static_embedding = StaticEmbedding(
        tokenizer=tokenizer,
        embedding_weights=torch.from_numpy(base.embedding.astype(np.float32, copy=True)),
        base_model=POTION_V2_ID,
    )
    return SentenceTransformer(modules=[static_embedding], device="cpu"), static_embedding


def _save_candidate(
    base: StaticModel,
    vectors: np.ndarray,
    output: Path,
    metadata: dict[str, Any],
    *,
    apply_sif: bool,
) -> StaticModel:
    """Save either raw MNRL vectors or the published v1 post-SIF variant."""
    weights: np.ndarray | None = None
    config = dict(base.config)
    config["contrastive_training"] = metadata
    tokenizer = Tokenizer.from_str(base.tokenizer.to_str())
    if apply_sif:
        from model2vec.distill.inference import post_process_embeddings

        vectors, weights = post_process_embeddings(vectors.astype(np.float32), pca_dims=256, sif_coefficient=1e-4)
        config["embedding_dtype"] = "float32"
        config["post_sif"] = {"pca_dims": 256, "sif_coefficient": 1e-4}
        output_vectors = vectors.astype(np.float32)
    else:
        config["embedding_dtype"] = "float16"
        config.pop("vocabulary_quantization", None)
        output_vectors = vectors.astype(np.float16)
    candidate = StaticModel(
        vectors=output_vectors,
        weights=weights,
        tokenizer=tokenizer,
        config=config,
        normalize=True,
        base_model_name=POTION_V2_ID,
        language=["code"],
    )
    candidate.save_pretrained(output)
    return candidate


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train from a pristine v2 model and save raw plus post-SIF candidates."""
    _prepare_output(args.output)
    if not (args.base_model / "config.json").is_file():
        raise ValueError(f"base model snapshot is incomplete: {args.base_model}")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(min(4, args.torch_threads))
    torch.manual_seed(args.seed)
    started_at = datetime.now(timezone.utc)
    overall_started = time.perf_counter()
    dataset, data_manifest = _load_training_dataset(
        args.training_data, args.training_manifest, args.mode, args.smoke_pairs
    )
    base = StaticModel.from_pretrained(str(args.base_model), force_download=False)
    if base.embedding.shape != (63_457, 256) or base.embedding.dtype != np.float16:
        raise ValueError(f"unexpected v2 base shape/dtype: {base.embedding.shape} {base.embedding.dtype}")
    base_tokenizer_sha = sha256_file(args.base_model / "tokenizer.json")
    model, static_embedding = _build_model(base)
    loss = MultipleNegativesRankingLoss(model)
    max_steps = args.smoke_steps if args.mode == "smoke" else -1
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=1 if args.mode == "smoke" else args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=64 if args.mode == "smoke" else args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=0.1,
        fp16=False,
        bf16=False,
        use_cpu=True,
        full_determinism=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=1 if args.mode == "smoke" else 50,
        logging_first_step=True,
        report_to=[],
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=False,
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=dataset, loss=loss)
    train_started = time.perf_counter()
    outcome = trainer.train()
    training_seconds = time.perf_counter() - train_started
    trained_vectors = static_embedding.embedding.weight.detach().cpu().float().numpy()
    changed_rows = int(np.any(trained_vectors != base.embedding.astype(np.float32), axis=1).sum())
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "source_revision": args.source_revision,
        "run_id": args.run_id,
        "base_model": {"id": POTION_V2_ID, "revision": POTION_V2_REVISION},
        "data_sha256": sha256_file(args.training_data),
        "seed": args.seed,
        "loss": "MultipleNegativesRankingLoss",
        "batch_sampler": "NO_DUPLICATES",
        "learning_rate": args.learning_rate,
        "batch_size": 64 if args.mode == "smoke" else args.batch_size,
        "epochs": 1 if args.mode == "smoke" else args.epochs,
        "warmup_steps_fraction": 0.1,
    }
    raw_path = args.output / "candidate-raw"
    sif_path = args.output / "candidate-post-sif"
    raw = _save_candidate(base, trained_vectors, raw_path, metadata, apply_sif=False)
    sif = _save_candidate(base, trained_vectors, sif_path, metadata, apply_sif=True)
    raw_loaded = StaticModel.from_pretrained(raw_path, force_download=False)
    sif_loaded = StaticModel.from_pretrained(sif_path, force_download=False)
    gates = {
        "cpu_only": not torch.cuda.is_available(),
        "full_row_count": args.mode == "smoke" or len(dataset) == FULL_TRAINING_ROWS,
        "raw_loads_256d": raw_loaded.dim == 256,
        "post_sif_loads_256d": sif_loaded.dim == 256,
        "raw_vectors_finite": bool(np.isfinite(raw.embedding).all()),
        "post_sif_vectors_finite": bool(np.isfinite(sif.embedding).all()),
        "tokenizer_preserved_raw": sha256_file(raw_path / "tokenizer.json") == base_tokenizer_sha,
        "tokenizer_preserved_post_sif": sha256_file(sif_path / "tokenizer.json") == base_tokenizer_sha,
        "training_loss_finite": bool(np.isfinite(outcome.training_loss)),
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "benchmarks.model2vec_training.train_contrastive",
        "status": "passed" if all(gates.values()) else "failed",
        "mode": args.mode,
        "run_id": args.run_id,
        "source_revision": args.source_revision,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - overall_started,
        "training_seconds": training_seconds,
        "training_rows": len(dataset),
        "full_training_rows": FULL_TRAINING_ROWS,
        "training_data": {
            "sha256": sha256_file(args.training_data),
            "manifest_sha256": sha256_file(args.training_manifest),
            "language_counts": {language: RUST_TRAIN_ROWS for language in data_manifest["languages"]},
        },
        "base_model": {
            "id": POTION_V2_ID,
            "revision": POTION_V2_REVISION,
            "shape": list(base.embedding.shape),
            "dtype": str(base.embedding.dtype),
            "tokenizer_sha256": base_tokenizer_sha,
        },
        "configuration": metadata,
        "trainer_metrics": outcome.metrics,
        "changed_vocabulary_rows": changed_rows,
        "environment": _training_environment(args.torch_threads),
        "artifacts": {
            "candidate_raw": _model_file_manifest(raw_path),
            "candidate_post_sif": _model_file_manifest(sif_path),
        },
        "gates": gates,
        "quality_boundary": "Training completion is not a quality verdict; evaluation is a separate required stage.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("%s training finished in %.3f seconds", args.mode, training_seconds)
    return receipt


def main() -> None:
    """Run training and return nonzero on a mechanical failure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    receipt = train(_parse_args())
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
