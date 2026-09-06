"""Verify an exported Model2Vec candidate through Semble's CPU search path."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from vicinity.backends.basic import BasicArgs

from benchmarks.model2vec_training.core import (
    read_pairs,
    retrieval_metrics,
    sha256_file,
    static_batch_invariance,
    static_retrieval_vectors,
)
from semble.index import dense as dense_module
from semble.index.dense import SelectableBasicBackend, embed_chunks, load_model
from semble.search import _search_semantic
from semble.types import Chunk


def _parse_args() -> argparse.Namespace:
    """Parse CPU verification arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _version(name: str) -> str:
    """Return an installed package version."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load, encode, search, and write the CPU verification receipt."""
    started = time.perf_counter()
    pairs = read_pairs(args.eval_pairs)
    model, resolved_path = load_model(str(args.model))
    queries, documents = static_retrieval_vectors(model, pairs)
    metrics = retrieval_metrics(queries, documents)
    chunks = [
        Chunk(content=pair.code, file_path=f"eval/{index:03d}.rs", start_line=1, end_line=1, language="rust")
        for index, pair in enumerate(pairs)
    ]
    semble_vectors = embed_chunks(model, chunks)
    backend = SelectableBasicBackend(semble_vectors, BasicArgs())
    results = _search_semantic(pairs[0].query, model, backend, chunks, top_k=5, selector=None)
    batch_cosine = static_batch_invariance(model, [pair.query for pair in pairs] + [pair.code for pair in pairs])
    gates = {
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "model_loads": model.dim == 256,
        "vectors_finite": bool(np.isfinite(semble_vectors).all()),
        "search_returns_results": bool(results),
        "batch_invariant": batch_cosine >= 0.99999,
    }
    source_path = Path(dense_module.__file__).resolve()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "tool": "benchmarks.model2vec_training.verify_smoke",
        "status": "passed" if all(gates.values()) else "failed",
        "run_id": args.run_id,
        "source_revision": args.source_revision,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "model_path": resolved_path,
        "model_dimension": model.dim,
        "model_files": {
            str(path.relative_to(args.model)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(args.model.rglob("*"))
            if path.is_file()
        },
        "eval_pairs": {
            "path": str(args.eval_pairs),
            "rows": len(pairs),
            "sha256": sha256_file(args.eval_pairs),
        },
        "metrics": metrics,
        "minimum_batch_invariance_cosine": batch_cosine,
        "first_query_top_files": [result.chunk.file_path for result in results],
        "environment": {
            "python": platform.python_version(),
            "model2vec": _version("model2vec"),
            "semble_distribution": _version("semble"),
            "semble_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        },
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    """Run CPU verification and return nonzero when a gate fails."""
    receipt = run(_parse_args())
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
