"""Streaming readers for local JSONL and Parquet dataset exports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1 << 20


def iter_jsonl_rows(path: Path) -> Iterator[Any]:
    """Yield parsed values from a UTF-8 JSON Lines file, one per line.

    Blank lines are skipped; unparsable lines yield ``None`` so callers can
    count them as invalid rows instead of crashing.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError:
                yield None


def iter_parquet_rows(path: Path, *, batch_size: int = 1024) -> Iterator[Any]:
    """Yield rows from a Parquet file in bounded batches.

    Requires ``pyarrow``; install the benchmark-local extra from
    ``benchmarks/model2vec_data/requirements.txt``.
    """
    import pyarrow.parquet as pq  # optional benchmark-only dependency

    parquet = pq.ParquetFile(str(path))
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_input_rows(path: Path) -> Iterator[Any]:
    """Yield rows from a .parquet, .jsonl, or .ndjson file, dispatching by suffix."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return iter_parquet_rows(path)
    if suffix in (".jsonl", ".ndjson"):
        return iter_jsonl_rows(path)
    raise ValueError(f"unsupported input format {suffix!r} for {path}")


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
