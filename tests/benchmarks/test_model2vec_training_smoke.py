"""End-to-end local smoke coverage without loading a transformer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from model2vec import StaticModel
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from benchmarks.model2vec_training import train_smoke, verify_smoke
from benchmarks.model2vec_training.core import sha256_file


class FakeTeacher:
    """Deterministic SentenceTransformer stand-in for orchestration coverage."""

    def __init__(self, *_args: Any, truncate_dim: int, **_kwargs: Any) -> None:
        """Record the configured target dimension."""
        self.dimension = truncate_dim
        self.max_seq_length = 0
        self.tokenizer = SimpleNamespace(padding_side="right")

    def encode(self, texts: list[str], *, prompt: str | None, **_kwargs: Any) -> np.ndarray:
        """Return stable normalized vectors derived from prompt plus text."""
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(f"{prompt or ''}{text}".encode()).digest()
            seed = int.from_bytes(digest[:8], "big")
            vector = np.random.default_rng(seed).normal(size=self.dimension).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)


def _save_student(path: Path) -> None:
    """Save a 256-dimensional static student fixture."""
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4, "delta": 5}
    tokenizer = Tokenizer(WordPiece(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    vectors = np.random.default_rng(5).normal(size=(len(vocabulary), 256)).astype(np.float32)
    StaticModel(vectors=vectors, tokenizer=tokenizer, normalize=True).save_pretrained(path)


def _write_pairs(path: Path, count: int, group_offset: int = 0) -> None:
    """Write small grouped pair fixtures."""
    words = ("alpha", "beta", "gamma", "delta")
    rows = [
        {
            "id": f"row-{group_offset}-{index}",
            "query": words[index % len(words)],
            "code": f"{words[(index + 1) % len(words)]} {words[(index + 2) % len(words)]}",
            "group": {"id": f"crate-{group_offset + index}"},
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_full_smoke_and_cpu_verification(tmp_path: Path, monkeypatch: Any) -> None:
    """The complete runner exports a candidate that Semble loads and searches on CPU."""
    train_path = tmp_path / "pairs.train.jsonl"
    eval_path = tmp_path / "pairs.eval.jsonl"
    manifest_path = tmp_path / "manifest.json"
    teacher_path = tmp_path / "teacher"
    student_path = tmp_path / "student"
    teacher_path.mkdir()
    (teacher_path / "config.json").write_text("{}\n", encoding="utf-8")
    _save_student(student_path)
    _write_pairs(train_path, 12)
    _write_pairs(eval_path, 4, group_offset=100)
    train_sha = sha256_file(train_path)
    eval_sha = sha256_file(eval_path)
    manifest = {
        "outputs": {
            "pairs.train.jsonl": {"rows": 12, "sha256": train_sha},
            "pairs.eval.jsonl": {"rows": 4, "sha256": eval_sha},
        },
        "sources": {"rust": {"pin": {"revision": train_smoke.DEFAULT_DATASET_REVISION}}},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(train_smoke, "SentenceTransformer", FakeTeacher)
    monkeypatch.setattr(train_smoke, "EXPECTED_TRAIN_SHA256", train_sha)
    monkeypatch.setattr(train_smoke, "EXPECTED_EVAL_SHA256", eval_sha)
    monkeypatch.setattr(train_smoke, "EXPECTED_TRAIN_ROWS", 12)
    monkeypatch.setattr(train_smoke, "EXPECTED_EVAL_ROWS", 4)

    output = tmp_path / "output"
    args = argparse.Namespace(
        train_pairs=train_path,
        eval_pairs=eval_path,
        sample_manifest=manifest_path,
        teacher_model=teacher_path,
        student_model=student_path,
        output=output,
        teacher_id="fake-teacher",
        teacher_revision="teacher-sha",
        student_id="fake-student",
        student_revision="student-sha",
        source_revision="source-sha",
        run_id="local-test",
        query_prompt=train_smoke.DEFAULT_QUERY_PROMPT,
        target_dim=256,
        max_length=16,
        teacher_batch_size=4,
        student_batch_size=4,
        learning_rate=1e-3,
        max_epochs=2,
        patience=1,
        validation_fraction=0.2,
        seed=42,
        device="cpu",
        teacher_dtype="float32",
        attention="eager",
    )
    receipt = train_smoke.run(args)
    assert receipt["status"] == "passed"
    assert receipt["counts"]["held_out_eval_pairs"] == 4

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    verification = verify_smoke.run(
        argparse.Namespace(
            model=output / "model",
            eval_pairs=eval_path,
            output=output / "cpu-verification.json",
            source_revision="source-sha",
            run_id="local-test",
        )
    )
    assert verification["status"] == "passed"
