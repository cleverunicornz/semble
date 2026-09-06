"""Focused tests for the bounded Model2Vec training smoke helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from model2vec import StaticModel
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from benchmarks.model2vec_training.core import (
    StaticStudent,
    TrainConfig,
    export_student,
    initialize_ridge_projection,
    pair_targets,
    probable_pad_id,
    read_pairs,
    retrieval_metrics,
    stable_group_split,
    static_batch_invariance,
    tokenize_texts,
    train_student,
)


def _static_model() -> StaticModel:
    """Build a tiny static model whose configured PAD row starts nonzero."""
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4, "delta": 5}
    tokenizer = Tokenizer(WordPiece(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    vectors = np.asarray(
        [
            [0.3, 0.4],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    return StaticModel(vectors=vectors, tokenizer=tokenizer, normalize=True)


def test_read_pairs_and_group_split_are_deterministic(tmp_path: Path) -> None:
    """Prepared JSONL validates and the validation split never crosses groups."""
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"id": f"row-{index}", "query": "alpha", "code": "beta", "group": {"id": f"crate-{index // 2}"}}
        for index in range(12)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    pairs = read_pairs(path)
    train, validation = stable_group_split(pairs, validation_fraction=0.25, seed=7)
    assert train == stable_group_split(pairs, validation_fraction=0.25, seed=7)[0]
    assert {pair.group_id for pair in train}.isdisjoint(pair.group_id for pair in validation)
    assert len(train) + len(validation) == len(pairs)


def test_read_pairs_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Duplicate provenance identities fail rather than silently collapsing."""
    path = tmp_path / "pairs.jsonl"
    row = {"id": "same", "query": "alpha", "code": "beta", "group": {"id": "crate"}}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate pair id"):
        read_pairs(path)


def test_tokenization_disables_inherited_batch_padding() -> None:
    """Training sequences contain real IDs only even when the saved tokenizer pads."""
    model = _static_model()
    sequences = tokenize_texts(model, ["alpha", "alpha beta gamma"], max_length=8)
    assert sequences == [[2], [2, 3, 4]]
    assert probable_pad_id(model) == 0


def test_pair_targets_interleave_query_and_code() -> None:
    """Target order matches flattened query/code training texts."""
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    codes = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    expected = np.asarray([[1, 0], [0.8, 0.2], [0, 1], [0.2, 0.8]], dtype=np.float32)
    assert np.array_equal(pair_targets(queries, codes), expected)


def test_training_export_zeros_pad_and_is_batch_invariant(tmp_path: Path) -> None:
    """A synthetic target mapping trains, folds, reloads, and removes PAD dependence."""
    torch.set_num_threads(1)
    base = _static_model()
    train_text = ["alpha", "beta", "gamma", "delta", "alpha beta", "gamma delta"]
    validation_text = ["alpha gamma", "beta delta"]
    train_sequences = tokenize_texts(base, train_text, max_length=8)
    validation_sequences = tokenize_texts(base, validation_text, max_length=8)
    transform = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    def targets(sequences: list[list[int]]) -> np.ndarray:
        vectors = np.stack([base.embedding[sequence].mean(axis=0) for sequence in sequences])
        values = vectors @ transform.T
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    train_target = targets(train_sequences)
    validation_target = targets(validation_sequences)
    student = StaticStudent(base.embedding, pad_id=0, output_dim=2)
    initialize_ridge_projection(student, train_sequences, train_target)
    result = train_student(
        student,
        train_sequences,
        train_target,
        validation_sequences,
        validation_target,
        TrainConfig(learning_rate=1e-2, batch_size=2, max_epochs=4, patience=2, min_delta=1e-6, seed=3),
        "cpu",
    )
    output = tmp_path / "model"
    export_student(student, base, output, {"test": True})
    loaded = StaticModel.from_pretrained(output, force_download=False)
    assert result.epochs_completed >= 2
    assert np.count_nonzero(loaded.embedding[0]) == 0
    assert static_batch_invariance(loaded, ["alpha", "alpha beta gamma"]) >= 0.99999


def test_retrieval_metrics_rank_matching_rows() -> None:
    """Perfect paired vectors report perfect recall and reciprocal rank."""
    values = np.eye(3, dtype=np.float32)
    metrics = retrieval_metrics(values, values)
    assert metrics["recall_at_1"] == 1
    assert metrics["recall_at_5"] == 1
    assert metrics["mrr"] == 1
