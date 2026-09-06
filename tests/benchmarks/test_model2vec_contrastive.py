"""Focused tests for full contrastive data and model helpers."""

from __future__ import annotations

import numpy as np
from model2vec import StaticModel
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from benchmarks.model2vec_training.contrastive_sources import ReplaySource
from benchmarks.model2vec_training.prepare_contrastive import select_replay_rows
from benchmarks.model2vec_training.train_contrastive import _build_model, _save_candidate


def _tiny_model() -> StaticModel:
    """Return a direct-vocabulary 256-dimensional static model."""
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4}
    tokenizer = Tokenizer(WordPiece(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    vectors = np.random.default_rng(42).normal(size=(len(vocabulary), 256)).astype(np.float16)
    return StaticModel(vectors=vectors, tokenizer=tokenizer, normalize=True)


def test_replay_selection_applies_length_dedup_and_holdout() -> None:
    """Selection retains exactly the eligible unique non-held-out pairs."""
    good_query = "q" * 32
    good_document = "d" * 32
    held_out_query = "h" * 32
    rows = [
        {"query": 4, "document": good_document},
        {"query": "short", "document": good_document},
        {"query": held_out_query, "document": "x" * 32},
        {"query": good_query, "document": good_document},
        {"query": good_query, "document": "e" * 32},
        {"query": "z" * 32, "document": good_document},
        {"query": "a" * 32, "document": "b" * 32},
    ]
    selected, stats = select_replay_rows(
        rows,
        ReplaySource("python", "example/python", "revision"),
        limit=2,
        held_out_query_digests={__import__("hashlib").sha256(held_out_query.encode()).hexdigest()},
        held_out_document_digests=set(),
    )
    assert [(pair.anchor, pair.positive) for pair in selected] == [
        (good_query, good_document),
        ("a" * 32, "b" * 32),
    ]
    assert stats.invalid_type == 1
    assert stats.too_short == 1
    assert stats.held_out_overlap == 1
    assert stats.duplicate_query == 1
    assert stats.duplicate_document == 1


def test_build_and_raw_export_preserve_tokenizer(tmp_path) -> None:
    """The trainable module is float32 and raw export retains tokenizer bytes."""
    base = _tiny_model()
    model, static = _build_model(base)
    assert static.embedding.weight.dtype.is_floating_point
    assert str(static.embedding.weight.dtype) == "torch.float32"
    assert model.get_embedding_dimension() == 256
    output = tmp_path / "candidate"
    candidate = _save_candidate(base, static.embedding.weight.detach().numpy(), output, {"test": True}, apply_sif=False)
    assert candidate.embedding.dtype == np.float16
    assert candidate.tokenizer.to_str() == base.tokenizer.to_str()
