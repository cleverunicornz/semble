"""Focused tests for statistical and paired-retrieval evaluation helpers."""

from __future__ import annotations

import numpy as np
from model2vec import StaticModel
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from benchmarks.model2vec_training.core import Pair
from benchmarks.model2vec_training.evaluate_full import (
    bootstrap_mean_delta,
    dense_ranks,
    dense_ranks_against_corpus,
    paired_rank_comparison,
    parse_model_specs,
    rank_summary,
    validate_comparison_specs,
)


def _identity_model() -> StaticModel:
    """Build a tiny static model with two orthogonal semantic groups."""
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}
    tokenizer = Tokenizer(WordPiece(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    vectors = np.zeros((4, 256), dtype=np.float32)
    vectors[2, 0] = 1
    vectors[3, 1] = 1
    return StaticModel(vectors=vectors, tokenizer=tokenizer, normalize=True)


def test_dense_ranks_find_matching_documents() -> None:
    """Dense evaluation ranks aligned orthogonal pairs first."""
    pairs = [Pair("a", "alpha", "alpha", "g1"), Pair("b", "beta", "beta", "g2")]
    ranks, geometry = dense_ranks(_identity_model(), pairs)
    assert ranks.tolist() == [1, 1]
    assert geometry["negative_margin_count"] == 0
    assert rank_summary(ranks)["recall_at_1"] == 1


def test_dense_ranks_use_supplied_full_corpus_positions() -> None:
    """Full-corpus evaluation finds positives after unrelated leading documents."""
    pairs = [Pair("a", "alpha", "alpha", "g1"), Pair("b", "beta", "beta", "g2")]
    documents = ["beta", "beta", "alpha", "beta"]
    ranks, geometry = dense_ranks_against_corpus(
        _identity_model(),
        pairs,
        documents,
        np.asarray([2, 3], dtype=np.int32),
    )
    assert ranks.tolist() == [1, 3]
    assert geometry["negative_margin_count"] == 0


def test_paired_statistics_detect_improvement_and_are_deterministic() -> None:
    """Bootstrap and McNemar summaries retain paired direction."""
    baseline = np.asarray([1, 2, 3, 1, 4], dtype=np.int32)
    candidate = np.asarray([1, 1, 2, 1, 1], dtype=np.int32)
    first = paired_rank_comparison(baseline, candidate, resamples=200, seed=7)
    second = paired_rank_comparison(baseline, candidate, resamples=200, seed=7)
    assert first == second
    assert first["top1"]["delta"] > 0
    assert first["rank_movements"] == {"improved": 3, "unchanged": 2, "worsened": 0}


def test_bootstrap_rejects_mismatched_rows() -> None:
    """Paired intervals require aligned observations."""
    with __import__("pytest").raises(ValueError, match="same-length"):
        bootstrap_mean_delta(np.ones(2), np.ones(3), 10, 1)


def test_model_spec_parsing_requires_unique_complete_paths(tmp_path) -> None:
    """Model labels are unique and local paths must contain config.json."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    assert parse_model_specs([f"baseline={model}"])[0].label == "baseline"
    with __import__("pytest").raises(ValueError, match="duplicate"):
        parse_model_specs([f"same={model}", f"same={model}"])


def test_comparison_requires_all_four_models_and_v2_baseline(tmp_path) -> None:
    """Decision evaluation rejects incomplete model sets and a shifted baseline."""
    paths = {}
    for label in ("potion-v1", "potion-v2", "candidate-raw", "candidate-post-sif"):
        path = tmp_path / label
        path.mkdir()
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        paths[label] = path
    specs = parse_model_specs([f"{label}={path}" for label, path in paths.items()])
    validate_comparison_specs(specs, "potion-v2")
    with __import__("pytest").raises(ValueError, match="model labels must be exactly"):
        validate_comparison_specs(specs[:-1], "potion-v2")
    with __import__("pytest").raises(ValueError, match="baseline label must be"):
        validate_comparison_specs(specs, "potion-v1")
