from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from model2vec import StaticModel

from semble.index.bm25 import BM25
from semble.index.dense import SelectableBasicBackend
from semble.index.sparse import selector_to_mask
from semble.ranking import apply_query_boost, boost_multi_chunk_files, rerank_topk, resolve_alpha
from semble.tokens import tokenize
from semble.types import Chunk, SearchResult

_RRF_K = 60


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    """Query representation shared across one or more compatible indexes."""

    text: str
    embedding: npt.NDArray[np.float32]
    tokens: tuple[str, ...]
    alpha_weight: float


def _rrf_scores(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    """Convert raw scores to RRF scores 1/(k + rank); higher raw score → rank 1."""
    if not scores:
        return scores
    ranked = sorted(scores, key=lambda c: -scores[c])
    return {chunk: 1.0 / (_RRF_K + rank) for rank, chunk in enumerate(ranked, 1)}


def _search_semantic_prepared(
    query_embedding: npt.NDArray[np.float32],
    semantic_index: SelectableBasicBackend,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
    excluded: npt.NDArray[np.int_] | None = None,
) -> list[SearchResult]:
    """Run semantic search with an already-computed query embedding."""
    indices, scores = semantic_index.query(
        query_embedding,
        k=top_k,
        selector=selector,
        excluded=excluded,
    )[0]
    return [SearchResult(chunk=chunks[index], score=1.0 - float(distance)) for index, distance in zip(indices, scores)]


def _search_semantic(
    query: str,
    model: StaticModel,
    semantic_index: SelectableBasicBackend,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
) -> list[SearchResult]:
    """Run semantic search for a query."""
    return _search_semantic_prepared(model.encode([query]), semantic_index, chunks, top_k, selector)


def _sort_top_k(arr: npt.NDArray, top_k: int) -> npt.NDArray[np.int_]:
    """Get the top k indices of an array in sort order."""
    neg_arr = -arr
    if top_k >= len(arr):
        return np.argsort(neg_arr)
    partitioned = np.argpartition(neg_arr, kth=top_k)[:top_k]
    return partitioned[np.argsort(neg_arr[partitioned])]


def _search_bm25_prepared(
    tokens: tuple[str, ...],
    bm25_index: BM25,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
    excluded: npt.NDArray[np.int_] | None = None,
) -> list[SearchResult]:
    """Return chunks ranked by BM25 score for pre-tokenized query text."""
    if not tokens:
        return []
    mask = selector_to_mask(selector, len(chunks))
    if excluded is not None:
        if mask is None:
            mask = np.ones(len(chunks), dtype=bool)
        mask[excluded] = False
    scores: npt.NDArray[np.float32] = bm25_index.get_scores(list(tokens), weight_mask=mask)
    indices = _sort_top_k(scores, top_k)
    return [SearchResult(chunk=chunks[i], score=float(scores[i])) for i in indices if scores[i] > 0]


def _search_bm25(
    query: str,
    bm25_index: BM25,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
) -> list[SearchResult]:
    """Return chunks ranked by BM25 score, excluding zero-score results."""
    return _search_bm25_prepared(tuple(tokenize(query)), bm25_index, chunks, top_k, selector)


def prepare_query(query: str, model: StaticModel, alpha: float | None = None) -> PreparedQuery:
    """Prepare one query for reuse across compatible indexes."""
    return PreparedQuery(
        text=query,
        embedding=model.encode([query]),
        tokens=tuple(tokenize(query)),
        alpha_weight=resolve_alpha(query, alpha),
    )


def search_prepared(
    prepared: PreparedQuery,
    semantic_index: SelectableBasicBackend,
    bm25_index: BM25,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None = None,
    excluded: npt.NDArray[np.int_] | None = None,
    rerank: bool = True,
) -> list[SearchResult]:
    """Hybrid search using a query representation shared across indexes."""
    candidate_count = top_k * 5
    semantic = _search_semantic_prepared(
        prepared.embedding,
        semantic_index,
        chunks,
        candidate_count,
        selector,
        excluded,
    )
    semantic_scores = {result.chunk: result.score for result in semantic}
    bm25_scores = {
        result.chunk: result.score
        for result in _search_bm25_prepared(
            prepared.tokens,
            bm25_index,
            chunks,
            candidate_count,
            selector,
            excluded,
        )
        if result.score
    }

    normalized_semantic = _rrf_scores(semantic_scores)
    normalized_bm25 = _rrf_scores(bm25_scores)
    all_candidates = sorted(
        {*normalized_semantic, *normalized_bm25},
        key=lambda chunk: chunk.start_line,
    )
    combined_scores = {
        chunk: prepared.alpha_weight * normalized_semantic.get(chunk, 0.0)
        + (1.0 - prepared.alpha_weight) * normalized_bm25.get(chunk, 0.0)
        for chunk in all_candidates
    }
    combined_scores = {chunk: score for chunk, score in combined_scores.items() if score}

    if rerank:
        boost_multi_chunk_files(combined_scores)
        combined_scores = apply_query_boost(combined_scores, prepared.text, chunks)
        ranked = rerank_topk(combined_scores, top_k, penalise_paths=prepared.alpha_weight < 1.0)
    else:
        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [SearchResult(chunk=chunk, score=score) for chunk, score in ranked]


def search(
    query: str,
    model: StaticModel,
    semantic_index: SelectableBasicBackend,
    bm25_index: BM25,
    chunks: list[Chunk],
    top_k: int,
    alpha: float | None = None,
    selector: npt.NDArray[np.int_] | None = None,
    rerank: bool = True,
) -> list[SearchResult]:
    """Hybrid search: alpha-weighted semantic and BM25 reciprocal-rank fusion."""
    return search_prepared(
        prepare_query(query, model, alpha),
        semantic_index,
        bm25_index,
        chunks,
        top_k,
        selector=selector,
        rerank=rerank,
    )
