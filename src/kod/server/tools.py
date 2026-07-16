"""MCP tools for KOD server."""

import logging

from fastmcp import Context

from kod.models import DocumentChunk
from kod.server.app import AppContext
from kod.server.app import embed_queries


logger = logging.getLogger(__name__)

_rrf_k: int = 60
_max_queries: int = 5
_max_top_k: int = 20


def configure(rrf_k: int = 60, max_queries: int = 5, max_top_k: int = 20) -> None:
    """Set search tuning parameters."""
    if rrf_k < 1:
        msg = f"rrf_k must be >= 1, got {rrf_k}"
        raise ValueError(msg)
    if max_queries < 1:
        msg = f"max_queries must be >= 1, got {max_queries}"
        raise ValueError(msg)
    if max_top_k < 1:
        msg = f"max_top_k must be >= 1, got {max_top_k}"
        raise ValueError(msg)
    global _rrf_k, _max_queries, _max_top_k  # noqa: W0603
    _rrf_k = rrf_k
    _max_queries = max_queries
    _max_top_k = max_top_k


def _normalize_queries(query: str | list[str]) -> list[str] | str:
    """Validate and normalize query input into a deduplicated list of strings."""
    if isinstance(query, str):
        raw = [query]
    elif isinstance(query, list):
        bad = [repr(q) for q in query if not isinstance(q, str)]
        if bad:
            return f"All query entries must be strings, got non-string items: {', '.join(bad)}"
        raw = query
    else:
        return f"query must be a string or list of strings, got {type(query).__name__}"

    seen: set[str] = set()
    cleaned: list[str] = []
    for q in raw:
        stripped = q.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            cleaned.append(stripped)

    return cleaned if cleaned else "Please provide a non-empty search query."


def _rrf_merge(
    ranked_lists: list[list[tuple[int, float]]],
    metadata: list[DocumentChunk],
    top_k: int,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists via Reciprocal Rank Fusion.

    Each ranked list contains (faiss_idx, faiss_score) tuples ordered by rank.
    Returns (faiss_idx, rrf_score) tuples sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    best_idx: dict[str, int] = {}
    best_len: dict[str, int] = {}

    for ranked_list in ranked_lists:
        for rank, (idx, _score) in enumerate(ranked_list):
            if idx < 0:
                continue
            doc_id = metadata[idx].document_id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_rrf_k + rank)

            content_len = len(metadata[idx].content)
            if doc_id not in best_idx or content_len > best_len[doc_id]:
                best_idx[doc_id] = idx
                best_len[doc_id] = content_len

    merged = [(best_idx[doc_id], score) for doc_id, score in scores.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def _format_result(chunk: DocumentChunk, score: float) -> dict:
    """Format a single search result."""
    return {
        "document_id": chunk.document_id,
        "title": chunk.section_title or chunk.file_path or chunk.document_id,
        "score": round(score, 4),
        "source_url": chunk.source_url,
        "content": chunk.content,
    }


async def search_knowledge(
    query: str | list[str],
    top_k: int = 5,
    *,
    ctx: Context,
) -> list[dict] | str:
    """Search Konflux documentation.

    Pass a single query string or a list of up to 5 query reformulations
    for better coverage. Multiple queries are merged via reciprocal rank
    fusion so documents matching across phrasings rank higher.
    """
    normalized = _normalize_queries(query)
    if isinstance(normalized, str):
        # _normalize_queries returns a str on validation failure (error message)
        return normalized

    queries = normalized[:_max_queries]
    top_k = max(1, min(top_k, _max_top_k))

    app: AppContext = ctx.request_context.lifespan_context["app"]

    if app.index.ntotal == 0:
        return []

    candidates = min(top_k * 2, app.index.ntotal)

    embeddings = embed_queries(app.model, queries)
    distances, indices = app.index.search(embeddings, k=candidates)

    ranked_lists = [
        list(zip(indices[i].tolist(), distances[i].tolist(), strict=True))
        for i in range(len(queries))
    ]

    if len(ranked_lists) == 1:
        results = [(idx, score) for idx, score in ranked_lists[0] if idx >= 0][:top_k]
    else:
        results = _rrf_merge(ranked_lists, app.metadata, top_k)

    return [_format_result(app.metadata[idx], score) for idx, score in results]
