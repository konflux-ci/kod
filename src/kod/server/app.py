"""Application context and resource loading for the KOD MCP server."""

import logging

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from fastembed import TextEmbedding

from kod.models import DocumentChunk
from kod.pipeline.io import read_chunks


logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Shared resources for MCP tool execution."""

    index: faiss.Index
    metadata: list[DocumentChunk]
    model: TextEmbedding


def load_app_context(data_dir: Path, embedding_model: str) -> AppContext:
    """Load FAISS index, metadata, and embedding model."""
    index_path = data_dir / "index" / "index.faiss"
    metadata_path = data_dir / "index" / "metadata.jsonl"

    logger.info("Loading FAISS index from %s", index_path)
    index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)

    logger.info("Loading metadata from %s", metadata_path)
    metadata = read_chunks(metadata_path)

    if len(metadata) != index.ntotal:
        msg = f"Metadata count ({len(metadata)}) does not match index vector count ({index.ntotal})"
        raise ValueError(msg)

    logger.info("Loading embedding model: %s", embedding_model)
    model = TextEmbedding(model_name=embedding_model)

    logger.info(
        "AppContext ready: %d vectors, %d dims",
        index.ntotal,
        index.d,
    )
    return AppContext(index=index, metadata=metadata, model=model)


def embed_queries(model: TextEmbedding, queries: list[str]) -> np.ndarray:
    """Embed query strings using the query prefix for asymmetric retrieval."""
    return np.array(list(model.query_embed(queries)), dtype=np.float32)


def apply_source_limit(
    candidates: list[tuple[int, float]],
    metadata: list[DocumentChunk],
    top_k: int,
    max_per_source: int = 0,
) -> list[tuple[int, float]]:
    """Filter ranked (idx, score) pairs to enforce top_k and per-source caps."""
    results = []
    source_counts: dict[str, int] = {}
    for idx, score in candidates:
        if max_per_source > 0:
            source = metadata[idx].source_name
            if source_counts.get(source, 0) >= max_per_source:
                continue
            source_counts[source] = source_counts.get(source, 0) + 1
        results.append((idx, score))
        if len(results) >= top_k:
            break
    return results


def faiss_k(ntotal: int, top_k: int, max_per_source: int) -> int:
    """How many FAISS neighbors to fetch.

    When max_per_source is active the nearest neighbors may be dominated
    by one source, so we must search the full index to fill top_k after
    the per-source cap.
    """
    if max_per_source > 0:
        return ntotal
    return min(top_k * 2, ntotal)


def search_index(
    app: AppContext, query_vec: np.ndarray, top_k: int, max_per_source: int = 0
) -> list[tuple[int, float]]:
    """Run a single-vector FAISS search and return (idx, score) pairs."""
    if app.index.ntotal == 0:
        return []
    k = faiss_k(app.index.ntotal, top_k, max_per_source)
    distances, indices = app.index.search(query_vec, k=k)
    raw = [
        (idx, score)
        for idx, score in zip(indices[0].tolist(), distances[0].tolist(), strict=True)
        if idx >= 0
    ]
    return apply_source_limit(raw, app.metadata, top_k, max_per_source)
