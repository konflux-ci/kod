"""Application context and resource loading for the KOD MCP server."""

import json
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


def _read_indexed_model(path: Path) -> str | None:
    """Return the embedding model name recorded at index time, if available.

    The metadata file is best-effort provenance, so an unreadable, malformed,
    or unexpected payload is logged and ignored rather than aborting startup.
    """
    try:
        meta = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable or malformed index metadata at %s", path)
        return None
    if not isinstance(meta, dict):
        logger.warning("Ignoring malformed index metadata at %s", path)
        return None
    model = meta.get("embedding_model")
    return model if isinstance(model, str) else None


def load_app_context(data_dir: Path, embedding_model: str) -> AppContext:
    """Load FAISS index, metadata, and embedding model."""
    index_path = data_dir / "index" / "index.faiss"
    metadata_path = data_dir / "index" / "metadata.jsonl"
    index_meta_path = data_dir / "index" / "index_meta.json"

    logger.info("Loading FAISS index from %s", index_path)
    index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)

    logger.info("Loading metadata from %s", metadata_path)
    metadata = read_chunks(metadata_path)

    if len(metadata) != index.ntotal:
        msg = f"Metadata count ({len(metadata)}) does not match index vector count ({index.ntotal})"
        raise ValueError(msg)

    if index_meta_path.exists():
        indexed_model = _read_indexed_model(index_meta_path)
        # FastEmbed resolves model names case-insensitively, so compare that way.
        if indexed_model and indexed_model.lower() != embedding_model.lower():
            logger.warning(
                "Configured embedding model '%s' differs from the model used at index "
                "time '%s'; query and document vectors may be incompatible, degrading "
                "search rankings",
                embedding_model,
                indexed_model,
            )

    # Check dimensions from FastEmbed's registry before loading the model, so a
    # mismatched-but-valid model fails fast instead of paying a full ONNX load
    # (which can OOM a memory-capped pod) only to be rejected afterwards.
    model_dim = TextEmbedding.get_embedding_size(embedding_model)
    if model_dim != index.d:
        msg = (
            f"Embedding model '{embedding_model}' produces {model_dim}-dim "
            f"vectors but the FAISS index has {index.d} dims; the index must be rebuilt "
            f"with this model or the server started with the model used to build it"
        )
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
