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
