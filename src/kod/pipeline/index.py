"""Index step - build FAISS vector index."""

import logging

from kod.config import KodConfig


logger = logging.getLogger(__name__)


def run_index(config: KodConfig) -> None:
    """Build FAISS index from embeddings."""
    logger.info("[index] Would build FAISS index from embeddings")
    logger.info("[index] Indexing complete (stub)")
