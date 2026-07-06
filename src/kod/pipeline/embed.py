"""Embed step - generate vector embeddings for chunks."""

import logging

from kod.config import KodConfig


logger = logging.getLogger(__name__)


def run_embed(config: KodConfig) -> None:
    """Generate embeddings for document chunks using FastEmbed."""
    logger.info(
        "[embed] Would generate embeddings for chunks from %d source(s)",
        len(config.sources),
    )
    logger.info("[embed] Embedding complete (stub)")
