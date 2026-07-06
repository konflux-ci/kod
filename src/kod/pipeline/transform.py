"""Transform step - chunk extracted documents."""

import logging

from kod.config import KodConfig


logger = logging.getLogger(__name__)


def run_transform(config: KodConfig) -> None:
    """Chunk extracted documents for embedding."""
    logger.info("[transform] Would chunk documents from %d source(s)", len(config.sources))
    logger.info("[transform] Transform complete (stub)")
