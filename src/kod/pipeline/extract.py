"""Extract step - fetch documents from configured sources."""

import logging

from kod.config import KodConfig


logger = logging.getLogger(__name__)


def run_extract(config: KodConfig) -> None:
    """Extract documents from all configured sources."""
    for source in config.sources:
        logger.info("[extract] Would extract from '%s' (%s)", source.name, source.url)
    logger.info("[extract] Extraction complete (stub)")
