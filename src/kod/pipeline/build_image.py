"""Build-image step - build container image with pre-built index."""

import logging

from kod.config import KodConfig


logger = logging.getLogger(__name__)


def run_build_image(config: KodConfig) -> None:
    """Build the container image using podman."""
    logger.info("[build-image] Would build container image with podman")
    logger.info("[build-image] Build complete (stub)")
