"""KOD ETL pipeline steps."""

import logging

from kod.config import KodConfig


logger = logging.getLogger(__name__)


def run_pipeline(config: KodConfig) -> None:
    """Run the full ETL pipeline: extract -> transform -> embed -> index."""
    from kod.pipeline.embed import run_embed
    from kod.pipeline.extract import run_extract
    from kod.pipeline.index import run_index
    from kod.pipeline.transform import run_transform

    for step in [run_extract, run_transform, run_embed, run_index]:
        step(config)
    logger.info("Pipeline complete")
