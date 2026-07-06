"""Configuration schema and loader for KOD."""

import logging

from pathlib import Path

import yaml

from pydantic import BaseModel
from pydantic import Field


logger = logging.getLogger(__name__)


class DocumentSource(BaseModel):
    """A single documentation source to extract and index."""

    name: str = Field(description="Human-readable name for this source")
    url: str = Field(description="URL of the documentation source")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional metadata attached to extracted documents"
    )


class KodConfig(BaseModel):
    """Top-level KOD configuration."""

    sources: list[DocumentSource] = Field(
        min_length=1, description="List of documentation sources to index"
    )


def load_config(path: str | Path) -> KodConfig:
    """Load and validate a KOD configuration file."""
    config_path = Path(path)
    logger.info("Loading configuration from %s", config_path)
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    config = KodConfig.model_validate(raw)
    source_names = ", ".join(s.name for s in config.sources)
    logger.info("Loaded %d source(s): %s", len(config.sources), source_names)
    return config
