"""Configuration schema and loader for KOD."""

import logging

from pathlib import Path

import yaml

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


logger = logging.getLogger(__name__)


class DocumentSource(BaseModel):
    """A single documentation source to extract and index."""

    name: str = Field(description="Human-readable name for this source")
    url: str = Field(description="URL of the documentation source")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if any(c in v for c in ("/", "\\", "\0")):
            msg = "name must not contain '/', '\\', or null bytes"
            raise ValueError(msg)
        if ".." in v:
            msg = "name must not contain '..'"
            raise ValueError(msg)
        return v

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata attached to extracted documents",
    )
    include_paths: list[str] = Field(
        default_factory=list,
        description="Only extract from these paths within the source",
    )
    exclude_paths: list[str] = Field(
        default_factory=list,
        description="Exclude these paths from extraction",
    )
    max_pages: int = Field(
        default=50,
        ge=1,
        description="Maximum number of pages to crawl for web sources",
    )
    use_sitemap: bool = Field(
        default=True,
        description="Try sitemap.xml before crawling links for web sources",
    )

    @model_validator(mode="after")
    def _check_paths_mutual_exclusivity(self):
        if self.include_paths and self.exclude_paths:
            msg = "include_paths and exclude_paths cannot both be set"
            raise ValueError(msg)
        return self


class KodConfig(BaseModel):
    """Top-level KOD configuration."""

    sources: list[DocumentSource] = Field(
        min_length=1, description="List of documentation sources to index"
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Working directory for pipeline intermediate files",
    )
    doc_extensions: set[str] = Field(
        default={".md", ".adoc", ".html", ".htm"},
        description="File extensions to extract from git sources",
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum characters per chunk",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Character overlap between consecutive chunks",
    )

    @model_validator(mode="after")
    def _check_chunk_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            msg = "chunk_overlap must be less than chunk_size"
            raise ValueError(msg)
        return self


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
