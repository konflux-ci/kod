"""Data models for KOD pipeline inter-step data."""

from pydantic import BaseModel
from pydantic import Field


class Document(BaseModel):
    """A single extracted document."""

    content: str = Field(description="Full text content")
    source_name: str = Field(description="Name of the document source")
    source_url: str = Field(description="URL of the document source")
    file_path: str | None = Field(
        default=None,
        description="Relative file path within a repo source",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Metadata from the source configuration",
    )
