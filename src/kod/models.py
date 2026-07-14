"""Data models for KOD pipeline inter-step data."""

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class Document(BaseModel):
    """A single extracted document with structured Unstructured elements."""

    elements: list[dict[str, Any]] = Field(
        description="Serialized Unstructured elements (via element.to_dict())",
    )
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


class DocumentChunk(BaseModel):
    """A chunk of a document, produced by the transform step."""

    document_id: str = Field(description="Unique document identifier ({source_name}:{file_path})")
    content: str = Field(description="Chunk text content")
    chunk_index: int = Field(description="Position of this chunk within the parent document")
    source_name: str = Field(description="Name of the document source")
    source_url: str = Field(description="URL of the document source")
    file_path: str | None = Field(
        default=None,
        description="Relative file path within a repo source",
    )
    section_title: str | None = Field(
        default=None,
        description="Title of the section this chunk belongs to",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Metadata from the source configuration",
    )
