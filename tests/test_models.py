"""Tests for KOD data models."""

from kod.models import Document
from kod.models import DocumentChunk


def _fake_elements(*texts):
    return [{"type": "NarrativeText", "text": t, "metadata": {}} for t in texts]


def test_document_all_fields():
    doc = Document(
        elements=_fake_elements("Hello"),
        source_name="src",
        source_url="https://example.com",
        file_path="doc.md",
        metadata={"k": "v"},
    )
    assert doc.elements[0]["text"] == "Hello"
    assert doc.source_name == "src"
    assert doc.source_url == "https://example.com"
    assert doc.file_path == "doc.md"
    assert doc.metadata == {"k": "v"}


def test_document_defaults():
    doc = Document(
        elements=_fake_elements("Hello"),
        source_name="src",
        source_url="https://example.com",
    )
    assert doc.file_path is None
    assert doc.metadata == {}


def test_document_chunk_all_fields():
    chunk = DocumentChunk(
        document_id="src:doc.md",
        content="Chunk text",
        chunk_index=0,
        source_name="src",
        source_url="https://example.com",
        file_path="doc.md",
        section_title="Introduction",
        metadata={"k": "v"},
    )
    assert chunk.document_id == "src:doc.md"
    assert chunk.content == "Chunk text"
    assert chunk.chunk_index == 0
    assert chunk.source_name == "src"
    assert chunk.section_title == "Introduction"


def test_document_chunk_defaults():
    chunk = DocumentChunk(
        document_id="src:",
        content="Chunk text",
        chunk_index=0,
        source_name="src",
        source_url="https://example.com",
    )
    assert chunk.file_path is None
    assert chunk.section_title is None
    assert chunk.metadata == {}
