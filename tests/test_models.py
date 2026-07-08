"""Tests for KOD data models."""

from kod.models import Document


def test_document_all_fields():
    doc = Document(
        content="Hello",
        source_name="src",
        source_url="https://example.com",
        file_path="doc.md",
        metadata={"k": "v"},
    )
    assert doc.content == "Hello"
    assert doc.source_name == "src"
    assert doc.source_url == "https://example.com"
    assert doc.file_path == "doc.md"
    assert doc.metadata == {"k": "v"}


def test_document_defaults():
    doc = Document(
        content="Hello",
        source_name="src",
        source_url="https://example.com",
    )
    assert doc.file_path is None
    assert doc.metadata == {}
