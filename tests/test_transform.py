"""Tests for KOD document chunking (transform step)."""

import json

from kod.config import DocumentSource
from kod.config import KodConfig
from kod.models import Document
from kod.pipeline.transform import _chunk_document
from kod.pipeline.transform import _get_section_title
from kod.pipeline.transform import _read_documents
from kod.pipeline.transform import run_transform


def _make_elements(*parts):
    """Create serialized Unstructured element dicts.

    Each part is either a string (NarrativeText) or a tuple of (category, text).
    """
    result = []
    for part in parts:
        if isinstance(part, tuple):
            category, text = part
        else:
            category, text = "NarrativeText", part
        result.append(
            {
                "type": category,
                "element_id": "test",
                "text": text,
                "metadata": {"category_depth": 0 if category == "Title" else None},
            }
        )
    return result


def _make_document(**overrides):
    defaults = {
        "elements": _make_elements("Hello world."),
        "source_name": "test-source",
        "source_url": "https://example.com",
        "file_path": "doc.md",
        "metadata": {"product": "Test"},
    }
    defaults.update(overrides)
    return Document(**defaults)


def _write_jsonl(path, documents):
    with path.open("w") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")


# --- _read_documents ---


def test_read_documents(tmp_path):
    path = tmp_path / "docs.jsonl"
    doc = _make_document()
    _write_jsonl(path, [doc])

    result = _read_documents(path)

    assert len(result) == 1
    assert result[0].source_name == "test-source"


def test_read_documents_multiple(tmp_path):
    path = tmp_path / "docs.jsonl"
    docs = [_make_document(file_path="a.md"), _make_document(file_path="b.md")]
    _write_jsonl(path, docs)

    result = _read_documents(path)

    assert len(result) == 2
    assert result[0].file_path == "a.md"
    assert result[1].file_path == "b.md"


def test_read_documents_skips_blank_lines(tmp_path):
    path = tmp_path / "docs.jsonl"
    doc = _make_document()
    path.write_text(doc.model_dump_json() + "\n\n")

    result = _read_documents(path)

    assert len(result) == 1


def test_read_documents_empty_file(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text("")

    result = _read_documents(path)

    assert result == []


# --- _get_section_title ---


def test_get_section_title_with_title():
    from unstructured.documents.elements import CompositeElement
    from unstructured.documents.elements import NarrativeText
    from unstructured.documents.elements import Title

    title_el = Title("Getting Started")
    narrative_el = NarrativeText("Some content here.")
    chunk = CompositeElement("Getting Started\n\nSome content here.")
    chunk.metadata.orig_elements = [title_el, narrative_el]

    assert _get_section_title(chunk) == "Getting Started"


def test_get_section_title_without_title():
    from unstructured.documents.elements import CompositeElement
    from unstructured.documents.elements import NarrativeText

    narrative_el = NarrativeText("Some content here.")
    chunk = CompositeElement("Some content here.")
    chunk.metadata.orig_elements = [narrative_el]

    assert _get_section_title(chunk) is None


def test_get_section_title_no_orig_elements():
    from unstructured.documents.elements import CompositeElement

    chunk = CompositeElement("Some content.")

    assert _get_section_title(chunk) is None


# --- _chunk_document ---


def test_chunk_document_single_section():
    doc = _make_document(
        elements=_make_elements(
            ("Title", "Introduction"),
            "This is the introduction text.",
        ),
    )

    chunks = _chunk_document(doc, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 1
    assert "Introduction" in chunks[0].content
    assert "introduction text" in chunks[0].content
    assert chunks[0].document_id == "test-source:doc.md"
    assert chunks[0].chunk_index == 0
    assert chunks[0].source_name == "test-source"
    assert chunks[0].source_url == "https://example.com"
    assert chunks[0].file_path == "doc.md"
    assert chunks[0].metadata == {"product": "Test"}


def test_chunk_document_multiple_sections():
    section_a_text = "Content for section A. " * 10
    section_b_text = "Content for section B. " * 10
    doc = _make_document(
        elements=_make_elements(
            ("Title", "Section A"),
            section_a_text,
            ("Title", "Section B"),
            section_b_text,
        ),
    )

    chunks = _chunk_document(doc, chunk_size=100, chunk_overlap=0)

    assert len(chunks) >= 2
    titles = [c.section_title for c in chunks]
    assert "Section A" in titles
    assert "Section B" in titles
    assert chunks[0].chunk_index == 0


def test_chunk_document_preserves_metadata():
    doc = _make_document(
        metadata={"product": "Konflux", "topic": "architecture"},
    )

    chunks = _chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert len(chunks) >= 1
    assert chunks[0].metadata == {"product": "Konflux", "topic": "architecture"}


def test_chunk_document_empty_elements():
    doc = _make_document(elements=[])

    chunks = _chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert chunks == []


def test_chunk_document_no_title_elements():
    doc = _make_document(
        elements=_make_elements("Just plain text content."),
    )

    chunks = _chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].section_title is None
    assert "plain text" in chunks[0].content


def test_chunk_document_long_content_splits():
    long_text = "Word " * 500
    doc = _make_document(
        elements=_make_elements(
            ("Title", "Long Section"),
            long_text,
        ),
    )

    chunks = _chunk_document(doc, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(c.source_name == "test-source" for c in chunks)
    assert all(c.section_title == "Long Section" for c in chunks)


def test_chunk_document_deeply_nested_headers():
    text_block = "Detailed content for this section. " * 10
    doc = _make_document(
        elements=_make_elements(
            ("Title", "H1"),
            text_block,
            ("Title", "H2"),
            text_block,
            ("Title", "H3"),
            text_block,
        ),
    )

    chunks = _chunk_document(doc, chunk_size=100, chunk_overlap=0)

    assert len(chunks) >= 3


def test_chunk_document_file_path_none():
    doc = _make_document(file_path=None)

    chunks = _chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert len(chunks) >= 1
    assert chunks[0].file_path is None
    assert chunks[0].document_id == "test-source:"


def test_chunk_document_document_id_format():
    doc_with_path = _make_document(source_name="my-source", file_path="path/to/doc.md")
    doc_no_path = _make_document(source_name="my-source", file_path=None)

    chunks_with = _chunk_document(doc_with_path, chunk_size=1000, chunk_overlap=0)
    chunks_without = _chunk_document(doc_no_path, chunk_size=1000, chunk_overlap=0)

    assert chunks_with[0].document_id == "my-source:path/to/doc.md"
    assert chunks_without[0].document_id == "my-source:"


def test_chunk_document_section_title_propagation():
    long_text = "Some detailed content for this section. " * 20
    doc = _make_document(
        elements=_make_elements(
            ("Title", "First"),
            long_text,
            ("Title", "Second"),
            long_text,
        ),
    )

    chunks = _chunk_document(doc, chunk_size=200, chunk_overlap=0)

    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.section_title in ("First", "Second")


# --- run_transform ---


def test_run_transform(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    doc = _make_document(
        elements=_make_elements(
            ("Title", "Getting Started"),
            "Introduction to the system.",
        ),
    )
    _write_jsonl(extracted / "test-source.jsonl", [doc])

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_transform(config)

    output = tmp_path / "chunked" / "test-source.jsonl"
    assert output.exists()
    lines = output.read_text().strip().split("\n")
    assert len(lines) >= 1
    chunk = json.loads(lines[0])
    assert chunk["source_name"] == "test-source"


def test_run_transform_multiple_sources(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    for name in ("source-a", "source-b"):
        doc = _make_document(source_name=name)
        _write_jsonl(extracted / f"{name}.jsonl", [doc])

    config = KodConfig(
        sources=[
            DocumentSource(name="source-a", url="https://example.com/a"),
            DocumentSource(name="source-b", url="https://example.com/b"),
        ],
        data_dir=tmp_path,
    )

    run_transform(config)

    assert (tmp_path / "chunked" / "source-a.jsonl").exists()
    assert (tmp_path / "chunked" / "source-b.jsonl").exists()


def test_run_transform_skips_missing_extracted(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()

    config = KodConfig(
        sources=[DocumentSource(name="missing", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_transform(config)

    assert not (tmp_path / "chunked" / "missing.jsonl").exists()


def test_run_transform_continues_after_failure(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "bad.jsonl").write_text("not valid json\n")
    good_doc = _make_document(source_name="good")
    _write_jsonl(extracted / "good.jsonl", [good_doc])

    config = KodConfig(
        sources=[
            DocumentSource(name="bad", url="https://example.com/bad"),
            DocumentSource(name="good", url="https://example.com/good"),
        ],
        data_dir=tmp_path,
    )

    run_transform(config)

    good_output = tmp_path / "chunked" / "good.jsonl"
    assert good_output.exists()
    assert good_output.read_text().strip() != ""


def test_run_transform_custom_chunk_size(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    long_text = "Word " * 500
    doc = _make_document(elements=_make_elements(long_text))
    _write_jsonl(extracted / "test-source.jsonl", [doc])

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
        chunk_size=200,
        chunk_overlap=50,
    )

    run_transform(config)

    output = tmp_path / "chunked" / "test-source.jsonl"
    lines = output.read_text().strip().split("\n")
    assert len(lines) > 1


def test_run_transform_empty_document(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    doc = _make_document(elements=[])
    _write_jsonl(extracted / "test-source.jsonl", [doc])

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_transform(config)

    output = tmp_path / "chunked" / "test-source.jsonl"
    assert output.exists()
    assert output.read_text() == ""
