"""Tests for shared pipeline I/O helpers."""

import json

from kod.models import DocumentChunk
from kod.pipeline.io import read_chunks
from kod.pipeline.io import write_chunks


def _make_chunk(**overrides):
    defaults = {
        "document_id": "test-source:doc.md",
        "content": "Hello world.",
        "chunk_index": 0,
        "source_name": "test-source",
        "source_url": "https://example.com",
        "file_path": "doc.md",
        "section_title": "Intro",
        "metadata": {"product": "Test"},
    }
    defaults.update(overrides)
    return DocumentChunk(**defaults)


# --- read_chunks ---


def test_read_chunks(tmp_path):
    path = tmp_path / "chunks.jsonl"
    chunk = _make_chunk()
    path.write_text(chunk.model_dump_json() + "\n")

    result = read_chunks(path)

    assert len(result) == 1
    assert result[0].content == "Hello world."


def test_read_chunks_multiple(tmp_path):
    path = tmp_path / "chunks.jsonl"
    chunks = [_make_chunk(content="A", chunk_index=0), _make_chunk(content="B", chunk_index=1)]
    path.write_text("".join(c.model_dump_json() + "\n" for c in chunks))

    result = read_chunks(path)

    assert len(result) == 2
    assert result[0].content == "A"
    assert result[1].content == "B"


def test_read_chunks_skips_blank_lines(tmp_path):
    path = tmp_path / "chunks.jsonl"
    chunk = _make_chunk()
    path.write_text(chunk.model_dump_json() + "\n\n")

    result = read_chunks(path)

    assert len(result) == 1


def test_read_chunks_empty_file(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("")

    result = read_chunks(path)

    assert result == []


# --- write_chunks ---


def test_write_chunks(tmp_path):
    chunks = [
        _make_chunk(content="Chunk 1"),
        _make_chunk(content="Chunk 2", chunk_index=1),
    ]
    path = tmp_path / "out.jsonl"

    write_chunks(chunks, path)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["content"] == "Chunk 1"
    assert json.loads(lines[1])["content"] == "Chunk 2"


def test_write_chunks_empty(tmp_path):
    path = tmp_path / "out.jsonl"

    write_chunks([], path)

    assert path.read_text() == ""


def test_write_chunks_roundtrip(tmp_path):
    chunk = _make_chunk()
    path = tmp_path / "out.jsonl"

    write_chunks([chunk], path)
    restored = read_chunks(path)

    assert len(restored) == 1
    assert restored[0] == chunk
