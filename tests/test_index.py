"""Tests for KOD FAISS index building step."""

import json

import faiss
import numpy as np

from kod.config import DocumentSource
from kod.config import KodConfig
from kod.models import DocumentChunk
from kod.pipeline.index import _build_index
from kod.pipeline.index import run_index


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


def _write_chunks(path, chunks):
    with path.open("w") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def _make_embeddings(n, dim=384):
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# --- _build_index ---


def test_build_index():
    embeddings = _make_embeddings(10)

    index = _build_index(embeddings)

    assert index.ntotal == 10
    assert index.d == 384
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT


def test_build_index_single_vector():
    embeddings = _make_embeddings(1)

    index = _build_index(embeddings)

    assert index.ntotal == 1


def test_build_index_search_returns_results():
    embeddings = _make_embeddings(5)
    index = _build_index(embeddings)

    distances, indices = index.search(embeddings[:1], k=3)

    assert indices.shape == (1, 3)
    assert indices[0][0] == 0


# --- run_index ---


def test_run_index(tmp_path):
    chunks = [_make_chunk()]
    embeddings = _make_embeddings(1)

    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", chunks)

    embedded = tmp_path / "embedded"
    embedded.mkdir()
    np.save(embedded / "test-source.npy", embeddings)

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_index(config)

    index_path = tmp_path / "index" / "index.faiss"
    metadata_path = tmp_path / "index" / "metadata.jsonl"
    assert index_path.exists()
    assert metadata_path.exists()

    index = faiss.read_index(str(index_path))
    assert index.ntotal == 1

    meta_lines = metadata_path.read_text().strip().split("\n")
    assert len(meta_lines) == 1


def test_run_index_multiple_sources(tmp_path):
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    embedded = tmp_path / "embedded"
    embedded.mkdir()

    for name in ("source-a", "source-b"):
        chunks = [_make_chunk(source_name=name, content=f"From {name}")]
        _write_chunks(chunked / f"{name}.jsonl", chunks)
        np.save(embedded / f"{name}.npy", _make_embeddings(1))

    config = KodConfig(
        sources=[
            DocumentSource(name="source-a", url="https://example.com/a"),
            DocumentSource(name="source-b", url="https://example.com/b"),
        ],
        data_dir=tmp_path,
    )

    run_index(config)

    index = faiss.read_index(str(tmp_path / "index" / "index.faiss"))
    assert index.ntotal == 2

    meta_lines = (tmp_path / "index" / "metadata.jsonl").read_text().strip().split("\n")
    assert len(meta_lines) == 2
    assert json.loads(meta_lines[0])["source_name"] == "source-a"
    assert json.loads(meta_lines[1])["source_name"] == "source-b"


def test_run_index_skips_missing_embeddings(tmp_path):
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", [_make_chunk()])

    embedded = tmp_path / "embedded"
    embedded.mkdir()

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_index(config)

    assert not (tmp_path / "index" / "index.faiss").exists()


def test_run_index_skips_missing_chunks(tmp_path):
    embedded = tmp_path / "embedded"
    embedded.mkdir()
    np.save(embedded / "test-source.npy", _make_embeddings(1))

    chunked = tmp_path / "chunked"
    chunked.mkdir()

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_index(config)

    assert not (tmp_path / "index" / "index.faiss").exists()


def test_run_index_skips_mismatched_counts(tmp_path):
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", [_make_chunk()])

    embedded = tmp_path / "embedded"
    embedded.mkdir()
    np.save(embedded / "test-source.npy", _make_embeddings(3))

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_index(config)

    assert not (tmp_path / "index" / "index.faiss").exists()


def test_run_index_mismatch_skips_but_valid_source_indexed(tmp_path):
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    embedded = tmp_path / "embedded"
    embedded.mkdir()

    _write_chunks(chunked / "bad.jsonl", [_make_chunk(source_name="bad")])
    np.save(embedded / "bad.npy", _make_embeddings(3))

    _write_chunks(chunked / "good.jsonl", [_make_chunk(source_name="good")])
    np.save(embedded / "good.npy", _make_embeddings(1))

    config = KodConfig(
        sources=[
            DocumentSource(name="bad", url="https://example.com/bad"),
            DocumentSource(name="good", url="https://example.com/good"),
        ],
        data_dir=tmp_path,
    )

    run_index(config)

    index = faiss.read_index(str(tmp_path / "index" / "index.faiss"))
    assert index.ntotal == 1

    meta_lines = (tmp_path / "index" / "metadata.jsonl").read_text().strip().split("\n")
    assert len(meta_lines) == 1
    assert json.loads(meta_lines[0])["source_name"] == "good"


def test_run_index_continues_after_corrupt_file(tmp_path):
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    embedded = tmp_path / "embedded"
    embedded.mkdir()

    (embedded / "corrupt.npy").write_bytes(b"not a numpy file")
    _write_chunks(chunked / "corrupt.jsonl", [_make_chunk(source_name="corrupt")])

    _write_chunks(chunked / "good.jsonl", [_make_chunk(source_name="good")])
    np.save(embedded / "good.npy", _make_embeddings(1))

    config = KodConfig(
        sources=[
            DocumentSource(name="corrupt", url="https://example.com/corrupt"),
            DocumentSource(name="good", url="https://example.com/good"),
        ],
        data_dir=tmp_path,
    )

    run_index(config)

    index = faiss.read_index(str(tmp_path / "index" / "index.faiss"))
    assert index.ntotal == 1


def test_run_index_mmap_loading(tmp_path):
    chunks = [_make_chunk(content=f"Chunk {i}", chunk_index=i) for i in range(5)]
    embeddings = _make_embeddings(5)

    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", chunks)

    embedded = tmp_path / "embedded"
    embedded.mkdir()
    np.save(embedded / "test-source.npy", embeddings)

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_index(config)

    index_path = str(tmp_path / "index" / "index.faiss")
    mmap_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    assert mmap_index.ntotal == 5

    distances, indices = mmap_index.search(embeddings[:1], k=3)
    assert indices[0][0] == 0


def test_run_index_metadata_position_aligned(tmp_path):
    chunks = [
        _make_chunk(content="First", chunk_index=0),
        _make_chunk(content="Second", chunk_index=1),
        _make_chunk(content="Third", chunk_index=2),
    ]
    embeddings = _make_embeddings(3)

    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", chunks)

    embedded = tmp_path / "embedded"
    embedded.mkdir()
    np.save(embedded / "test-source.npy", embeddings)

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_index(config)

    index = faiss.read_index(str(tmp_path / "index" / "index.faiss"))
    _, indices = index.search(embeddings[1:2], k=1)
    hit_idx = indices[0][0]

    meta_lines = (tmp_path / "index" / "metadata.jsonl").read_text().strip().split("\n")
    hit_meta = json.loads(meta_lines[hit_idx])
    assert hit_meta["content"] == "Second"


def test_run_index_no_embeddings_found(tmp_path):
    embedded = tmp_path / "embedded"
    embedded.mkdir()
    chunked = tmp_path / "chunked"
    chunked.mkdir()

    config = KodConfig(
        sources=[DocumentSource(name="missing", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_index(config)

    assert not (tmp_path / "index" / "index.faiss").exists()
