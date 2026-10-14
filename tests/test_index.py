"""Tests for KOD FAISS index building step."""

import json

from unittest.mock import patch

import faiss
import numpy as np
import pytest

from kod.config import DEFAULT_EMBEDDING_MODEL
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


def _save_embeddings(embedded_dir, name, embeddings, model=DEFAULT_EMBEDDING_MODEL):
    """Write a source's `.npy` and its provenance sidecar, like run_embed does."""
    np.save(embedded_dir / f"{name}.npy", embeddings)
    (embedded_dir / f"{name}.meta.json").write_text(
        json.dumps({"embedding_model": model, "dim": int(embeddings.shape[1])})
    )


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
    _save_embeddings(embedded, "test-source", embeddings)

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


def test_run_index_writes_index_meta(tmp_path):
    chunks = [_make_chunk()]
    embeddings = _make_embeddings(1)

    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", chunks)

    embedded = tmp_path / "embedded"
    embedded.mkdir()
    _save_embeddings(embedded, "test-source", embeddings, model="custom/embedding-model")

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
        embedding_model="custom/embedding-model",
    )

    run_index(config)

    index_meta = json.loads((tmp_path / "index" / "index_meta.json").read_text())
    assert index_meta == {"embedding_model": "custom/embedding-model"}


def test_run_index_no_meta_when_nothing_indexed(tmp_path):
    embedded = tmp_path / "embedded"
    embedded.mkdir()
    chunked = tmp_path / "chunked"
    chunked.mkdir()

    config = KodConfig(
        sources=[DocumentSource(name="missing", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_index(config)

    assert not (tmp_path / "index" / "index_meta.json").exists()


def test_run_index_multiple_sources(tmp_path):
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    embedded = tmp_path / "embedded"
    embedded.mkdir()

    for name in ("source-a", "source-b"):
        chunks = [_make_chunk(source_name=name, content=f"From {name}")]
        _write_chunks(chunked / f"{name}.jsonl", chunks)
        _save_embeddings(embedded, name, _make_embeddings(1))

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
    _save_embeddings(embedded, "test-source", _make_embeddings(3))

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
    _save_embeddings(embedded, "bad", _make_embeddings(3))

    _write_chunks(chunked / "good.jsonl", [_make_chunk(source_name="good")])
    _save_embeddings(embedded, "good", _make_embeddings(1))

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

    # Valid provenance so the source passes the model check and fails later, at
    # np.load, exercising the load-failure path rather than the provenance guard.
    (embedded / "corrupt.npy").write_bytes(b"not a numpy file")
    (embedded / "corrupt.meta.json").write_text(
        json.dumps({"embedding_model": DEFAULT_EMBEDDING_MODEL, "dim": 384})
    )
    _write_chunks(chunked / "corrupt.jsonl", [_make_chunk(source_name="corrupt")])

    _write_chunks(chunked / "good.jsonl", [_make_chunk(source_name="good")])
    _save_embeddings(embedded, "good", _make_embeddings(1))

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
    _save_embeddings(embedded, "test-source", embeddings)

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
    _save_embeddings(embedded, "test-source", embeddings)

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


# --- run_index provenance verification ---


def test_run_index_raises_on_model_change_between_embed_and_index(tmp_path):
    """A same-dimension model swap between embed and index must abort the build."""
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", [_make_chunk()])

    embedded = tmp_path / "embedded"
    embedded.mkdir()
    # Embedded with one model...
    _save_embeddings(embedded, "test-source", _make_embeddings(1), model="other/model-384d")

    # ...but indexed with another (same dimension, so np.vstack would succeed).
    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
        embedding_model="BAAI/bge-small-en-v1.5",
    )

    with pytest.raises(ValueError, match="provenance check failed"):
        run_index(config)

    assert not (tmp_path / "index" / "index.faiss").exists()
    assert not (tmp_path / "index" / "index_meta.json").exists()


def test_run_index_raises_on_stale_npy_without_sidecar(tmp_path):
    """A stale `.npy` left by a failed re-embed (no sidecar) must abort the build."""
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", [_make_chunk()])

    embedded = tmp_path / "embedded"
    embedded.mkdir()
    np.save(embedded / "test-source.npy", _make_embeddings(1))  # no sidecar

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="provenance check failed"):
        run_index(config)

    assert not (tmp_path / "index" / "index.faiss").exists()


@pytest.mark.parametrize("body", ["not json", "[]", "null", '{"embedding_model": 123}'])
def test_run_index_raises_on_malformed_sidecar(tmp_path, body):
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", [_make_chunk()])

    embedded = tmp_path / "embedded"
    embedded.mkdir()
    np.save(embedded / "test-source.npy", _make_embeddings(1))
    (embedded / "test-source.meta.json").write_text(body)

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="provenance check failed"):
        run_index(config)


def test_run_index_raises_naming_only_offending_source(tmp_path):
    """One mismatched source aborts the whole build and is named in the error."""
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    embedded = tmp_path / "embedded"
    embedded.mkdir()

    _write_chunks(chunked / "good.jsonl", [_make_chunk(source_name="good")])
    _save_embeddings(embedded, "good", _make_embeddings(1))

    _write_chunks(chunked / "wrong.jsonl", [_make_chunk(source_name="wrong")])
    _save_embeddings(embedded, "wrong", _make_embeddings(1), model="other/model-384d")

    config = KodConfig(
        sources=[
            DocumentSource(name="good", url="https://example.com/good"),
            DocumentSource(name="wrong", url="https://example.com/wrong"),
        ],
        data_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="wrong") as excinfo:
        run_index(config)

    assert "good" not in str(excinfo.value)
    assert not (tmp_path / "index" / "index.faiss").exists()


def test_run_index_replaces_stale_index_meta_atomically(tmp_path):
    chunks = [_make_chunk()]
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", chunks)
    embedded = tmp_path / "embedded"
    embedded.mkdir()
    _save_embeddings(embedded, "test-source", _make_embeddings(1), model="custom/embedding-model")

    # A record from a previous build with a different model.
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index_meta.json").write_text(json.dumps({"embedding_model": "old/model"}))

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
        embedding_model="custom/embedding-model",
    )

    run_index(config)

    assert json.loads((index_dir / "index_meta.json").read_text()) == {
        "embedding_model": "custom/embedding-model"
    }
    # The atomic temp file is not left behind.
    assert not (index_dir / "index_meta.json.tmp").exists()


def test_run_index_interrupted_rebuild_leaves_no_stale_meta(tmp_path):
    """A crash mid-rebuild must not leave an old model record beside new vectors."""
    chunks = [_make_chunk()]
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", chunks)
    embedded = tmp_path / "embedded"
    embedded.mkdir()
    _save_embeddings(embedded, "test-source", _make_embeddings(1))

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index_meta.json").write_text(json.dumps({"embedding_model": "old/model"}))

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    with (
        patch("kod.pipeline.index.write_chunks", side_effect=OSError("boom")),
        pytest.raises(OSError),
    ):
        run_index(config)

    # The stale record was cleared before the crash, so serve falls back to the
    # back-compat default rather than trusting the old model against new vectors.
    assert not (index_dir / "index_meta.json").exists()
