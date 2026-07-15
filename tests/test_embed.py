"""Tests for KOD embedding step."""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from kod.config import DocumentSource
from kod.config import KodConfig
from kod.models import DocumentChunk
from kod.pipeline.embed import _embed_chunks
from kod.pipeline.embed import _get_embedding_model
from kod.pipeline.embed import run_embed


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


def _mock_model(dim=384):
    model = MagicMock()

    def fake_passage_embed(texts):
        for _ in texts:
            yield np.random.default_rng(42).random(dim, dtype=np.float32)

    model.passage_embed.side_effect = fake_passage_embed
    return model


# --- _get_embedding_model ---


@patch("kod.pipeline.embed.TextEmbedding")
def test_get_embedding_model(mock_cls):
    _get_embedding_model("BAAI/bge-small-en-v1.5")

    mock_cls.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")


# --- _embed_chunks ---


def test_embed_chunks():
    model = _mock_model(dim=384)
    chunks = [_make_chunk(content="First"), _make_chunk(content="Second")]

    result = _embed_chunks(chunks, model)

    assert result.shape == (2, 384)
    assert result.dtype == np.float32
    model.passage_embed.assert_called_once_with(["First", "Second"])


def test_embed_chunks_single():
    model = _mock_model(dim=384)
    chunks = [_make_chunk(content="Only one")]

    result = _embed_chunks(chunks, model)

    assert result.shape == (1, 384)


# --- run_embed ---


@patch("kod.pipeline.embed._get_embedding_model")
def test_run_embed(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", [_make_chunk()])

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_embed(config)

    output = tmp_path / "embedded" / "test-source.npy"
    assert output.exists()
    data = np.load(output)
    assert data.shape == (1, 384)


@patch("kod.pipeline.embed._get_embedding_model")
def test_run_embed_multiple_sources(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    for name in ("source-a", "source-b"):
        _write_chunks(chunked / f"{name}.jsonl", [_make_chunk(source_name=name)])

    config = KodConfig(
        sources=[
            DocumentSource(name="source-a", url="https://example.com/a"),
            DocumentSource(name="source-b", url="https://example.com/b"),
        ],
        data_dir=tmp_path,
    )

    run_embed(config)

    assert (tmp_path / "embedded" / "source-a.npy").exists()
    assert (tmp_path / "embedded" / "source-b.npy").exists()


@patch("kod.pipeline.embed._get_embedding_model")
def test_run_embed_skips_missing_chunked(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    chunked = tmp_path / "chunked"
    chunked.mkdir()

    config = KodConfig(
        sources=[DocumentSource(name="missing", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_embed(config)

    assert not (tmp_path / "embedded" / "missing.npy").exists()


@patch("kod.pipeline.embed._get_embedding_model")
def test_run_embed_skips_empty_chunks(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    (chunked / "empty.jsonl").write_text("")

    config = KodConfig(
        sources=[DocumentSource(name="empty", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_embed(config)

    assert not (tmp_path / "embedded" / "empty.npy").exists()


@patch("kod.pipeline.embed._get_embedding_model")
def test_run_embed_continues_after_failure(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    (chunked / "bad.jsonl").write_text("not valid json\n")
    _write_chunks(chunked / "good.jsonl", [_make_chunk(source_name="good")])

    config = KodConfig(
        sources=[
            DocumentSource(name="bad", url="https://example.com/bad"),
            DocumentSource(name="good", url="https://example.com/good"),
        ],
        data_dir=tmp_path,
    )

    run_embed(config)

    assert (tmp_path / "embedded" / "good.npy").exists()


@patch("kod.pipeline.embed._get_embedding_model")
def test_run_embed_multiple_chunks_per_source(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    chunks = [_make_chunk(content=f"Chunk {i}", chunk_index=i) for i in range(5)]
    _write_chunks(chunked / "test-source.jsonl", chunks)

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
    )

    run_embed(config)

    data = np.load(tmp_path / "embedded" / "test-source.npy")
    assert data.shape == (5, 384)


@patch("kod.pipeline.embed._get_embedding_model")
def test_run_embed_uses_config_model(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    chunked = tmp_path / "chunked"
    chunked.mkdir()
    _write_chunks(chunked / "test-source.jsonl", [_make_chunk()])

    config = KodConfig(
        sources=[DocumentSource(name="test-source", url="https://example.com")],
        data_dir=tmp_path,
        embedding_model="BAAI/bge-base-en-v1.5",
    )

    run_embed(config)

    mock_get_model.assert_called_once_with("BAAI/bge-base-en-v1.5")
