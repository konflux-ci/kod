"""Tests for KOD MCP server."""

import asyncio

from unittest.mock import MagicMock
from unittest.mock import patch

import faiss
import numpy as np
import pytest

from kod.models import DocumentChunk
from kod.pipeline.io import write_chunks
from kod.server import tools as _tools_module
from kod.server.app import AppContext
from kod.server.app import embed_queries
from kod.server.app import load_app_context
from kod.server.tools import _normalize_queries
from kod.server.tools import _rrf_merge
from kod.server.tools import configure
from kod.server.tools import search_knowledge


@pytest.fixture(autouse=True)
def _reset_tools_globals():
    """Reset search tuning globals after each test."""
    yield
    _tools_module._rrf_k = 60
    _tools_module._max_queries = 5
    _tools_module._max_top_k = 20


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


def _make_embeddings(n, dim=384):
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _build_test_index(tmp_path, chunks, embeddings):
    """Write a FAISS index and metadata to tmp_path/index/."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    write_chunks(chunks, index_dir / "metadata.jsonl")


def _mock_model(dim=384):
    model = MagicMock()

    def fake_query_embed(texts):
        rng = np.random.default_rng(99)
        for _ in texts:
            vec = rng.random(dim, dtype=np.float32)
            vec /= np.linalg.norm(vec)
            yield vec

    model.query_embed.side_effect = fake_query_embed
    return model


def _make_ctx(app):
    """Build a minimal FastMCP-compatible context mock."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"app": app}
    return ctx


# --- load_app_context ---


@patch("kod.server.app.TextEmbedding")
def test_load_app_context_success(mock_te, tmp_path):
    chunks = [_make_chunk()]
    embeddings = _make_embeddings(1)
    _build_test_index(tmp_path, chunks, embeddings)

    app = load_app_context(tmp_path, "BAAI/bge-small-en-v1.5")

    assert app.index.ntotal == 1
    assert len(app.metadata) == 1
    assert app.metadata[0].document_id == "test-source:doc.md"
    mock_te.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")


@patch("kod.server.app.TextEmbedding")
def test_load_app_context_mmap_search(mock_te, tmp_path):
    embeddings = _make_embeddings(5)
    chunks = [_make_chunk(chunk_index=i, content=f"Chunk {i}") for i in range(5)]
    _build_test_index(tmp_path, chunks, embeddings)

    app = load_app_context(tmp_path, "BAAI/bge-small-en-v1.5")

    distances, indices = app.index.search(embeddings[:1], k=3)
    assert indices[0][0] == 0


@patch("kod.server.app.TextEmbedding")
def test_load_app_context_mismatched_counts(mock_te, tmp_path):
    chunks = [_make_chunk()]
    embeddings = _make_embeddings(3)
    _build_test_index(tmp_path, chunks, embeddings)

    with pytest.raises(ValueError, match="does not match"):
        load_app_context(tmp_path, "BAAI/bge-small-en-v1.5")


def test_load_app_context_missing_index(tmp_path):
    with pytest.raises(RuntimeError):
        load_app_context(tmp_path, "BAAI/bge-small-en-v1.5")


@patch("kod.server.app.TextEmbedding")
def test_load_app_context_empty_index(mock_te, tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(384)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    write_chunks([], index_dir / "metadata.jsonl")

    app = load_app_context(tmp_path, "BAAI/bge-small-en-v1.5")

    assert app.index.ntotal == 0
    assert len(app.metadata) == 0


# --- embed_queries ---


def test_embed_queries():
    model = _mock_model(dim=384)

    result = embed_queries(model, ["hello", "world"])

    assert result.shape == (2, 384)
    assert result.dtype == np.float32
    model.query_embed.assert_called_once_with(["hello", "world"])


# --- configure ---


def test_configure_sets_globals():
    configure(rrf_k=30, max_queries=3, max_top_k=10)

    assert _tools_module._rrf_k == 30
    assert _tools_module._max_queries == 3
    assert _tools_module._max_top_k == 10


@pytest.mark.parametrize("param,value", [("rrf_k", 0), ("max_queries", 0), ("max_top_k", -1)])
def test_configure_rejects_invalid(param, value):
    with pytest.raises(ValueError, match="must be >= 1"):
        configure(**{param: value})


def test_configure_rrf_k_affects_merge():
    metadata = [
        _make_chunk(document_id="a"),
        _make_chunk(document_id="b"),
    ]
    ranked_lists = [
        [(0, 0.9), (1, 0.8)],
        [(1, 0.85), (0, 0.7)],
    ]

    result_default = _rrf_merge(ranked_lists, metadata, top_k=2)
    scores_default = {metadata[idx].document_id: s for idx, s in result_default}

    configure(rrf_k=1)
    result_custom = _rrf_merge(ranked_lists, metadata, top_k=2)
    scores_custom = {metadata[idx].document_id: s for idx, s in result_custom}

    assert scores_default != scores_custom


def test_configure_max_queries_affects_search():
    embeddings = _make_embeddings(5)
    chunks = [_make_chunk(document_id=f"doc-{i}", chunk_index=i) for i in range(5)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    model = _mock_model()
    app = AppContext(index=index, metadata=chunks, model=model)
    ctx = _make_ctx(app)

    configure(max_queries=2)
    asyncio.run(search_knowledge([f"q{i}" for i in range(4)], top_k=3, ctx=ctx))

    embedded = model.query_embed.call_args[0][0]
    assert len(embedded) == 2


def test_configure_max_top_k_affects_search():
    embeddings = _make_embeddings(10)
    chunks = [_make_chunk(document_id=f"doc-{i}", chunk_index=i) for i in range(10)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    configure(max_top_k=3)
    results = asyncio.run(search_knowledge("test", top_k=10, ctx=ctx))

    assert len(results) <= 3


# --- _normalize_queries ---


def test_normalize_single_string():
    assert _normalize_queries("hello") == ["hello"]


def test_normalize_list():
    assert _normalize_queries(["hello", "world"]) == ["hello", "world"]


def test_normalize_strips_whitespace():
    assert _normalize_queries(["  hello  ", "world"]) == ["hello", "world"]


def test_normalize_deduplicates():
    assert _normalize_queries(["hello", "hello", "world"]) == ["hello", "world"]


def test_normalize_drops_empty():
    assert _normalize_queries(["", "  ", "hello"]) == ["hello"]


def test_normalize_all_empty():
    result = _normalize_queries(["", "  "])
    assert isinstance(result, str)


def test_normalize_non_string_items():
    result = _normalize_queries([123, "hello"])
    assert isinstance(result, str)
    assert "non-string" in result


def test_normalize_wrong_type():
    result = _normalize_queries(42)
    assert isinstance(result, str)
    assert "must be a string" in result


def test_normalize_empty_list():
    result = _normalize_queries([])
    assert isinstance(result, str)
    assert "non-empty" in result


# --- _rrf_merge ---


def test_rrf_single_list():
    metadata = [
        _make_chunk(document_id="a"),
        _make_chunk(document_id="b"),
        _make_chunk(document_id="c"),
    ]
    ranked_lists = [[(0, 0.9), (1, 0.8), (2, 0.7)]]

    result = _rrf_merge(ranked_lists, metadata, top_k=3)

    assert [idx for idx, _ in result] == [0, 1, 2]


def test_rrf_merge_two_lists():
    metadata = [
        _make_chunk(document_id="a"),
        _make_chunk(document_id="b"),
        _make_chunk(document_id="c"),
    ]
    ranked_lists = [
        [(0, 0.9), (1, 0.8)],
        [(1, 0.85), (2, 0.7)],
    ]

    result = _rrf_merge(ranked_lists, metadata, top_k=3)

    doc_ids = [metadata[idx].document_id for idx, _ in result]
    assert doc_ids[0] == "b"


def test_rrf_deduplicates_by_document_id():
    metadata = [
        _make_chunk(document_id="a", content="short"),
        _make_chunk(document_id="a", content="this is much longer content"),
    ]
    ranked_lists = [
        [(0, 0.9)],
        [(1, 0.8)],
    ]

    result = _rrf_merge(ranked_lists, metadata, top_k=3)

    assert len(result) == 1
    assert result[0][0] == 1


def test_rrf_respects_top_k():
    metadata = [_make_chunk(document_id=f"doc-{i}") for i in range(10)]
    ranked_lists = [[(i, 0.9 - i * 0.1) for i in range(10)]]

    result = _rrf_merge(ranked_lists, metadata, top_k=3)

    assert len(result) == 3


def test_rrf_skips_negative_indices():
    metadata = [_make_chunk(document_id="a")]
    ranked_lists = [[(-1, 0.0), (0, 0.9)]]

    result = _rrf_merge(ranked_lists, metadata, top_k=3)

    assert len(result) == 1
    assert result[0][0] == 0


def test_rrf_empty_ranked_lists():
    metadata = [_make_chunk(document_id="a")]

    result = _rrf_merge([], metadata, top_k=3)

    assert result == []


# --- search_knowledge ---


def test_search_single_query():
    embeddings = _make_embeddings(5)
    chunks = [_make_chunk(document_id=f"doc-{i}", chunk_index=i) for i in range(5)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(search_knowledge("test query", top_k=3, ctx=ctx))

    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_result_fields():
    embeddings = _make_embeddings(3)
    chunks = [
        _make_chunk(
            document_id=f"doc-{i}",
            chunk_index=i,
            section_title=f"Title {i}",
            source_url="https://example.com/docs",
            content=f"Content {i}",
        )
        for i in range(3)
    ]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(search_knowledge("test", top_k=3, ctx=ctx))

    for r in results:
        assert "document_id" in r
        assert "title" in r
        assert "score" in r
        assert "source_url" in r
        assert "content" in r


def test_search_title_fallback_to_file_path():
    embeddings = _make_embeddings(1)
    chunks = [_make_chunk(section_title=None, file_path="docs/guide.md")]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(search_knowledge("test", ctx=ctx))

    assert results[0]["title"] == "docs/guide.md"


def test_search_title_fallback_to_document_id():
    embeddings = _make_embeddings(1)
    chunks = [_make_chunk(section_title=None, file_path=None)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(search_knowledge("test", ctx=ctx))

    assert results[0]["title"] == "test-source:doc.md"


def test_search_multi_query_rrf():
    embeddings = _make_embeddings(5)
    chunks = [_make_chunk(document_id=f"doc-{i}", chunk_index=i) for i in range(5)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(
        search_knowledge(["query one", "query two"], top_k=3, ctx=ctx)
    )

    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_top_k_respected():
    embeddings = _make_embeddings(10)
    chunks = [_make_chunk(document_id=f"doc-{i}", chunk_index=i) for i in range(10)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(search_knowledge("test", top_k=2, ctx=ctx))

    assert len(results) <= 2


def test_search_top_k_clamped_high():
    embeddings = _make_embeddings(3)
    chunks = [_make_chunk(document_id=f"doc-{i}", chunk_index=i) for i in range(3)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(search_knowledge("test", top_k=100, ctx=ctx))

    assert len(results) <= 3


def test_search_top_k_zero_clamped_to_one():
    embeddings = _make_embeddings(3)
    chunks = [_make_chunk(document_id=f"doc-{i}", chunk_index=i) for i in range(3)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(search_knowledge("test", top_k=0, ctx=ctx))

    assert len(results) == 1


def test_search_top_k_negative_clamped_to_one():
    embeddings = _make_embeddings(3)
    chunks = [_make_chunk(document_id=f"doc-{i}", chunk_index=i) for i in range(3)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(search_knowledge("test", top_k=-1, ctx=ctx))

    assert len(results) == 1


def test_search_max_queries_truncation():
    embeddings = _make_embeddings(5)
    chunks = [_make_chunk(document_id=f"doc-{i}", chunk_index=i) for i in range(5)]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    model = _mock_model()
    app = AppContext(index=index, metadata=chunks, model=model)
    ctx = _make_ctx(app)

    queries = [f"query {i}" for i in range(7)]
    asyncio.run(search_knowledge(queries, top_k=3, ctx=ctx))

    embedded = model.query_embed.call_args[0][0]
    assert len(embedded) == 5


def test_search_empty_query():
    embeddings = _make_embeddings(1)
    chunks = [_make_chunk()]
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)

    app = AppContext(index=index, metadata=chunks, model=_mock_model())
    ctx = _make_ctx(app)

    result = asyncio.run(search_knowledge("", ctx=ctx))

    assert isinstance(result, str)


def test_search_empty_index():
    index = faiss.IndexFlatIP(384)

    app = AppContext(index=index, metadata=[], model=_mock_model())
    ctx = _make_ctx(app)

    results = asyncio.run(search_knowledge("test", ctx=ctx))

    assert results == []


# --- server module ---


def test_mcp_has_search_knowledge_tool():
    from kod.server.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool_names = [t.name for t in tools]
    assert "search_knowledge" in tool_names


# --- run_server ---


@patch("kod.server.server._configure_tools")
@patch("kod.server.server.mcp")
def test_run_server_defaults(mock_mcp, mock_configure):
    from kod.server import server
    from kod.server.server import run_server

    run_server()

    assert server._server_data_dir == "data"
    assert server._server_embedding_model == "BAAI/bge-small-en-v1.5"
    mock_configure.assert_called_once_with(rrf_k=60, max_queries=5, max_top_k=20)
    mock_mcp.run.assert_called_once_with(
        transport="streamable-http",
        host="0.0.0.0",  # noqa: S104
        port=8000,
    )


@patch("kod.server.server._configure_tools")
@patch("kod.server.server.mcp")
def test_run_server_custom_args(mock_mcp, mock_configure):
    from kod.server import server
    from kod.server.server import run_server

    run_server(
        data_dir="/custom/path",
        embedding_model="custom/model",
        rrf_k=30,
        max_queries=3,
        max_top_k=10,
    )

    assert server._server_data_dir == "/custom/path"
    assert server._server_embedding_model == "custom/model"
    mock_configure.assert_called_once_with(rrf_k=30, max_queries=3, max_top_k=10)
    mock_mcp.run.assert_called_once()


# --- lifespan ---


@patch("kod.server.server.load_app_context")
def test_lifespan_yields_app(mock_load):
    from kod.server.server import _lifespan

    mock_app = MagicMock()
    mock_load.return_value = mock_app

    async def _run():
        async with _lifespan(MagicMock()) as ctx:
            assert ctx["app"] is mock_app

    asyncio.run(_run())
    mock_load.assert_called_once()


@patch("kod.server.server.load_app_context")
def test_lifespan_reads_module_globals(mock_load):
    from kod.server import server
    from kod.server.server import _lifespan

    server._server_data_dir = "/test/data"
    server._server_embedding_model = "test/model"
    mock_load.return_value = MagicMock()

    async def _run():
        async with _lifespan(MagicMock()):
            pass

    asyncio.run(_run())

    from pathlib import Path

    mock_load.assert_called_once_with(Path("/test/data"), "test/model")
