"""Tests for KOD benchmark module."""

import textwrap

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from pydantic import ValidationError

from kod.config import DocumentSource
from kod.config import KodConfig
from kod.models import DocumentChunk
from kod.pipeline.benchmark import _run_pipeline_for_chunk_size
from kod.pipeline.benchmark import _search
from kod.pipeline.benchmark import BenchmarkQuery
from kod.pipeline.benchmark import ChunkSizeResult
from kod.pipeline.benchmark import compute_recall
from kod.pipeline.benchmark import evaluate_chunk_size
from kod.pipeline.benchmark import format_results_table
from kod.pipeline.benchmark import load_queries
from kod.pipeline.benchmark import run_benchmark
from kod.server.app import AppContext


class TestComputeRecall:
    def test_all_found(self):
        results = [
            {"source_name": "docs"},
            {"source_name": "arch"},
        ]
        assert compute_recall(results, ["docs", "arch"]) == 1.0

    def test_none_found(self):
        results = [{"source_name": "other"}]
        assert compute_recall(results, ["docs", "arch"]) == 0.0

    def test_partial(self):
        results = [
            {"source_name": "docs"},
            {"source_name": "other"},
        ]
        assert compute_recall(results, ["docs", "arch"]) == 0.5

    def test_empty_results(self):
        assert compute_recall([], ["docs"]) == 0.0


class TestLoadQueries:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "queries.yaml"
        f.write_text(
            textwrap.dedent("""\
            queries:
              - query: "test query"
                expected_sources:
                  - docs
              - query: "another query"
                expected_sources:
                  - arch
                  - docs
        """)
        )
        queries = load_queries(f)
        assert len(queries) == 2
        assert queries[0].query == "test query"
        assert queries[0].expected_sources == ["docs"]
        assert queries[1].expected_sources == ["arch", "docs"]

    def test_empty_queries_fails(self, tmp_path):
        f = tmp_path / "queries.yaml"
        f.write_text("queries: []\n")
        with pytest.raises(ValidationError):
            load_queries(f)

    def test_missing_expected_sources_fails(self, tmp_path):
        f = tmp_path / "queries.yaml"
        f.write_text(
            textwrap.dedent("""\
            queries:
              - query: "test"
                expected_sources: []
        """)
        )
        with pytest.raises(ValidationError):
            load_queries(f)

    def test_malformed_yaml_fails(self, tmp_path):
        f = tmp_path / "queries.yaml"
        f.write_text("queries: [{")
        with pytest.raises(yaml.YAMLError):
            load_queries(f)


class TestFormatResultsTable:
    def test_single_result(self):
        results = [
            ChunkSizeResult(chunk_size=1000, chunk_count=50, recall_at_k={5: 0.8}),
        ]
        table = format_results_table(results)
        assert "chunk_size" in table
        assert "chunks" in table
        assert "recall@5" in table
        assert "1000" in table
        assert "50" in table
        assert "80.00%" in table

    def test_multiple_results(self):
        results = [
            ChunkSizeResult(chunk_size=500, chunk_count=100, recall_at_k={5: 0.6, 10: 0.8}),
            ChunkSizeResult(chunk_size=1000, chunk_count=50, recall_at_k={5: 0.8, 10: 0.9}),
        ]
        table = format_results_table(results)
        assert "recall@5" in table
        assert "recall@10" in table
        assert "500" in table
        assert "1000" in table

    def test_empty_results(self):
        assert format_results_table([]) == "No results."


def _make_chunk(source_name="docs", document_id="docs:file.md", content="text"):
    return DocumentChunk(
        document_id=document_id,
        content=content,
        chunk_index=0,
        source_name=source_name,
        source_url="https://example.com",
    )


class TestSearch:
    def test_returns_results(self):
        import faiss

        dim = 4
        index = faiss.IndexFlatIP(dim)
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        index.add(vecs)
        metadata = [
            _make_chunk("docs", "docs:a.md", "first"),
            _make_chunk("arch", "arch:b.md", "second"),
        ]
        model = MagicMock()
        model.query_embed.return_value = [np.array([1, 0, 0, 0], dtype=np.float32)]
        app = AppContext(index=index, metadata=metadata, model=model)

        results = _search(app, "test", top_k=2)
        assert len(results) == 2
        assert results[0]["source_name"] == "docs"

    def test_empty_index(self):
        import faiss

        index = faiss.IndexFlatIP(4)
        app = AppContext(index=index, metadata=[], model=MagicMock())
        assert _search(app, "test", top_k=5) == []

    def test_skips_negative_indices(self):
        import faiss

        dim = 4
        index = faiss.IndexFlatIP(dim)
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        index.add(vecs)
        metadata = [
            _make_chunk("docs", "docs:a.md", "first"),
            _make_chunk("arch", "arch:b.md", "second"),
        ]
        model = MagicMock()
        model.query_embed.return_value = [np.array([1, 0, 0, 0], dtype=np.float32)]

        with patch.object(index, "search") as mock_search:
            mock_search.return_value = (
                np.array([[0.9, -1.0]], dtype=np.float32),
                np.array([[0, -1]], dtype=np.int64),
            )
            app = AppContext(index=index, metadata=metadata, model=model)
            results = _search(app, "test", top_k=2)

        assert len(results) == 1
        assert results[0]["source_name"] == "docs"

    def test_respects_top_k(self):
        import faiss

        dim = 4
        index = faiss.IndexFlatIP(dim)
        vecs = np.random.rand(5, dim).astype(np.float32)
        index.add(vecs)
        metadata = [_make_chunk(f"s{i}") for i in range(5)]
        model = MagicMock()
        model.query_embed.return_value = [np.random.rand(dim).astype(np.float32)]
        app = AppContext(index=index, metadata=metadata, model=model)

        results = _search(app, "test", top_k=2)
        assert len(results) == 2


class TestRunPipelineForChunkSize:
    @patch("kod.pipeline.benchmark.run_index")
    @patch("kod.pipeline.benchmark.run_embed")
    @patch("kod.pipeline.benchmark.run_transform")
    def test_copies_extracted_and_runs_pipeline(
        self, mock_transform, mock_embed, mock_index, tmp_path
    ):
        config = KodConfig(
            sources=[DocumentSource(name="test", url="https://example.com/docs.git")],
            data_dir=tmp_path / "data",
        )
        extracted = config.data_dir / "extracted"
        extracted.mkdir(parents=True)
        (extracted / "test.jsonl").write_text('{"test": true}\n')
        (extracted / "subdir").mkdir()

        bench_dir = _run_pipeline_for_chunk_size(config, 1000)

        assert (bench_dir / "extracted" / "test.jsonl").exists()
        assert not (bench_dir / "extracted" / "subdir").exists()
        mock_transform.assert_called_once()
        mock_embed.assert_called_once()
        mock_index.assert_called_once()
        called_config = mock_transform.call_args[0][0]
        assert called_config.chunk_size == 1000
        assert called_config.data_dir == bench_dir

    @patch("kod.pipeline.benchmark.run_index")
    @patch("kod.pipeline.benchmark.run_embed")
    @patch("kod.pipeline.benchmark.run_transform")
    def test_cleans_existing_bench_dir(self, mock_transform, mock_embed, mock_index, tmp_path):
        config = KodConfig(
            sources=[DocumentSource(name="test", url="https://example.com/docs.git")],
            data_dir=tmp_path / "data",
        )
        extracted = config.data_dir / "extracted"
        extracted.mkdir(parents=True)
        (extracted / "test.jsonl").write_text("{}\n")

        bench_dir = config.data_dir / "benchmark" / "cs_1000"
        bench_dir.mkdir(parents=True)
        (bench_dir / "stale.txt").write_text("old")

        _run_pipeline_for_chunk_size(config, 1000)
        assert not (bench_dir / "stale.txt").exists()


class TestEvaluateChunkSize:
    @patch("kod.pipeline.benchmark.load_app_context")
    @patch("kod.pipeline.benchmark._run_pipeline_for_chunk_size")
    def test_evaluates_queries(self, mock_run, mock_load, tmp_path):
        import faiss

        mock_run.return_value = tmp_path / "bench"

        dim = 4
        index = faiss.IndexFlatIP(dim)
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        index.add(vecs)
        metadata = [
            _make_chunk("docs", "docs:a.md"),
            _make_chunk("arch", "arch:b.md"),
        ]
        model = MagicMock()
        model.query_embed.return_value = [np.array([1, 0, 0, 0], dtype=np.float32)]
        mock_load.return_value = AppContext(index=index, metadata=metadata, model=model)

        config = KodConfig(
            sources=[DocumentSource(name="docs", url="https://example.com/docs.git")],
            data_dir=tmp_path / "data",
        )
        queries = [BenchmarkQuery(query="test", expected_sources=["docs"])]

        result = evaluate_chunk_size(config, 1000, queries, [1, 2])

        assert result.chunk_size == 1000
        assert result.chunk_count == 2
        assert result.recall_at_k[1] == 1.0
        assert result.recall_at_k[2] == 1.0

    @patch("kod.pipeline.benchmark.load_app_context")
    @patch("kod.pipeline.benchmark._run_pipeline_for_chunk_size")
    def test_averages_recall_across_queries(self, mock_run, mock_load, tmp_path):
        import faiss

        mock_run.return_value = tmp_path / "bench"

        dim = 4
        index = faiss.IndexFlatIP(dim)
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        index.add(vecs)
        metadata = [
            _make_chunk("docs", "docs:a.md"),
            _make_chunk("arch", "arch:b.md"),
        ]
        model = MagicMock()
        model.query_embed.return_value = [np.array([1, 0, 0, 0], dtype=np.float32)]
        mock_load.return_value = AppContext(index=index, metadata=metadata, model=model)

        config = KodConfig(
            sources=[DocumentSource(name="docs", url="https://example.com/docs.git")],
            data_dir=tmp_path / "data",
        )
        queries = [
            BenchmarkQuery(query="match docs", expected_sources=["docs"]),
            BenchmarkQuery(query="miss other", expected_sources=["other"]),
        ]

        result = evaluate_chunk_size(config, 1000, queries, [2])

        assert result.recall_at_k[2] == 0.5


class TestRunPipelineValidation:
    @patch("kod.pipeline.benchmark.run_index")
    @patch("kod.pipeline.benchmark.run_embed")
    @patch("kod.pipeline.benchmark.run_transform")
    def test_rejects_chunk_size_below_overlap(
        self, mock_transform, mock_embed, mock_index, tmp_path
    ):
        config = KodConfig(
            sources=[DocumentSource(name="test", url="https://example.com/docs.git")],
            data_dir=tmp_path / "data",
            chunk_overlap=200,
        )
        extracted = config.data_dir / "extracted"
        extracted.mkdir(parents=True)
        (extracted / "test.jsonl").write_text("{}\n")

        with pytest.raises(Exception, match="chunk_overlap"):
            _run_pipeline_for_chunk_size(config, 100)


class TestRunBenchmark:
    def test_missing_extracted_data(self, tmp_path):
        config = KodConfig(
            sources=[DocumentSource(name="test", url="https://example.com/docs.git")],
            data_dir=tmp_path / "data",
        )
        queries_file = tmp_path / "queries.yaml"
        queries_file.write_text(
            textwrap.dedent("""\
            queries:
              - query: "test"
                expected_sources:
                  - test
        """)
        )
        with pytest.raises(FileNotFoundError, match="No extracted data"):
            run_benchmark(config, [1000], queries_file, [5])

    def test_empty_extracted_dir(self, tmp_path):
        config = KodConfig(
            sources=[DocumentSource(name="test", url="https://example.com/docs.git")],
            data_dir=tmp_path / "data",
        )
        (tmp_path / "data" / "extracted").mkdir(parents=True)
        queries_file = tmp_path / "queries.yaml"
        queries_file.write_text(
            textwrap.dedent("""\
            queries:
              - query: "test"
                expected_sources:
                  - test
        """)
        )
        with pytest.raises(FileNotFoundError, match="No extracted data"):
            run_benchmark(config, [1000], queries_file, [5])

    @patch("kod.pipeline.benchmark.evaluate_chunk_size")
    def test_runs_for_each_chunk_size(self, mock_eval, tmp_path):
        config = KodConfig(
            sources=[DocumentSource(name="test", url="https://example.com/docs.git")],
            data_dir=tmp_path / "data",
        )
        extracted = tmp_path / "data" / "extracted"
        extracted.mkdir(parents=True)
        (extracted / "test.jsonl").write_text("{}\n")

        queries_file = tmp_path / "queries.yaml"
        queries_file.write_text(
            textwrap.dedent("""\
            queries:
              - query: "test"
                expected_sources:
                  - test
        """)
        )

        mock_eval.return_value = ChunkSizeResult(
            chunk_size=1000, chunk_count=10, recall_at_k={5: 0.8}
        )

        results = run_benchmark(config, [500, 1000, 1500], queries_file, [5])
        assert len(results) == 3
        assert mock_eval.call_count == 3
