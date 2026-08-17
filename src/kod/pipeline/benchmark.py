"""Benchmark step - evaluate retrieval quality across chunk sizes."""

import logging
import shutil

from dataclasses import dataclass
from pathlib import Path

import yaml

from pydantic import BaseModel
from pydantic import Field

from kod.config import KodConfig
from kod.pipeline.embed import run_embed
from kod.pipeline.index import run_index
from kod.pipeline.transform import run_transform
from kod.server.app import AppContext
from kod.server.app import embed_queries
from kod.server.app import load_app_context
from kod.server.app import search_index


logger = logging.getLogger(__name__)


class BenchmarkQuery(BaseModel):
    """A single benchmark query with expected source matches."""

    query: str
    expected_sources: list[str] = Field(min_length=1)


class BenchmarkQueries(BaseModel):
    """Collection of benchmark queries loaded from a YAML file."""

    queries: list[BenchmarkQuery] = Field(min_length=1)


@dataclass
class ChunkSizeResult:
    """Benchmark results for a single chunk_size value."""

    chunk_size: int
    chunk_count: int
    recall_at_k: dict[int, float]


def load_queries(path: Path) -> list[BenchmarkQuery]:
    """Load benchmark queries from a YAML file."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    return BenchmarkQueries.model_validate(raw).queries


def compute_recall(results: list[dict], expected_sources: list[str]) -> float:
    """Compute recall: fraction of expected sources found in results."""
    found = {r["source_name"] for r in results}
    hits = sum(1 for s in expected_sources if s in found)
    return hits / len(expected_sources)


def _search(app: AppContext, query: str, top_k: int) -> list[dict]:
    """Run a single vector search and return results with source_name."""
    query_vec = embed_queries(app.model, [query])
    hits = search_index(app, query_vec, top_k)
    return [
        {
            "source_name": app.metadata[idx].source_name,
            "document_id": app.metadata[idx].document_id,
            "score": score,
        }
        for idx, score in hits
    ]


def _run_pipeline_for_chunk_size(config: KodConfig, chunk_size: int) -> Path:
    """Run transform -> embed -> index with a specific chunk_size into a temp dir."""
    bench_dir = config.data_dir / "benchmark" / f"cs_{chunk_size}"
    if bench_dir.exists():
        shutil.rmtree(bench_dir)

    bench_config = KodConfig.model_validate(
        config.model_dump() | {"chunk_size": chunk_size, "data_dir": bench_dir}
    )

    extracted_dir = config.data_dir / "extracted"
    bench_extracted = bench_dir / "extracted"
    bench_extracted.mkdir(parents=True, exist_ok=True)
    for f in extracted_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, bench_extracted / f.name)

    run_transform(bench_config)
    run_embed(bench_config)
    run_index(bench_config)

    return bench_dir


def evaluate_chunk_size(
    config: KodConfig,
    chunk_size: int,
    queries: list[BenchmarkQuery],
    top_k_values: list[int],
) -> ChunkSizeResult:
    """Run the pipeline for a chunk_size and evaluate recall on the query set."""
    logger.info("[benchmark] Evaluating chunk_size=%d", chunk_size)

    bench_dir = _run_pipeline_for_chunk_size(config, chunk_size)
    app = load_app_context(bench_dir, config.embedding_model)

    max_k = max(top_k_values)
    recall_sums: dict[int, float] = dict.fromkeys(top_k_values, 0.0)

    for bq in queries:
        results = _search(app, bq.query, max_k)
        for k in top_k_values:
            recall_sums[k] += compute_recall(results[:k], bq.expected_sources)

    n = len(queries)
    recall_at_k = {k: recall_sums[k] / n for k in top_k_values}

    logger.info(
        "[benchmark] chunk_size=%d chunks=%d recall=%s",
        chunk_size,
        app.index.ntotal,
        {k: f"{v:.2%}" for k, v in recall_at_k.items()},
    )

    return ChunkSizeResult(
        chunk_size=chunk_size,
        chunk_count=app.index.ntotal,
        recall_at_k=recall_at_k,
    )


def format_results_table(results: list[ChunkSizeResult]) -> str:
    """Format benchmark results as a text table."""
    if not results:
        return "No results."

    all_k = sorted(results[0].recall_at_k.keys())
    recall_headers = [f"recall@{k}" for k in all_k]
    headers = ["chunk_size", "chunks"] + recall_headers

    rows = []
    for r in results:
        row = [str(r.chunk_size), str(r.chunk_count)]
        row.extend(f"{r.recall_at_k[k]:.2%}" for k in all_k)
        rows.append(row)

    col_widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    header_line = "  ".join(h.rjust(w) for h, w in zip(headers, col_widths, strict=True))
    separator = "  ".join("-" * w for w in col_widths)
    data_lines = [
        "  ".join(cell.rjust(w) for cell, w in zip(row, col_widths, strict=True)) for row in rows
    ]

    return "\n".join([header_line, separator, *data_lines])


def run_benchmark(
    config: KodConfig,
    chunk_sizes: list[int],
    queries_path: Path,
    top_k_values: list[int],
) -> list[ChunkSizeResult]:
    """Run the full benchmark across chunk sizes and print results."""
    extracted_dir = config.data_dir / "extracted"
    if not extracted_dir.exists() or not any(extracted_dir.iterdir()):
        msg = f"No extracted data found in {extracted_dir}. Run 'kod extract' first."
        raise FileNotFoundError(msg)

    queries = load_queries(queries_path)
    logger.info(
        "[benchmark] %d queries, %d chunk sizes: %s",
        len(queries),
        len(chunk_sizes),
        chunk_sizes,
    )

    results = []
    for cs in chunk_sizes:
        result = evaluate_chunk_size(config, cs, queries, top_k_values)
        results.append(result)

    table = format_results_table(results)
    logger.info("\n%s\n", table)

    return results
