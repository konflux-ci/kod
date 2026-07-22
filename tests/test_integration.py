"""Integration tests for the KOD pipeline and MCP server.

These tests run the full ETL pipeline on fixture docs with real FastEmbed
embeddings and query the MCP tools against the resulting FAISS index.
"""

import os
import shutil
import subprocess

from pathlib import Path

import pytest

from conftest import make_ctx

from kod.config import DocumentSource
from kod.config import KodConfig
from kod.pipeline import run_pipeline
from kod.server.app import load_app_context
from kod.server.tools import get_document
from kod.server.tools import search_knowledge


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "docs"


def _create_fixture_repo(tmp_path):
    """Create a local git repo containing the fixture docs."""
    repo_dir = tmp_path / "test-docs.git"
    repo_dir.mkdir()
    shutil.copytree(FIXTURES_DIR, repo_dir / "docs")
    subprocess.run(  # noqa: S603
        ["git", "init", str(repo_dir)],  # noqa: S607
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_dir), "add", "."],  # noqa: S607
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_dir), "commit", "-m", "init"],  # noqa: S607
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_COMMITTER_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(tmp_path),
        },
    )
    return str(repo_dir)


@pytest.fixture(scope="session")
def integration_env(tmp_path_factory):
    """Run the full pipeline once and return (app_context, data_dir)."""
    tmp_path = tmp_path_factory.mktemp("integration")
    repo_url = _create_fixture_repo(tmp_path)

    config = KodConfig(
        sources=[
            DocumentSource(name="test-docs", url=repo_url),
        ],
        data_dir=tmp_path / "data",
    )

    run_pipeline(config)

    index_path = config.data_dir / "index" / "index.faiss"
    metadata_path = config.data_dir / "index" / "metadata.jsonl"
    assert index_path.exists(), "Pipeline failed: index.faiss not created"
    assert metadata_path.exists(), "Pipeline failed: metadata.jsonl not created"
    assert index_path.stat().st_size > 0, "Pipeline failed: index.faiss is empty"

    app = load_app_context(config.data_dir, config.embedding_model)
    return app, config.data_dir


@pytest.mark.integration
def test_pipeline_builds_index(integration_env):
    app, data_dir = integration_env

    assert (data_dir / "index" / "index.faiss").exists()
    assert (data_dir / "index" / "metadata.jsonl").exists()
    assert app.index.ntotal > 0
    assert len(app.metadata) == app.index.ntotal


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_returns_relevant_results(integration_env):
    app, _ = integration_env
    ctx = make_ctx(app)

    results = await search_knowledge("troubleshooting build failures", top_k=5, ctx=ctx)

    assert isinstance(results, list)
    assert len(results) > 0
    assert "troubleshooting" in results[0]["document_id"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_multiple_queries_rrf(integration_env):
    app, _ = integration_env
    ctx = make_ctx(app)

    results = await search_knowledge(
        ["enterprise contract policy", "release pipeline validation"],
        top_k=5,
        ctx=ctx,
    )

    assert isinstance(results, list)
    assert len(results) > 0
    for result in results:
        assert "document_id" in result
        assert "content" in result
        assert "score" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_document_returns_content(integration_env):
    app, _ = integration_env
    ctx = make_ctx(app)

    search_results = await search_knowledge("workspace", top_k=1, ctx=ctx)
    assert len(search_results) > 0
    document_id = search_results[0]["document_id"]

    content = await get_document(document_id, ctx=ctx)

    assert isinstance(content, str)
    assert document_id in content
    assert len(content) > 100


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_document_with_query_ranks_sections(integration_env):
    app, _ = integration_env
    ctx = make_ctx(app)

    content = await get_document(
        "test-docs:docs/pipelines.md",
        query="customizing pipeline",
        ctx=ctx,
    )

    assert isinstance(content, str)
    assert "ranked by relevance" in content
