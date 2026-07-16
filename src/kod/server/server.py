"""FastMCP server instance for KOD."""

import logging

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP

from kod.server.app import load_app_context
from kod.server.tools import configure as _configure_tools
from kod.server.tools import search_knowledge


logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "data"
_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_server_data_dir: str = _DEFAULT_DATA_DIR
_server_embedding_model: str = _DEFAULT_EMBEDDING_MODEL


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Load FAISS index, metadata, and embedding model at startup."""
    del server
    app = load_app_context(Path(_server_data_dir), _server_embedding_model)
    yield {"app": app}


mcp = FastMCP(
    "KOD - Konflux Offline Documentation",
    instructions="Search Konflux documentation using the search_knowledge tool.",
    lifespan=_lifespan,
)

mcp.tool()(search_knowledge)


def run_server(
    data_dir: str = _DEFAULT_DATA_DIR,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    rrf_k: int = 60,
    max_queries: int = 5,
    max_top_k: int = 20,
) -> None:
    """Start the MCP server on port 8000 with streamable-http transport."""
    global _server_data_dir, _server_embedding_model  # noqa: W0603
    _server_data_dir = data_dir
    _server_embedding_model = embedding_model
    _configure_tools(rrf_k=rrf_k, max_queries=max_queries, max_top_k=max_top_k)
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",  # noqa: S104
        port=8000,
    )
