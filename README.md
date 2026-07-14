# KOD - Konflux Offline Documentation

<p align="center">
  <img src="images/kod.png" alt="KOD logo" width="200">
</p>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)

KOD is an ETL pipeline and MCP server that indexes Konflux documentation for
RAG (Retrieval-Augmented Generation). It extracts documentation from configured
sources, chunks it, generates embeddings, builds a FAISS vector index, and
serves it as an MCP server with `search_knowledge` and `get_document` tools.

## Quickstart

Install dependencies:

```
uv sync --group dev
```

Run a pipeline step:

```
uv run kod --config config.example.yaml extract
```

Run the full pipeline:

```
uv run kod --config config.example.yaml pipeline
```

## CLI

```
kod [OPTIONS] COMMAND

Options:
  -c, --config PATH  Path to the KOD configuration file  [default: config.yaml]
  -v, --verbose      Enable debug logging
  --version          Show version and exit
  --help             Show this message and exit

Commands:
  extract      Extract documents from configured sources
  transform    Chunk extracted documents for embedding
  embed        Generate embeddings for document chunks
  index        Build FAISS index from embeddings
  build-image  Build the container image with the pre-built index
  serve        Start the MCP server
  pipeline     Run the full ETL pipeline (extract -> transform -> embed -> index)
```

## Configuration

KOD reads document sources from a YAML configuration file. See
[config.example.yaml](config.example.yaml) for a complete example.

Each source has:
- **name** - human-readable identifier
- **url** - git repository (`.git` suffix) or web page URL
- **metadata** - key-value pairs attached to extracted documents (optional)
- **include_paths** / **exclude_paths** - filter files within a git source (mutually exclusive, optional)
- **max_pages** - maximum pages to crawl for web sources (default: 50)
- **use_sitemap** - try `sitemap.xml` before crawling links (default: true)

Global pipeline settings:
- **chunk_size** - maximum characters per chunk (default: 1000)
- **chunk_overlap** - character overlap between consecutive chunks (default: 200)

## Development

Install dev dependencies and pre-commit hooks:

```
make setup
```

Run the full CI suite locally:

```
make ci
```

Individual targets:

```
make lint    # ruff check
make format  # ruff format
make test    # pytest with coverage
make fix     # auto-fix lint issues + format
```

## License

See [LICENSE](LICENSE) for details.
