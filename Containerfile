# --- Builder stage ---
FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock README.md ./

RUN uv sync --locked --no-dev --no-install-project

# Copy source and build a wheel install (not editable)
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable

# --- Runtime stage ---
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download the FastEmbed model (~67 MB, rarely changes)
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding(os.environ['EMBEDDING_MODEL'])" \
    && chown -R 65532:0 /app/.cache

# Copy pre-built FAISS index and metadata (LAST layer for efficient daily pulls)
ARG DATA_DIR=data
COPY ${DATA_DIR}/index/ /app/data/index/

ENV KOD_DATA_DIR=/app/data

EXPOSE 8000

USER 65532

ENTRYPOINT ["kod", "serve"]
