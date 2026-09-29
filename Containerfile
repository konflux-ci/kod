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

# Reuse the FastEmbed model cache (~67 MB) built by the ETL pipeline instead of
# re-downloading it here. This avoids a second download per build and works on
# build networks that terminate TLS with an internal CA (the ETL injects the
# cluster CA when downloading; this build stage has no such trust). Run fully
# offline at runtime so a missing cache fails fast rather than hitting the network.
ARG DATA_DIR=data
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HUB_OFFLINE=1
COPY --chown=65532:0 ${DATA_DIR}/model-cache/ /app/.cache/fastembed/

# Copy pre-built FAISS index and metadata (LAST layer for efficient daily pulls)
COPY ${DATA_DIR}/index/ /app/data/index/

ENV KOD_DATA_DIR=/app/data

EXPOSE 8000

USER 65532

ENTRYPOINT ["kod", "serve"]
