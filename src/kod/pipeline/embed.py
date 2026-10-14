"""Embed step - generate vector embeddings for document chunks."""

import json
import logging
import os

from pathlib import Path

import numpy as np

from fastembed import TextEmbedding

from kod.config import KodConfig
from kod.pipeline.io import read_chunks


logger = logging.getLogger(__name__)


def run_embed(config: KodConfig) -> None:
    """Generate embeddings for document chunks using FastEmbed."""
    chunked_dir = config.data_dir / "chunked"
    embedded_dir = config.data_dir / "embedded"
    embedded_dir.mkdir(parents=True, exist_ok=True)

    # Cache the model under the data dir so buildah's `COPY data/model-cache/`
    # (and local `kod build-image`) reuses it. Honor FASTEMBED_CACHE_PATH when set
    # so callers can point every load at one stable cache (e.g. benchmark, which
    # otherwise re-downloads per chunk size into a freshly-wiped data dir).
    cache_dir = Path(os.environ.get("FASTEMBED_CACHE_PATH") or config.data_dir / "model-cache")
    model = _get_embedding_model(config.embedding_model, cache_dir)

    failures = []
    for source in config.sources:
        input_path = chunked_dir / f"{source.name}.jsonl"
        output_path = embedded_dir / f"{source.name}.npy"
        meta_path = embedded_dir / f"{source.name}.meta.json"

        # Invalidate any previous output for this source up front, before reading
        # or embedding. Otherwise a failed or skipped re-embed would leave stale
        # vectors (and a sidecar naming the old model) that run_index could pair
        # with freshly-regenerated chunks when their counts happen to match. With
        # the files gone, a failure instead drops the source (run_index skips a
        # source with no .npy) rather than indexing stale vectors.
        output_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

        if not input_path.exists():
            logger.warning("[embed] No chunked data for '%s', skipping", source.name)
            continue

        logger.info("[embed] Embedding chunks from '%s'", source.name)
        try:
            chunks = read_chunks(input_path)
            if not chunks:
                logger.warning("[embed] No chunks in '%s', skipping", source.name)
                continue
            embeddings = _embed_chunks(chunks, model)
            np.save(output_path, embeddings)
            # Record which model produced these vectors so run_index can verify
            # provenance. Written after the .npy so an interrupted write leaves a
            # missing sidecar (rejected by run_index), never a stale one.
            meta_path.write_text(
                json.dumps(
                    {"embedding_model": config.embedding_model, "dim": int(embeddings.shape[1])}
                )
            )
            logger.info(
                "[embed] Wrote %d embedding(s) (%d dims) to %s",
                embeddings.shape[0],
                embeddings.shape[1],
                output_path,
            )
        except Exception:
            logger.exception("[embed] Failed to embed '%s', skipping", source.name)
            failures.append(source.name)

    if failures:
        names = ", ".join(failures)
        logger.error("[embed] Embedding finished with %d failure(s): %s", len(failures), names)
    else:
        logger.info("[embed] Embedding complete")


def _get_embedding_model(model_name: str, cache_dir: Path) -> TextEmbedding:
    """Instantiate a FastEmbed text embedding model.

    The model is cached under ``cache_dir`` (inside the data dir) so the same
    cache the ETL downloads is reused by ``kod build-image`` / buildah's
    ``COPY`` instead of re-downloading the model during the container build.

    When ``KOD_EMBED_THREADS`` is set, cap the ONNX Runtime intra-op thread
    pool. ONNX Runtime otherwise sizes it to the host's physical core count
    (ignoring cgroup CPU limits), and each thread's arena can balloon memory
    into an OOM on many-core build nodes.
    """
    threads = os.environ.get("KOD_EMBED_THREADS")
    if threads:
        return TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), threads=int(threads))
    return TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))


def _embed_chunks(chunks, model) -> np.ndarray:
    """Generate embeddings for a list of chunks using passage_embed."""
    texts = [chunk.content for chunk in chunks]
    kwargs = {}
    # FastEmbed's default batch_size (256) drives O(seq^2) attention memory
    # (batch x heads x seq x seq) into an OOM under tight memory limits; allow
    # capping it. Unset -> FastEmbed default, so local behavior is unchanged.
    batch_size = os.environ.get("KOD_EMBED_BATCH_SIZE")
    if batch_size:
        kwargs["batch_size"] = int(batch_size)
    # passage_embed() adds the passage prefix for asymmetric retrieval models
    embeddings = list(model.passage_embed(texts, **kwargs))
    return np.array(embeddings, dtype=np.float32)
