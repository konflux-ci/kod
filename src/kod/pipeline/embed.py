"""Embed step - generate vector embeddings for document chunks."""

import logging

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

    model = _get_embedding_model(config.embedding_model)

    failures = []
    for source in config.sources:
        input_path = chunked_dir / f"{source.name}.jsonl"
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
            output_path = embedded_dir / f"{source.name}.npy"
            np.save(output_path, embeddings)
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


def _get_embedding_model(model_name: str) -> TextEmbedding:
    """Instantiate a FastEmbed text embedding model."""
    return TextEmbedding(model_name=model_name)


def _embed_chunks(chunks, model) -> np.ndarray:
    """Generate embeddings for a list of chunks using passage_embed."""
    texts = [chunk.content for chunk in chunks]
    # passage_embed() adds the passage prefix for asymmetric retrieval models
    embeddings = list(model.passage_embed(texts))
    return np.array(embeddings, dtype=np.float32)
