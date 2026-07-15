"""Index step - build FAISS vector index from embeddings."""

import logging

import faiss
import numpy as np

from kod.config import KodConfig
from kod.pipeline.io import read_chunks
from kod.pipeline.io import write_chunks


logger = logging.getLogger(__name__)


def run_index(config: KodConfig) -> None:
    """Build a unified FAISS index from all source embeddings."""
    embedded_dir = config.data_dir / "embedded"
    chunked_dir = config.data_dir / "chunked"
    index_dir = config.data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    all_chunks = []

    failures = []
    for source in config.sources:
        emb_path = embedded_dir / f"{source.name}.npy"
        chunk_path = chunked_dir / f"{source.name}.jsonl"

        if not emb_path.exists():
            logger.warning("[index] No embeddings for '%s', skipping", source.name)
            continue
        if not chunk_path.exists():
            logger.warning("[index] No chunk metadata for '%s', skipping", source.name)
            continue

        try:
            embeddings = np.load(emb_path)
            chunks = read_chunks(chunk_path)

            if len(embeddings) != len(chunks):
                logger.error(
                    "[index] Embedding/chunk count mismatch for '%s' (%d vs %d), skipping",
                    source.name,
                    len(embeddings),
                    len(chunks),
                )
                continue

            logger.info("[index] Loading %d vectors from '%s'", len(embeddings), source.name)
            all_embeddings.append(embeddings)
            all_chunks.extend(chunks)
        except Exception:
            logger.exception("[index] Failed to load '%s', skipping", source.name)
            failures.append(source.name)

    if not all_embeddings:
        logger.warning("[index] No embeddings found, nothing to index")
        return

    combined = np.vstack(all_embeddings)
    index = _build_index(combined)

    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.jsonl"
    faiss.write_index(index, str(index_path))
    write_chunks(all_chunks, metadata_path)

    logger.info(
        "[index] Built index with %d vectors (%d dims) at %s",
        index.ntotal,
        index.d,
        index_path,
    )

    if failures:
        names = ", ".join(failures)
        logger.error("[index] Indexing finished with %d failure(s): %s", len(failures), names)
    else:
        logger.info("[index] Indexing complete")


def _build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from an embedding matrix."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index
