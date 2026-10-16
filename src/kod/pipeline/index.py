"""Index step - build FAISS vector index from embeddings."""

import json
import logging

import faiss
import numpy as np

from kod.config import KodConfig
from kod.pipeline.io import read_chunks
from kod.pipeline.io import write_chunks


logger = logging.getLogger(__name__)


def _read_embedding_provenance(path) -> str | None:
    """Return the embedding model recorded alongside a source's `.npy`.

    Returns ``None`` when the sidecar is absent, unreadable, or malformed so the
    caller can treat missing provenance as a verification failure.
    """
    try:
        meta = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    model = meta.get("embedding_model")
    return model if isinstance(model, str) else None


def run_index(config: KodConfig) -> None:
    """Build a unified FAISS index from all source embeddings."""
    embedded_dir = config.data_dir / "embedded"
    chunked_dir = config.data_dir / "chunked"
    index_dir = config.data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    all_chunks = []

    failures = []
    provenance_errors = []
    for source in config.sources:
        emb_path = embedded_dir / f"{source.name}.npy"
        chunk_path = chunked_dir / f"{source.name}.jsonl"
        meta_path = embedded_dir / f"{source.name}.meta.json"

        if not emb_path.exists():
            logger.warning("[index] No embeddings for '%s', skipping", source.name)
            continue
        if not chunk_path.exists():
            logger.warning("[index] No chunk metadata for '%s', skipping", source.name)
            continue

        # Verify the vectors were produced by the configured model. np.vstack
        # happily stacks same-width vectors from different models, so without
        # this a same-dimension model swap (or a stale .npy from a failed
        # re-embed) would silently build a mixed-space index. FastEmbed resolves
        # model names case-insensitively, so compare that way.
        recorded_model = _read_embedding_provenance(meta_path)
        if recorded_model is None:
            logger.error("[index] Missing or unreadable embedding provenance for '%s'", source.name)
            provenance_errors.append(source.name)
            continue
        if recorded_model.lower() != config.embedding_model.lower():
            logger.error(
                "[index] '%s' was embedded with '%s' but the configured model is '%s'",
                source.name,
                recorded_model,
                config.embedding_model,
            )
            provenance_errors.append(source.name)
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

    # Refuse to build a mixed-model index: any provenance failure aborts before
    # writing anything, so a bad source never ships in a daily rebuild.
    if provenance_errors:
        names = ", ".join(provenance_errors)
        msg = (
            f"Embedding-model provenance check failed for: {names}. The FAISS index "
            f"must contain only vectors from the configured model "
            f"('{config.embedding_model}'); re-run `kod embed` so every source is "
            f"embedded with that model."
        )
        raise ValueError(msg)

    if not all_embeddings:
        logger.warning("[index] No embeddings found, nothing to index")
        return

    combined = np.vstack(all_embeddings)
    index = _build_index(combined)

    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.jsonl"
    index_meta_path = index_dir / "index_meta.json"

    # Remove any stale provenance record before publishing the new index, so an
    # interrupted rebuild cannot leave a new index.faiss described by an *old*
    # model record (which serve now trusts when no model is given, and which the
    # dimension check cannot catch for a same-width model). The interrupted window
    # then falls back to the "no record -> default" back-compat path rather than
    # silently loading a wrong model.
    index_meta_path.unlink(missing_ok=True)
    faiss.write_index(index, str(index_path))
    write_chunks(all_chunks, metadata_path)
    # Publish the record atomically so a crash mid-write can't leave a truncated
    # file (which load_app_context treats as a hard error).
    tmp_meta_path = index_meta_path.with_name(index_meta_path.name + ".tmp")
    tmp_meta_path.write_text(json.dumps({"embedding_model": config.embedding_model}))
    tmp_meta_path.replace(index_meta_path)

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
