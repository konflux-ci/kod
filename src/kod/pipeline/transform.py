"""Transform step - chunk extracted documents for embedding."""

import logging

from pathlib import Path

from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_dicts

from kod.config import KodConfig
from kod.models import Document
from kod.models import DocumentChunk


logger = logging.getLogger(__name__)


def run_transform(config: KodConfig) -> None:
    """Chunk extracted documents from all sources."""
    extracted_dir = config.data_dir / "extracted"
    chunked_dir = config.data_dir / "chunked"
    chunked_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for source in config.sources:
        input_path = extracted_dir / f"{source.name}.jsonl"
        if not input_path.exists():
            logger.warning("[transform] No extracted data for '%s', skipping", source.name)
            continue

        logger.info("[transform] Chunking documents from '%s'", source.name)
        try:
            documents = _read_documents(input_path)
            chunks = []
            for doc in documents:
                chunks.extend(_chunk_document(doc, config.chunk_size, config.chunk_overlap))
            output_path = chunked_dir / f"{source.name}.jsonl"
            _write_chunks(chunks, output_path)
            logger.info(
                "[transform] Wrote %d chunk(s) from %d document(s) to %s",
                len(chunks),
                len(documents),
                output_path,
            )
        except Exception:
            logger.exception("[transform] Failed to chunk '%s', skipping", source.name)
            failures.append(source.name)

    if failures:
        names = ", ".join(failures)
        logger.error("[transform] Transform finished with %d failure(s): %s", len(failures), names)
    else:
        logger.info("[transform] Transform complete")


def _read_documents(path: Path) -> list[Document]:
    """Deserialize Documents from a JSONL file."""
    documents = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(Document.model_validate_json(line))
    return documents


def _chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    """Split a document into chunks using Unstructured's title-based chunking."""
    elements = elements_from_dicts(doc.elements)
    if not elements:
        return []

    # combine_text_under_n_chars=0 disables merging of short consecutive
    # sections. The default (= max_characters) silently combines small
    # sections across heading boundaries, mixing content from different
    # topics into a single chunk.
    chunks = chunk_by_title(
        elements,
        max_characters=chunk_size,
        overlap=chunk_overlap,
        combine_text_under_n_chars=0,
    )

    document_id = f"{doc.source_name}:{doc.file_path or ''}"
    # Propagate the most recent section title to continuation chunks so
    # every chunk carries its section context, not just the ones that
    # start with a heading.
    current_section = None
    result = []
    for i, chunk in enumerate(chunks):
        title = _get_section_title(chunk)
        if title:
            current_section = title
        result.append(
            DocumentChunk(
                document_id=document_id,
                content=chunk.text,
                chunk_index=i,
                source_name=doc.source_name,
                source_url=doc.source_url,
                file_path=doc.file_path,
                section_title=current_section,
                metadata=dict(doc.metadata),
            )
        )
    return result


def _get_section_title(chunk) -> str | None:
    """Extract the section title from the first original element if it is a Title."""
    orig = getattr(chunk.metadata, "orig_elements", None)
    if orig and orig[0].category == "Title":
        return orig[0].text
    return None


def _write_chunks(chunks: list[DocumentChunk], path: Path) -> None:
    """Serialize chunks to a JSONL file."""
    with path.open("w") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
