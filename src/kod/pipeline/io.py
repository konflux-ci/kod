"""Shared I/O helpers for pipeline steps."""

from pathlib import Path

from kod.models import DocumentChunk


def read_chunks(path: Path) -> list[DocumentChunk]:
    """Deserialize DocumentChunks from a JSONL file."""
    chunks = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(DocumentChunk.model_validate_json(line))
    return chunks


def write_chunks(chunks: list[DocumentChunk], path: Path) -> None:
    """Serialize DocumentChunks to a JSONL file, one per line."""
    with path.open("w") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
