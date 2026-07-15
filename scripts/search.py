"""Search the FAISS index for debugging.

Usage:
    uv run python scripts/search.py "your query here"
    uv run python scripts/search.py -k 5 "your query here"
    uv run python scripts/search.py --data-dir /path/to/data "your query here"
"""

import argparse
import json

import faiss
import numpy as np
from fastembed import TextEmbedding


def main():
    parser = argparse.ArgumentParser(description="Search the KOD FAISS index")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", type=int, default=3, help="Number of results (default: 3)")
    parser.add_argument("--data-dir", default="data", help="Path to data directory (default: data)")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="FastEmbed model name")
    args = parser.parse_args()

    index = faiss.read_index(f"{args.data_dir}/index/index.faiss", faiss.IO_FLAG_MMAP)
    with open(f"{args.data_dir}/index/metadata.jsonl") as f:
        metadata = [json.loads(line) for line in f]

    model = TextEmbedding(model_name=args.model)
    embedding = np.array(list(model.query_embed([args.query])), dtype=np.float32)
    k = min(args.k, index.ntotal)
    distances, indices = index.search(embedding, k=k)

    print(f'Query: "{args.query}"')
    print(f"Index: {index.ntotal} vectors, {index.d} dims\n")

    for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0:
            continue
        m = metadata[idx]
        print(f"#{rank + 1} (score={score:.4f}) {m['source_name']}:{m.get('file_path', '?')}")
        print(f"  Section: {m.get('section_title', '?')}")
        print(f"  {m['content'][:500]}")
        print()


if __name__ == "__main__":
    main()
