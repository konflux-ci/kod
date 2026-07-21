"""Test the KOD MCP server via the FastMCP client.

Usage:
    uv run python scripts/mcp_client.py "your query here"
    uv run python scripts/mcp_client.py -k 5 "your query here"
    uv run python scripts/mcp_client.py --url http://localhost:8000/mcp "your query"
    uv run python scripts/mcp_client.py --list-tools
    uv run python scripts/mcp_client.py --get-document "source:file.md"
    uv run python scripts/mcp_client.py --get-document "source:file.md" "optional query"
"""

import argparse
import asyncio
import json

from fastmcp import Client


async def main():
    parser = argparse.ArgumentParser(description="Test the KOD MCP server")
    parser.add_argument("query", nargs="*", help="Search query (or multiple queries)")
    parser.add_argument("-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp", help="MCP server URL")
    parser.add_argument("--list-tools", action="store_true", help="List available tools and exit")
    parser.add_argument("--get-document", metavar="DOC_ID", help="Retrieve a document by ID")
    args = parser.parse_args()

    async with Client(args.url) as client:
        if args.list_tools:
            tools = await client.list_tools()
            for t in tools:
                print(f"{t.name}: {t.description}")
            return

        if args.get_document:
            params = {"document_id": args.get_document}
            if args.query:
                params["query"] = args.query[0]
            result = await client.call_tool("get_document", params)
            for block in result.content:
                print(block.text)
            return

        if not args.query:
            parser.error("query is required (unless using --list-tools or --get-document)")

        query = args.query if len(args.query) > 1 else args.query[0]
        result = await client.call_tool("search_knowledge", {"query": query, "top_k": args.k})

        for block in result.content:
            data = json.loads(block.text)
            if isinstance(data, str):
                print(data)
                return
            for rank, item in enumerate(data):
                print(f"#{rank + 1} (score={item['score']:.4f}) {item['document_id']}")
                print(f"  Title: {item['title']}")
                print(f"  URL:   {item['source_url']}")
                print(f"  {item['content'][:500]}")
                print()


if __name__ == "__main__":
    asyncio.run(main())
