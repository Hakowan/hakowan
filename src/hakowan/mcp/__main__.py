"""Run the Hakowan MCP server over stdio or Streamable HTTP."""

from __future__ import annotations

import argparse
from pathlib import Path

from .server import create_server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse MCP server transport and workspace options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root allowed for every MCP file read and write.",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        help="Optional hakowan-gallery checkout used by search_gallery.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Create and run the configured Hakowan MCP server."""
    args = parse_args(argv)
    mcp = create_server(root=args.root, gallery=args.gallery)
    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
