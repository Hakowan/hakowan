"""Default MCP application for ``mcp dev`` and ``mcp run`` discovery."""

from .server import create_server

mcp = create_server()

__all__ = ["mcp"]
