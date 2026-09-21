"""MCP v2 server exposing Hakowan's deterministic agent workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .service import HakowanMCPService

_SERVER_INSTRUCTIONS = """Hakowan authors deterministic 3D data visualizations.
Always inspect source data before writing a spec; never invent attributes. Use
canonical FigureSpec JSON, validate strictly, repair with minimal JSON Pointer
patches, and render or observe the result before completion. All file paths are
restricted to the configured workspace root. Arbitrary Python functions are not
accepted through this server; use safe expression specs instead.
"""


def create_server(
    *,
    root: str | Path | None = None,
    gallery: str | Path | None = None,
):
    """Create an MCPServer bound to one workspace and optional gallery checkout."""
    try:
        from mcp.server import MCPServer
        from mcp.types import ToolAnnotations
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Hakowan MCP support requires MCP Python SDK v2: "
            "pip install 'hakowan[mcp]'"
        ) from exc

    service = HakowanMCPService(root=root, gallery=gallery)
    mcp = MCPServer(
        "Hakowan",
        title="Hakowan 3D Visualization",
        description=(
            "Inspect 3D data, author and validate FigureSpec JSON, render and "
            "observe figures, and apply safe patches."
        ),
        instructions=_SERVER_INSTRUCTIONS,
    )
    read_only = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
    writes_files = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @mcp.tool(name="get_schema", annotations=read_only)
    def get_schema() -> dict[str, Any]:
        """Return the canonical Hakowan FigureSpec JSON Schema."""
        return service.get_schema()

    @mcp.tool(name="get_backends", annotations=read_only)
    def get_backends() -> dict[str, Any]:
        """Return declared features and limitations for every rendering backend."""
        return service.get_backends()

    @mcp.tool(name="inspect_data", annotations=read_only)
    def inspect_data(source: str) -> dict[str, Any]:
        """Inspect a workspace mesh before selecting attributes."""
        return service.inspect_data(source)

    @mcp.tool(name="search_gallery", annotations=read_only)
    def search_gallery(
        query: str = "",
        features: list[str] | None = None,
        limit: int = 5,
        include_spec: bool = False,
    ) -> dict[str, Any]:
        """Find feature-matched canonical gallery recipes and optional specs."""
        return service.search_gallery(query, features, limit, include_spec)

    @mcp.tool(name="validate_spec", annotations=read_only)
    def validate_spec(
        spec: dict[str, Any],
        backend: str = "webgl",
        strict: bool = True,
        data_bindings: dict[str, str] | None = None,
        compile_check: bool = True,
    ) -> dict[str, Any]:
        """Validate FigureSpec schema, resources, semantics, backend, and compile."""
        return service.validate_spec(
            spec, backend, strict, data_bindings, compile_check
        )

    @mcp.tool(name="compile_spec", annotations=read_only)
    def compile_spec(
        spec: dict[str, Any], data_bindings: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Compile a FigureSpec and return resolved view and overlay metadata."""
        return service.compile_spec(spec, data_bindings)

    @mcp.tool(name="render_spec", annotations=writes_files)
    def render_spec(
        spec: dict[str, Any],
        output: str,
        backend: str = "webgl",
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
        offline: bool = False,
    ) -> dict[str, Any]:
        """Validate and render a FigureSpec inside the configured workspace."""
        return service.render_spec(
            spec, output, backend, data_bindings, strict, offline
        )

    @mcp.tool(name="observe_spec", annotations=writes_files)
    def observe_spec(
        spec: dict[str, Any],
        output_dir: str,
        views: list[str] | None = None,
        passes: list[str] | None = None,
        resolution: list[int] | None = None,
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Capture multi-view WebGL evidence and structured visibility metadata."""
        return service.observe_spec(
            spec,
            output_dir,
            views,
            passes,
            resolution,
            data_bindings,
            strict,
        )

    @mcp.tool(name="apply_patch", annotations=read_only)
    def apply_patch(
        spec: dict[str, Any],
        operations: list[dict[str, Any]],
        backend: str = "webgl",
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
        semantic: bool = True,
    ) -> dict[str, Any]:
        """Apply atomic JSON Pointer patches and optionally validate semantics."""
        return service.apply_patch(
            spec, operations, backend, data_bindings, strict, semantic
        )

    @mcp.resource("hakowan://schema", name="Hakowan FigureSpec schema")
    def schema_resource() -> str:
        """Return the canonical JSON Schema as formatted JSON text."""
        return json.dumps(service.get_schema()["schema"], indent=2, sort_keys=True)

    @mcp.resource("hakowan://agent-instructions", name="Hakowan agent workflow")
    def instructions_resource() -> str:
        """Return the safe inspect-author-validate-render-patch workflow."""
        return service.agent_instructions()

    @mcp.prompt(name="author_figure")
    def author_figure(request: str, source: str, backend: str = "webgl") -> str:
        """Create an agent prompt for authoring one grounded Hakowan figure."""
        return (
            f"Create a Hakowan FigureSpec for this request: {request}\n\n"
            f"Source: {source}\nBackend: {backend}\n\n"
            f"{service.agent_instructions()}"
        )

    return mcp


__all__ = ["create_server"]
