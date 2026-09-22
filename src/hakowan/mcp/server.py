"""MCP v2 server exposing Hakowan's deterministic agent workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .service import HakowanMCPService

_SERVER_INSTRUCTIONS = """Hakowan authors deterministic 3D data visualizations.
Always inspect source data before writing a spec; never invent attributes. Use
canonical FigureSpec JSON, validate strictly, repair with minimal JSON Pointer
patches, and fit geometry-dependent cameras with fit_camera. Render or observe
before completion. Use evaluate_visual_patch for bounded visual repairs and keep
only accepted candidates. All file paths are restricted to the configured
workspace root. Arbitrary Python functions are not accepted through this server;
use safe expression specs instead.
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
            "Hakowan MCP support requires MCP Python SDK v2: pip install 'hakowan[mcp]'"
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
    def get_schema(fragment: str | None = None) -> dict[str, Any]:
        """Return the full FigureSpec schema or a focused fragment."""
        return service.get_schema(fragment)

    @mcp.tool(name="get_spec_template", annotations=read_only)
    def get_spec_template(
        name: str | None = None,
        data_id: str = "data",
        attribute: str = "value",
    ) -> dict[str, Any]:
        """List or return minimal canonical FigureSpec templates."""
        return service.get_spec_template(name, data_id, attribute)

    @mcp.tool(name="get_spec", annotations=read_only)
    def get_spec(spec_id: str) -> dict[str, Any]:
        """Resolve a session-local content-addressed FigureSpec handle."""
        return service.get_spec(spec_id)

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
        """Search local or published recipes; return no matches if unavailable."""
        return service.search_gallery(query, features, limit, include_spec)

    @mcp.tool(name="validate_spec", annotations=read_only)
    def validate_spec(
        spec: dict[str, Any] | str,
        backend: str = "webgl",
        strict: bool = True,
        data_bindings: dict[str, str] | None = None,
        compile_check: bool = True,
    ) -> dict[str, Any]:
        """Validate FigureSpec schema, resources, semantics, backend, and compile."""
        return service.validate_spec(
            spec, backend, strict, data_bindings, compile_check
        )

    @mcp.tool(name="fit_camera", annotations=read_only)
    def fit_camera(
        spec: dict[str, Any] | str,
        data_bindings: dict[str, str] | None = None,
        backend: str = "webgl",
        direction: str | list[float] = "isometric",
        projection: str = "perspective",
        margin: float = 0.08,
        resolution: list[int] | None = None,
        fov: float = 35.0,
        fov_axis: str = "smaller",
        up_axis: str = "y",
    ) -> dict[str, Any]:
        """Fit and validate a concrete camera, returning a minimal spec patch."""
        return service.fit_camera(
            spec,
            data_bindings,
            backend,
            direction,
            projection,
            margin,
            resolution,
            fov,
            fov_axis,
            up_axis,
        )

    @mcp.tool(name="compile_spec", annotations=read_only)
    def compile_spec(
        spec: dict[str, Any] | str, data_bindings: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Compile a FigureSpec and return resolved view and overlay metadata."""
        return service.compile_spec(spec, data_bindings)

    @mcp.tool(name="render_spec", annotations=writes_files)
    def render_spec(
        spec: dict[str, Any] | str,
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
        spec: dict[str, Any] | str,
        output_dir: str,
        views: list[str] | None = None,
        passes: list[str] | None = None,
        resolution: list[int] | None = None,
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
        visual_criteria: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Capture multi-view WebGL evidence and structured visibility metadata."""
        return service.observe_spec(
            spec,
            output_dir,
            views=views,
            passes=passes,
            resolution=resolution,
            data_bindings=data_bindings,
            strict=strict,
            visual_criteria=visual_criteria,
        )

    @mcp.tool(name="evaluate_visual_patch", annotations=writes_files)
    def evaluate_visual_patch(
        spec: dict[str, Any] | str,
        operations: list[dict[str, Any]],
        output_dir: str,
        views: list[str] | None = None,
        resolution: list[int] | None = None,
        data_bindings: dict[str, str] | None = None,
        visual_criteria: dict[str, float] | None = None,
        max_operations: int = 3,
    ) -> dict[str, Any]:
        """Accept a patch only if deterministic visual evidence improves."""
        return service.evaluate_visual_patch(
            spec,
            operations,
            output_dir,
            views,
            resolution,
            data_bindings,
            visual_criteria,
            max_operations,
        )

    @mcp.tool(name="apply_patch", annotations=read_only)
    def apply_patch(
        spec: dict[str, Any] | str,
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
