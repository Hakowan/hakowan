from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import lagrange
import numpy as np
import pytest

import hakowan as hkw
from hakowan.mcp import HakowanMCPService, PathPolicy, create_server


def _mesh(path: Path) -> Path:
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    mesh.add_triangle(0, 1, 2)
    mesh.create_attribute(
        "temperature",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([[1.0], [2.0], [3.0]]),
    )
    lagrange.io.save_mesh(path, mesh)
    return path


def _spec(path: Path) -> dict:
    figure = hkw.figure(hkw.layer(path).color_by("temperature")).camera(
        "fit", direction="front"
    )
    return hkw.to_spec(figure).to_dict()


def test_path_policy_rejects_escape_and_symlinks(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PathPolicy.create(workspace)
    inside = workspace / "inside.txt"
    inside.write_text("ok", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = workspace / "link.txt"
    link.symlink_to(outside)

    assert policy.resolve("inside.txt", must_exist=True) == inside
    with pytest.raises(PermissionError):
        policy.resolve("link.txt", must_exist=True)
    with pytest.raises(PermissionError):
        policy.resolve("../outside.txt")


def test_inspect_validate_compile_render_patch_tools(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    spec = _spec(mesh_path)
    service = HakowanMCPService(root=tmp_path)

    inspected = service.inspect_data("mesh.ply")
    validated = service.validate_spec(spec)
    compiled = service.compile_spec(spec)
    patched = service.apply_patch(
        spec,
        [{"op": "replace", "path": "/scene/camera/fov", "value": 40}],
    )
    rendered = service.render_spec(spec, "viewer.html")

    assert inspected["ok"]
    assert inspected["summary"]["attributes"][0]["name"] == "temperature"
    assert validated["ok"] and validated["validation"]["valid"]
    assert compiled["ok"] and compiled["scene"]["views"][0]["facet_count"] == 1
    assert patched["ok"]
    assert patched["spec"]["scene"]["camera"]["fov"] == 40
    assert rendered["ok"]
    assert (tmp_path / "viewer.html").is_file()


def test_resource_paths_and_outputs_stay_in_workspace(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    spec = _spec(mesh_path)
    spec["root"]["child"]["spec"]["data"]["path"] = "../outside.ply"
    service = HakowanMCPService(root=tmp_path)

    validation = service.validate_spec(spec)
    rendered = service.render_spec(_spec(mesh_path), "../viewer.html")

    assert not validation["ok"]
    assert validation["error"]["type"] == "PermissionError"
    assert not rendered["ok"]
    assert rendered["error"]["type"] == "PermissionError"


def test_external_data_bindings_are_resolved_inside_workspace(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    mesh = lagrange.io.load_mesh(mesh_path)
    figure = hkw.figure(hkw.layer(mesh).color_by("temperature"))
    spec = hkw.to_spec(figure, data_ids={id(mesh): "input"}).to_dict()
    service = HakowanMCPService(root=tmp_path)

    missing = service.validate_spec(spec)
    bound = service.validate_spec(spec, data_bindings={"input": "mesh.ply"})

    assert not missing["ok"]
    assert bound["ok"] and bound["validation"]["valid"]


def test_gallery_search_returns_feature_matched_canonical_spec(tmp_path):
    gallery = tmp_path / "hakowan-gallery"
    recipe = gallery / "gallery" / "Scalar"
    artifacts = recipe / "artifacts"
    artifacts.mkdir(parents=True)
    (recipe / "recipe.toml").write_text(
        '\n'.join(
            [
                'id = "scalar"',
                'title = "Scalar field"',
                'script = "scalar.py"',
                'summary = "Color a surface scalar field."',
                'features = ["surface", "scalar-field", "legend"]',
                'backends = ["webgl"]',
                'inputs = []',
                'outputs = []',
            ]
        ),
        encoding="utf-8",
    )
    (artifacts / "figure.json").write_text(
        json.dumps({"version": "1.1", "root": {"kind": "layer", "spec": {}}}),
        encoding="utf-8",
    )
    (artifacts / "inspect.json").write_text("{}", encoding="utf-8")
    service = HakowanMCPService(root=tmp_path, gallery=gallery)

    result = service.search_gallery(features=["scalar-field"], include_spec=True)

    assert result["ok"]
    assert result["matches"][0]["id"] == "scalar"
    assert result["matches"][0]["spec"]["version"] == "1.1"
    assert "inputs" in result["matches"][0]
    assert result["matches"][0]["inspection"] == {}


def test_observe_tool_writes_structured_evidence(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    service = HakowanMCPService(root=tmp_path)

    result = service.observe_spec(
        _spec(mesh_path),
        "observation",
        views=["front"],
        passes=["depth", "element_id", "layer_id"],
        resolution=[32, 32],
    )

    assert result["ok"]
    assert (tmp_path / "observation" / "manifest.json").is_file()
    assert "visibility" in result["manifest"]


def test_mcp_server_lists_and_calls_structured_tools(tmp_path):
    pytest.importorskip("mcp")
    try:
        from mcp import Client
        from mcp.server import MCPServer  # noqa: F401
    except ImportError:
        pytest.skip("MCP Python SDK v2 is unavailable")
    _mesh(tmp_path / "mesh.ply")
    server = create_server(root=tmp_path)

    async def exercise():
        async with Client(server) as client:
            listed = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()
            result = await client.call_tool("inspect_data", {"source": "mesh.ply"})
            tools = {tool.name: tool for tool in listed.tools}
            return tools, resources, prompts, result.structured_content

    tools, resources, prompts, payload = asyncio.run(exercise())

    assert {
        "get_schema",
        "inspect_data",
        "search_gallery",
        "validate_spec",
        "compile_spec",
        "get_backends",
        "render_spec",
        "observe_spec",
    } <= set(tools)
    assert tools["inspect_data"].annotations.read_only_hint is True
    assert tools["render_spec"].annotations.read_only_hint is False
    assert {str(item.uri) for item in resources.resources} == {
        "hakowan://agent-instructions",
        "hakowan://schema",
    }
    assert {item.name for item in prompts.prompts} == {"author_figure"}
    assert payload["ok"]
    assert payload["summary"]["vertex_count"] == 3


def test_mcp_stdio_entrypoint(tmp_path):
    pytest.importorskip("mcp")
    try:
        from mcp import Client, StdioServerParameters
        from mcp.server import MCPServer  # noqa: F401
    except ImportError:
        pytest.skip("MCP Python SDK v2 is unavailable")

    async def exercise():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "hakowan.mcp", "--root", str(tmp_path)],
            cwd=Path(__file__).parents[1],
        )
        async with Client(parameters) as client:
            result = await client.call_tool("get_schema")
            return result.structured_content

    payload = asyncio.run(exercise())
    assert payload["ok"]
    assert payload["schema"]["$id"].endswith("/schema/v1.json")
