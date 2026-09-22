from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import re

import sys
import urllib.error

import lagrange
import numpy as np
import pytest

import hakowan as hkw
import hakowan.mcp.service as service_module

from hakowan.mcp import HakowanMCPService, PathPolicy, create_server


def _mesh(path: Path) -> Path:
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
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


def test_schema_fragments_are_self_contained_and_smaller(tmp_path):
    service = HakowanMCPService(root=tmp_path)

    full = service.get_schema()
    fragment = service.get_schema("transform.clip")
    missing = service.get_schema("transform.unknown")

    assert full["ok"] and "transform.clip" in full["available_fragments"]
    assert fragment["ok"] and fragment["schema"]["$ref"].endswith("/ClipTransformSpec")
    assert set(fragment["schema"]["$defs"]) == {"ClipTransformSpec"}
    assert len(json.dumps(fragment["schema"])) < len(json.dumps(full["schema"]))
    vector = service.get_schema("channel.vector_field")["schema"]
    references = set(re.findall(r'"#/$defs/([^\"]+)"', json.dumps(vector)))
    assert references <= set(vector["$defs"])
    assert "AttributeSpec" in vector["$defs"]
    assert missing["error"]["code"] == "schema.fragment_unknown"


def test_spec_templates_are_minimal_and_schema_valid(tmp_path):
    service = HakowanMCPService(root=tmp_path)

    catalog = service.get_spec_template()
    names = {item["name"] for item in catalog["templates"]}
    assert {"surface-scalar", "vector-glyphs", "wireframe-overlay"} <= names
    for name in names:
        result = service.get_spec_template(
            name, data_id="input", attribute="temperature"
        )
        assert result["ok"]
        assert "scene" not in result["spec"]
        assert "roi_box" not in json.dumps(result["spec"])
        hkw.FigureSpec.model_validate(result["spec"])


def test_spec_handles_chain_without_resending_json(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    service = HakowanMCPService(root=tmp_path)

    validated = service.validate_spec(_spec(mesh_path))
    same = service.validate_spec(validated["spec"])
    compiled = service.compile_spec(validated["spec_id"])
    fitted = service.fit_camera(validated["spec_id"], direction="isometric")
    patched = service.apply_patch(
        fitted["spec_id"],
        [{"op": "replace", "path": "/scene/camera/fov", "value": 40}],
    )
    fetched = service.get_spec(patched["spec_id"])
    rendered = service.render_spec(patched["spec_id"], "handled.html")

    assert validated["spec_id"].startswith("sha256:")
    assert same["spec_id"] == validated["spec_id"]
    assert compiled["spec_id"] == validated["spec_id"]
    assert fitted["spec_id"] != validated["spec_id"]
    assert patched["spec_id"] != fitted["spec_id"]
    assert fetched["spec"] == patched["spec"]
    assert rendered["spec_id"] == patched["spec_id"]
    assert not service.get_spec("sha256:missing")["ok"]


def test_spec_handle_store_is_bounded_and_lru(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    service = HakowanMCPService(root=tmp_path)
    base = _spec(mesh_path)
    handles = []
    for index in range(129):
        candidate = json.loads(json.dumps(base))
        candidate["root"]["child"]["spec"]["name"] = f"layer-{index}"
        handles.append(service.validate_spec(candidate, compile_check=False)["spec_id"])

    assert not service.get_spec(handles[0])["ok"]
    assert service.get_spec(handles[-1])["ok"]


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


def test_fit_camera_returns_valid_minimal_patch(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    spec = hkw.to_spec(hkw.layer(mesh_path)).to_dict()
    service = HakowanMCPService(root=tmp_path)

    result = service.fit_camera(
        spec,
        direction="isometric",
        projection="orthographic",
        resolution=[640, 480],
    )

    assert result["ok"]
    assert result["camera"]["kind"] == "orthographic"
    assert result["camera"]["scale"] > 0
    assert result["patch"] == [
        {"op": "replace", "path": "/scene", "value": {"camera": result["camera"]}}
    ]
    assert result["validation"]["valid"]


def test_apply_patch_repairs_schema_invalid_input(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    spec = _spec(mesh_path)
    spec["scene"]["unexpected"] = True
    service = HakowanMCPService(root=tmp_path)

    invalid = service.validate_spec(spec)
    repaired = service.apply_patch(
        spec,
        [{"op": "remove", "path": "/scene/unexpected"}],
    )

    assert invalid["error"]["code"] == "validate.schema"
    assert invalid["error"]["diagnostics"] == [
        {
            "code": "schema.extra_forbidden",
            "path": "/scene/unexpected",
            "message": "Extra inputs are not permitted",
            "suggested_patch": {"op": "remove", "path": "/scene/unexpected"},
        }
    ]
    assert repaired["ok"]
    assert "unexpected" not in repaired["spec"]["scene"]
    assert spec["scene"]["unexpected"] is True

    nested = _spec(mesh_path)
    nested["root"]["child"]["spec"]["channels"]["unexpected"] = True
    nested_error = service.validate_spec(nested)
    assert nested_error["error"]["diagnostics"][0]["path"] == (
        "/root/child/spec/channels/unexpected"
    )
    assert nested_error["error"]["diagnostics"][0]["suggested_patch"]["path"] == (
        "/root/child/spec/channels/unexpected"
    )


def test_apply_patch_reports_structured_schema_failure(tmp_path):
    spec = _spec(_mesh(tmp_path / "mesh.ply"))
    service = HakowanMCPService(root=tmp_path)

    result = service.apply_patch(
        spec,
        [{"op": "add", "path": "/scene/unexpected", "value": True}],
    )

    assert not result["ok"]
    assert result["error"]["code"] == "patch.schema"
    assert result["error"]["diagnostics"][0]["path"] == "/scene/unexpected"


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


def test_missing_gallery_is_non_fatal(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    service = HakowanMCPService(
        root=tmp_path,
        gallery=tmp_path / "missing-gallery",
        gallery_url="",
    )

    gallery = service.search_gallery(query="scalar field", include_spec=True)
    validated = service.validate_spec(_spec(mesh_path))

    assert gallery == {
        "ok": True,
        "available": False,
        "gallery": None,
        "source": None,
        "matches": [],
        "message": (
            "Gallery examples are unavailable; continue using data inspection, "
            "the FigureSpec schema, validation, compilation, and rendering."
        ),
    }
    assert validated["ok"] and validated["validation"]["valid"]


def test_remote_gallery_failure_is_non_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        service_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("offline")
        ),
    )
    service = HakowanMCPService(root=tmp_path)

    result = service.search_gallery(query="surface")

    assert result["ok"]
    assert result["available"] is False
    assert result["matches"] == []


def test_remote_gallery_uses_validated_cached_artifacts(tmp_path, monkeypatch):
    figure = json.dumps({"version": "1.1", "root": {"kind": "layer"}}).encode()
    inspection = b"{}"
    index = json.dumps(
        {
            "corpus_digest": "corpus-v1",
            "recipes": [
                {
                    "id": "scalar",
                    "title": "Scalar field",
                    "summary": "Color a surface scalar field.",
                    "features": ["surface", "scalar-field", "legend"],
                    "backends": ["webgl"],
                    "inputs": [],
                    "outputs": [],
                    "artifacts": {
                        "figure": "recipes/scalar/figure.json",
                        "inspect": "recipes/scalar/inspect.json",
                    },
                    "sha256": {
                        "figure": hashlib.sha256(figure).hexdigest(),
                        "inspect": hashlib.sha256(inspection).hexdigest(),
                    },
                }
            ],
        }
    ).encode()
    calls = []

    class Response:
        def __init__(self, payload, headers=None):
            self.payload = payload
            self.headers = headers or {}

        def read(self, _limit):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def urlopen(request, timeout):
        calls.append((request.full_url, dict(request.header_items()), timeout))
        if request.full_url.endswith("index.json"):
            if len([call for call in calls if call[0].endswith("index.json")]) > 1:
                raise urllib.error.HTTPError(
                    request.full_url, 304, "Not Modified", {}, None
                )
            return Response(index, {"ETag": '"corpus-v1"'})
        if request.full_url.endswith("figure.json"):
            return Response(figure)
        if request.full_url.endswith("inspect.json"):
            return Response(inspection)
        raise AssertionError(request.full_url)

    monkeypatch.setattr(service_module.urllib.request, "urlopen", urlopen)
    service = HakowanMCPService(
        root=tmp_path,
        gallery_url="https://example.test/agent/v1/index.json",
    )

    first = service.search_gallery(features=["scalar-field"], include_spec=True)
    second = service.search_gallery(features=["scalar-field"], include_spec=True)

    assert first == second
    assert first["available"] is True
    assert first["source"] == "remote"
    assert first["corpus_digest"] == "corpus-v1"
    assert first["matches"][0]["spec"]["version"] == "1.1"
    assert first["matches"][0]["inspection"] == {}
    assert len(calls) == 4
    assert calls[-1][1]["If-none-match"] == '"corpus-v1"'


def test_gallery_search_returns_feature_matched_canonical_spec(tmp_path, monkeypatch):
    gallery = tmp_path / "hakowan-gallery"
    recipe = gallery / "gallery" / "Scalar"
    artifacts = recipe / "artifacts"
    artifacts.mkdir(parents=True)
    (recipe / "recipe.toml").write_text(
        "\n".join(
            [
                'id = "scalar"',
                'title = "Scalar field"',
                'script = "scalar.py"',
                'summary = "Color a surface scalar field."',
                'features = ["surface", "scalar-field", "legend"]',
                'backends = ["webgl"]',
                "inputs = []",
                "outputs = []",
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
    monkeypatch.setattr(
        service_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local gallery must take precedence")
        ),
    )

    result = service.search_gallery(features=["scalar-field"], include_spec=True)

    assert result["ok"]
    assert result["available"] is True
    assert result["matches"][0]["id"] == "scalar"
    assert result["matches"][0]["spec"]["version"] == "1.1"
    assert "inputs" in result["matches"][0]
    assert result["matches"][0]["inspection"] == {}


def test_observe_tool_writes_structured_evidence(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    service = HakowanMCPService(root=tmp_path)
    validated = service.validate_spec(_spec(mesh_path))

    result = service.observe_spec(
        validated["spec_id"],
        "observation",
        views=["front"],
        passes=["depth", "element_id", "layer_id"],
        resolution=[32, 32],
    )

    assert result["ok"]
    assert (tmp_path / "observation" / "manifest.json").is_file()
    assert "visibility" in result["manifest"]
    assert result["evidence"]["summary"]["view_count"] == 1
    assert isinstance(result["visual_diagnostics"], list)
    assert all(
        not Path(item["path"]).is_absolute() for item in result["manifest"]["snapshots"]
    )
    assert all(len(item["sha256"]) == 64 for item in result["manifest"]["snapshots"])


def test_visual_patch_accepts_improvement_and_rolls_back_regression(tmp_path):
    mesh_path = _mesh(tmp_path / "mesh.ply")
    original = hkw.to_spec(
        hkw.figure(hkw.layer(mesh_path)).camera(
            "perspective",
            eye=(0.5, 0.5, 100.0),
            target=(0.5, 0.5, 0.0),
            fov=35.0,
        )
    ).to_dict()
    service = HakowanMCPService(root=tmp_path)
    validated = service.validate_spec(original)
    fitted = service.fit_camera(validated["spec_id"], direction="front")

    accepted = service.evaluate_visual_patch(
        validated["spec_id"],
        fitted["patch"],
        "visual-comparison",
        resolution=[64, 64],
    )
    rejected = service.evaluate_visual_patch(
        accepted["spec_id"],
        [
            {
                "op": "replace",
                "path": "/scene/camera/eye",
                "value": [0.5, 0.5, 100.0],
            },
            {"op": "replace", "path": "/scene/camera/far", "value": 200.0},
        ],
        "visual-regression",
        resolution=[64, 64],
    )

    assert accepted["accepted"]
    assert not accepted["rolled_back"]
    assert "occupancy_distance" in accepted["comparison"]["improvements"]
    assert not rejected["accepted"]
    assert rejected["rolled_back"]
    assert rejected["spec"] == accepted["spec"]
    assert "occupancy_distance" in rejected["comparison"]["regressions"]


def test_mcp_server_lists_and_calls_structured_tools(tmp_path):
    pytest.importorskip("mcp")
    try:
        from mcp import Client
        from mcp.server import MCPServer  # noqa: F401
    except ImportError:
        pytest.skip("MCP Python SDK v2 is unavailable")
    mesh_path = _mesh(tmp_path / "mesh.ply")
    spec = hkw.to_spec(hkw.layer(mesh_path)).to_dict()
    server = create_server(root=tmp_path)

    async def exercise():
        async with Client(server) as client:
            listed = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()
            inspected = await client.call_tool("inspect_data", {"source": "mesh.ply"})
            fragment = await client.call_tool(
                "get_schema", {"fragment": "scene.camera.perspective"}
            )
            template = await client.call_tool(
                "get_spec_template",
                {"name": "surface-scalar", "attribute": "temperature"},
            )
            validated = await client.call_tool("validate_spec", {"spec": spec})
            spec_id = validated.structured_content["spec_id"]
            fetched = await client.call_tool("get_spec", {"spec_id": spec_id})
            fitted = await client.call_tool(
                "fit_camera",
                {"spec": spec_id, "direction": "front", "projection": "perspective"},
            )
            assessed = await client.call_tool(
                "evaluate_visual_patch",
                {"spec": spec_id, "operations": [], "output_dir": "visual-assessment"},
            )
            tools = {tool.name: tool for tool in listed.tools}
            return (
                tools,
                resources,
                prompts,
                inspected.structured_content,
                fragment.structured_content,
                template.structured_content,
                fetched.structured_content,
                fitted.structured_content,
                assessed.structured_content,
            )

    (
        tools,
        resources,
        prompts,
        payload,
        fragment,
        template,
        fetched,
        fitted,
        assessed,
    ) = asyncio.run(exercise())

    assert {
        "get_schema",
        "get_spec_template",
        "get_spec",
        "inspect_data",
        "search_gallery",
        "validate_spec",
        "compile_spec",
        "get_backends",
        "render_spec",
        "observe_spec",
        "fit_camera",
        "evaluate_visual_patch",
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
    assert fragment["schema"]["$ref"].endswith("/PerspectiveCameraSpec")
    assert template["spec"]["root"]["spec"]["mark"] == "surface"
    assert fetched["spec_id"].startswith("sha256:")
    assert fetched["spec"] == spec
    assert fitted["ok"]
    assert fitted["camera"]["kind"] == "perspective"
    assert not assessed["ok"]
    assert assessed["error"]["code"] == "visual.patch_failed"


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
