from types import SimpleNamespace
import logging
from pathlib import Path

import lagrange
import numpy as np
import pytest

import hakowan.__main__ as cli


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        (["webgl", "blender", "mitsuba"], "mitsuba"),
        (["webgl", "blender"], "blender"),
    ],
)
def test_render_selects_first_available_native_backend(
    monkeypatch, installed, expected
):
    monkeypatch.setattr(cli.hkw, "list_backends", lambda: installed)

    args = cli.parse_args(["render", "mesh.obj"])

    assert args.action == "render"
    assert args.backend == expected
    assert args.input_mesh == ["mesh.obj"]


def test_render_requires_mitsuba_or_blender(monkeypatch):
    monkeypatch.setattr(cli.hkw, "list_backends", lambda: ["webgl"])
    monkeypatch.setattr(cli.sys, "argv", ["hkw", "render", "mesh.obj"])

    with pytest.raises(SystemExit, match="requires Mitsuba or Blender"):
        cli.main()


def test_render_selects_blender_for_facet_id(monkeypatch):
    monkeypatch.setattr(
        cli.hkw, "list_backends", lambda: ["mitsuba", "webgl", "blender"]
    )

    args = cli.parse_args(["render", "mesh.obj", "--facet-id"])

    assert args.backend == "blender"


def test_render_facet_id_requires_blender(monkeypatch):
    monkeypatch.setattr(cli.hkw, "list_backends", lambda: ["mitsuba", "webgl"])

    with pytest.raises(SystemExit, match="requires the Blender backend"):
        cli.parse_args(["render", "mesh.obj", "--facet-id"])


def test_view_rejects_facet_id():
    with pytest.raises(SystemExit, match="only available with hkw render"):
        cli.parse_args(["view", "mesh.obj", "--facet-id"])


def test_view_always_selects_webgl(monkeypatch):
    monkeypatch.setattr(cli.hkw, "list_backends", lambda: ["mitsuba"])

    args = cli.parse_args(["view", "mesh.obj"])

    assert args.action == "view"
    assert args.backend == "webgl"
    assert args.input_mesh == ["mesh.obj"]


@pytest.mark.parametrize(
    ("facet_counts", "expected_point_cloud"),
    [([0], True), ([0, 0], True), ([0, 1], False)],
)
def test_main_enables_point_mode_only_for_facet_free_inputs(
    monkeypatch, tmp_path, facet_counts, expected_point_cloud
):
    paths = [tmp_path / f"input-{index}.ply" for index in range(len(facet_counts))]
    loaded_paths = []
    point_cloud_modes = []

    def load_mesh(path, *, quiet, stitch_vertices):
        loaded_paths.append(path)
        return SimpleNamespace(num_facets=facet_counts[len(loaded_paths) - 1])

    def build_layer(args, mesh_path, *, normalize, mesh):
        point_cloud_modes.append(args.point_cloud)
        return object()

    monkeypatch.setattr(cli.lagrange.io, "load_mesh", load_mesh)
    monkeypatch.setattr(cli, "build_layer", build_layer)
    monkeypatch.setattr(cli.hkw, "grid", lambda layers, **kwargs: object())
    monkeypatch.setattr(cli.hkw, "render", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["hkw", "view", *(str(path) for path in paths), "--no-open"],
    )

    cli.main()

    assert loaded_paths == [str(path) for path in paths]
    assert point_cloud_modes == [expected_point_cloud] * len(paths)


def test_point_size_uses_screen_pixels(triangle):
    args = cli.parse_args(
        ["view", "unused.ply", "--point-cloud", "--point-size", "3.5"]
    )

    layer = cli.build_layer(args, "unused.ply", mesh=triangle)
    size = next(
        channel
        for channel in layer._spec.channels
        if isinstance(channel, cli.hkw.channel.Size)
    )

    assert size.data == 3.5
    assert size.space == "screen"


@pytest.mark.parametrize("argv", [[], ["mesh.obj"], ["compile", "mesh.obj"]])
def test_action_is_required_and_limited(argv):
    with pytest.raises(SystemExit):
        cli.parse_args(argv)


@pytest.mark.parametrize(
    ("mesh_count", "expected_columns"),
    [(1, 1), (2, 2), (3, 3), (4, 2), (5, 3), (6, 3)],
)
def test_main_grid_columns(monkeypatch, tmp_path, mesh_count, expected_columns):
    paths = [tmp_path / f"input-{i}.ply" for i in range(mesh_count)]
    captured_grid_kwargs = {}

    monkeypatch.setattr(
        cli.lagrange.io,
        "load_mesh",
        lambda path, *, quiet, stitch_vertices: SimpleNamespace(num_facets=1),
    )
    monkeypatch.setattr(
        cli, "build_layer", lambda args, path, *, normalize, mesh: cli.hkw.layer()
    )

    def fake_grid(layers, **kwargs):
        captured_grid_kwargs.update(kwargs)
        return cli.hkw.layer()

    monkeypatch.setattr(cli.hkw, "grid", fake_grid)
    monkeypatch.setattr(cli.hkw, "render", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["hkw", "view", *(str(path) for path in paths), "--no-open"],
    )

    cli.main()

    assert captured_grid_kwargs["columns"] == expected_columns


def test_clip_with_seams_extracts_curve_from_clipped_surface():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    mesh.add_triangle(0, 1, 2)
    mesh.add_triangle(0, 2, 3)
    mesh.create_attribute(
        "uv",
        element=lagrange.AttributeElement.Indexed,
        usage=lagrange.AttributeUsage.UV,
        initial_values=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.1, 0.1],
                [1.1, 1.1],
                [0.0, 1.0],
            ]
        ),
        initial_indices=np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32),
    )
    args = cli.parse_args(["view", "unused.mesh", "--clip", "x", "--seams"])

    layer = cli.build_layer(args, "unused.mesh", mesh=mesh)
    scene = cli.hkw.compiler.compile(layer)

    surface_mesh = next(
        view.data_frame.mesh for view in scene if view.mark == cli.hkw.mark.Surface
    )
    curve_mesh = next(
        view.data_frame.mesh for view in scene if view.mark == cli.hkw.mark.Curve
    )
    assert np.all(surface_mesh.vertices[:, 0] >= 0.5 - 1e-9)
    assert curve_mesh.num_facets > 0
    assert np.all(curve_mesh.vertices[:, 0] >= 0.5 - 1e-9)
    assert any(
        {
            tuple(np.round(curve_mesh.vertices[index, :2], 6))
            for index in curve_mesh.get_facet_vertices(facet)
        }
        == {(0.5, 0.5), (1.0, 1.0)}
        for facet in range(curve_mesh.num_facets)
    )
    assert mesh.is_attribute_indexed("uv")


@pytest.mark.parametrize("field_style", ["streamline", "arrow", "fur"])
def test_clip_with_vector_field_clips_before_overlay_generation(field_style):
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array(
            [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            dtype=np.float64,
        )
    )
    mesh.add_triangles(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32))
    mesh.create_attribute(
        "flow",
        element=lagrange.AttributeElement.Facet,
        usage=lagrange.AttributeUsage.Vector,
        initial_values=np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float64),
    )
    args = cli.parse_args(
        [
            "render",
            "--vector-field",
            "flow",
            "--field-style",
            field_style,
            "--num-streamlines",
            "4",
            "--clip",
            "x",
            "unused.mesh",
        ]
    )

    scene = cli.hkw.compiler.compile(cli.build_layer(args, "unused.mesh", mesh=mesh))
    curve = next(view for view in scene if view.mark == cli.hkw.mark.Curve)

    assert curve.data_frame.mesh.num_facets > 0
    assert np.all(curve.data_frame.mesh.vertices[:, 0] >= 0.5 - 1e-9)


def test_view_reports_temporary_html_if_browser_open_fails(
    monkeypatch, tmp_path, triangle, caplog
):
    mesh_path = tmp_path / "mesh.ply"
    lagrange.io.save_mesh(mesh_path, triangle)
    rendered = []

    def fake_render(root, config, *, filename, backend, **kwargs):
        output = Path(filename)
        output.write_text("viewer")
        rendered.append(output)

    monkeypatch.setattr(cli.hkw, "render", fake_render)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: False)
    monkeypatch.setattr(cli.sys, "argv", ["hkw", "view", str(mesh_path)])
    caplog.set_level(logging.WARNING, logger="hakowan")

    cli.main()

    assert len(rendered) == 1
    assert str(rendered[0]) in caplog.text
    assert "Could not open WebGL viewer" in caplog.text
    rendered[0].unlink()
