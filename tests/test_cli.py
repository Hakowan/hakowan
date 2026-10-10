from types import SimpleNamespace
import logging
import sys
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
    monkeypatch.setattr(sys, "argv", ["hkw", "render", "mesh.obj"])

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


@pytest.mark.parametrize("action", ["view", "render"])
def test_turn_table_rejects_negative_counts(monkeypatch, capsys, action):
    monkeypatch.setattr(cli.hkw, "list_backends", lambda: ["mitsuba", "webgl"])

    with pytest.raises(SystemExit) as error:
        cli.parse_args([action, "mesh.obj", "--turn-table", "-1"])

    assert error.value.code == 2
    assert "--turn-table" in capsys.readouterr().err


@pytest.mark.parametrize("count", [0, 1, 12])
def test_turn_table_accepts_nonnegative_counts(count):
    args = cli.parse_args(["view", "mesh.obj", "--turn-table", str(count)])

    assert args.turn_table == count


def test_clip_accepts_negative_axis_with_equals_form():
    args = cli.parse_args(["view", "mesh.obj", "--clip=-y:0.3"])

    assert args.clip == ("-y", 0.3)


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
        sys,
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
        sys,
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


def test_singularity_adds_valence_point_view():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
        )
    )
    mesh.add_triangles(
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    )
    args = cli.parse_args(["view", "unused.mesh", "--singularity", "--no-open"])

    scene = cli.hkw.compiler.compile(cli.build_layer(args, "unused.mesh", mesh=mesh))

    point_views = [view for view in scene if view.mark == cli.hkw.mark.Point]
    assert len(point_views) == 1
    assert point_views[0].data_frame is not None
    assert point_views[0].data_frame.mesh.num_vertices == 4


def test_view_preserves_accessible_temporary_html_if_browser_open_fails(
    monkeypatch, tmp_path, triangle, caplog
):
    from urllib.parse import urlsplit
    from urllib.request import url2pathname

    mesh_path = tmp_path / "mesh.ply"
    lagrange.io.save_mesh(mesh_path, triangle)
    attempted_urls = []

    def failed_open(url):
        attempted_urls.append(url)
        return False

    monkeypatch.setattr(cli.webbrowser, "open", failed_open)
    monkeypatch.setattr(sys, "argv", ["hkw", "view", str(mesh_path)])
    caplog.set_level(logging.WARNING, logger="hakowan")

    cli.main()

    saved_path = Path(url2pathname(urlsplit(attempted_urls[0]).path))
    try:
        warnings = [
            record for record in caplog.records if record.levelno == logging.WARNING
        ]
        reported_path = Path(warnings[-1].args[0])
        assert saved_path.is_file()
        assert reported_path.samefile(saved_path)
        assert not mesh_path.with_suffix(".html").exists()
    finally:
        saved_path.unlink(missing_ok=True)


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_khr_texture_conversion_preserves_external_source(tmp_path, suffix):
    from PIL import Image
    from hakowan.cli.materials import extract_material

    source = tmp_path / f"source{suffix}"
    Image.new("RGB", (2, 2), "red").save(source)
    original = source.read_bytes()
    decoded = np.full((2, 2, 3), 128, dtype=np.uint8)
    scene = lagrange.scene.Scene()
    material = lagrange.scene.Material()
    material.extensions.data = {
        "KHR_materials_pbrSpecularGlossiness": {"diffuseTexture": {"index": 0}}
    }
    image = lagrange.scene.Image()
    image.uri = str(source)
    image.image.data = decoded
    texture = lagrange.scene.Texture()
    texture.image = 0
    scene.materials.append(material)
    scene.images.append(image)
    scene.textures.append(texture)

    converted = extract_material(scene)[0].color.filename
    try:
        assert source.read_bytes() == original
        assert converted != source
        with Image.open(converted) as result:
            assert result.format == "PNG"
            np.testing.assert_array_equal(np.asarray(result)[..., :3], decoded)
            np.testing.assert_array_equal(np.asarray(result)[..., 3], 255)
    finally:
        if converted != source:
            converted.unlink(missing_ok=True)


def test_khr_texture_conversion_rejects_empty_image():
    from hakowan.cli.materials import extract_material

    scene = lagrange.scene.Scene()
    material = lagrange.scene.Material()
    material.extensions.data = {
        "KHR_materials_pbrSpecularGlossiness": {"diffuseTexture": {"index": 0}}
    }
    image = lagrange.scene.Image()
    image.image.data = np.empty((0, 2, 3), dtype=np.uint8)
    texture = lagrange.scene.Texture()
    texture.image = 0
    scene.materials.append(material)
    scene.images.append(image)
    scene.textures.append(texture)

    with pytest.raises(ValueError, match="KHR texture image data is empty"):
        extract_material(scene)
