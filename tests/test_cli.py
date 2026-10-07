from types import SimpleNamespace

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
    monkeypatch.setattr(cli, "build_layer", lambda args, path, *, normalize, mesh: cli.hkw.layer())
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
