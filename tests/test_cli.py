import pytest

import hakowan.__main__ as cli


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        (["webgl", "blender", "mitsuba"], "mitsuba"),
        (["webgl", "blender"], "blender"),
        (["webgl"], "webgl"),
    ],
)
def test_render_selects_first_available_backend(monkeypatch, installed, expected):
    monkeypatch.setattr(cli.hkw, "list_backends", lambda: installed)

    args = cli.parse_args(["render", "mesh.obj"])

    assert args.action == "render"
    assert args.backend == expected
    assert args.input_mesh == ["mesh.obj"]


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


@pytest.mark.parametrize("argv", [[], ["mesh.obj"], ["compile", "mesh.obj"]])
def test_action_is_required_and_limited(argv):
    with pytest.raises(SystemExit):
        cli.parse_args(argv)
