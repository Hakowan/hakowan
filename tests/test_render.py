import os
import sys

import pytest
import pathlib
import lagrange
import numpy as np
import hakowan as hkw

if sys.platform == "win32" and os.environ.get("CI") == "true":
    pytest.skip(
        "mitsuba/Dr.Jit causes process crash on Windows CI during Python shutdown",
        allow_module_level=True,
    )

pytest.importorskip("mitsuba", reason="mitsuba not installed")

from hakowan.backends.mitsuba.render import generate_scene_config


class TestRender:
    def test_render(self, triangle):
        mesh = triangle
        base = hkw.layer().data(mesh).mark(hkw.mark.Surface)

        scene = hkw.compiler.compile(base)
        scene_config = generate_scene_config(scene)
        assert len(scene_config) == 1

        for shape_id, shape in scene_config.items():
            assert shape["type"] == "ply"
            assert pathlib.Path(shape["filename"]).exists()

    def test_mitsuba_render_produces_image(self, triangle, tmp_path):
        """End-to-end smoke test that actually invokes ``mi.render``.

        This exercises Dr.Jit's render path (which dlopens libLLVM even for the
        scalar variant in Mitsuba 3.8); the config-only tests above never call
        ``mi.render`` and so would not catch a broken LLVM backend.
        """
        config = hkw.config()
        config.film.width = 16
        config.film.height = 16
        base = hkw.layer().data(triangle).mark(hkw.mark.Surface)
        out = tmp_path / "smoke.png"
        image = hkw.render(base, config, filename=out, backend="mitsuba")
        assert image is not None
        assert tuple(image.shape[:2]) == (16, 16)
        assert out.exists() and out.stat().st_size > 0

    def test_point_cloud(self, triangle):
        mesh = triangle
        base = hkw.layer().data(mesh).mark(hkw.mark.Point)

        scene = hkw.compiler.compile(base)
        scene_config = generate_scene_config(scene)
        assert len(scene_config) == 3

        for shape_id, shape in scene_config.items():
            assert shape["type"] == "sphere"
            assert shape["radius"] > 0
            for key, value in shape.items():
                if key.startswith("bsdf"):
                    assert value["type"] == "plastic"

    def test_point_cloud_with_size(self, triangle):
        mesh = triangle
        mesh.create_attribute(
            "size",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.array([1, 2, 3], dtype=np.float32),
        )
        base = hkw.layer().data(mesh)
        l0 = base.mark(hkw.mark.Point).channel(size="size")
        l1 = base.mark(hkw.mark.Surface)
        scene = hkw.compiler.compile(l0 + l1)
        generate_scene_config(scene)

    def test_identity_colormap(self, triangle):
        mesh = triangle
        mesh.create_attribute(
            "color",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Color,
            initial_values=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )
        base = hkw.layer(mesh).channel(
            material=hkw.material.Diffuse(
                reflectance=hkw.texture.ScalarField(data="color", colormap="identity")
            )
        )
        scene = hkw.compiler.compile(base)
        generate_scene_config(scene)


class TestBackSide:
    @staticmethod
    def _surface_layer(front_color, back_color):
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertices(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64))
        mesh.add_triangles(np.array([[0, 1, 2]], dtype=np.uint32))
        front = hkw.material.Diffuse(front_color)
        front.back_side = hkw.material.Diffuse(back_color)
        return hkw.layer(mesh).mark(hkw.mark.Surface).channel(material=front)

    def test_back_side_twosided_split(self):
        layer = self._surface_layer("red", "blue")
        scene_config = generate_scene_config(hkw.compiler.compile(layer))
        shape = list(scene_config.values())[0]
        bsdf = shape["bsdf"]
        assert bsdf["type"] == "twosided"
        children = [v for v in bsdf.values() if isinstance(v, dict)]
        assert len(children) == 2
        front_color = list(children[0]["reflectance"]["value"])
        back_color = list(children[1]["reflectance"]["value"])
        assert front_color != back_color

    def test_back_side_ignored_per_primitive(self):
        # The per-primitive path can't pair a back face with the per-facet BSDF
        # dict, so back_side is dropped (with a warning) and a single BSDF stays.
        from hakowan.backends.mitsuba.bsdf import generate_bsdf_config

        layer = self._surface_layer("red", "blue")
        view = hkw.compiler.compile(layer)[0]
        config = generate_bsdf_config(view, is_primitive=True)
        assert config["type"] == "diffuse"
