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
        result = hkw.render(base, config, filename=out, backend="mitsuba")
        assert result.image is not None
        assert tuple(result.image.shape[:2]) == (16, 16)
        assert result.path == out
        assert out.exists() and out.stat().st_size > 0

    @pytest.mark.parametrize("ext", [".png", ".webp", ".jpg", ".tif", ".bmp"])
    def test_mitsuba_writes_pillow_formats(self, triangle, tmp_path, ext):
        """Non-EXR output is encoded by Pillow, so any Pillow format works."""
        from PIL import Image

        config = hkw.config()
        config.film.width = config.film.height = 16
        base = hkw.layer().data(triangle).mark(hkw.mark.Surface)
        out = tmp_path / f"img{ext}"
        result = hkw.render(base, config, filename=out, backend="mitsuba")
        assert result.path == out
        assert out.exists() and out.stat().st_size > 0
        with Image.open(out) as im:
            assert im.size == (16, 16)

    def test_mitsuba_unsupported_format_raises(self, triangle, tmp_path):
        config = hkw.config()
        config.film.width = config.film.height = 16
        base = hkw.layer().data(triangle).mark(hkw.mark.Surface)
        with pytest.raises(ValueError, match="Unsupported output image format"):
            hkw.render(base, config, filename=tmp_path / "img.xyz", backend="mitsuba")

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

    def test_hair_melanin_config(self):
        from hakowan.backends.mitsuba.bsdf import generate_hair_bsdf_config
        from hakowan.grammar.channel.material import Hair

        cfg = generate_hair_bsdf_config(lagrange.SurfaceMesh(), Hair())
        assert cfg["type"] == "hair"
        assert "eumelanin" in cfg and "pheomelanin" in cfg
        assert "sigma_a" not in cfg

    def test_hair_constant_color_config(self):
        # A constant RGB color inverts to a hair absorption coefficient; a blue
        # target absorbs red most (largest sigma_a) and blue least.
        from hakowan.backends.mitsuba.bsdf import generate_hair_bsdf_config
        from hakowan.grammar.channel.material import Hair

        cfg = generate_hair_bsdf_config(
            lagrange.SurfaceMesh(), Hair(color=[0.15, 0.35, 0.95])
        )
        assert "sigma_a" in cfg and "eumelanin" not in cfg
        sigma = list(cfg["sigma_a"]["value"])
        assert sigma[0] > sigma[2]  # red absorbed more than blue

    def test_hair_gradient_collapses_to_average_color(self):
        # A root/tip gradient can't be per-strand in Mitsuba: it collapses to a
        # single averaged absorption (still overriding melanin).
        from hakowan.backends.mitsuba.bsdf import generate_hair_bsdf_config
        from hakowan.grammar.channel.material import Hair

        cfg = generate_hair_bsdf_config(
            lagrange.SurfaceMesh(),
            Hair(root_color=[0.02, 0.01, 0.005], tip_color=[0.9, 0.65, 0.3]),
        )
        assert "sigma_a" in cfg and "eumelanin" not in cfg


class TestPointOrientation:
    """Point-cloud discs oriented by a normal field.

    Regression: PCD-imported normals are frequently *not* unit length (observed
    magnitudes up to ~1400). ``rotation`` used to assume unit inputs, so a
    non-unit normal became a huge scale/shear instead of a rotation, producing
    giant degenerate disc triangles that made Mitsuba's BVH pathologically slow
    (an effective hang). The orientation transform must stay a bounded rotation.
    """

    @staticmethod
    def _matrix(transform) -> np.ndarray:
        return np.array(transform.matrix).reshape(4, 4)

    def test_rotation_normalizes_non_unit_inputs(self):
        from hakowan.backends.mitsuba.utils import rotation

        z = np.array([0.0, 0.0, 1.0])
        rng = np.random.default_rng(0)
        for _ in range(64):
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            scaled = direction * rng.uniform(0.01, 1400.0)
            R = rotation(z, scaled)[:3, :3]
            # A genuine rotation: orthonormal, det 1, and maps +Z onto the
            # *normalized* target direction regardless of the input magnitude.
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
            assert abs(np.linalg.det(R) - 1.0) < 1e-6
            np.testing.assert_allclose(R @ z, direction, atol=1e-6)

    def test_rotation_flips_antiparallel(self):
        from hakowan.backends.mitsuba.utils import rotation

        z = np.array([0.0, 0.0, 1.0])
        R = rotation(z, np.array([0.0, 0.0, -7.0]))[:3, :3]
        np.testing.assert_allclose(R @ z, [0.0, 0.0, -1.0], atol=1e-6)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_disc_transform_bounded_for_non_unit_normals(self):
        # Build a point cloud whose normals have wildly varying magnitude and
        # confirm every generated disc carries a bounded (radius-scaled) linear
        # transform — never the ~magnitude-scaled blowup that caused the hang.
        radius = 0.5
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertices(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        )
        normals = np.array(
            [[0.0, 0.0, 1000.0], [300.0, 400.0, 0.0], [0.0, 0.0, -1400.0]],
            dtype=np.float64,
        )
        mesh.create_attribute(
            "normal",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Normal,
            initial_values=normals,
        )
        layer = (
            hkw.layer(mesh)
            .mark(hkw.mark.Point)
            .channel(
                size=radius,
                shape=hkw.channel.Shape(base_shape="disk", orientation="normal"),
            )
        )
        scene_config = generate_scene_config(hkw.compiler.compile(layer))
        discs = [s for s in scene_config.values() if s.get("type") == "ply"]
        assert len(discs) == mesh.num_vertices
        for shape in discs:
            linear = self._matrix(shape["to_world"])[:3, :3]
            column_scales = np.linalg.norm(linear, axis=0)
            # A correct disc transform is rotation × uniform scale: all three
            # column norms are equal and, once divided out, the remainder is
            # orthonormal. The pre-fix bug fed the un-normalized normal straight
            # into the Rodrigues terms, yielding an anisotropic ~magnitude-scaled
            # shear (hundreds of times larger) that fails both checks.
            scale = column_scales.mean()
            np.testing.assert_allclose(column_scales, scale, rtol=1e-4)
            R = linear / scale
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-4)
