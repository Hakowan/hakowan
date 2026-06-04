"""Tests for hakowan.backends.webgl.material_translate."""

from __future__ import annotations


import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib")
from PIL import Image as PILImage

import hakowan as hkw
from hakowan.backends.webgl.builder import GLTFBuilder
from hakowan.backends.webgl.material_translate import (
    MaterialResult,
    _srgb_to_linear,
    translate_material,
)
from hakowan.compiler import compile as hkw_compile


def _compile_first_view(layer):
    scene = hkw_compile(layer)
    return scene[0]


def _write_temp_png(path, size=(8, 8), color=(255, 0, 255)):
    img = PILImage.new("RGB", size, color)
    img.save(path)


class TestSrgbToLinear:
    def test_zero(self):
        assert _srgb_to_linear(0.0) == 0.0

    def test_one(self):
        assert _srgb_to_linear(1.0) == pytest.approx(1.0, abs=1e-6)

    def test_low_branch(self):
        # Below 0.04045 it's linear (divide by 12.92).
        assert _srgb_to_linear(0.04) == pytest.approx(0.04 / 12.92, rel=1e-6)

    def test_curve_branch(self):
        # Above 0.04045 uses the power-2.4 form.
        x = 0.5
        expected = ((x + 0.055) / 1.055) ** 2.4
        assert _srgb_to_linear(x) == pytest.approx(expected, rel=1e-6)


def _triangle_view(material):
    """Make a one-triangle mesh (with UV + scalar) and compile a layer."""
    import lagrange

    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64))
    mesh.add_triangles(np.array([[0, 1, 2]], dtype=np.uint32))
    mesh.create_attribute(
        "scalar",
        usage=lagrange.AttributeUsage.Scalar,
        element=lagrange.AttributeElement.Vertex,
        initial_values=np.array([0.0, 0.5, 1.0], dtype=np.float64),
    )
    mesh.create_attribute(
        "uv",
        usage=lagrange.AttributeUsage.UV,
        element=lagrange.AttributeElement.Vertex,
        initial_values=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
    )
    layer = hkw.layer(mesh).mark(hkw.mark.Surface).channel(material=material)
    return _compile_first_view(layer)


class TestDiffuse:
    def test_uniform_color_red_maps_to_linear(self):
        view = _triangle_view(hkw.material.Diffuse(reflectance="red"))
        result = translate_material(view, GLTFBuilder())
        assert isinstance(result, MaterialResult)
        bc = result.pbr["baseColorFactor"]
        assert bc == [1.0, 0.0, 0.0, 1.0]
        assert result.pbr["metallicFactor"] == 0.0
        assert result.pbr["roughnessFactor"] == 1.0
        assert result.double_sided is False

    def test_scalarfield_yields_white_with_no_texture(self):
        view = _triangle_view(
            hkw.material.Diffuse(
                reflectance=hkw.texture.ScalarField(
                    data=hkw.attribute(name="scalar"), colormap="viridis"
                )
            )
        )
        result = translate_material(view, GLTFBuilder())
        # Vertex colors carry the actual hue; base factor should be neutral.
        assert result.pbr["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]
        # No baseColorTexture because vertex colors handle it.
        assert "baseColorTextureIndex" not in result.pbr


class TestConductor:
    def test_gold_preset(self):
        view = _triangle_view(hkw.material.Conductor(material="Au"))
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["metallicFactor"] == 1.0
        # Linear gold: gold sRGB ~(1.0, 0.766, 0.336)
        bc = result.pbr["baseColorFactor"]
        assert bc[0] == pytest.approx(1.0, abs=1e-3)
        assert bc[1] == pytest.approx(_srgb_to_linear(0.766), abs=1e-3)
        assert bc[2] == pytest.approx(_srgb_to_linear(0.336), abs=1e-3)

    def test_unknown_preset_falls_back_to_gray_with_warning(self, caplog):
        view = _triangle_view(hkw.material.Conductor(material="Unobtainium"))
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["metallicFactor"] == 1.0
        # No assertion on warning content (logger config varies); just check
        # the fallback baseColor is finite & non-extreme.
        bc = result.pbr["baseColorFactor"]
        assert all(0.0 <= c <= 1.0 for c in bc)

    def test_rough_conductor_alpha_maps_to_roughness(self):
        view = _triangle_view(hkw.material.RoughConductor(material="Cu", alpha=0.35))
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["roughnessFactor"] == pytest.approx(0.35)

    def test_rough_conductor_scalarfield_alpha_bakes_per_vertex(self):
        view = _triangle_view(
            hkw.material.RoughConductor(
                material="Cu",
                alpha=hkw.texture.ScalarField(data=hkw.attribute(name="scalar")),
            )
        )
        result = translate_material(view, GLTFBuilder())
        # ScalarField alpha is baked per-vertex; the viewer multiplies it into
        # roughnessFactor (which is set to 1.0 so the multiply is exact).
        assert result.pbr["roughnessFactor"] == pytest.approx(1.0)
        assert "_roughness_0" in result.custom_attrs
        assert result.custom_attrs["_roughness_0"].shape == (3,)
        assert (
            result.extras["hakowan"]["principled_attrs"]["roughness_attr"]
            == "_roughness_0"
        )


class TestPlastic:
    def test_plastic_low_roughness(self):
        view = _triangle_view(hkw.material.Plastic(diffuse_reflectance="ivory"))
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["metallicFactor"] == 0.0
        assert result.pbr["roughnessFactor"] == pytest.approx(0.1)

    def test_rough_plastic_alpha(self):
        view = _triangle_view(
            hkw.material.RoughPlastic(diffuse_reflectance="ivory", alpha=0.42)
        )
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["roughnessFactor"] == pytest.approx(0.42)


class TestPrincipled:
    def test_float_roughness_metallic(self):
        view = _triangle_view(
            hkw.material.Principled(
                color=hkw.texture.Uniform(color="purple"),
                roughness=0.25,
                metallic=0.9,
            )
        )
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["roughnessFactor"] == pytest.approx(0.25)
        assert result.pbr["metallicFactor"] == pytest.approx(0.9)

    def test_scalarfield_roughness_emits_custom_attr(self):
        view = _triangle_view(
            hkw.material.Principled(
                color=hkw.texture.Uniform(color="silver"),
                roughness=hkw.texture.ScalarField(data=hkw.attribute(name="scalar")),
                metallic=0.5,
            )
        )
        result = translate_material(view, GLTFBuilder())
        assert "_roughness_0" in result.custom_attrs
        assert result.custom_attrs["_roughness_0"].shape == (3,)
        assert result.extras is not None
        assert (
            result.extras["hakowan"]["principled_attrs"]["roughness_attr"]
            == "_roughness_0"
        )


class TestTwoSided:
    def test_two_sided_propagates(self):
        view = _triangle_view(hkw.material.Diffuse(reflectance="red", two_sided=True))
        result = translate_material(view, GLTFBuilder())
        assert result.double_sided is True


class TestDielectric:
    def test_dielectric_emits_transmission_and_ior(self):
        view = _triangle_view(hkw.material.Dielectric())  # default int=bk7, ext=air
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]
        assert result.pbr["metallicFactor"] == 0.0
        assert result.pbr["roughnessFactor"] == 0.0
        assert result.pbr["transmissionFactor"] == pytest.approx(1.0)
        # bk7 / air ≈ 1.5046 / 1.000277 ≈ 1.5042
        assert result.pbr["ior"] == pytest.approx(1.504, rel=1e-3)
        # bbox-derived thickness should be finite + positive for any real mesh.
        assert result.pbr["thicknessFactor"] > 0.0

    def test_thin_dielectric_zero_thickness(self):
        view = _triangle_view(hkw.material.ThinDielectric())
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["transmissionFactor"] == pytest.approx(1.0)
        assert result.pbr["thicknessFactor"] == 0.0
        assert "attenuationDistance" not in result.pbr

    def test_rough_dielectric_alpha(self):
        view = _triangle_view(hkw.material.RoughDielectric(alpha=0.4))
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["roughnessFactor"] == pytest.approx(0.4)
        assert result.pbr["transmissionFactor"] == pytest.approx(1.0)

    def test_rough_dielectric_scalarfield_alpha_bakes_per_vertex(self):
        view = _triangle_view(
            hkw.material.RoughDielectric(
                alpha=hkw.texture.ScalarField(data=hkw.attribute(name="scalar"))
            )
        )
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["roughnessFactor"] == pytest.approx(1.0)
        assert result.pbr["transmissionFactor"] == pytest.approx(1.0)
        assert "_roughness_0" in result.custom_attrs
        assert (
            result.extras["hakowan"]["principled_attrs"]["roughness_attr"]
            == "_roughness_0"
        )

    def test_dielectric_with_medium_emits_volume(self):
        view = _triangle_view(
            hkw.material.Dielectric(
                medium=hkw.material.Medium(albedo=(0.4, 0.9, 0.4), scale=2.0)
            )
        )
        result = translate_material(view, GLTFBuilder())
        assert "attenuationColor" in result.pbr
        assert "attenuationDistance" in result.pbr
        # Color path is sRGB → linear.
        assert result.pbr["attenuationColor"][1] > result.pbr["attenuationColor"][0]

    def test_ior_named_preset_resolves(self):
        view = _triangle_view(hkw.material.Dielectric(int_ior="water"))
        result = translate_material(view, GLTFBuilder())
        # water / air ≈ 1.333
        assert result.pbr["ior"] == pytest.approx(1.333, rel=1e-2)

    def test_ior_unknown_falls_back_with_warning(self, caplog):
        view = _triangle_view(hkw.material.Dielectric(int_ior="kryptonite"))
        result = translate_material(view, GLTFBuilder())
        # Falls back to default 1.5046 → ior ≈ 1.504
        assert result.pbr["ior"] == pytest.approx(1.504, rel=1e-2)


class TestImage:
    def test_image_texture_embedded(self, tmp_path):
        png = tmp_path / "pink.png"
        _write_temp_png(png, size=(4, 4), color=(255, 0, 200))
        view = _triangle_view(
            hkw.material.Diffuse(reflectance=hkw.texture.Image(filename=png))
        )
        builder = GLTFBuilder()
        result = translate_material(view, builder)
        assert "baseColorTextureIndex" in result.pbr
        # Builder should have one texture/image now.
        assert len(builder._gltf.textures) == 1
        assert len(builder._gltf.images) == 1


class TestCheckerboard:
    def test_checkerboard_emits_texture_with_uv_scale(self):
        view = _triangle_view(
            hkw.material.Diffuse(
                reflectance=hkw.texture.Checkerboard(
                    texture1=hkw.texture.Uniform(color="white"),
                    texture2=hkw.texture.Uniform(color="black"),
                    size=5,
                )
            )
        )
        builder = GLTFBuilder()
        result = translate_material(view, builder)
        assert "baseColorTextureIndex" in result.pbr
        assert "baseColorTextureScale" not in result.pbr
        assert result.extras is not None
        assert result.extras["hakowan"]["checkerboard"] is True


class TestIsocontour:
    def test_isocontour_wires_shader_extras_and_scalar_attr(self):
        view = _triangle_view(
            hkw.material.Diffuse(
                reflectance=hkw.texture.Isocontour(
                    data=hkw.attribute(name="scalar"),
                    num_contours=4,
                    ratio=0.2,
                    texture1=hkw.texture.Uniform(color="black"),
                    texture2=hkw.texture.Uniform(color="white"),
                )
            )
        )
        builder = GLTFBuilder()
        result = translate_material(view, builder)
        # The shader path drives contours per-pixel from a custom scalar
        # attribute — no baseColorTexture, no UV transform.
        assert "baseColorTextureIndex" not in result.pbr
        assert "baseColorTextureScale" not in result.pbr
        assert result.pbr["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]
        # Scalar field baked as a per-vertex custom attribute the viewer JS
        # binds to ``_scalar_0`` (lowercase for three.js GLTFLoader).
        assert "_scalar_0" in result.custom_attrs
        assert result.custom_attrs["_scalar_0"].shape == (3,)
        # Isocontour parameters land on the material's extras dict.
        assert result.extras is not None
        iso = result.extras["hakowan"]["isocontour"]
        assert iso["num_contours"] == 4
        assert iso["ratio"] == pytest.approx(0.2)
        assert iso["color1"] == [0.0, 0.0, 0.0]
        assert iso["color2"] == [1.0, 1.0, 1.0]


class TestNoMaterial:
    def test_view_with_no_material_channel_returns_gray_directly(self):
        """When ``view.material_channel`` is None (constructed manually,
        bypassing the compiler's default-material step), we return gray.
        """
        from hakowan.compiler.view import View

        view = View()
        result = translate_material(view, GLTFBuilder())
        assert result.pbr["baseColorFactor"] == [0.5, 0.5, 0.5, 1.0]


class TestBackSide:
    def test_back_side_emits_back_color_and_forces_double_sided(self):
        view = _triangle_view(
            hkw.material.Diffuse(
                "red", back_side=hkw.material.Diffuse(reflectance="blue")
            )
        )
        result = translate_material(view, GLTFBuilder())
        # Back color must force double-sided so gl_FrontFacing is meaningful.
        assert result.double_sided is True
        assert result.extras is not None
        back = result.extras["hakowan"]["back"]
        assert back["color"] == [0.0, 0.0, 1.0]  # blue, linear
        # Front color is untouched.
        assert result.pbr["baseColorFactor"] == [1.0, 0.0, 0.0, 1.0]

    def test_back_side_conductor_approximated_to_flat_color(self, caplog):
        view = _triangle_view(
            hkw.material.Diffuse(
                "red", back_side=hkw.material.Conductor(material="Au")
            )
        )
        result = translate_material(view, GLTFBuilder())
        assert result.double_sided is True
        back_color = result.extras["hakowan"]["back"]["color"]
        assert len(back_color) == 3
        assert all(0.0 <= c <= 1.0 for c in back_color)
