"""Render-pass registry and per-backend capability declaration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import lagrange

import hakowan as hkw
from hakowan.setup import render_pass as rp
from hakowan.backends import RenderBackend
from hakowan.backends.webgl.render import WebGLBackend
from hakowan.render import _manifest_for


class TestRegistry:
    def test_known_passes(self):
        assert set(rp.RENDER_PASSES) == {"albedo", "depth", "normal", "facet_id"}

    def test_descriptor_metadata(self):
        assert rp.ALBEDO.channels == 3 and rp.ALBEDO.mitsuba_aov == "albedo:albedo"
        assert rp.DEPTH.channels == 1 and rp.DEPTH.mitsuba_aov == "depth:depth"
        assert rp.NORMAL.mitsuba_aov == "sh_normal:sh_normal"
        # facet_id is a discrete (lossless) pass with no Mitsuba AOV.
        assert rp.FACET_ID.discrete is True
        assert rp.FACET_ID.mitsuba_aov is None
        assert rp.ALBEDO.discrete is False

    def test_get_render_pass_unknown(self):
        with pytest.raises(ValueError, match="Unknown render pass"):
            rp.get_render_pass("bogus")


class TestCapabilities:
    def test_webgl_supports_continuous_not_facet_id(self):
        names = {p.name for p in WebGLBackend.SUPPORTED_PASSES}
        assert names == {"albedo", "depth", "normal"}
        assert rp.FACET_ID not in WebGLBackend.SUPPORTED_PASSES

    def test_webgl_delivers_passes_interactively(self):
        assert WebGLBackend.PASS_DELIVERY == "interactive"


class TestAovPath:
    def test_with_descriptor(self):
        assert rp.aov_path("out.png", rp.ALBEDO) == Path("out_albedo.png")

    def test_with_name(self):
        assert rp.aov_path("out.png", "facet_id") == Path("out_facet_id.png")

    def test_preserves_directory_and_suffix(self):
        assert rp.aov_path(Path("/a/b/bust.exr"), rp.DEPTH) == Path("/a/b/bust_depth.exr")


# A minimal file-delivery backend so manifest behavior can be tested without
# importing a heavy backend (mitsuba/bpy/pygltflib).
class _StubFileBackend(RenderBackend):
    SUPPORTED_PASSES = frozenset({rp.ALBEDO, rp.DEPTH, rp.NORMAL})
    PASS_DELIVERY = "file"

    def render(self, scene, config, filename=None, **kwargs):  # pragma: no cover
        return None


class TestManifest:
    def test_file_backend_maps_passes_to_paths(self):
        cfg = hkw.config()
        cfg.render_passes = {"albedo", "normal"}
        m = _manifest_for(_StubFileBackend(), cfg, "viz.png")
        assert m["main"] == Path("viz.png")
        assert m["albedo"] == Path("viz_albedo.png")
        assert m["normal"] == Path("viz_normal.png")
        assert "depth" not in m  # not requested

    def test_unsupported_pass_omitted_from_manifest(self):
        cfg = hkw.config()
        cfg.render_passes = {"albedo", "facet_id"}  # facet_id unsupported here
        m = _manifest_for(_StubFileBackend(), cfg, "viz.png")
        assert "facet_id" not in m
        assert m["albedo"] == Path("viz_albedo.png")

    def test_none_filename_yields_empty_manifest(self):
        cfg = hkw.config()
        cfg.render_passes = {"albedo"}
        assert _manifest_for(_StubFileBackend(), cfg, None) == {}

    def test_interactive_backend_marks_passes(self):
        cfg = hkw.config()
        cfg.render_passes = {"albedo", "depth"}
        m = _manifest_for(WebGLBackend(), cfg, "viz.html")
        assert m["main"] == Path("viz.html")
        assert m["albedo"] == "interactive"
        assert m["depth"] == "interactive"


@pytest.fixture
def single_triangle_rr():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.eye(3))
    mesh.add_triangle(0, 1, 2)
    return mesh


class TestRenderResult:
    def test_webgl_render_returns_result(self, single_triangle_rr, tmp_path):
        cfg = hkw.config()
        cfg.render_passes = {"albedo"}
        layer = hkw.layer().data(single_triangle_rr).mark(hkw.mark.Surface)
        out = tmp_path / "viz.html"
        result = hkw.render(layer, cfg, filename=out, backend="webgl")

        assert isinstance(result, hkw.RenderResult)
        assert result.backend == "webgl"
        assert result.path == out
        assert result.image is None  # webgl has no in-memory image
        assert result.outputs["main"] == out
        assert result.outputs["albedo"] == "interactive"

    def test_result_is_path_like(self, single_triangle_rr, tmp_path):
        layer = hkw.layer().data(single_triangle_rr).mark(hkw.mark.Surface)
        out = tmp_path / "viz.html"
        result = hkw.render(layer, hkw.config(), filename=out, backend="webgl")
        # __fspath__ lets the result stand in for its main output path.
        from pathlib import Path as _P

        assert _P(result) == out
        assert result.path.read_bytes() == _P(result).read_bytes()

    def test_fspath_raises_without_filename(self):
        result = hkw.RenderResult(backend="webgl")
        with pytest.raises(TypeError, match="no output path"):
            _ = result.__fspath__()


@pytest.fixture
def single_triangle():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.eye(3))
    mesh.add_triangle(0, 1, 2)
    return mesh


def _render_webgl(mesh, passes, tmp_path):
    config = hkw.config()
    config.render_passes = passes
    layer = hkw.layer().data(mesh).mark(hkw.mark.Surface)
    hkw.render(layer, config, filename=tmp_path / "out.html", backend="webgl")


class TestCapabilityWarning:
    def test_supported_pass_does_not_warn(self, single_triangle, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="hakowan"):
            _render_webgl(single_triangle, {"albedo", "depth", "normal"}, tmp_path)
        assert not any(
            "does not support render pass" in r.message for r in caplog.records
        ), "warned about passes the webgl backend actually supports"

    def test_unsupported_pass_warns_with_name(self, single_triangle, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="hakowan"):
            _render_webgl(single_triangle, {"albedo", "facet_id"}, tmp_path)
        msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        # Only facet_id is unsupported; albedo is supported and must not appear
        # in the unsupported list (it may appear in the "Supported passes" tail).
        assert any(
            "does not support render pass(es) ['facet_id']" in m and "webgl" in m
            for m in msgs
        ), f"expected an unsupported-pass warning listing only facet_id, got: {msgs}"
