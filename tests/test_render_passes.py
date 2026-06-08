"""Render-pass registry and per-backend capability declaration."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import lagrange

import hakowan as hkw
from hakowan.setup import render_pass as rp
from hakowan.backends.webgl.render import WebGLBackend


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
