"""The facet-ID pass is Blender-only; other backends must warn, not pretend."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import lagrange

import hakowan as hkw


@pytest.fixture
def single_triangle():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.eye(3))
    mesh.add_triangle(0, 1, 2)
    return mesh


def _render_webgl(mesh, facet_id, tmp_path):
    config = hkw.config()
    config.facet_id = facet_id
    layer = hkw.layer().data(mesh).mark(hkw.mark.Surface)
    hkw.render(
        layer, config, filename=tmp_path / "out.html", backend="webgl"
    )


def test_warns_when_facet_id_on_unsupported_backend(single_triangle, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="hakowan"):
        _render_webgl(single_triangle, True, tmp_path)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("facet_id" in r.message and "webgl" in r.message for r in warnings), (
        f"expected a facet_id/webgl warning, got: {[r.message for r in warnings]}"
    )


def test_no_warning_without_facet_id(single_triangle, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="hakowan"):
        _render_webgl(single_triangle, False, tmp_path)
    assert not any(
        "facet_id" in r.message for r in caplog.records
    ), "facet_id warning emitted even though facet_id was disabled"
