"""Tests for hakowan.backends.webgl.camera."""

from __future__ import annotations

import math

import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib")

from hakowan.backends.webgl.builder import GLTFBuilder
from hakowan.backends.webgl.camera import _yfov_radians, add_camera
from hakowan.setup.sensor import Orthographic, Perspective, ThinLens
from hakowan.setup import Config


def _config_with_sensor(sensor, width=1024, height=512):
    cfg = Config()
    cfg.sensor = sensor
    cfg.film.width = width
    cfg.film.height = height
    return cfg


class TestYFovComputation:
    def test_fov_y_axis_passes_through(self):
        sensor = Perspective(fov=30.0, fov_axis="y")
        yfov = _yfov_radians(sensor, aspect=16 / 9)
        assert yfov == pytest.approx(math.radians(30.0))

    def test_fov_x_axis_inverts_with_aspect(self):
        sensor = Perspective(fov=60.0, fov_axis="x")
        # half_x = tan(30°); half_y = half_x / aspect
        yfov = _yfov_radians(sensor, aspect=2.0)
        expected = 2.0 * math.atan(math.tan(math.radians(30.0)) / 2.0)
        assert yfov == pytest.approx(expected, rel=1e-6)

    def test_fov_smaller_axis_when_landscape(self):
        sensor = Perspective(fov=45.0, fov_axis="smaller")
        # aspect = w/h = 2 (landscape) → height is smaller → yfov = fov
        yfov_landscape = _yfov_radians(sensor, aspect=2.0)
        assert yfov_landscape == pytest.approx(math.radians(45.0))

    def test_fov_smaller_axis_when_portrait(self):
        sensor = Perspective(fov=45.0, fov_axis="smaller")
        # aspect = 0.5 (portrait) → width is smaller → fov along x
        yfov_portrait = _yfov_radians(sensor, aspect=0.5)
        expected = 2.0 * math.atan(math.tan(math.radians(22.5)) / 0.5)
        assert yfov_portrait == pytest.approx(expected, rel=1e-6)

    def test_diagonal_axis(self):
        sensor = Perspective(fov=60.0, fov_axis="diagonal")
        # diag_tan = tan(30°); y_tan = diag_tan / sqrt(1+a²)
        aspect = 16 / 9
        yfov = _yfov_radians(sensor, aspect)
        expected = 2.0 * math.atan(
            math.tan(math.radians(30.0)) / math.sqrt(1 + aspect**2)
        )
        assert yfov == pytest.approx(expected, rel=1e-6)


class TestAddCamera:
    def test_perspective_camera_registered(self):
        b = GLTFBuilder()
        cfg = _config_with_sensor(Perspective(fov=45.0))
        node_idx, initial_view = add_camera(b, cfg)
        assert node_idx >= 0
        assert len(b._gltf.cameras) == 1
        assert b._gltf.cameras[0].type == "perspective"
        assert initial_view["eye"] == list(cfg.sensor.location)
        assert initial_view["target"] == list(cfg.sensor.target)

    def test_orthographic_camera_registered(self):
        b = GLTFBuilder()
        cfg = _config_with_sensor(Orthographic())
        add_camera(b, cfg)
        assert b._gltf.cameras[0].type == "orthographic"

    def test_thinlens_degrades_to_perspective(self):
        b = GLTFBuilder()
        cfg = _config_with_sensor(ThinLens(fov=30.0))
        add_camera(b, cfg)
        assert b._gltf.cameras[0].type == "perspective"
