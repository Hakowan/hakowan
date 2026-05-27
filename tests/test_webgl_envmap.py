"""Tests for hakowan.backends.webgl.envmap rotation/up handling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hakowan.backends.webgl.envmap import (
    _align_y_to,
    _build_rotation_matrix,
    _rotate_y,
    envmap_descriptor,
)
from hakowan.setup import Config
from hakowan.setup.emitter import Envmap


class TestRotateY:
    def test_zero_is_identity(self):
        np.testing.assert_allclose(_rotate_y(0.0), np.eye(3), atol=1e-12)

    def test_quarter_turn_maps_x_to_neg_z(self):
        R = _rotate_y(math.pi / 2)
        # Right-handed rotation around +Y maps +X → -Z, +Z → +X.
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 1, 0], atol=1e-12)


class TestAlignYTo:
    def test_aligning_to_y_is_identity(self):
        np.testing.assert_allclose(
            _align_y_to(np.array([0, 1, 0])), np.eye(3), atol=1e-12
        )

    def test_aligning_to_negative_y_flips(self):
        R = _align_y_to(np.array([0, -1, 0]))
        np.testing.assert_allclose(R @ [0, 1, 0], [0, -1, 0], atol=1e-9)

    def test_aligning_to_z_rotates_y_onto_z(self):
        R = _align_y_to(np.array([0, 0, 1]))
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-9)

    def test_aligning_to_negative_z(self):
        R = _align_y_to(np.array([0, 0, -1]))
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, -1], atol=1e-9)

    def test_aligning_to_x(self):
        R = _align_y_to(np.array([1, 0, 0]))
        np.testing.assert_allclose(R @ [0, 1, 0], [1, 0, 0], atol=1e-9)

    def test_non_unit_up_is_normalised(self):
        R = _align_y_to(np.array([0, 0, 5.0]))
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-9)

    def test_zero_up_returns_identity(self):
        R = _align_y_to(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


class TestBuildRotationMatrix:
    def test_default_y_up_180_rotation(self):
        """Default config: rotation=180°, up=[0,1,0] → 180° around Y."""
        R = _build_rotation_matrix(180.0, [0, 1, 0])
        np.testing.assert_allclose(R @ [1, 0, 0], [-1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-9)
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 1, 0], atol=1e-9)

    def test_z_up_rotation_180(self):
        """Z-up config (up=[0,0,1]) with rotation=180° → align Y to Z then
        rotate 180° around the original Y. The combined behaviour is what
        Mitsuba's ``config.z_up()`` ships with."""
        R = _build_rotation_matrix(180.0, [0, 0, 1])
        # The 180°-around-Y first flips X and Z; the alignment then maps Y→Z
        # and (−Z)→Y. So:
        np.testing.assert_allclose(R @ [1, 0, 0], [-1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(R @ [0, 0, 1], [0, 1, 0], atol=1e-9)
        # Compare with the pure-alignment (rotation=0) case to witness that
        # the 180° is actually composed in: X flips, Z is sent to +Y not −Y.
        R0 = _build_rotation_matrix(0.0, [0, 0, 1])
        np.testing.assert_allclose(R0 @ [1, 0, 0], [1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(R0 @ [0, 0, 1], [0, -1, 0], atol=1e-9)
        assert not np.allclose(R, R0)

    def test_zero_rotation_z_up_pure_alignment(self):
        R = _build_rotation_matrix(0.0, [0, 0, 1])
        # No Y-rotation, just alignment.
        np.testing.assert_allclose(R, _align_y_to(np.array([0, 0, 1])), atol=1e-12)

    def test_zero_rotation_default_up_is_identity(self):
        R = _build_rotation_matrix(0.0, [0, 1, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_composed_order_matches_mitsuba(self):
        """Mitsuba composes as ``align(up) @ rotate_y(deg)``; verify the
        order matters for non-degenerate cases."""
        R_correct = _build_rotation_matrix(90.0, [0, 0, 1])
        R_wrong = _rotate_y(math.radians(90.0)) @ _align_y_to(np.array([0, 0, 1]))
        # They differ — composition is non-commutative for non-trivial cases.
        assert not np.allclose(R_correct, R_wrong)


class TestEnvmapDescriptor:
    def test_no_envmap_emitter_returns_none(self):
        cfg = Config()
        cfg.emitters = []
        assert envmap_descriptor(cfg) is None

    def test_default_envmap_descriptor_has_rotation_matrix(self):
        cfg = Config()  # has default Envmap() with rotation=180, up=Y.
        desc = envmap_descriptor(cfg)
        assert desc is not None
        assert "rotation_matrix" in desc
        assert len(desc["rotation_matrix"]) == 9
        # Matches the 180° around Y case above.
        R = np.array(desc["rotation_matrix"]).reshape(3, 3)
        np.testing.assert_allclose(R @ [1, 0, 0], [-1, 0, 0], atol=1e-9)

    def test_descriptor_reflects_custom_up(self):
        cfg = Config()
        cfg.z_up()  # sets envmap up=[0,0,1], rotation=180
        desc = envmap_descriptor(cfg)
        assert desc is not None
        R = np.array(desc["rotation_matrix"]).reshape(3, 3)
        # Same as the z-up case in TestBuildRotationMatrix.
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-9)

    def test_missing_envmap_file_returns_none(self, tmp_path):
        cfg = Config()
        cfg.emitters = [Envmap(filename=tmp_path / "does_not_exist.exr")]
        assert envmap_descriptor(cfg) is None

    def test_unsupported_format_returns_none(self, tmp_path):
        bogus = tmp_path / "envmap.png"
        bogus.write_bytes(b"\x89PNG\r\n")
        cfg = Config()
        cfg.emitters = [Envmap(filename=bogus)]
        assert envmap_descriptor(cfg) is None
