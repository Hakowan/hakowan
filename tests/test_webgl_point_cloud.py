"""Tests for hakowan.backends.webgl.point_cloud base shapes & orientation."""

from __future__ import annotations

import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib")

from hakowan.backends.webgl.point_cloud import (
    _cube,
    _disk,
    _icosphere,
    _rotation_matrix_z_to,
)


class TestIcosphere:
    def test_unit_radius(self):
        positions, normals, tris = _icosphere(refinement=0)
        # 12 base vertices.
        assert positions.shape == (12, 3)
        radii = np.linalg.norm(positions, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-5)

    def test_normals_equal_positions(self):
        positions, normals, _ = _icosphere(refinement=0)
        np.testing.assert_allclose(normals, positions, atol=1e-7)

    def test_refinement_subdivides(self):
        _, _, tris_0 = _icosphere(refinement=0)
        _, _, tris_1 = _icosphere(refinement=1)
        assert tris_1.shape[0] == tris_0.shape[0] * 4


class TestDisk:
    def test_disk_face_in_xy_plane(self):
        positions, normals, tris = _disk(segments=8)
        assert positions[0].tolist() == [0.0, 0.0, 0.0]  # centre
        # Ring vertices on the unit circle in z=0 plane.
        ring = positions[1:]
        np.testing.assert_allclose(ring[:, 2], 0.0, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(ring[:, :2], axis=1), 1.0, atol=1e-7)
        # Normals all +Z.
        np.testing.assert_allclose(normals[:, 2], 1.0, atol=1e-7)

    def test_disk_triangle_count(self):
        _, _, tris = _disk(segments=8)
        assert tris.shape == (8, 3)


class TestCube:
    def test_cube_has_24_vertices(self):
        positions, normals, tris = _cube()
        assert positions.shape == (24, 3)  # 6 faces × 4 corners
        assert tris.shape == (12, 3)  # 6 faces × 2 triangles
        # Each position is at extents ±1.
        assert np.all(np.abs(positions) == 1.0)

    def test_cube_normals_axis_aligned(self):
        _, normals, _ = _cube()
        # Each normal should have exactly one non-zero component (±1).
        for n in normals:
            nz = np.count_nonzero(n)
            assert nz == 1


class TestRotationMatrix:
    def test_z_to_z_is_identity(self):
        R = _rotation_matrix_z_to(np.array([0, 0, 1]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_z_to_neg_z_handles_flip(self):
        R = _rotation_matrix_z_to(np.array([0, 0, -1]))
        # Should map +Z to -Z (i.e. R @ [0,0,1] ≈ [0,0,-1])
        result = R @ np.array([0, 0, 1])
        np.testing.assert_allclose(result, [0, 0, -1], atol=1e-6)

    def test_z_to_x_rotates(self):
        R = _rotation_matrix_z_to(np.array([1.0, 0.0, 0.0]))
        result = R @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)

    def test_z_to_zero_returns_identity(self):
        R = _rotation_matrix_z_to(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
