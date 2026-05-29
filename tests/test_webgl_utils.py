"""Tests for hakowan.backends.webgl.utils."""

from __future__ import annotations

import base64

import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib")

from hakowan.backends.webgl.utils import (
    glb_to_data_uri,
    gltf_matrix,
    look_at,
    np_to_bytes,
)


class TestNpToBytes:
    def test_float32_roundtrip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        data = np_to_bytes(arr)
        # 2*2 floats * 4 bytes = 16
        assert len(data) == 16
        recovered = np.frombuffer(data, dtype=np.float32).reshape(2, 2)
        np.testing.assert_array_equal(arr, recovered)

    def test_non_contiguous_input(self):
        """Strided slices are made contiguous on the way out."""
        big = np.arange(12, dtype=np.float32).reshape(3, 4)
        sliced = big[:, ::2]  # non-contiguous
        data = np_to_bytes(sliced)
        # 3*2 floats * 4 bytes = 24
        assert len(data) == 24


class TestGlbToDataUri:
    def test_data_uri_prefix_and_decodes(self):
        glb = b"glTF dummy bytes \x00\x01\x02"
        uri = glb_to_data_uri(glb)
        assert uri.startswith("data:model/gltf-binary;base64,")
        payload = uri.split(",", 1)[1]
        assert base64.b64decode(payload) == glb


class TestLookAt:
    def test_identity_when_eye_at_origin_looking_neg_z(self):
        m = look_at([0, 0, 0], [0, 0, -1], [0, 1, 0])
        # Camera at origin, local -Z = world -Z → x_axis = world +X, y = +Y
        np.testing.assert_allclose(m[:3, 0], [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(m[:3, 1], [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(m[:3, 2], [0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(m[:3, 3], [0, 0, 0], atol=1e-7)

    def test_position_set_to_eye(self):
        m = look_at([5, 6, 7], [0, 0, 0], [0, 1, 0])
        np.testing.assert_allclose(m[:3, 3], [5, 6, 7], atol=1e-7)

    def test_degenerate_zero_forward_doesnt_crash(self):
        m = look_at([1, 1, 1], [1, 1, 1], [0, 1, 0])
        assert m.shape == (4, 4)
        assert np.all(np.isfinite(m))

    def test_up_parallel_to_forward_picks_fallback(self):
        # Looking along +Y with up = +Y → degenerate, should not return NaNs.
        m = look_at([0, -5, 0], [0, 0, 0], [0, 1, 0])
        assert np.all(np.isfinite(m))


class TestGltfMatrix:
    def test_column_major_flatten(self):
        m = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.float64,
        )
        flat = gltf_matrix(m)
        # glTF wants column-major: [m00, m10, m20, m30, m01, m11, ...]
        assert flat == [
            1, 5, 9, 13,
            2, 6, 10, 14,
            3, 7, 11, 15,
            4, 8, 12, 16,
        ]
