"""Tests for the WebGL backend's covariance channel support (point marks).

The covariance channel applies a per-point 3x3 stretch/rotation to the baked
base shape. We verify the matrix extraction (raw vs ``full`` SVD form) and that
``add_point_view`` actually deforms the baked geometry accordingly while keeping
the shading normals unit-length under anisotropic stretch.
"""

from __future__ import annotations

import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib")
import lagrange

import hakowan as hkw
from hakowan.backends.webgl import point_cloud as pc
from hakowan.backends.webgl.builder import GLTFBuilder


def _point_view(cov_matrices: np.ndarray, full: bool, centers: np.ndarray):
    """Compile a point-mark view with a per-vertex covariance attribute.

    ``cov_matrices`` is (N, 3, 3); it is flattened to the 9-component vertex
    attribute the channel expects.
    """
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.asarray(centers, dtype=np.float64))
    mesh.create_attribute(
        "cov",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Vector,
        initial_values=np.asarray(cov_matrices, dtype=np.float64).reshape(-1, 9),
    )
    layer = (
        hkw.layer()
        .data(mesh)
        .mark(hkw.mark.Point)
        .channel(
            covariance=hkw.channel.Covariance(data="cov", full=full),
            material=hkw.material.Diffuse(reflectance="ivory"),
        )
    )
    return list(hkw.compiler.compile(layer))[0]


class _CapturingBuilder(GLTFBuilder):
    """GLTFBuilder that records the mesh node arguments instead of emitting it."""

    def __init__(self):
        super().__init__()
        self.captured: dict = {}

    def add_mesh_node(self, positions, indices, normals=None, **kwargs):
        self.captured = {
            "positions": np.asarray(positions),
            "normals": None if normals is None else np.asarray(normals),
        }
        return 0


class TestCovarianceMatrices:
    def test_raw_form_returns_matrix_directly(self):
        m = np.diag([3.0, 1.0, 1.0])
        cov = np.stack([m, np.eye(3)])
        view = _point_view(cov, full=False, centers=[[0, 0, 0], [10, 0, 0]])
        out = pc._covariance_matrices(view, 2)
        np.testing.assert_allclose(out[0], m, atol=1e-6)

    def test_full_form_takes_matrix_square_root(self):
        # Sigma = diag(9,4,1) -> M = diag(3,2,1) so that M @ M^T == Sigma.
        sigma = np.diag([9.0, 4.0, 1.0])
        cov = np.stack([sigma, np.eye(3)])
        view = _point_view(cov, full=True, centers=[[0, 0, 0], [10, 0, 0]])
        out = pc._covariance_matrices(view, 2)
        recovered = out[0] @ out[0].T
        np.testing.assert_allclose(recovered, sigma, atol=1e-5)

    def test_no_covariance_returns_none(self):
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertices(np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64))
        view = list(
            hkw.compiler.compile(hkw.layer().data(mesh).mark(hkw.mark.Point))
        )[0]
        assert pc._covariance_matrices(view, 2) is None


class TestCovarianceGeometry:
    def _bake(self, cov, full):
        # Two well-separated points so the scene bbox is non-degenerate.
        view = _point_view(cov, full=full, centers=[[0, 0, 0], [10, 0, 0]])
        builder = _CapturingBuilder()
        pc.add_point_view(builder, view)
        return builder.captured

    def test_anisotropic_stretch_deforms_first_point(self):
        # Point 0 stretched 3x along x; point 1 left as identity.
        cov = np.stack([np.diag([3.0, 1.0, 1.0]), np.eye(3)])
        cap = self._bake(cov, full=False)
        n_base = pc._icosphere(refinement=0)[0].shape[0]  # 12 sphere vertices
        p0 = cap["positions"][:n_base]
        ext = p0.max(0) - p0.min(0)
        # Uniform scene normalization preserves the x:y extent ratio (~3:1).
        assert ext[0] / ext[1] == pytest.approx(3.0, rel=1e-3)

    def test_normals_unit_length_under_stretch(self):
        cov = np.stack([np.diag([3.0, 0.5, 1.0]), np.eye(3)])
        cap = self._bake(cov, full=False)
        lengths = np.linalg.norm(cap["normals"], axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-4)
