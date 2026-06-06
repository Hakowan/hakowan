"""Regression tests for hakowan.backends.webgl.mesh_extract."""

from __future__ import annotations

import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib")
import lagrange

import hakowan as hkw
from hakowan.backends.webgl.mesh_extract import extract_surface_arrays


def _quad() -> lagrange.SurfaceMesh:
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    )
    mesh.add_triangles(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32))
    return mesh


def _surface_arrays(mesh, material):
    layer = hkw.layer().data(mesh).mark(hkw.mark.Surface).channel(material=material)
    return extract_surface_arrays(list(hkw.compiler.compile(layer))[0])


class TestFacetColor:
    """A per-facet ScalarField color must not crash and must color per-face.

    Regression: previously the facet-length color array was indexed by vertex
    ids (IndexError) or handed to the builder with a vertex-count mismatch.
    """

    def test_facet_scalarfield_color_does_not_crash_and_is_per_face(self):
        mesh = _quad()
        mesh.create_attribute(
            "fscalar",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.array([0.0, 1.0], dtype=np.float64),
        )
        arrays = _surface_arrays(
            mesh,
            hkw.material.Diffuse(
                reflectance=hkw.texture.ScalarField(data="fscalar", colormap="viridis")
            ),
        )
        colors = arrays["colors"]
        positions = arrays["positions"]
        # De-indexed per-corner layout: 2 triangles × 3 corners.
        assert colors is not None
        assert colors.shape == (6, 4)
        assert positions.shape == (6, 3)
        # Each triangle's three corners share one facet color.
        np.testing.assert_allclose(colors[0], colors[1])
        np.testing.assert_allclose(colors[1], colors[2])
        np.testing.assert_allclose(colors[3], colors[4])
        np.testing.assert_allclose(colors[4], colors[5])
        # The two facets carry different colors (0.0 vs 1.0 on viridis).
        assert not np.allclose(colors[0], colors[3])

    def test_vertex_scalarfield_color_still_per_vertex_layout(self):
        mesh = _quad()
        mesh.create_attribute(
            "vscalar",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.array([0.0, 0.3, 0.6, 1.0], dtype=np.float64),
        )
        arrays = _surface_arrays(
            mesh,
            hkw.material.Diffuse(
                reflectance=hkw.texture.ScalarField(data="vscalar", colormap="viridis")
            ),
        )
        # Colors line up with the emitted vertex/corner positions.
        assert arrays["colors"] is not None
        assert arrays["colors"].shape[0] == arrays["positions"].shape[0]


class TestNormalChannel:
    """Custom Normal channel overrides auto-computed normals."""

    def test_custom_per_vertex_normals_used(self):
        mesh = _quad()
        custom = np.array(
            [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32
        )
        custom /= np.linalg.norm(custom, axis=1, keepdims=True)
        mesh.create_attribute(
            "my_normals",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Normal,
            initial_values=custom.astype(np.float64),
        )
        layer = (
            hkw.layer()
            .data(mesh)
            .mark(hkw.mark.Surface)
            .channel(normal="my_normals")
        )
        view = list(hkw.compiler.compile(layer))[0]
        arrays = extract_surface_arrays(view)
        normals = arrays["normals"]
        assert normals is not None
        assert normals.shape == (4, 3)
        np.testing.assert_allclose(normals, custom, atol=1e-5)
