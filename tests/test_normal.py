"""Tests that custom mesh normals are honored consistently across backends.

A flat quad in the ``z = 0`` plane is used so the geometric normal is always
``+z``.  By attaching *bogus* normals (e.g. ``+x``) that differ from the
geometric one, we can tell whether a backend actually transports the authored
normal values or silently recomputes them from the geometry.
"""

import os
import sys

import numpy as np
import pytest
import lagrange

import hakowan as hkw

X = [1.0, 0.0, 0.0]
Y = [0.0, 1.0, 0.0]
Z = [0.0, 0.0, 1.0]


def _quad() -> lagrange.SurfaceMesh:
    """Unit quad in the z=0 plane, two triangles, geometric normal +z."""
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    )
    # tri0 = (0,0),(1,0),(1,1) [lower-right]; tri1 = (0,0),(1,1),(0,1) [upper-left]
    mesh.add_triangles(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32))
    return mesh


def _quad_vertex_normal(value=X) -> lagrange.SurfaceMesh:
    mesh = _quad()
    mesh.create_attribute(
        "N",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Normal,
        initial_values=np.tile(value, (4, 1)).astype(np.float64),
    )
    return mesh


def _quad_facet_normal(value0=X, value1=Y) -> lagrange.SurfaceMesh:
    mesh = _quad()
    mesh.create_attribute(
        "N",
        element=lagrange.AttributeElement.Facet,
        usage=lagrange.AttributeUsage.Normal,
        initial_values=np.array([value0, value1], dtype=np.float64),
    )
    return mesh


def _quad_corner_normal(value=Y) -> lagrange.SurfaceMesh:
    mesh = _quad()
    mesh.create_attribute(
        "N",
        element=lagrange.AttributeElement.Corner,
        usage=lagrange.AttributeUsage.Normal,
        initial_values=np.tile(value, (6, 1)).astype(np.float64),
    )
    return mesh


def _multiset(arr: np.ndarray) -> list:
    """Sorted list of rounded rows — order-independent comparison of normals."""
    return sorted(tuple(np.round(row, 3)) for row in np.asarray(arr).reshape(-1, 3))


# ---------------------------------------------------------------------------
# Mitsuba
# ---------------------------------------------------------------------------


class TestMitsubaNormal:
    @staticmethod
    def _vertex_normals(mesh_in) -> np.ndarray:
        """Compile + serialize to ply, reload through Mitsuba, return its normals."""
        mi = pytest.importorskip("mitsuba")
        if mi.variant() is None:
            mi.set_variant("scalar_rgb")
        from hakowan.backends.mitsuba.render import generate_scene_config

        scene = hkw.compiler.compile(
            hkw.layer().data(mesh_in).mark(hkw.mark.Surface)
        )
        cfg = generate_scene_config(scene)
        assert len(cfg) == 1
        _, shape = next(iter(cfg.items()))
        # Normals now always travel as per-vertex ply normals.
        assert shape["face_normals"] is False
        s = mi.load_dict(
            {
                "type": "ply",
                "filename": shape["filename"],
                "face_normals": shape["face_normals"],
            }
        )
        return np.array(mi.traverse(s)["vertex_normals"]).reshape(-1, 3)

    def test_vertex_normal_preserved(self):
        n = self._vertex_normals(_quad_vertex_normal(X))
        assert np.allclose(n, X, atol=1e-5)

    def test_facet_normal_preserved(self):
        # Each face keeps its own normal; vertices are duplicated across the
        # crease so both +x and +y survive (3 vertices each).
        n = self._vertex_normals(_quad_facet_normal(X, Y))
        assert _multiset(n) == sorted([tuple(X)] * 3 + [tuple(Y)] * 3)

    def test_corner_normal_preserved(self):
        n = self._vertex_normals(_quad_corner_normal(Y))
        assert np.allclose(n, Y, atol=1e-5)

    def test_no_normal_uses_geometric(self):
        # Without an authored normal, Mitsuba computes the geometric (+z) normal.
        n = self._vertex_normals(_quad())
        assert np.allclose(n, Z, atol=1e-5)


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32" and os.environ.get("CI") == "true",
    reason="bpy crashes on Windows CI runners",
)
class TestBlenderNormal:
    @staticmethod
    def _loop_normals(mesh_in) -> np.ndarray:
        """Build the Blender surface object and return per-loop split normals."""
        bpy = pytest.importorskip("bpy")
        from hakowan.backends.blender.render import BlenderBackend

        scene = hkw.compiler.compile(
            hkw.layer().data(mesh_in).mark(hkw.mark.Surface)
        )
        backend = BlenderBackend()
        backend._clear_scene()
        backend._create_surface_object(list(scene)[0], 0)
        mesh = bpy.data.meshes["mesh_000"]
        # Loops are ordered tri0 (3) then tri1 (3); Blender quantizes normals,
        # hence the loose tolerance in the assertions below.
        return np.array([tuple(loop.normal) for loop in mesh.loops])

    def test_vertex_normal_preserved(self):
        n = self._loop_normals(_quad_vertex_normal(X))
        assert np.allclose(n, X, atol=1e-3)

    def test_facet_normal_preserved(self):
        n = self._loop_normals(_quad_facet_normal(X, Y))
        assert np.allclose(n[:3], X, atol=1e-3)  # tri0 loops
        assert np.allclose(n[3:], Y, atol=1e-3)  # tri1 loops

    def test_corner_normal_preserved(self):
        n = self._loop_normals(_quad_corner_normal(Y))
        assert np.allclose(n, Y, atol=1e-3)

    def test_no_normal_uses_geometric(self):
        n = self._loop_normals(_quad())
        assert np.allclose(n, Z, atol=1e-3)


# ---------------------------------------------------------------------------
# WebGL
# ---------------------------------------------------------------------------


class TestWebglNormal:
    @staticmethod
    def _normals(mesh_in) -> np.ndarray:
        from hakowan.backends.webgl.mesh_extract import extract_surface_arrays

        scene = hkw.compiler.compile(
            hkw.layer().data(mesh_in).mark(hkw.mark.Surface)
        )
        arrays = extract_surface_arrays(list(scene)[0])
        assert arrays["normals"] is not None
        return np.asarray(arrays["normals"])

    def test_vertex_normal_preserved(self):
        n = self._normals(_quad_vertex_normal(X))
        assert np.allclose(n, X, atol=1e-5)

    def test_facet_normal_preserved(self):
        # Facet normals expand to per-corner; first triangle's 3 corners are +x,
        # second triangle's 3 corners are +y.
        n = self._normals(_quad_facet_normal(X, Y))
        assert _multiset(n) == sorted([tuple(X)] * 3 + [tuple(Y)] * 3)

    def test_corner_normal_preserved(self):
        n = self._normals(_quad_corner_normal(Y))
        assert np.allclose(n, Y, atol=1e-5)

    def test_no_normal_uses_geometric(self):
        n = self._normals(_quad())
        assert np.allclose(n, Z, atol=1e-5)
