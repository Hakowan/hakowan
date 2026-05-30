import pytest
from hakowan import transform, scale
from hakowan.compiler.transform import principal_axes_affine_matrix
from hakowan.compiler.streamline import _compute_streamlines
import copy
import lagrange
import numpy as np


class TestTransform:
    def test_filter(self):
        attr = scale.Attribute(name="index")
        t = transform.Filter(data=attr, condition=lambda x: True)
        assert t.data is attr
        assert t.condition(0)
        assert t._child is None

    def test_chaining_and_copy(self):
        attr0 = scale.Attribute(name="index")
        t0 = transform.Filter(data=attr0, condition=lambda x: True)
        attr1 = scale.Attribute(name="curvature")
        t1 = transform.Filter(data=attr1, condition=lambda x: True)
        t1 *= t0

        assert t1.data is attr1
        assert t1._child.data == attr0

        t2 = copy.deepcopy(t1)
        assert t2 is not t1
        assert t2.data is not t1.data
        assert t2._child is not t1._child
        assert t2._child.data is not t1._child.data

    def test_uv_mesh(self):
        t = transform.UVMesh(uv="@uv")
        assert t.uv == "@uv"
        assert t._child is None

    def test_affine(self):
        t = transform.Affine(matrix=np.eye(4))
        assert np.all(t.matrix == np.eye(4))
        assert t._child is None

    def test_principal_axes(self):
        t = transform.PrincipalAxes(frame=np.eye(3), orthonormalize_frame=False)
        assert np.allclose(t.frame, np.eye(3))
        assert t.orthonormalize_frame is False
        assert t._child is None

    def test_principal_axes_affine_matrix_maps_major_to_frame_column0(self):
        v = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0]])
        frame = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        m = principal_axes_affine_matrix(v, frame, orthonormalize_frame=False)
        w = (m[:3, :3] @ v.T).T + m[:3, 3]
        assert w.std(0)[1] > max(w.std(0)[0], w.std(0)[2])

    def test_principal_axes_affine_matrix_single_vertex(self):
        # n < 2: identity rotation, translate centroid to origin
        v = np.array([[1.0, 2.0, 3.0]])
        m = principal_axes_affine_matrix(v, np.eye(3))
        assert np.allclose(m[:3, :3], np.eye(3))
        assert np.allclose(m[:3, 3], -v[0])

    def test_streamline_grammar(self):
        t = transform.Streamline(vec_field="velocity", n=10, cross_field=False)
        assert t.vec_field == "velocity"
        assert t.n == 10
        assert t.cross_field is False
        assert t.length is None
        assert t.id_attr_name == "_hakowan_streamline_id"
        assert t._child is None


class TestStreamlineCompiler:
    def _make_grid_mesh(self, attr_name="vec", with_attr=True):
        # Simple two-triangle grid w/ optional per-facet vector field.
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertex([0.0, 0.0, 0.0])
        mesh.add_vertex([1.0, 0.0, 0.0])
        mesh.add_vertex([1.0, 1.0, 0.0])
        mesh.add_vertex([0.0, 1.0, 0.0])
        mesh.add_triangle(0, 1, 2)
        mesh.add_triangle(0, 2, 3)
        if with_attr:
            mesh.create_attribute(
                attr_name,
                element=lagrange.AttributeElement.Facet,
                usage=lagrange.AttributeUsage.Vector,
                initial_values=np.array(
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
                ),
            )
        return mesh

    def test_missing_attribute_raises(self):
        mesh = self._make_grid_mesh(with_attr=False)
        with pytest.raises(ValueError, match="no attribute"):
            _compute_streamlines(mesh, "missing_attr")

    def test_non_triangle_mesh_raises(self):
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertex([0.0, 0.0, 0.0])
        mesh.add_vertex([1.0, 0.0, 0.0])
        mesh.add_vertex([1.0, 1.0, 0.0])
        mesh.add_vertex([0.0, 1.0, 0.0])
        mesh.add_polygon(np.array([0, 1, 2, 3], dtype=np.uint32))
        mesh.create_attribute(
            "vec",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        )
        with pytest.raises(ValueError, match="triangle mesh"):
            _compute_streamlines(mesh, "vec")

    def test_zero_seeds_returns_empty_mesh(self):
        mesh = self._make_grid_mesh()
        out = _compute_streamlines(mesh, "vec", n=0)
        assert out.num_vertices == 0
        assert out.num_facets == 0

    def test_valid_input_produces_output(self):
        mesh = self._make_grid_mesh()
        out = _compute_streamlines(mesh, "vec", n=2, cross_field=False, min_length=2)
        # Should produce at least one streamline segment.
        assert out.num_vertices > 0
        assert out.num_facets > 0
        assert out.has_attribute("_hakowan_streamline_id")
