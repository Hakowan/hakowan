import pytest
import hakowan as hkw
import hakowan.compiler
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

    def test_clip(self):
        t = transform.Clip(point=[0.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
        assert np.allclose(np.asarray(t.point), [0.0, 0.0, 0.0])
        assert np.allclose(np.asarray(t.normal), [1.0, 0.0, 0.0])
        assert t._child is None

    def test_clip_defaults(self):
        t = transform.Clip()
        assert np.allclose(np.asarray(t.point), [0.0, 0.0, 0.0])
        assert np.allclose(np.asarray(t.normal), [1.0, 0.0, 0.0])

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


class TestClipCompiler:
    def _make_square(self):
        # Unit square in the z=0 plane, split into two triangles, with a
        # per-vertex scalar equal to the x coordinate and a per-facet id.
        mesh = lagrange.SurfaceMesh()
        for v in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]:
            mesh.add_vertex(v)
        mesh.add_triangle(0, 1, 2)
        mesh.add_triangle(0, 2, 3)
        mesh.create_attribute(
            "xcoord",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=mesh.vertices[:, 0].copy(),
        )
        mesh.create_attribute(
            "fid",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.array([10.0, 20.0], dtype=np.float64),
        )
        return mesh

    def test_clip_keeps_positive_side(self):
        from hakowan.compiler.transform import _clip_mesh

        mesh = self._make_square()
        out = _clip_mesh(mesh, np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        assert out.num_vertices > 0
        # Every surviving vertex is on (or on the boundary of) the kept side.
        assert np.all(out.vertices[:, 0] >= 0.5 - 1e-9)

    def test_clip_interpolates_vertex_attribute(self):
        from hakowan.compiler.transform import _clip_mesh

        mesh = self._make_square()
        out = _clip_mesh(mesh, np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        # The xcoord attribute must track the (re-cut) vertex x coordinate.
        xc = out.attribute("xcoord").data.ravel()
        assert np.allclose(xc, out.vertices[:, 0])

    def test_clip_copies_facet_attribute(self):
        from hakowan.compiler.transform import _clip_mesh

        mesh = self._make_square()
        out = _clip_mesh(mesh, np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        fid = out.attribute("fid").data.ravel()
        # Child triangles inherit their parent's facet value (10 from tri 0, 20 from tri 1).
        assert set(fid.tolist()).issubset({10.0, 20.0})
        assert out.num_facets == len(fid)

    def test_clip_integer_attribute_not_averaged(self):
        from hakowan.compiler.transform import _clip_mesh

        mesh = self._make_square()
        mesh.create_attribute(
            "vid",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.array([0, 1, 2, 3], dtype=np.int32),
        )
        out = _clip_mesh(mesh, np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        vid = out.attribute("vid").data
        assert np.issubdtype(vid.dtype, np.integer)
        # Dominant-corner pick => values stay in the original integer set.
        assert set(vid.ravel().tolist()).issubset({0, 1, 2, 3})

    def test_clip_indexed_attribute(self):
        from hakowan.compiler.transform import _clip_mesh

        mesh = self._make_square()
        lagrange.compute_normal(mesh, output_attribute_name="nrm")
        assert mesh.is_attribute_indexed("nrm")
        out = _clip_mesh(mesh, np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        nrm = out.attribute("nrm").data
        assert nrm.shape == (out.num_vertices, 3)
        assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-6)

    def test_clip_fully_inside_is_unchanged(self):
        from hakowan.compiler.transform import _clip_mesh

        mesh = self._make_square()
        out = _clip_mesh(mesh, np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        assert out.num_facets == mesh.num_facets

    def test_clip_fully_outside_is_empty(self):
        from hakowan.compiler.transform import _clip_mesh

        mesh = self._make_square()
        out = _clip_mesh(mesh, np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        assert out.num_vertices == 0
        assert out.num_facets == 0

    def test_clip_zero_normal_raises(self):
        from hakowan.compiler.transform import _clip_mesh

        mesh = self._make_square()
        with pytest.raises(ValueError, match="non-zero"):
            _clip_mesh(mesh, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))


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

    def _make_tent_mesh(self):
        """A strip folded 90° along a ridge, with a per-facet field flowing
        across the ridge. The flat half lies in z=0 (field +x); the wall half is
        vertical at x=1 (field +z). A streamline crossing the ridge therefore
        bends ~90° in 3D — the crease/kink scenario.
        """
        ny = 6
        verts = []
        idx = {}

        def add(key, p):
            idx[key] = len(verts)
            verts.append(p)

        for j in range(ny + 1):
            y = j / ny
            add(("flat", 0, j), [0.0, y, 0.0])
            add(("flat", 1, j), [1.0, y, 0.0])  # ridge line at x=1
            add(("wall", 1, j), [1.0, y, 1.0])  # top of the vertical wall

        tris = []
        field = []
        for j in range(ny):
            a, b = idx[("flat", 0, j)], idx[("flat", 1, j)]
            c, d = idx[("flat", 1, j + 1)], idx[("flat", 0, j + 1)]
            tris += [[a, b, c], [a, c, d]]
            field += [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]  # flow +x toward ridge
            e, f = idx[("flat", 1, j)], idx[("wall", 1, j)]
            g, h = idx[("wall", 1, j + 1)], idx[("flat", 1, j + 1)]
            tris += [[e, f, g], [e, g, h]]
            field += [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]  # flow +z up the wall

        mesh = lagrange.SurfaceMesh()
        mesh.add_vertices(np.array(verts, dtype=np.float64))
        mesh.add_triangles(np.array(tris, dtype=np.uint32))
        mesh.create_attribute(
            "vec",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.array(field, dtype=np.float64),
        )
        return mesh

    @staticmethod
    def _max_turn_deg(out):
        """Largest angle (deg) between consecutive segments of any streamline."""
        if out.num_vertices == 0:
            return 0.0
        ids = np.asarray(out.attribute("_hakowan_streamline_id").data)
        V = np.asarray(out.vertices)
        worst = 0.0
        for sid in np.unique(ids):
            pts = V[ids == sid]
            if len(pts) < 3:
                continue
            seg = np.diff(pts, axis=0)
            seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-20)
            dots = np.einsum("ij,ij->i", seg[:-1], seg[1:]).clip(-1, 1)
            worst = max(worst, float(np.degrees(np.arccos(dots)).max()))
        return worst

    def test_crease_bends_geometry_not_reversal(self):
        # Tracing across the 90° fold yields a ~90° turn in the 3D-embedded
        # polyline — that is faithful surface geometry (the transported tangent
        # is continuous; only the embedding folds). What must NOT happen is a
        # backward reversal (~180°): the forward-direction check guarantees every
        # emitted segment advances along the travel direction. The tent's only
        # sharp feature is the 90° crease, so a >150° turn would flag the
        # backward-crossing bug.
        mesh = self._make_tent_mesh()
        out = _compute_streamlines(mesh, "vec", n=20, cross_field=False, min_length=2)
        assert out.num_vertices > 0
        assert self._max_turn_deg(out) < 150.0


class TestExplodeCompiler:
    def test_explode_accepts_attribute_pieces(self, two_triangles):
        # ``pieces`` given as an Attribute (not a bare string) must resolve to
        # its name rather than being passed straight to ``mesh.has_attribute``.
        mesh = two_triangles
        l1 = hkw.layer(data=mesh, mark=hkw.mark.Surface).transform(
            hkw.transform.Explode(pieces=hkw.attribute(name="facet_index"))
        )
        scene = hkw.compiler.compile(l1)
        assert len(scene) == 1
        # The two facet groups are displaced apart but both survive.
        assert scene[0].data_frame.mesh.num_facets == 2

    def test_explode_accepts_string_pieces(self, two_triangles):
        mesh = two_triangles
        l1 = hkw.layer(data=mesh, mark=hkw.mark.Surface).transform(
            hkw.transform.Explode(pieces="facet_index")
        )
        scene = hkw.compiler.compile(l1)
        assert len(scene) == 1
        assert scene[0].data_frame.mesh.num_facets == 2
