import pytest
import hakowan as hkw
import hakowan.compiler
from hakowan import transform, scale
from hakowan.compiler.transform import principal_axes_affine_matrix
from hakowan.compiler.streamline import _compute_streamlines
from hakowan.compiler.fur import (
    FUR_CHILDREN_ATTR,
    STRAND_ID_ATTR,
    STRAND_RADIUS_ATTR,
    _compute_fur,
)
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

    def test_filter_accepts_indexed_scalar_labels_on_surfaces(self):
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertices(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                ]
            )
        )
        mesh.add_triangles(np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32))
        mesh.create_attribute(
            "region",
            element=lagrange.AttributeElement.Indexed,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.array([[0], [1]], dtype=np.int32),
            initial_indices=np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32),
        )
        layer = hkw.layer(mesh).transform(
            transform.Filter(data="region", condition=lambda value: value == 1)
        )

        report = hkw.validate(layer)
        result = hkw.compile(layer)[0].data_frame.mesh

        assert report.valid
        assert result.num_facets == 1
        assert np.min(np.asarray(result.vertices)[:, 0]) >= 2.0

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

    def test_normalize_defaults(self):
        t = transform.Normalize()
        assert t.normalize_normals is True
        assert t.normalize_tangents_bitangents is True
        assert t._child is None

    def test_normalize(self):
        t = transform.Normalize(
            normalize_normals=False, normalize_tangents_bitangents=False
        )
        assert t.normalize_normals is False
        assert t.normalize_tangents_bitangents is False
        assert t._child is None

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


class TestNormalizeCompiler:
    def _make_offset_box(self):
        # An axis-aligned box from (10,10,10) to (14,12,20): off-center and
        # non-cubic so both the recentering and the uniform scaling are exercised.
        mesh = lagrange.SurfaceMesh()
        verts = np.array(
            [
                [10, 10, 10],
                [14, 10, 10],
                [14, 12, 10],
                [10, 12, 10],
                [10, 10, 20],
                [14, 10, 20],
                [14, 12, 20],
                [10, 12, 20],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.uint32,
        )
        mesh.add_vertices(verts)
        mesh.add_triangles(faces)
        return mesh

    def test_normalize_recenters_and_scales(self):
        mesh = self._make_offset_box()
        layer = hkw.layer(data=mesh, mark=hkw.mark.Surface).transform(
            hkw.transform.Normalize()
        )
        scene = hkw.compiler.compile(layer)
        assert len(scene) == 1
        v = scene[0].data_frame.mesh.vertices
        bbox_min = v.min(axis=0)
        bbox_max = v.max(axis=0)
        # Recentered at the origin ...
        assert np.allclose((bbox_min + bbox_max) / 2.0, 0.0, atol=1e-9)
        # ... and scaled so the geometry fits the unit sphere (bbox diagonal == 2).
        assert np.isclose(np.linalg.norm(bbox_max - bbox_min), 2.0, atol=1e-9)

    def test_normalize_equalizes_size_across_meshes(self):
        # Two boxes at very different scales should end up the same size.
        mesh_small = self._make_offset_box()
        mesh_big = self._make_offset_box()
        mesh_big.vertices = mesh_big.vertices * 100.0

        def normalized_diag(mesh):
            layer = hkw.layer(data=mesh, mark=hkw.mark.Surface).transform(
                hkw.transform.Normalize()
            )
            v = hkw.compiler.compile(layer)[0].data_frame.mesh.vertices
            return np.linalg.norm(v.max(axis=0) - v.min(axis=0))

        assert np.isclose(normalized_diag(mesh_small), normalized_diag(mesh_big))

    def test_normalize_empty_mesh_is_noop(self):
        mesh = lagrange.SurfaceMesh()
        layer = hkw.layer(data=mesh, mark=hkw.mark.Surface).transform(
            hkw.transform.Normalize()
        )
        scene = hkw.compiler.compile(layer)
        assert len(scene) == 1
        assert scene[0].data_frame.mesh.num_vertices == 0


class TestFurGrammar:
    def test_fur_grammar_defaults(self):
        t = transform.Fur(vec_field="flow")
        assert t.vec_field == "flow"
        assert t.n == 2000
        assert t.length is None
        assert t.lift == 30.0
        assert t.curl == 0.35
        assert t.segments == 6
        assert t.root_radius is None
        assert t.tip_radius == 0.0
        assert t.seed == 0
        assert t._child is None

    def test_fur_compile_produces_strand_view(self):
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertex([0.0, 0.0, 0.0])
        mesh.add_vertex([1.0, 0.0, 0.0])
        mesh.add_vertex([1.0, 1.0, 0.0])
        mesh.add_vertex([0.0, 1.0, 0.0])
        mesh.add_triangle(0, 1, 2)
        mesh.add_triangle(0, 2, 3)
        mesh.create_attribute(
            "vec",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.array(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
            ),
        )
        layer = (
            hkw.layer(mesh)
            .transform(hkw.transform.Fur(vec_field="vec", n=8, seed=1))
            .mark("curve")
        )
        scene = hkw.compiler.compile(layer)
        assert len(scene) == 1
        out = scene[0].data_frame.mesh
        assert out.has_attribute(STRAND_ID_ATTR)
        assert out.has_attribute(STRAND_RADIUS_ATTR)


class TestFurCompiler:
    def _make_grid_mesh(self, attr_name="vec", with_attr=True):
        # Two-triangle grid in the z=0 plane (outward normal +z), field +x.
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
            _compute_fur(mesh, "missing_attr")

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
            _compute_fur(mesh, "vec")

    def test_zero_strands_returns_empty_mesh(self):
        mesh = self._make_grid_mesh()
        out = _compute_fur(mesh, "vec", n=0)
        assert out.num_vertices == 0
        assert out.num_facets == 0

    def test_produces_expected_geometry(self):
        mesh = self._make_grid_mesh()
        n, segments = 20, 6
        out = _compute_fur(mesh, "vec", n=n, segments=segments, seed=1)
        # n strands, each with (segments + 1) points and ``segments`` line segs.
        assert out.num_vertices == n * (segments + 1)
        assert out.num_facets == n * segments
        assert out.has_attribute(STRAND_ID_ATTR)
        assert out.has_attribute(STRAND_RADIUS_ATTR)
        ids = np.asarray(out.attribute(STRAND_ID_ATTR).data).reshape(-1)
        assert len(np.unique(ids)) == n

    def test_radius_tapers_from_root_to_tip(self):
        mesh = self._make_grid_mesh()
        out = _compute_fur(
            mesh, "vec", n=10, segments=6, root_radius=0.05, tip_radius=0.0, seed=2
        )
        ids = np.asarray(out.attribute(STRAND_ID_ATTR).data).reshape(-1)
        radii = np.asarray(out.attribute(STRAND_RADIUS_ATTR).data).reshape(-1)
        for s in np.unique(ids):
            r = radii[ids == s]
            assert r[0] == pytest.approx(0.05)
            assert r[-1] == pytest.approx(0.0)
            assert np.all(np.diff(r) <= 1e-12)  # monotonically non-increasing

    @pytest.mark.parametrize("children", [0, 1, 2])
    def test_child_count_is_preserved(self, children):
        out = _compute_fur(self._make_grid_mesh(), "vec", n=2, children=children)

        assert out.has_attribute(FUR_CHILDREN_ATTR) is (children > 0)
        if children > 0:
            values = np.asarray(out.attribute(FUR_CHILDREN_ATTR).data).reshape(-1)
            np.testing.assert_array_equal(values, children)

    def test_strands_lift_off_surface_and_flow_along_field(self):
        # Flat grid (normal +z), field +x, no randomness => deterministic shape.
        mesh = self._make_grid_mesh()
        out = _compute_fur(
            mesh, "vec", n=30, segments=6, lift=30.0, randomness=0.0, seed=3
        )
        ids = np.asarray(out.attribute(STRAND_ID_ATTR).data).reshape(-1)
        verts = np.asarray(out.vertices)
        for s in np.unique(ids):
            pts = verts[ids == s]
            root, tip = pts[0], pts[-1]
            # Rises off the z=0 surface (lift > 0).
            assert tip[2] > root[2] + 1e-9
            # Flows along the +x field direction.
            assert tip[0] > root[0] + 1e-9

    def test_follow_surface_hugs_surface_and_flows(self):
        # Flat grid (normal +z, field +x). Surface-following strands trace along
        # the plane (roots exactly on it) and lift only gently at the tip.
        mesh = self._make_grid_mesh()
        length, lift, curl = 0.05, 15.0, 0.1
        out = _compute_fur(
            mesh,
            "vec",
            n=20,
            segments=6,
            length=length,
            lift=lift,
            curl=curl,
            follow_surface=True,
            randomness=0.0,
            seed=3,
        )
        assert out.has_attribute(STRAND_ID_ATTR)
        ids = np.asarray(out.attribute(STRAND_ID_ATTR).data).reshape(-1)
        verts = np.asarray(out.vertices)
        # Tip lift is bounded by length * (sin(lift) + curl) — it hugs the plane.
        max_rise = length * (np.sin(np.radians(lift)) + curl) + 1e-6
        for s in np.unique(ids):
            pts = verts[ids == s]
            root, tip = pts[0], pts[-1]
            assert abs(root[2]) < 1e-9  # root lies on the z=0 surface
            assert 0.0 <= tip[2] <= max_rise  # gentle, bounded lift
            assert tip[0] > root[0] + 1e-9  # flows along +x

    def test_follow_surface_reaches_requested_length_on_dense_mesh(self):
        count = 600
        mesh = lagrange.SurfaceMesh()
        for index in range(count + 1):
            x = index * 0.01
            mesh.add_vertex([x, 0.0, 0.0])
            mesh.add_vertex([x, 0.05, 0.0])
        for index in range(count):
            a, b, c, d = 2 * index, 2 * index + 1, 2 * index + 2, 2 * index + 3
            mesh.add_triangle(a, c, d)
            mesh.add_triangle(a, d, b)
        mesh.create_attribute(
            "vec",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.tile(np.array([1.0, 0.0, 0.0]), (mesh.num_facets, 1)),
        )

        requested_length = 2.0
        out = _compute_fur(
            mesh,
            "vec",
            n=1,
            segments=6,
            length=requested_length,
            lift=0.0,
            curl=0.0,
            follow_surface=True,
            randomness=0.0,
            seed=0,
        )
        vertices = np.asarray(out.vertices)
        strand_ids = np.asarray(out.attribute(STRAND_ID_ATTR).data).reshape(-1)
        points = vertices[strand_ids == 0]
        arc_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

        assert arc_length >= 0.9 * requested_length

    def test_follow_surface_lift_zero_stays_on_surface_with_randomness(self):
        # With lift=0/curl=0, randomness must not push strands off the surface:
        # lift/curl vary multiplicatively, so on a flat grid every point stays
        # exactly on the z=0 plane regardless of ``randomness``.
        mesh = self._make_grid_mesh()
        out = _compute_fur(
            mesh,
            "vec",
            n=40,
            segments=6,
            lift=0.0,
            curl=0.0,
            follow_surface=True,
            randomness=0.6,
            seed=5,
        )
        verts = np.asarray(out.vertices)
        assert np.max(np.abs(verts[:, 2])) < 1e-9


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
