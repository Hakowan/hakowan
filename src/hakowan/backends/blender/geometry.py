"""Geometry extraction and Blender object creation."""

from typing import Literal, cast

from ...common import logger
from ...common.color import srgb_to_linear, srgb_to_linear_array
from ...common.vector_field import filter_zero_length_vectors
from ...compiler import View
from ...compiler.fur import (
    FUR_CHILDREN_ATTR,
    FUR_CLUMP_ATTR,
    FUR_SPREAD_ATTR,
    STRAND_ID_ATTR,
    STRAND_NORMAL_ATTR,
    STRAND_RADIUS_ATTR,
)
from ...grammar import mark
from ...grammar.scale import Attribute
from ...grammar.channel import DEFAULT_COVARIANCE_SIZE, DEFAULT_MARK_SIZE
from ...grammar.channel.curvestyle import Bend

import numpy as np
import lagrange

import bpy
import mathutils
from .materials import _MaterialMixin


def _decompose_rotation_scale(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a 3x3 ``M`` into the rotation + per-axis scale reproducing the
    ellipsoid ``M`` maps the unit sphere to.

    The glyph is a symmetric sphere, so ``M·sphere`` depends only on the
    ellipsoid ``M·Mᵀ``; the SVD ``M = U·diag(σ)·Vᵀ`` gives rotation ``U`` and
    scale ``σ`` (the trailing ``Vᵀ`` rotation is absorbed by the sphere's
    symmetry). This stays exact even for a sheared covariance ``M`` (non-``full``
    covariance), which a column-wise normalisation would not. A reflection
    (``det U < 0``) is folded into a flipped axis — harmless by the same symmetry
    — so ``U`` is a proper rotation a quaternion can encode.
    """
    u, sigma, _ = np.linalg.svd(m)
    if np.linalg.det(u) < 0.0:
        u = u.copy()
        u[:, -1] = -u[:, -1]
    return u, sigma.astype(np.float32)


def _matrix3_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """3x3 proper rotation → Blender quaternion array ``(w, x, y, z)``."""
    q = mathutils.Matrix([list(row) for row in rotation]).to_quaternion()
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)


class _GeometryMixin(_MaterialMixin):
    def _extract_size(self, view: View, default_size: float = DEFAULT_MARK_SIZE):
        """Extract size attribute from a view (scalar or per-vertex).

        Returns:
            Either a float (uniform size) or a list of floats (per-vertex).
        """
        assert view.data_frame is not None
        mesh = view.data_frame.mesh

        if view.size_channel is not None:
            data = view.size_channel.data
            if isinstance(data, (int, float)):
                return float(data)
            if isinstance(data, Attribute):
                assert data._internal_name is not None
                return mesh.attribute(data._internal_name).data.tolist()
            raise NotImplementedError(f"Unsupported size channel type: {type(data)}")
        return default_size

    def _extract_edges(self, view: View):
        """Extract edge segments with base, tip, base_size, tip_size and vertex indices.

        Returns:
            Tuple (base, tip, base_size, tip_size, base_idx, tip_idx).
            base/tip are (n_edges, 3) positions; base_idx/tip_idx are (n_edges,) vertex indices.
        """
        assert view.data_frame is not None
        mesh = view.data_frame.mesh
        mesh.initialize_edges()

        sizes = self._extract_size(view)
        if np.isscalar(sizes):
            # np.isscalar guarantees a scalar here, but mypy can't narrow it.
            sizes = [float(sizes)] * mesh.num_vertices  # type: ignore[arg-type]

        n_edges = mesh.num_edges
        vertices = mesh.vertices
        base = np.zeros((n_edges, 3), dtype=np.float64)
        tip = np.zeros((n_edges, 3), dtype=np.float64)
        base_size = np.zeros(n_edges, dtype=np.float64)
        tip_size = np.zeros(n_edges, dtype=np.float64)
        base_idx = np.zeros(n_edges, dtype=np.int32)
        tip_idx = np.zeros(n_edges, dtype=np.int32)

        for i in range(n_edges):
            edge_vts = mesh.get_edge_vertices(i)
            base_idx[i] = edge_vts[0]
            tip_idx[i] = edge_vts[1]
            base[i] = vertices[edge_vts[0]]
            tip[i] = vertices[edge_vts[1]]
            base_size[i] = sizes[edge_vts[0]]
            tip_size[i] = sizes[edge_vts[1]]

        return base, tip, base_size, tip_size, base_idx, tip_idx

    def _extract_vector_field(self, view: View):
        """Extract vector field data from a view.

        Returns:
            Tuple (base, ctrl_pts_1, ctrl_pts_2, tip, base_size, tip_size).
        """
        assert view.data_frame is not None
        mesh = view.data_frame.mesh

        assert view.vector_field_channel is not None
        assert isinstance(view.vector_field_channel.data, Attribute)
        attr_name = view.vector_field_channel.data._internal_name
        assert attr_name is not None
        assert mesh.has_attribute(attr_name)

        attr = mesh.attribute(attr_name)
        match attr.element_type:
            case lagrange.AttributeElement.Vertex:
                base = mesh.vertices.copy()
                size = self._extract_size(view)
                if np.isscalar(size):
                    size = [size] * mesh.num_vertices
            case lagrange.AttributeElement.Facet:
                centroid_attr_id = lagrange.compute_facet_centroid(mesh)
                base = np.array(mesh.attribute(centroid_attr_id).data)
                size = self._extract_size(view)
                if np.isscalar(size):
                    size = [size] * mesh.num_facets
            case _:
                raise NotImplementedError(
                    f"Unsupported vector field element type: {attr.element_type}"
                )

        tip = attr.data + base
        ctrl_pts_1 = None
        ctrl_pts_2 = None

        match view.vector_field_channel.style:
            case Bend():
                direction = view.vector_field_channel.style.direction
                assert isinstance(direction, Attribute)
                assert direction._internal_name is not None
                assert mesh.has_attribute(direction._internal_name)
                dir_attr = mesh.attribute(direction._internal_name)
                assert dir_attr.element_type == attr.element_type
                dirs = dir_attr.data
                assert np.all(dirs.shape == base.shape)

                bend_type = view.vector_field_channel.style.bend_type
                if bend_type == "n":
                    ctrl_pts_1 = base + dirs
                    ctrl_pts_2 = tip + dirs
                elif bend_type == "r":
                    ctrl_pts_1 = base + dirs
                    ctrl_pts_2 = base + dirs + 0.5 * (tip - base)
                    tip = tip + dirs
                elif bend_type == "s":
                    ctrl_pts_1 = base + dirs
                    ctrl_pts_2 = tip
                    tip = tip + dirs
                else:
                    raise NotImplementedError(f"Unsupported bend type: {bend_type}")

        def refine(mesh, data, level):
            assert mesh.is_triangle_mesh, "Only triangle mesh is supported."
            facets = mesh.facets
            n = level + 1
            refined_data = []
            B0, B1, B2 = np.mgrid[0 : n + 1, 0 : n + 1, 0 : n + 1]
            for b0, b1, b2 in zip(B0.ravel(), B1.ravel(), B2.ravel()):
                s = b0 + b1 + b2
                if s != n:
                    continue
                d = (
                    data[facets[:, 0]] * b0
                    + data[facets[:, 1]] * b1
                    + data[facets[:, 2]] * b2
                )
                d /= n
                refined_data.append(d)
            return np.vstack(refined_data)

        if view.vector_field_channel.refinement_level > 0:
            base = refine(mesh, base, view.vector_field_channel.refinement_level)
            tip = refine(mesh, tip, view.vector_field_channel.refinement_level)
            if ctrl_pts_1 is not None:
                assert ctrl_pts_2 is not None
                ctrl_pts_1 = refine(
                    mesh, ctrl_pts_1, view.vector_field_channel.refinement_level
                )
                ctrl_pts_2 = refine(
                    mesh, ctrl_pts_2, view.vector_field_channel.refinement_level
                )
            size = refine(
                mesh, np.array(size), view.vector_field_channel.refinement_level
            ).ravel()

        base_size = size
        if view.vector_field_channel.end_type == "point":
            tip_size = np.zeros_like(size)
        elif view.vector_field_channel.end_type == "arrow":
            assert ctrl_pts_1 is None
            assert ctrl_pts_2 is None
            stem_point = 0.25 * base + 0.75 * tip
            base = np.vstack([base, stem_point])
            tip = np.vstack([stem_point, tip])
            size = np.array(size)
            base_size = np.hstack([size, 2 * size])
            tip_size = np.hstack([size, np.zeros_like(size)])
        else:  # "flat": constant radius at both ends
            tip_size = size

        base, tip, base_size, tip_size, ctrl_pts_1, ctrl_pts_2 = (
            filter_zero_length_vectors(
                base, tip, base_size, tip_size, ctrl_pts_1, ctrl_pts_2
            )
        )

        return (
            base,
            ctrl_pts_1,
            ctrl_pts_2,
            tip,
            np.atleast_1d(base_size),
            np.atleast_1d(tip_size),
        )

    def _create_view_object(self, view: View, index: int):
        """Create Blender object from a view.

        Args:
            view: View to convert.
            index: View index for naming.
        """
        match view.mark:
            case mark.Surface:
                self._create_surface_object(view, index)
            case mark.Point:
                self._create_point_object(view, index)
            case mark.Curve:
                self._create_curve_object(view, index)
            case _:
                logger.warning(f"Unknown mark type: {view.mark}")

    def _create_surface_object(self, view: View, index: int):
        """Create Blender mesh object from surface view.

        Args:
            view: Surface view.
            index: Object index.
        """
        if view.data_frame is None:
            return

        mesh_data = view.data_frame.mesh

        # Extract vertices and faces
        vertices = mesh_data.vertices
        facets = []
        for i in range(mesh_data.num_facets):
            facet = mesh_data.facets[i].tolist()
            facets.append(facet)

        # Create Blender mesh
        mesh = bpy.data.meshes.new(name=f"mesh_{index:03d}")
        mesh.from_pydata(vertices.tolist(), [], facets)
        mesh.update()
        # Sanitize geometry built by from_pydata. Meshes can carry degenerate
        # facets (e.g. a UV sphere's pole quads with a repeated vertex);
        # validate() drops them. Without it, downstream ops like
        # normals_split_custom_set operate on invalid loops and *segfault*
        # Blender rather than raising.
        mesh.validate(verbose=False)
        mesh.update()

        # Apply custom shading normals (if the mesh carries a normal attribute)
        # so authored/smoothed normals match the other backends instead of
        # falling back to Blender's default flat shading.
        self._apply_custom_normals(mesh_data, mesh, facets)

        # Scalar field texture: copy vertex/face color attribute to Blender mesh
        color_layer_name = None
        scalar_info = self._get_scalar_field_color_attr(view)
        if scalar_info is not None:
            attr_name, element_type = scalar_info
            self._copy_mesh_color_to_blender(
                mesh_data, mesh, attr_name, element_type, color_layer_name="Color"
            )
            color_layer_name = "Color"

        # Copy the UV attribute used by any active texture (checkerboard, image
        # base color, normal map, or bump map) onto the Blender mesh.
        uv_layer_name = None
        uv_attr_name = self._resolve_uv_attr_name(view)
        if uv_attr_name is not None:
            self._copy_mesh_uv_to_blender(
                mesh_data, mesh, uv_attr_name, uv_layer_name="UVMap"
            )
            uv_layer_name = "UVMap"

        # Create object
        obj = bpy.data.objects.new(f"object_{index:03d}", mesh)
        bpy.context.collection.objects.link(obj)

        # Apply global transformation
        if hasattr(view, "global_transform") and view.global_transform is not None:
            # Convert numpy 4x4 matrix to Blender Matrix
            transform_matrix = mathutils.Matrix(view.global_transform.tolist())
            obj.matrix_world = transform_matrix
            logger.debug(f"Applied global transform to object {index}")

        # Apply material
        if view.material_channel is not None:
            mat = self._create_material(
                view,
                index,
                color_layer_name=color_layer_name,
                uv_layer_name=uv_layer_name,
            )
            if mat:
                obj.data.materials.append(mat)

        logger.debug(f"Created surface object {index} with {len(vertices)} vertices")

    def _apply_custom_normals(
        self,
        mesh_data,
        mesh: "bpy.types.Mesh",
        facets: list[list[int]],
    ):
        """Apply the lagrange mesh's normal attribute as Blender custom split normals.

        Mirrors the WebGL backend's element-type handling so authored normals
        render consistently. Vertex normals become per-vertex custom normals
        (smooth shading); facet, corner and indexed normals are expanded to
        per-loop normals. When the mesh has no normal attribute, Blender's
        default geometric shading is left untouched.

        Args:
            mesh_data: Source ``lagrange.SurfaceMesh`` carrying the normals.
            mesh: Target Blender mesh (already populated via ``from_pydata``).
            facets: Per-facet vertex-index lists, used to expand facet normals
                to the matching per-loop count for polygonal meshes.
        """
        normal_ids = mesh_data.get_matching_attribute_ids(
            usage=lagrange.AttributeUsage.Normal
        )
        if not normal_ids:
            return

        normal_name = mesh_data.get_attribute_name(normal_ids[0])
        per_vertex: np.ndarray | None = None
        per_corner: np.ndarray | None = None

        if mesh_data.is_attribute_indexed(normal_name):
            # Indexed normal: per-corner lookup into the value table, preserving
            # creases by giving a shared vertex different normals across facets.
            indexed = mesh_data.indexed_attribute(normal_name)
            values = np.asarray(indexed.values.data, dtype=np.float32)
            idx = np.asarray(indexed.indices.data, dtype=np.uint32).reshape(-1)
            per_corner = values[idx]
        else:
            normal_attr = mesh_data.attribute(normal_ids[0])
            element_type = normal_attr.element_type
            if element_type == lagrange.AttributeElement.Vertex:
                per_vertex = np.asarray(normal_attr.data, dtype=np.float32)
            elif element_type == lagrange.AttributeElement.Corner:
                per_corner = np.asarray(normal_attr.data, dtype=np.float32)
            elif element_type == lagrange.AttributeElement.Facet:
                facet_normals = np.asarray(normal_attr.data, dtype=np.float32)
                counts = [len(f) for f in facets]
                per_corner = np.repeat(facet_normals, counts, axis=0)
            else:
                logger.warning(
                    "Blender backend: unsupported normal element type "
                    f"'{element_type}', leaving default shading."
                )
                return

        # Custom split normals require smooth shading to take effect.
        mesh.shade_smooth()
        if per_vertex is not None:
            if per_vertex.shape[0] != len(mesh.vertices):
                logger.warning(
                    "Blender backend: vertex normal count "
                    f"({per_vertex.shape[0]}) != vertex count ({len(mesh.vertices)}); "
                    "skipping custom normals."
                )
                return
            mesh.normals_split_custom_set_from_vertices(per_vertex.tolist())
        else:
            assert per_corner is not None
            if per_corner.shape[0] == len(mesh.loops):
                mesh.normals_split_custom_set(per_corner.tolist())
                return
            # The per-corner array no longer lines up with the Blender loops —
            # typically because mesh.validate() dropped degenerate facets (e.g. a
            # UV sphere's pole quads). Fall back to per-vertex normals, which are
            # keyed on vertices (validate keeps them) and still shade smoothly.
            corner_vertices = np.concatenate(
                [np.asarray(f, dtype=np.int64) for f in facets]
            )
            num_vertices = len(mesh.vertices)
            if (
                corner_vertices.shape[0] == per_corner.shape[0]
                and int(corner_vertices.max()) < num_vertices
            ):
                per_vertex = np.zeros((num_vertices, 3), dtype=np.float32)
                per_vertex[corner_vertices] = per_corner
                mesh.normals_split_custom_set_from_vertices(per_vertex.tolist())
                return
            logger.warning(
                "Blender backend: corner normal count "
                f"({per_corner.shape[0]}) != loop count ({len(mesh.loops)}) and no "
                "per-vertex fallback; using Blender's default smooth shading."
            )

    def _create_point_base_mesh(
        self, base_shape: Literal["sphere", "disk", "cube"], index: int
    ):
        """Create the shared base mesh (instance prototype) for a point shape.

        The mesh is built once and instanced onto every point by a geometry-nodes
        modifier, so detail is baked into the prototype rather than added per
        point: the sphere is a subdivided icosphere (no per-instance Subdivision
        modifier, which would defeat Cycles' instance de-duplication).

        Args:
            base_shape: One of "sphere", "cube", or "disk".
            index: Object index (used for naming).

        Returns:
            Tuple of (base_object, base_mesh, use_smooth).
        """
        match base_shape:
            case "sphere":
                bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)
                base_obj = bpy.context.active_object
                base_obj.name = f"base_ico_{index:03d}"
                base_obj.data.name = f"ico_sphere_{index:03d}"
                return base_obj, base_obj.data, True
            case "cube":
                bpy.ops.mesh.primitive_cube_add(size=2.0)
                base_obj = bpy.context.active_object
                base_obj.name = f"base_cube_{index:03d}"
                base_obj.data.name = f"cube_{index:03d}"
                return base_obj, base_obj.data, False
            case "disk":
                bpy.ops.mesh.primitive_circle_add(
                    vertices=32, radius=1.0, fill_type="NGON"
                )
                base_obj = bpy.context.active_object
                base_obj.name = f"base_disk_{index:03d}"
                base_obj.data.name = f"disk_{index:03d}"
                return base_obj, base_obj.data, False
            case _:
                raise ValueError(f"Unsupported base shape: {base_shape}")

    def _compute_orientation_rotation(self, normal: np.ndarray) -> mathutils.Matrix:
        """Compute rotation matrix that aligns Z-axis to the given normal.

        Args:
            normal: Target normal direction (will be normalized).

        Returns:
            4x4 rotation matrix.
        """
        # Normalize the input normal
        normal_normalized = normal / np.linalg.norm(normal)

        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, normal_normalized)
        sin_a = np.linalg.norm(axis)
        cos_a = np.dot(z, normal_normalized)

        # Handle edge cases
        if sin_a < 1e-9:
            # Either parallel or antiparallel
            if cos_a > 0:
                # Parallel: normal ≈ (0, 0, 1)
                return mathutils.Matrix.Identity(4)
            else:
                # Antiparallel: normal ≈ (0, 0, -1)
                # Rotate 180° around X-axis
                rot = mathutils.Matrix.Identity(4)
                rot[1][1] = -1.0
                rot[2][2] = -1.0
                return rot

        # General case: use Rodrigues' rotation formula
        v = axis / sin_a
        eye3 = np.eye(3)
        H = np.outer(v, v)
        S = np.cross(eye3, v)
        M = eye3 * cos_a + S * sin_a + H * (1 - cos_a)
        rot = mathutils.Matrix.Identity(4)
        for r in range(3):
            for c in range(3):
                rot[r][c] = float(M[r, c])
        return rot

    def _extract_covariance_transforms(self, view: View):
        """Extract per-vertex 3x3 transforms from covariance channel.

        Args:
            view: The view with covariance channel.

        Returns:
            Array of shape (n, 3, 3) transformation matrices M where covariance is M @ M^T.
        """
        assert view.data_frame is not None
        mesh = view.data_frame.mesh

        assert view.covariance_channel is not None
        assert isinstance(view.covariance_channel.data, Attribute)
        attr_name = view.covariance_channel.data._internal_name
        assert attr_name is not None
        assert mesh.has_attribute(attr_name)

        attr = mesh.attribute(attr_name)
        assert attr.element_type == lagrange.AttributeElement.Vertex
        assert attr.data.shape[1] == 9
        if view.covariance_channel.full:
            sigma = attr.data.reshape(-1, 3, 3)
            U, S, Vh = np.linalg.svd(sigma)
            S_diag = np.apply_along_axis(lambda _s: np.diag(_s), 1, S)
            return U @ np.sqrt(S_diag)
        else:
            return attr.data.reshape(-1, 3, 3)

    def _create_point_object(self, view: View, index: int):
        """Create a GPU-instanced point cloud (spheres/cubes/disks) from a view.

        One shared base shape is instanced onto every vertex by a geometry-nodes
        "Instance on Points" modifier. Per-point placement rides on point-domain
        attributes — ``hkw_scale`` (FLOAT_VECTOR), ``hkw_rotation`` (QUATERNION),
        and optional ``hkw_color`` (FLOAT_COLOR). This keeps geometry to a single
        prototype mesh that Cycles instances, instead of one baked mesh per point.

        Args:
            view: Point view.
            index: Object index.
        """
        if view.data_frame is None:
            return

        mesh_data = view.data_frame.mesh
        vertices = np.asarray(mesh_data.vertices, dtype=np.float64)
        n_points = mesh_data.num_vertices
        if n_points == 0:
            logger.debug(f"Point view {index} has no vertices; skipping.")
            return

        # Extract covariance transforms if specified
        covariance_transforms = None
        if view.covariance_channel is not None:
            covariance_transforms = self._extract_covariance_transforms(view)

        radii = self._extract_size(
            view,
            default_size=DEFAULT_MARK_SIZE
            if covariance_transforms is None
            else DEFAULT_COVARIANCE_SIZE,
        )
        if np.isscalar(radii):
            # np.isscalar guarantees a scalar here, but mypy can't narrow it.
            radii = np.full(n_points, float(radii))  # type: ignore[arg-type]
        radii = np.atleast_1d(np.asarray(radii, dtype=np.float64))
        assert len(radii) == n_points

        # Determine base shape (shared instance prototype)
        base_shape = "sphere"
        if view.shape_channel is not None:
            base_shape = view.shape_channel.base_shape
        base_obj, base_mesh, use_smooth = self._create_point_base_mesh(
            cast(Literal["sphere", "disk", "cube"], base_shape), index
        )
        if use_smooth:
            for poly in base_mesh.polygons:
                poly.use_smooth = True
        # Keep the base only as a hidden instance source for the modifier.
        base_obj.hide_render = True
        base_obj.hide_viewport = True

        # Extract orientation normals if specified
        normals = None
        if (
            view.shape_channel is not None
            and view.shape_channel.orientation is not None
        ):
            assert isinstance(view.shape_channel.orientation, Attribute)
            normal_attr_name = view.shape_channel.orientation._internal_name
            assert normal_attr_name is not None
            assert mesh_data.has_attribute(normal_attr_name)
            normals = np.asarray(mesh_data.attribute(normal_attr_name).data)

        # Per-instance rotation (quaternion w,x,y,z) and scale (x,y,z).
        quaternions, scales = self._compute_point_instance_transforms(
            radii, covariance_transforms, normals
        )

        # Scalar field: per-vertex colour, one RGBA per point.
        point_colors = None
        scalar_info = self._get_scalar_field_color_attr(view)
        if scalar_info is not None:
            attr_name, element_type = scalar_info
            if element_type == lagrange.AttributeElement.Vertex:
                data = np.asarray(mesh_data.attribute(attr_name).data)
                if data.ndim == 2 and data.shape[0] == n_points and data.shape[1] >= 3:
                    # Baked colors are sRGB; FLOAT_COLOR attributes are linear.
                    rgb = srgb_to_linear_array(data[:, :3])
                    if data.shape[1] >= 4:
                        alpha = data[:, 3:4]
                    else:
                        alpha = np.ones((n_points, 1))
                    point_colors = np.hstack([rgb, alpha]).astype(np.float32)

        # Points mesh carrying the per-instance attributes.
        points_mesh = bpy.data.meshes.new(f"points_{index:03d}")
        points_mesh.from_pydata(vertices.tolist(), [], [])
        points_mesh.update()
        scale_attr = points_mesh.attributes.new("hkw_scale", "FLOAT_VECTOR", "POINT")
        scale_attr.data.foreach_set("vector", scales.reshape(-1))
        rot_attr = points_mesh.attributes.new("hkw_rotation", "QUATERNION", "POINT")
        rot_attr.data.foreach_set("value", quaternions.reshape(-1))
        if point_colors is not None:
            color_attr = points_mesh.attributes.new("hkw_color", "FLOAT_COLOR", "POINT")
            color_attr.data.foreach_set("color", point_colors.reshape(-1))

        points_obj = bpy.data.objects.new(f"points_{index:03d}", points_mesh)
        bpy.context.collection.objects.link(points_obj)
        if hasattr(view, "global_transform") and view.global_transform is not None:
            points_obj.matrix_world = mathutils.Matrix(view.global_transform.tolist())

        self._add_instance_on_points_modifier(points_obj, base_obj, index)

        # One material on the shared prototype; per-point colour (when present)
        # is read off the instancer via an INSTANCER-domain attribute node.
        if view.material_channel is not None:
            mat = self._create_material(
                view,
                index,
                color_layer_name="hkw_color" if point_colors is not None else None,
                color_attribute_type="INSTANCER",
            )
            if mat:
                base_mesh.materials.append(mat)

        logger.debug(
            f"Created instanced point object {index} with {n_points} {base_shape}s"
        )

    def _compute_point_instance_transforms(
        self,
        radii: np.ndarray,
        covariance_transforms: np.ndarray | None,
        normals: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-point (quaternion w,x,y,z) and (scale x,y,z) for the instancer.

        Covariance ``M = U·diag(√S)`` decomposes to rotation + per-axis scale;
        an orientation normal becomes a rotation with uniform scale; otherwise
        the rotation is identity and the scale is the uniform radius.
        """
        n = len(radii)
        scales = np.empty((n, 3), dtype=np.float32)
        quaternions = np.zeros((n, 4), dtype=np.float32)
        quaternions[:, 0] = 1.0  # identity (w, x, y, z)

        if covariance_transforms is not None:
            linear = covariance_transforms * radii[:, None, None]
            for i in range(n):
                rotation, scale = _decompose_rotation_scale(linear[i])
                quaternions[i] = _matrix3_to_quaternion(rotation)
                scales[i] = scale
        elif normals is not None:
            for i in range(n):
                q = self._compute_orientation_rotation(normals[i]).to_quaternion()
                quaternions[i] = (q.w, q.x, q.y, q.z)
                scales[i] = radii[i]
        else:
            scales[:] = radii[:, None]
        return quaternions, scales

    def _add_instance_on_points_modifier(self, points_obj, base_obj, index: int):
        """Attach a geometry-nodes modifier instancing ``base_obj`` on the points.

        Reads ``hkw_scale`` / ``hkw_rotation`` point attributes for the per-point
        Scale and Rotation sockets of the Instance-on-Points node.
        """
        node_group = bpy.data.node_groups.new(
            f"hkw_instance_{index:03d}", "GeometryNodeTree"
        )
        node_group.interface.new_socket(
            "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        node_group.interface.new_socket(
            "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
        nodes = node_group.nodes
        links = node_group.links

        group_in = nodes.new("NodeGroupInput")
        group_out = nodes.new("NodeGroupOutput")
        instance_on_points = nodes.new("GeometryNodeInstanceOnPoints")
        object_info = nodes.new("GeometryNodeObjectInfo")
        object_info.inputs["Object"].default_value = base_obj
        object_info.transform_space = "RELATIVE"

        scale_input = nodes.new("GeometryNodeInputNamedAttribute")
        scale_input.data_type = "FLOAT_VECTOR"
        scale_input.inputs["Name"].default_value = "hkw_scale"
        rotation_input = nodes.new("GeometryNodeInputNamedAttribute")
        rotation_input.data_type = "QUATERNION"
        rotation_input.inputs["Name"].default_value = "hkw_rotation"

        links.new(group_in.outputs["Geometry"], instance_on_points.inputs["Points"])
        links.new(
            object_info.outputs["Geometry"], instance_on_points.inputs["Instance"]
        )
        links.new(scale_input.outputs["Attribute"], instance_on_points.inputs["Scale"])
        links.new(
            rotation_input.outputs["Attribute"], instance_on_points.inputs["Rotation"]
        )
        links.new(instance_on_points.outputs["Instances"], group_out.inputs["Geometry"])

        modifier = points_obj.modifiers.new(f"hkw_instance_{index:03d}", "NODES")
        modifier.node_group = node_group

    def _create_curve_object(self, view: View, index: int):
        """Create Blender curve object from view (mesh edges or vector field).

        Args:
            view: Curve view.
            index: Object index.
        """
        if view.data_frame is None:
            return

        mesh_data = view.data_frame.mesh

        # Fur/strand meshes carry a per-vertex strand id: render each strand as a
        # single continuous, tapered curve so the Principled Hair BSDF sees a
        # proper hair strand (root -> tip) rather than disconnected segments.
        if mesh_data.has_attribute(STRAND_ID_ATTR):
            self._create_strand_object(view, index)
            return

        if view.vector_field_channel is not None:
            base, ctrl_pts_1, ctrl_pts_2, tip, base_size, tip_size = (
                self._extract_vector_field(view)
            )
            n_segments = len(base)
            base_idx = tip_idx = None
        else:
            base, tip, base_size, tip_size, base_idx, tip_idx = self._extract_edges(
                view
            )
            n_segments = len(base)
            ctrl_pts_1 = ctrl_pts_2 = None

        curve_data = bpy.data.curves.new(name=f"curve_{index:03d}", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.bevel_mode = "ROUND"
        curve_data.bevel_depth = 1.0
        curve_data.fill_mode = "FULL"

        if ctrl_pts_1 is not None and ctrl_pts_2 is not None:
            # Bezier curves for bent vector fields
            for i in range(n_segments):
                spline = curve_data.splines.new("BEZIER")
                spline.bezier_points.add(1)  # starts with 1, add(1) gives 2
                bp0 = spline.bezier_points[0]
                bp1 = spline.bezier_points[1]
                bp0.co = tuple(base[i])
                bp0.handle_left_type = "FREE"
                bp0.handle_right_type = "FREE"
                bp0.handle_left = tuple(base[i])
                bp0.handle_right = tuple(ctrl_pts_1[i])
                bp0.radius = float(base_size[i])
                bp1.co = tuple(tip[i])
                bp1.handle_left_type = "FREE"
                bp1.handle_right_type = "FREE"
                bp1.handle_left = tuple(ctrl_pts_2[i])
                bp1.handle_right = tuple(tip[i])
                bp1.radius = float(tip_size[i])
        else:
            # Linear segments (POLY splines)
            for i in range(n_segments):
                spline = curve_data.splines.new("POLY")
                spline.points.add(1)  # POLY starts with 1 point, add(1) gives 2
                spline.points[0].co = (base[i][0], base[i][1], base[i][2], 1.0)
                spline.points[1].co = (tip[i][0], tip[i][1], tip[i][2], 1.0)
                spline.points[0].radius = float(base_size[i])
                spline.points[1].radius = float(tip_size[i])

        # Scalar field: set per-point colors on curve (2 points per segment)
        color_layer_name = None
        if base_idx is not None and tip_idx is not None:
            scalar_info = self._get_scalar_field_color_attr(view)
            if scalar_info is not None:
                attr_name, element_type = scalar_info
                if element_type == lagrange.AttributeElement.Vertex:
                    attr = mesh_data.attribute(attr_name)
                    data = np.asarray(attr.data)
                    if data.ndim == 2 and data.shape[1] >= 3:
                        try:
                            if hasattr(curve_data, "color_attributes"):
                                curve_data.color_attributes.new(
                                    name="Color",
                                    type="FLOAT_COLOR",
                                    domain="POINT",
                                )
                                layer = curve_data.color_attributes["Color"]
                                point_idx = 0
                                for i in range(n_segments):
                                    for vi in (base_idx[i], tip_idx[i]):
                                        c = data[vi]
                                        # sRGB → linear for FLOAT_COLOR.
                                        r = srgb_to_linear(float(c[0]))
                                        g = srgb_to_linear(float(c[1]))
                                        b = srgb_to_linear(float(c[2]))
                                        a = float(c[3]) if c.shape[0] >= 4 else 1.0
                                        if point_idx < len(layer.data):
                                            layer.data[point_idx].color = (r, g, b, a)
                                        point_idx += 1
                                color_layer_name = "Color"
                        except (AttributeError, TypeError):
                            pass

        obj = bpy.data.objects.new(f"curve_{index:03d}", curve_data)
        bpy.context.collection.objects.link(obj)

        if hasattr(view, "global_transform") and view.global_transform is not None:
            obj.matrix_world = mathutils.Matrix(view.global_transform.tolist())

        if view.material_channel is not None:
            mat = self._create_material(view, index, color_layer_name=color_layer_name)
            if mat:
                obj.data.materials.append(mat)

        logger.debug(f"Created curve object {index} with {n_segments} segments")

    def _create_strand_object(self, view: View, index: int):
        """Create a native Blender hair-Curves object for fur/hair strands.

        Each strand (a run of vertices sharing the same ``_hakowan_strand_id``)
        becomes one hair curve in a :class:`bpy.types.Curves` datablock, which
        Cycles renders as a true thin hair *primitive* (not a beveled mesh tube).
        This gives realistic thin-strand shading, self-shadowing and
        translucency under the Principled Hair BSDF.  Per-point radius is taken
        from the ``_hakowan_strand_radius`` attribute (root-to-tip taper); an
        explicit ``size`` channel, when set, overrides it.

        Args:
            view: Curve view whose data frame holds a strand mesh.
            index: Object index.
        """
        assert view.data_frame is not None
        mesh_data = view.data_frame.mesh

        vertices = np.asarray(mesh_data.vertices, dtype=np.float64)
        ids = np.asarray(mesh_data.attribute(STRAND_ID_ATTR).data).reshape(-1)

        # Radius: an explicit size channel wins, otherwise the baked taper.
        if view.size_channel is not None:
            size = self._extract_size(view)
            size_array = np.asarray(size, dtype=np.float64)
            radius = (
                np.full(len(vertices), float(size_array.item()))
                if size_array.ndim == 0
                else size_array
            )
        elif mesh_data.has_attribute(STRAND_RADIUS_ATTR):
            radius = np.asarray(
                mesh_data.attribute(STRAND_RADIUS_ATTR).data, dtype=np.float64
            ).reshape(-1)
        else:
            radius = np.full(len(vertices), DEFAULT_MARK_SIZE)

        # Strand vertices are stored consecutively (root first), so contiguous
        # runs of equal id delimit strands. Keep only strands with >= 2 points.
        if len(ids) > 0:
            boundaries = np.flatnonzero(np.diff(ids) != 0) + 1
            runs = [r for r in np.split(np.arange(len(ids)), boundaries) if len(r) >= 2]
        else:
            runs = []

        if not runs:
            logger.debug(f"Fur object {index} has no strands; skipping.")
            return

        # ``order`` concatenates the per-strand point index runs; the Curves
        # datablock stores points strand-by-strand in exactly this order.
        order = np.concatenate(runs)
        sizes = [len(r) for r in runs]

        curves_data = bpy.data.hair_curves.new(f"fur_{index:03d}")
        curves_data.add_curves(sizes)
        curves_data.points.foreach_set(
            "position", vertices[order].astype(np.float32).ravel()
        )
        radius_attr = curves_data.attributes.get("radius")
        if radius_attr is None:
            radius_attr = curves_data.attributes.new("radius", "FLOAT", "POINT")
        radius_attr.data.foreach_set("value", radius[order].astype(np.float32))

        # Carry the per-strand seed normal onto the hair points so the child-hair
        # Geometry Nodes modifier can scatter roots tangentially (along the
        # surface) rather than in an arbitrary 3D direction.
        if mesh_data.has_attribute(STRAND_NORMAL_ATTR):
            ndata = np.asarray(
                mesh_data.attribute(STRAND_NORMAL_ATTR).data, dtype=np.float32
            )
            normal_attr = curves_data.attributes.new(
                STRAND_NORMAL_ATTR, "FLOAT_VECTOR", "POINT"
            )
            normal_attr.data.foreach_set("vector", ndata[order].ravel())

        obj = bpy.data.objects.new(f"fur_{index:03d}", curves_data)
        bpy.context.collection.objects.link(obj)

        if hasattr(view, "global_transform") and view.global_transform is not None:
            obj.matrix_world = mathutils.Matrix(view.global_transform.tolist())

        if view.material_channel is not None:
            mat = self._create_material(view, index)
            if mat:
                obj.data.materials.append(mat)

        # Optional child/clump hairs via a Geometry Nodes modifier: the transform
        # bakes the settings (constant across vertices) when children are asked
        # for, keeping the guide count low while GN multiplies them at render.
        if mesh_data.has_attribute(FUR_CHILDREN_ATTR):
            children = int(np.asarray(mesh_data.attribute(FUR_CHILDREN_ATTR).data)[0])
            clump = float(np.asarray(mesh_data.attribute(FUR_CLUMP_ATTR).data)[0])
            spread = float(np.asarray(mesh_data.attribute(FUR_SPREAD_ATTR).data)[0])
            if children > 1:
                self._apply_fur_child_modifier(obj, index, children, clump, spread)

        logger.debug(f"Created fur object {index} with {len(runs)} hair strands")

    def _apply_fur_child_modifier(
        self, obj, index: int, children: int, clump: float, spread: float
    ):
        """Attach a Geometry Nodes modifier that grows child hairs per guide.

        The node tree duplicates every guide strand ``children`` times, scatters
        each child's root within a ``spread`` radius, and pulls the child tips
        back toward their guide (``clump``) — so a small set of guide strands
        expands into dense, clumped fur only at render time.

        Args:
            obj: The hair-Curves object to modify.
            index: Object index (used for unique naming).
            children: Number of child hairs per guide strand.
            clump: Tip convergence strength in ``[0, 1]``.
            spread: Child-root scatter radius in object space.
        """
        ng = bpy.data.node_groups.new(f"fur_children_{index:03d}", "GeometryNodeTree")
        ng.interface.new_socket(
            "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        ng.interface.new_socket(
            "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
        nodes = ng.nodes
        links = ng.links

        group_in = nodes.new("NodeGroupInput")
        group_in.location = (-800, 0)
        group_out = nodes.new("NodeGroupOutput")
        group_out.location = (960, 0)

        # Duplicate each guide spline into ``children`` copies.
        dup = nodes.new("GeometryNodeDuplicateElements")
        dup.domain = "SPLINE"
        dup.location = (-560, 0)
        dup.inputs["Amount"].default_value = children
        links.new(group_in.outputs[0], dup.inputs["Geometry"])

        # Capture a per-child spline index (constant along each child's points)
        # so the random root offset below is coherent per strand.
        capture = nodes.new("GeometryNodeCaptureAttribute")
        capture.domain = "CURVE"
        capture.location = (-360, 0)
        capture.capture_items.new("INT", "cid")
        spline_index = nodes.new("GeometryNodeInputIndex")
        spline_index.location = (-560, -200)
        links.new(dup.outputs["Geometry"], capture.inputs["Geometry"])
        links.new(spline_index.outputs["Index"], capture.inputs["cid"])

        # Per-child random root offset direction.
        rand = nodes.new("FunctionNodeRandomValue")
        rand.data_type = "FLOAT_VECTOR"
        rand.location = (-140, -220)
        rand.inputs["Min"].default_value = (-1.0, -1.0, -1.0)
        rand.inputs["Max"].default_value = (1.0, 1.0, 1.0)
        links.new(capture.outputs["cid"], rand.inputs["ID"])

        # Clump falloff: offset scale = spread * (1 - clump * factor), so roots
        # (factor 0) spread fully and tips (factor 1) converge toward the guide.
        spline_param = nodes.new("GeometryNodeSplineParameter")
        spline_param.location = (-360, 240)
        mul = nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.location = (-160, 240)
        mul.inputs[1].default_value = clump
        links.new(spline_param.outputs["Factor"], mul.inputs[0])
        sub = nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.location = (20, 240)
        sub.inputs[0].default_value = 1.0
        links.new(mul.outputs["Value"], sub.inputs[1])
        scale = nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.location = (200, 240)
        scale.inputs[1].default_value = spread
        links.new(sub.outputs["Value"], scale.inputs[0])

        # offset = random_direction * scale
        offset = nodes.new("ShaderNodeVectorMath")
        offset.operation = "SCALE"
        offset.location = (200, 0)
        links.new(rand.outputs["Value"], offset.inputs[0])
        links.new(scale.outputs["Value"], offset.inputs["Scale"])

        # Project the offset onto the surface tangent plane (remove the component
        # along the seed normal) so child roots scatter *along* the surface and
        # the fur keeps hugging it: offset_t = offset - (offset . n) n.
        normal_attr = nodes.new("GeometryNodeInputNamedAttribute")
        normal_attr.data_type = "FLOAT_VECTOR"
        normal_attr.location = (0, -420)
        normal_attr.inputs["Name"].default_value = STRAND_NORMAL_ATTR
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.location = (220, -300)
        links.new(offset.outputs["Vector"], dot.inputs[0])
        links.new(normal_attr.outputs["Attribute"], dot.inputs[1])
        normal_comp = nodes.new("ShaderNodeVectorMath")
        normal_comp.operation = "SCALE"
        normal_comp.location = (380, -300)
        links.new(normal_attr.outputs["Attribute"], normal_comp.inputs[0])
        links.new(dot.outputs["Value"], normal_comp.inputs["Scale"])
        tangential = nodes.new("ShaderNodeVectorMath")
        tangential.operation = "SUBTRACT"
        tangential.location = (540, 0)
        links.new(offset.outputs["Vector"], tangential.inputs[0])
        links.new(normal_comp.outputs["Vector"], tangential.inputs[1])

        set_pos = nodes.new("GeometryNodeSetPosition")
        set_pos.location = (720, 0)
        links.new(capture.outputs["Geometry"], set_pos.inputs["Geometry"])
        links.new(tangential.outputs["Vector"], set_pos.inputs["Offset"])
        links.new(set_pos.outputs["Geometry"], group_out.inputs[0])

        modifier = obj.modifiers.new(f"fur_children_{index:03d}", "NODES")
        modifier.node_group = ng
        logger.debug(f"Fur object {index}: added {children} child hairs per guide")

    # Approximate base colors for common Mitsuba conductor presets.
