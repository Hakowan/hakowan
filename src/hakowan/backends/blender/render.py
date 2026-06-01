"""Blender rendering backend implementation."""

from ...common import logger
from ...compiler import Scene, View
from ...setup import Config
from ...grammar import mark
from ...grammar.channel.material import (
    Conductor,
    Dielectric,
    Diffuse,
    Hair,
    Plastic,
    Principled,
    RoughConductor,
    RoughDielectric,
    RoughPlastic,
    ThinDielectric,
    ThinPrincipled,
)
from ...grammar.scale import Attribute
from ...grammar.texture import (
    ScalarField,
    Checkerboard,
    Image,
    Texture,
    Uniform,
)
from ...grammar.channel.curvestyle import Bend
from .. import RenderBackend

from pathlib import Path
from typing import Any
import numpy as np
import lagrange

import bpy
import mathutils


def _ensure_nodes(datablock) -> None:
    """Ensure a material/world has a shader node tree.

    Newer Blender (4.x+) creates the node tree automatically, so the legacy
    ``use_nodes = True`` toggle is deprecated (slated for removal in 6.0).
    Only fall back to it when the node tree is genuinely absent, which keeps
    compatibility with older Blender without emitting deprecation warnings.
    """
    if datablock.node_tree is None:
        datablock.use_nodes = True


class BlenderBackend(RenderBackend):
    """Blender rendering backend.

    Supports:
    - Surface marks (meshes) and curve marks (edges and vector fields)
    - Materials: Diffuse, Plastic, RoughPlastic, Principled, ThinPrincipled,
      Conductor, RoughConductor, Dielectric, ThinDielectric, RoughDielectric, Hair
    - Two-sided materials
    - Textures: ScalarField, Checkerboard, Image
    - Normal maps and bump maps
    - Camera: Perspective (with fov_axis), Orthographic, ThinLens (DOF),
      arbitrary up-vector
    - Render passes: albedo, depth, normal, facet ID
    - Lighting setup
    """

    def render(
        self,
        scene: Scene,
        config: Config,
        filename: Path | str | None = None,
        **kwargs,
    ) -> Any:
        """Render scene using Blender.

        The main image is always rendered first.  When ``config`` enables
        optional passes, additional outputs are written alongside it:

        - **albedo / depth / normal** – extracted from view-layer passes via
          the Blender compositor in the *same* render call.
        - **facet_id** – a second, independent render pass using flat
          Emission shaders and EEVEE (no gamma, no AA).  Written to
          ``<stem>_facet_id<ext>``.

        Args:
            scene: Compiled scene (iterable of :class:`View` objects).
            config: Rendering configuration, including pass flags and camera
                / film / sampler settings.
            filename: Output image path.  Parent directories are created
                automatically.  Pass ``None`` to render without saving.
            **kwargs: Additional backend options:
                - ``engine`` (str): Render engine for the main pass;
                  ``"CYCLES"`` (default) or ``"BLENDER_EEVEE"``.
                - ``blend_file`` (str | Path): If provided, save the Blender
                  scene to this path before rendering (useful for debugging).
                - ``yaml_file`` (str | Path): If provided, serialize the
                  scene configuration to a YAML file at this path.

        Returns:
            ``None`` — Blender writes directly to *filename*.
        """
        logger.info("Starting Blender rendering...")

        # Clear existing scene
        self._clear_scene()

        # Create objects from views
        for i, view in enumerate(scene):
            self._create_view_object(view, i)

        # Setup camera
        self._setup_camera(config)

        # Setup lighting
        self._setup_lighting(config, **kwargs)

        # Setup render settings
        self._setup_render_settings(config, **kwargs)

        # Save .blend file if requested (for debugging)
        blend_file = kwargs.get("blend_file")
        if blend_file is not None:
            if isinstance(blend_file, str):
                blend_file = Path(blend_file)
            self._save_blend_file(blend_file)

        # Render
        renames = []
        _temp_blend: Path | None = None
        if filename is not None:
            if isinstance(filename, str):
                filename = Path(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)
            bpy.context.scene.render.filepath = str(filename.resolve())
            # In headless bpy, bpy.data.filepath is empty so Blender has no
            # "home directory" and the compositor File Output node writes bare
            # filenames with no directory.  Save a temporary .blend file into
            # the output directory so that '//' resolves to filename.parent.
            _temp_blend = filename.parent / ".hakowan_compositor.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(_temp_blend), compress=False)
            renames = self._setup_compositor_passes(config, filename)

        logger.info("Rendering with Blender...")
        bpy.ops.render.render(write_still=True)
        logger.info(f"Rendering saved to {filename}")

        # Move pass files if needed (src == final means already in the right place).
        if renames:
            import shutil

            assert filename is not None
            output_dir = filename.parent
            for pass_name, final in renames:
                src = output_dir / pass_name
                if src.exists():
                    if src != final:
                        shutil.move(str(src), str(final))
                    logger.info(f"Pass saved to {final}")
                else:
                    logger.warning(f"Pass file not found: {pass_name}")

        # Facet-ID pass: second render with flat ID-color materials.
        if config.facet_id and filename is not None:
            self._render_facet_id_pass(filename)

        # Remove the temporary .blend file used to anchor '//' path resolution.
        if _temp_blend is not None and _temp_blend.exists():
            _temp_blend.unlink()

        return None

    def _clear_scene(self):
        """Clear all objects, meshes, materials from Blender scene."""
        # Delete all objects
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        # Clear orphaned data
        for mesh in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)
        for light in bpy.data.lights:
            bpy.data.lights.remove(light)

    def _setup_facet_id_mode(self):
        """Replace every mesh object's materials with flat facet-ID shaders.

        For each ``MESH`` object in the scene:

        1. A ``FLOAT_COLOR`` / ``CORNER``-domain color attribute named
           ``_hakowan_facet_id`` is created on the mesh data block (or
           recreated if it already exists).
        2. Every loop corner of each polygon is set to the linear-float RGB
           encoding of that polygon's zero-based index::

               R = ((fid >> 16) & 0xFF) / 255.0
               G = ((fid >>  8) & 0xFF) / 255.0
               B = ( fid        & 0xFF) / 255.0

        3. All existing materials are replaced by a single ``ShaderNodeEmission``
           material that reads the attribute through ``ShaderNodeAttribute``
           (``attribute_type = "GEOMETRY"``), so lighting has no effect on the
           output color.

        Values are stored as linear floats so that when rendered with the
        ``"Raw"`` view transform the 8-bit PNG channel values equal the
        original byte components of the face index, allowing lossless
        round-trip recovery via ``fid = (R << 16) | (G << 8) | B``.
        """
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH":
                continue
            mesh = obj.data
            if len(mesh.polygons) == 0:
                continue

            # Build a CORNER-domain color attribute (one color per loop index).
            attr_name = "_hakowan_facet_id"
            if attr_name in mesh.color_attributes:
                mesh.color_attributes.remove(mesh.color_attributes[attr_name])
            color_attr = mesh.color_attributes.new(
                name=attr_name, type="FLOAT_COLOR", domain="CORNER"
            )

            for poly in mesh.polygons:
                fid = poly.index
                r = ((fid >> 16) & 0xFF) / 255.0
                g = ((fid >> 8) & 0xFF) / 255.0
                b = (fid & 0xFF) / 255.0
                for loop_idx in range(
                    poly.loop_start, poly.loop_start + poly.loop_total
                ):
                    color_attr.data[loop_idx].color = (r, g, b, 1.0)

            # Flat Emission shader — reads the attribute color, ignores lighting.
            mat = bpy.data.materials.new(name=f"_hakowan_facet_id_{obj.name}")
            _ensure_nodes(mat)
            ntree = mat.node_tree
            ntree.nodes.clear()

            out_node = ntree.nodes.new(type="ShaderNodeOutputMaterial")
            emit_node = ntree.nodes.new(type="ShaderNodeEmission")
            emit_node.inputs["Strength"].default_value = 1.0
            attr_node = ntree.nodes.new(type="ShaderNodeAttribute")
            attr_node.attribute_name = attr_name
            attr_node.attribute_type = "GEOMETRY"

            ntree.links.new(attr_node.outputs["Color"], emit_node.inputs["Color"])
            ntree.links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

            obj.data.materials.clear()
            obj.data.materials.append(mat)

    def _render_facet_id_pass(self, filename: Path):
        """Perform a second Blender render that writes per-facet ID colors.

        Called automatically by :meth:`render` when ``config.facet_id`` is
        ``True``, *after* the main render has completed.

        The output file is placed next to *filename* with a ``_facet_id``
        suffix, e.g. ``bust.png`` → ``bust_facet_id.png``.

        The following settings are applied exclusively for this pass and
        restored afterward so they do not affect any subsequent call:

        - **Engine**: ``BLENDER_EEVEE`` — deterministic rasterisation, no
          path-tracing noise.
        - **TAA samples**: 1 — disables temporal anti-aliasing / blending.
        - **Pixel filter** (``filter_size``): 0.0 — no Gaussian spread across
          pixel boundaries.
        - **View transform**: ``"Raw"`` — bypasses all gamma and tone-mapping
          so pixel channel values equal the stored linear float values.
        - **Compositor node group**: ``None`` — disconnects the compositor so
          albedo / depth / normal file-output nodes are not re-triggered and
          do not overwrite the outputs from the main render.

        Args:
            filename: Main output image path used to derive the facet-ID
                output path (``<stem>_facet_id<ext>``).
        """
        scene = bpy.context.scene

        # Override materials with flat facet-ID emission shaders.
        self._setup_facet_id_mode()

        # Save engine / filter / color-management / compositor state.
        prev_engine = scene.render.engine
        prev_filter = scene.render.filter_size
        prev_transform = scene.view_settings.view_transform
        prev_compositor = scene.compositing_node_group
        prev_taa_samples = scene.eevee.taa_render_samples

        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1
        scene.render.filter_size = 0.0
        scene.view_settings.view_transform = "Raw"
        # Disconnect the compositor so albedo/depth/normal passes are not
        # re-rendered and do not overwrite the outputs from the main render.
        scene.compositing_node_group = None

        # Derive output path: <stem>_facet_id<suffix>
        facet_id_path = filename.parent / (
            filename.stem + "_facet_id" + filename.suffix
        )
        scene.render.filepath = str(facet_id_path.resolve())

        logger.info("Rendering facet-ID pass...")
        bpy.ops.render.render(write_still=True)
        logger.info(f"Facet-ID pass saved to {facet_id_path}")

        # Restore render settings and compositor for any subsequent operations.
        scene.render.engine = prev_engine
        scene.render.filter_size = prev_filter
        scene.view_settings.view_transform = prev_transform
        scene.compositing_node_group = prev_compositor
        scene.eevee.taa_render_samples = prev_taa_samples

    def _extract_size(self, view: View, default_size: float = 0.01):
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
        else:
            tip_size = size

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
            if per_corner.shape[0] != len(mesh.loops):
                logger.warning(
                    "Blender backend: corner normal count "
                    f"({per_corner.shape[0]}) != loop count ({len(mesh.loops)}); "
                    "skipping custom normals."
                )
                return
            mesh.normals_split_custom_set(per_corner.tolist())

    def _create_point_base_mesh(self, base_shape: str, index: int):
        """Create the base mesh for a point mark shape.

        Args:
            base_shape: One of "sphere", "cube", or "disk".
            index: Object index (used for naming).

        Returns:
            Tuple of (base_object, base_mesh, use_smooth, use_subdiv).
        """
        match base_shape:
            case "sphere":
                bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=0, radius=1.0)
                base_obj = bpy.context.active_object
                base_obj.name = f"base_ico_{index:03d}"
                base_obj.data.name = f"ico_sphere_{index:03d}"
                return base_obj, base_obj.data, True, True
            case "cube":
                bpy.ops.mesh.primitive_cube_add(size=2.0)
                base_obj = bpy.context.active_object
                base_obj.name = f"base_cube_{index:03d}"
                base_obj.data.name = f"cube_{index:03d}"
                return base_obj, base_obj.data, False, False
            case "disk":
                bpy.ops.mesh.primitive_circle_add(
                    vertices=32, radius=1.0, fill_type="NGON"
                )
                base_obj = bpy.context.active_object
                base_obj.name = f"base_disk_{index:03d}"
                base_obj.data.name = f"disk_{index:03d}"
                return base_obj, base_obj.data, False, False
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
        """Create Blender point cloud (spheres/cubes/disks at each vertex) from a view.

        Args:
            view: Point view.
            index: Object index.
        """
        if view.data_frame is None:
            return

        mesh_data = view.data_frame.mesh
        vertices = mesh_data.vertices
        n_points = mesh_data.num_vertices

        # Extract covariance transforms if specified
        covariance_transforms = None
        if view.covariance_channel is not None:
            covariance_transforms = self._extract_covariance_transforms(view)

        radii = self._extract_size(
            view, default_size=0.01 if covariance_transforms is None else 1.0
        )
        if np.isscalar(radii):
            # np.isscalar guarantees a scalar here, but mypy can't narrow it.
            radii = [float(radii)] * n_points  # type: ignore[arg-type]
        radii = np.atleast_1d(radii)
        assert len(radii) == n_points

        # Determine base shape
        base_shape = "sphere"
        if view.shape_channel is not None:
            base_shape = view.shape_channel.base_shape

        base_obj, base_mesh, use_smooth, use_subdiv = self._create_point_base_mesh(
            base_shape, index
        )

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

        # Scalar field: get per-vertex color array for points
        point_colors = None
        scalar_info = self._get_scalar_field_color_attr(view)
        if scalar_info is not None:
            attr_name, element_type = scalar_info
            if element_type == lagrange.AttributeElement.Vertex:
                attr = mesh_data.attribute(attr_name)
                data = np.asarray(attr.data)
                if data.ndim == 2 and data.shape[0] == n_points and data.shape[1] >= 3:
                    rgb = data[:, :3]
                    a = data[:, 3] if data.shape[1] >= 4 else np.ones(n_points)
                    point_colors = [
                        (
                            float(rgb[i, 0]),
                            float(rgb[i, 1]),
                            float(rgb[i, 2]),
                            float(a[i]),
                        )
                        for i in range(n_points)
                    ]

        # Parent empty to apply global transform
        empty = bpy.data.objects.new(f"points_empty_{index:03d}", None)
        bpy.context.collection.objects.link(empty)
        if hasattr(view, "global_transform") and view.global_transform is not None:
            empty.matrix_world = mathutils.Matrix(view.global_transform.tolist())

        # Create one shape per point, parent to empty
        for i in range(n_points):
            obj = bpy.data.objects.new(f"point_{index:03d}_{i:06d}", base_mesh.copy())
            bpy.context.collection.objects.link(obj)
            r = float(radii[i])
            obj.parent = empty

            if covariance_transforms is not None:
                local_transform = np.eye(4)
                local_transform[:3, :3] = covariance_transforms[i] * r
                local_transform[:3, 3] = vertices[i]
                obj.matrix_local = mathutils.Matrix(local_transform.tolist())
            elif normals is not None:
                obj.matrix_local = (
                    mathutils.Matrix.Translation(vertices[i].tolist())
                    @ self._compute_orientation_rotation(normals[i])
                    @ mathutils.Matrix.Diagonal((r, r, r, 1.0))
                )
            else:
                obj.location = mathutils.Vector(vertices[i].tolist())
                obj.scale = (r, r, r)

            if use_smooth:
                for poly in obj.data.polygons:
                    poly.use_smooth = True

            if use_subdiv:
                subd_mod = obj.modifiers.new(name="Subdiv", type="SUBSURF")
                subd_mod.levels = 1  # viewport
                subd_mod.render_levels = 3  # render

            if view.material_channel is not None:
                mat = self._create_material(
                    view,
                    index,
                    override_color=point_colors[i] if point_colors else None,
                    material_suffix=f"{i:06d}" if point_colors else None,
                )
                if mat:
                    obj.data.materials.append(mat)

        # Remove the temporary base object (mesh is still used by instances)
        bpy.data.objects.remove(base_obj, do_unlink=True)

        logger.debug(f"Created point object {index} with {n_points} {base_shape}s")

    def _create_curve_object(self, view: View, index: int):
        """Create Blender curve object from view (mesh edges or vector field).

        Args:
            view: Curve view.
            index: Object index.
        """
        if view.data_frame is None:
            return

        mesh_data = view.data_frame.mesh

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
                                        r, g, b = float(c[0]), float(c[1]), float(c[2])
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

    # Approximate base colors for common Mitsuba conductor presets.
    _conductor_colors: dict[str, tuple[float, float, float]] = {
        "Au": (1.0, 0.78, 0.34),
        "Ag": (0.97, 0.96, 0.91),
        "Cu": (0.95, 0.64, 0.54),
        "Al": (0.91, 0.92, 0.92),
        "Fe": (0.56, 0.57, 0.58),
        "Cr": (0.55, 0.55, 0.55),
        "Pt": (0.67, 0.64, 0.59),
        "W": (0.50, 0.50, 0.50),
        "Ti": (0.62, 0.58, 0.54),
        "Ni": (0.66, 0.63, 0.58),
        "V": (0.55, 0.55, 0.55),
        "none": (0.8, 0.8, 0.8),
    }

    # Common IOR values for named Mitsuba dielectric presets.
    _ior_presets: dict[str, float] = {
        "vacuum": 1.0,
        "air": 1.000277,
        "water": 1.333,
        "ice": 1.31,
        "glass": 1.5,
        "bk7": 1.5046,
        "diamond": 2.419,
        "fused_quartz": 1.458,
        "polycarbonate": 1.584,
        "acrylic": 1.49,
        "sodium_chloride": 1.544,
        "amber": 1.55,
        "pet": 1.575,
    }

    def _resolve_ior(self, ior: str | float) -> float:
        """Resolve an IOR value from a string preset name or float."""
        if isinstance(ior, (int, float)):
            return float(ior)
        return self._ior_presets.get(ior, 1.5)

    def _create_material(
        self,
        view: View,
        index: int,
        *,
        color_layer_name: str | None = None,
        override_color: tuple[float, float, float, float] | None = None,
        material_suffix: str | None = None,
        uv_layer_name: str | None = None,
    ):
        """Create Blender material from view's material channel.

        Args:
            view: View with material channel.
            index: Material index.
            color_layer_name: If set, use mesh/curve color attribute (ScalarField).
            override_color: If set, use this RGBA as base color (e.g. per-point).
            material_suffix: Optional suffix for material name (e.g. point index).
            uv_layer_name: If set, use UV layer for texture mapping (Checkerboard).

        Returns:
            Blender material or None.
        """
        if view.material_channel is None:
            return None

        mat_data = view.material_channel
        mat_name = f"material_{index:03d}"
        if material_suffix is not None:
            mat_name = f"{mat_name}_{material_suffix}"
        mat = bpy.data.materials.new(name=mat_name)
        _ensure_nodes(mat)
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Hair and ThinDielectric use dedicated shader node graphs
        if isinstance(mat_data, Hair):
            return self._create_hair_material(mat, mat_data, nodes, links)
        if isinstance(mat_data, ThinDielectric):
            return self._create_thin_dielectric_material(mat, mat_data, nodes, links)

        # Create Principled BSDF
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)

        # Material output
        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        # Base color: checkerboard, scalar field, override, or material default.
        checkerboard_tex = self._get_checkerboard_texture(view)
        if checkerboard_tex is not None and uv_layer_name is not None:
            # Create checkerboard shader node network
            checker_node = self._create_checkerboard_shader(
                nodes, links, checkerboard_tex, uv_layer_name
            )
            if checker_node is not None:
                links.new(checker_node.outputs["Color"], bsdf.inputs["Base Color"])
            else:
                # Fallback if checkerboard creation failed
                bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        elif (
            image_tex := self._get_image_texture(view)
        ) is not None and uv_layer_name is not None:
            tex_node = self._build_image_texture_node(
                image_tex, nodes, links, uv_layer_name, is_data=False
            )
            if tex_node is not None:
                links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
            else:
                bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        elif color_layer_name is not None:
            attr_node = nodes.new(type="ShaderNodeAttribute")
            attr_node.location = (-200, 0)
            attr_node.attribute_name = color_layer_name
            links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
        elif override_color is not None:
            bsdf.inputs["Base Color"].default_value = override_color
        else:
            color = self._extract_material_color(mat_data)
            if color is not None:
                bsdf.inputs["Base Color"].default_value = color
            else:
                bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

        # Configure based on material type (non-color inputs)
        match mat_data:
            case Diffuse():
                bsdf.inputs["Roughness"].default_value = 1.0
                bsdf.inputs["Metallic"].default_value = 0.0

            case Plastic() | RoughPlastic():
                bsdf.inputs["Roughness"].default_value = 1
                bsdf.inputs["Metallic"].default_value = 0.0
                bsdf.inputs["Coat Weight"].default_value = 1
                bsdf.inputs["Coat IOR"].default_value = 1.49
                if isinstance(mat_data, RoughPlastic):
                    # ``alpha`` is the microfacet roughness of the glossy coat.
                    alpha = (
                        mat_data.alpha
                        if isinstance(mat_data.alpha, (int, float))
                        else 0.1
                    )
                    bsdf.inputs["Coat Roughness"].default_value = float(alpha)
                else:
                    bsdf.inputs["Coat Roughness"].default_value = 0

            case RoughConductor():
                bsdf.inputs["Metallic"].default_value = 1.0
                alpha = (
                    mat_data.alpha if isinstance(mat_data.alpha, (int, float)) else 0.1
                )
                bsdf.inputs["Roughness"].default_value = float(alpha)

            case Conductor():
                bsdf.inputs["Metallic"].default_value = 1.0
                bsdf.inputs["Roughness"].default_value = 0.0

            case RoughDielectric():
                bsdf.inputs[
                    "Transmission Weight"
                ].default_value = mat_data.specular_transmittance
                # ``specular_reflectance`` is a [0,1] reflectance multiplier where
                # 1.0 means "unchanged" (Mitsuba convention). Blender's "Specular
                # IOR Level" is neutral at 0.5 (1.0 ≈ double specular), so map
                # multiplier → 0.5 * multiplier.
                bsdf.inputs["Specular IOR Level"].default_value = (
                    0.5 * mat_data.specular_reflectance
                )
                bsdf.inputs["IOR"].default_value = self._resolve_ior(mat_data.int_ior)
                alpha = (
                    mat_data.alpha if isinstance(mat_data.alpha, (int, float)) else 0.1
                )
                bsdf.inputs["Roughness"].default_value = float(alpha)
                bsdf.inputs["Metallic"].default_value = 0.0

            case Dielectric():
                bsdf.inputs[
                    "Transmission Weight"
                ].default_value = mat_data.specular_transmittance
                # See RoughDielectric: remap [0,1] reflectance multiplier (1.0 =
                # unchanged) onto Blender's 0.5-neutral "Specular IOR Level".
                bsdf.inputs["Specular IOR Level"].default_value = (
                    0.5 * mat_data.specular_reflectance
                )
                bsdf.inputs["IOR"].default_value = self._resolve_ior(mat_data.int_ior)
                bsdf.inputs["Roughness"].default_value = 0.0
                bsdf.inputs["Metallic"].default_value = 0.0

            case ThinPrincipled():
                bsdf.inputs["Roughness"].default_value = float(
                    mat_data.roughness
                    if isinstance(mat_data.roughness, (int, float))
                    else 0.5
                )
                bsdf.inputs["Metallic"].default_value = float(
                    mat_data.metallic
                    if isinstance(mat_data.metallic, (int, float))
                    else 0.0
                )
                bsdf.inputs["Transmission Weight"].default_value = mat_data.spec_trans
                bsdf.inputs["IOR"].default_value = mat_data.eta
                mat.use_backface_culling = False

            case Principled():
                bsdf.inputs["Roughness"].default_value = float(
                    mat_data.roughness
                    if isinstance(mat_data.roughness, (int, float))
                    else 0.5
                )
                bsdf.inputs["Metallic"].default_value = float(
                    mat_data.metallic
                    if isinstance(mat_data.metallic, (int, float))
                    else 0.0
                )
                bsdf.inputs["Anisotropic"].default_value = mat_data.anisotropic
                bsdf.inputs["Transmission Weight"].default_value = mat_data.spec_trans
                bsdf.inputs["IOR"].default_value = mat_data.eta
                bsdf.inputs["Sheen Weight"].default_value = mat_data.sheen

            case _:
                if color_layer_name is None and override_color is None:
                    logger.warning(
                        f"Material type {type(mat_data)} not fully supported, using default"
                    )

        # Normal / bump maps drive the Principled BSDF's "Normal" input. When
        # both are present the normal map feeds the bump node so they compose
        # (mirrors Mitsuba's bumpmap-over-normalmap nesting). Both require UVs.
        normal_socket = None
        if uv_layer_name is not None and view.normal_map is not None:
            normal_socket = self._build_normal_map_node(
                view.normal_map, nodes, links, uv_layer_name
            )
        if uv_layer_name is not None and view.bump_map is not None:
            bump_socket = self._build_bump_map_node(
                view.bump_map, nodes, links, uv_layer_name, normal_socket
            )
            if bump_socket is not None:
                normal_socket = bump_socket
        if normal_socket is not None and "Normal" in bsdf.inputs:
            links.new(normal_socket, bsdf.inputs["Normal"])

        # Two-sided rendering: disable backface culling
        if mat_data.two_sided:
            mat.use_backface_culling = False

        return mat

    def _create_hair_material(self, mat, mat_data: Hair, nodes, links):
        """Create a Principled Hair BSDF material.

        Args:
            mat: Blender material.
            mat_data: Hair material data.
            nodes: Shader node tree nodes.
            links: Shader node tree links.

        Returns:
            Blender material.
        """
        hair_bsdf = nodes.new(type="ShaderNodeBsdfHairPrincipled")
        hair_bsdf.location = (0, 0)
        # Use melanin concentration parametrization
        hair_bsdf.parametrization = "MELANIN"
        hair_bsdf.inputs["Melanin"].default_value = mat_data.eumelanin / 8.0
        hair_bsdf.inputs["Melanin Redness"].default_value = (
            mat_data.pheomelanin / (mat_data.eumelanin + mat_data.pheomelanin)
            if (mat_data.eumelanin + mat_data.pheomelanin) > 0
            else 0.5
        )
        hair_bsdf.inputs["Roughness"].default_value = 0.3
        hair_bsdf.inputs["Coat"].default_value = 0.0

        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)
        links.new(hair_bsdf.outputs["BSDF"], output.inputs["Surface"])

        return mat

    def _create_thin_dielectric_material(
        self, mat, mat_data: ThinDielectric, nodes, links
    ):
        """Create a thin dielectric (thin glass) material.

        Models a thin sheet of glass where the two refractions cancel out:
        light passes straight through while Fresnel reflections are preserved.
        Built as a Fresnel-driven mix of Transparent BSDF and Glossy BSDF.

        Args:
            mat: Blender material.
            mat_data: ThinDielectric material data.
            nodes: Shader node tree nodes.
            links: Shader node tree links.

        Returns:
            Blender material.
        """
        ior = self._resolve_ior(mat_data.int_ior)

        # Transparent BSDF: light passes straight through
        transparent = nodes.new(type="ShaderNodeBsdfTransparent")
        transparent.location = (-200, 100)

        # Glass BSDF: reflection + refraction for a thicker glass appearance
        glass = nodes.new(type="ShaderNodeBsdfGlass")
        glass.location = (-200, -100)
        glass.inputs["Roughness"].default_value = 0.0
        glass.inputs["IOR"].default_value = ior

        # Fresnel node drives the mix based on viewing angle and IOR
        fresnel = nodes.new(type="ShaderNodeFresnel")
        fresnel.location = (-400, 0)
        fresnel.inputs["IOR"].default_value = ior

        # Scale Fresnel by specular_reflectance to control reflection intensity
        scale = nodes.new(type="ShaderNodeMath")
        scale.location = (-200, 0)
        scale.operation = "MULTIPLY"
        links.new(fresnel.outputs["Fac"], scale.inputs[0])
        scale.inputs[1].default_value = mat_data.specular_reflectance

        # Mix Shader: factor=0 → transparent, factor=1 → glass
        mix = nodes.new(type="ShaderNodeMixShader")
        mix.location = (0, 0)
        links.new(scale.outputs["Value"], mix.inputs["Fac"])
        links.new(transparent.outputs["BSDF"], mix.inputs[1])
        links.new(glass.outputs["BSDF"], mix.inputs[2])

        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (200, 0)
        links.new(mix.outputs["Shader"], output.inputs["Surface"])

        # Thin dielectric is inherently two-sided
        mat.use_backface_culling = False

        return mat

    def _create_checkerboard_shader(
        self, nodes, links, checkerboard_tex: Checkerboard, uv_layer_name: str
    ):
        """Create a checkerboard shader node network.

        Args:
            nodes: Shader node tree nodes.
            links: Shader node tree links.
            checkerboard_tex: Checkerboard texture configuration.
            uv_layer_name: Name of the UV layer to use.

        Returns:
            Output node that provides the checkerboard color, or None if failed.
        """
        # UV Map node
        uv_node = nodes.new(type="ShaderNodeUVMap")
        uv_node.location = (-800, 0)
        uv_node.uv_map = uv_layer_name

        # Mapping node for scaling
        mapping_node = nodes.new(type="ShaderNodeMapping")
        mapping_node.location = (-600, 0)
        mapping_node.inputs["Scale"].default_value = (
            float(checkerboard_tex.size),
            float(checkerboard_tex.size),
            1.0,
        )
        links.new(uv_node.outputs["UV"], mapping_node.inputs["Vector"])

        # Checker Texture node
        checker_node = nodes.new(type="ShaderNodeTexChecker")
        checker_node.location = (-400, 0)
        checker_node.inputs["Scale"].default_value = 1.0  # Scale is handled by mapping
        links.new(mapping_node.outputs["Vector"], checker_node.inputs["Vector"])

        # Extract colors from texture1 and texture2
        color1 = self._texture_to_color(checkerboard_tex.texture1)
        color2 = self._texture_to_color(checkerboard_tex.texture2)

        # Set checker colors
        if color1 is not None:
            checker_node.inputs["Color1"].default_value = color1
        if color2 is not None:
            checker_node.inputs["Color2"].default_value = color2

        return checker_node

    def _texture_to_color(self, texture) -> tuple[float, float, float, float] | None:
        """Convert a TextureLike to an RGBA color.

        Args:
            texture: Texture or color value.

        Returns:
            RGBA tuple or None if cannot be converted.
        """
        if isinstance(texture, Uniform):
            return self._extract_color(texture.color)
        elif isinstance(texture, (str, int, float, list, tuple)):
            return self._extract_color(texture)
        else:
            # Cannot convert complex textures like ScalarField or nested Checkerboard
            logger.warning(
                f"Cannot convert texture type {type(texture)} to color for checkerboard"
            )
            return None

    def _extract_material_color(
        self, mat_data
    ) -> tuple[float, float, float, float] | None:
        """Extract base color from any supported material type.

        Args:
            mat_data: Material channel data.

        Returns:
            RGBA tuple or None.
        """
        match mat_data:
            case Diffuse():
                return self._extract_color(mat_data.reflectance)
            case Plastic() | RoughPlastic():
                return self._extract_color(mat_data.diffuse_reflectance)
            case Principled() | ThinPrincipled():
                return self._extract_color(mat_data.color)
            case RoughConductor() | Conductor():
                name = mat_data.material
                rgb = self._conductor_colors.get(name, (0.8, 0.8, 0.8))
                return (rgb[0], rgb[1], rgb[2], 1.0)
            case Dielectric() | ThinDielectric() | RoughDielectric():
                # Glass-like materials: white/clear base
                return (1.0, 1.0, 1.0, 1.0)
            case _:
                return None

    def _extract_color(self, color_data) -> tuple[float, float, float, float] | None:
        """Extract RGBA color from various color representations.

        Args:
            color_data: Color data (str, tuple, or texture).

        Returns:
            RGBA tuple or None.
        """
        from ...common.to_color import to_color

        if isinstance(color_data, str):
            rgb = to_color(color_data)
            return (rgb[0], rgb[1], rgb[2], 1.0)
        elif isinstance(color_data, (list, tuple)):
            if len(color_data) == 3:
                return (color_data[0], color_data[1], color_data[2], 1.0)
            elif len(color_data) == 4:
                return tuple(color_data)
        # Texture types (e.g. ScalarField) are handled via mesh color attributes
        return None

    def _get_scalar_field_color_attr(self, view: View):
        """If the view's material uses a ScalarField color, return (attr_name, element_type).

        Returns:
            (attr_name, element_type) from the mesh, or None if not a scalar field color.
        """
        if view.material_channel is None or view.data_frame is None:
            return None
        mat_data = view.material_channel
        mesh = view.data_frame.mesh

        def check(tex):
            if isinstance(tex, ScalarField):
                if not isinstance(tex.data, Attribute) or not getattr(
                    tex.data, "_internal_color_field", None
                ):
                    return None
                name = tex.data._internal_color_field
                if not mesh.has_attribute(name):
                    return None
                attr = mesh.attribute(name)
                return (name, attr.element_type)
            return None

        match mat_data:
            case Diffuse():
                out = check(mat_data.reflectance)
            case Plastic() | RoughPlastic():
                out = check(mat_data.diffuse_reflectance)
            case Principled() | ThinPrincipled():
                out = check(mat_data.color)
            case _:
                out = None
        return out

    def _get_checkerboard_texture(self, view: View):
        """If the view's material uses a Checkerboard texture, return the texture.

        Returns:
            Checkerboard texture or None.
        """
        if view.material_channel is None or view.data_frame is None:
            return None
        mat_data = view.material_channel

        def check(tex):
            if isinstance(tex, Checkerboard):
                return tex
            return None

        match mat_data:
            case Diffuse():
                return check(mat_data.reflectance)
            case Plastic() | RoughPlastic():
                return check(mat_data.diffuse_reflectance)
            case Principled() | ThinPrincipled():
                return check(mat_data.color)
            case _:
                return None

    def _get_image_texture(self, view: View) -> Image | None:
        """Return the material's reflectance/color ``Image`` texture, or None."""
        if view.material_channel is None:
            return None
        mat_data = view.material_channel

        def check(tex):
            return tex if isinstance(tex, Image) else None

        match mat_data:
            case Diffuse():
                return check(mat_data.reflectance)
            case Plastic() | RoughPlastic():
                return check(mat_data.diffuse_reflectance)
            case Principled() | ThinPrincipled():
                return check(mat_data.color)
            case _:
                return None

    def _resolve_uv_attr_name(self, view: View) -> str | None:
        """Internal name of the UV attribute used by any UV-bearing texture.

        Checks the base-color Checkerboard/Image plus the normal/bump map
        textures (the compiler stores the resolved UV on ``texture._uv``).
        Returns the first available, or None.
        """
        candidates: list = []
        cb = self._get_checkerboard_texture(view)
        if cb is not None:
            candidates.append(cb)
        img = self._get_image_texture(view)
        if img is not None:
            candidates.append(img)
        for channel in (view.normal_map, view.bump_map):
            if channel is not None and isinstance(channel.texture, Texture):
                candidates.append(channel.texture)
        for tex in candidates:
            uv = getattr(tex, "_uv", None)
            if isinstance(uv, Attribute) and uv._internal_name is not None:
                return uv._internal_name
        return None

    def _load_blender_image(self, image: Image, *, is_data: bool):
        """Load an ``Image`` texture into a Blender image datablock.

        Applies the texture's ``saturation`` / ``whiteness`` adjustments via PIL
        when non-default, and sets the colour space ("Non-Color" for raw/data
        images such as normal and bump maps; "sRGB" otherwise).
        """
        from pathlib import Path as _Path

        path = str(_Path(image.filename).resolve())
        if image.saturation != 1.0 or image.whiteness != 0.0:
            import tempfile
            from PIL import Image as PILImage, ImageEnhance

            img = PILImage.open(path).convert("RGBA")
            if image.saturation != 1.0:
                img = ImageEnhance.Color(img).enhance(image.saturation)
            if image.whiteness != 0.0:
                white = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
                img = PILImage.blend(img, white, alpha=image.whiteness)
            tmp = tempfile.mktemp(suffix=".png")
            img.save(tmp)
            path = tmp
        bimg = bpy.data.images.load(path)
        bimg.colorspace_settings.name = (
            "Non-Color" if (is_data or image.raw) else "sRGB"
        )
        return bimg

    def _build_image_texture_node(
        self,
        image: Image,
        nodes,
        links,
        uv_layer_name: str,
        *,
        is_data: bool,
        y: int = 0,
    ):
        """Create a UV-mapped ``ShaderNodeTexImage`` node, or None on failure."""
        try:
            bimg = self._load_blender_image(image, is_data=is_data)
        except Exception as e:  # pragma: no cover - bad path / unreadable image
            logger.warning(
                f"Blender backend: failed to load image '{image.filename}': {e}"
            )
            return None
        uv_node = nodes.new(type="ShaderNodeUVMap")
        uv_node.location = (-800, y)
        uv_node.uv_map = uv_layer_name
        tex_node = nodes.new(type="ShaderNodeTexImage")
        tex_node.location = (-600, y)
        tex_node.image = bimg
        links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
        return tex_node

    def _build_normal_map_node(self, normal_map, nodes, links, uv_layer_name: str):
        """Build a normal-map node chain; return its ``Normal`` output socket."""
        tex = normal_map.texture
        if not isinstance(tex, Image):
            logger.warning(
                f"Blender backend: NormalMap.texture type {type(tex).__name__} "
                "not supported (only Image is wired)."
            )
            return None
        tex_node = self._build_image_texture_node(
            tex, nodes, links, uv_layer_name, is_data=True, y=-400
        )
        if tex_node is None:
            return None
        nm = nodes.new(type="ShaderNodeNormalMap")
        nm.location = (-400, -400)
        nm.uv_map = uv_layer_name
        links.new(tex_node.outputs["Color"], nm.inputs["Color"])
        return nm.outputs["Normal"]

    def _build_bump_map_node(
        self, bump_map, nodes, links, uv_layer_name: str, base_normal_socket
    ):
        """Build a bump node chain; return its ``Normal`` output socket.

        ``base_normal_socket`` (e.g. from a normal map) is fed into the bump
        node's ``Normal`` input so the two compose.
        """
        tex = bump_map.texture
        if not isinstance(tex, Image):
            logger.warning(
                f"Blender backend: BumpMap.texture type {type(tex).__name__} "
                "not supported (only Image is wired)."
            )
            return None
        tex_node = self._build_image_texture_node(
            tex, nodes, links, uv_layer_name, is_data=True, y=-700
        )
        if tex_node is None:
            return None
        bump = nodes.new(type="ShaderNodeBump")
        bump.location = (-400, -700)
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = float(bump_map.scale)
        links.new(tex_node.outputs["Color"], bump.inputs["Height"])
        if base_normal_socket is not None:
            links.new(base_normal_socket, bump.inputs["Normal"])
        return bump.outputs["Normal"]

    def _copy_mesh_color_to_blender(
        self,
        lagrange_mesh,
        bpy_mesh,
        attr_name: str,
        element_type,
        color_layer_name: str = "Color",
    ):
        """Copy Lagrange mesh color attribute to a Blender mesh color attribute."""
        attr = lagrange_mesh.attribute(attr_name)
        data = np.asarray(attr.data)
        if data.ndim == 2 and data.shape[1] >= 3:
            colors = (
                data[:, :4]
                if data.shape[1] >= 4
                else np.column_stack([data[:, :3], np.ones(len(data))])
            )
        else:
            return

        if element_type == lagrange.AttributeElement.Vertex:
            bpy_mesh.color_attributes.new(
                name=color_layer_name,
                type="FLOAT_COLOR",
                domain="POINT",
            )
            layer = bpy_mesh.color_attributes[color_layer_name]
            if len(colors) != len(layer.data):
                logger.warning(
                    f"Color attribute '{attr_name}' has {len(colors)} entries but "
                    f"Blender mesh has {len(layer.data)} vertices; truncating to the shorter length."
                )
            for i, c in enumerate(colors[: len(layer.data)]):
                layer.data[i].color = (
                    float(c[0]),
                    float(c[1]),
                    float(c[2]),
                    float(c[3]),
                )
        elif element_type == lagrange.AttributeElement.Facet:
            bpy_mesh.color_attributes.new(
                name=color_layer_name,
                type="FLOAT_COLOR",
                domain="CORNER",
            )
            layer = bpy_mesh.color_attributes[color_layer_name]
            loop_idx = 0
            for face in bpy_mesh.polygons:
                c = colors[face.index]
                for _ in range(face.loop_total):
                    if loop_idx < len(layer.data):
                        layer.data[loop_idx].color = (
                            float(c[0]),
                            float(c[1]),
                            float(c[2]),
                            float(c[3]),
                        )
                    loop_idx += 1
        else:
            logger.warning(
                f"Blender backend: unsupported color element type {element_type}"
            )

    def _copy_mesh_uv_to_blender(
        self,
        lagrange_mesh,
        bpy_mesh,
        attr_name: str,
        uv_layer_name: str = "UVMap",
    ):
        """Copy Lagrange mesh UV attribute to a Blender mesh UV map.

        Args:
            lagrange_mesh: Lagrange mesh with UV attribute.
            bpy_mesh: Blender mesh to copy UV data into.
            attr_name: Name of the UV attribute in the Lagrange mesh.
            uv_layer_name: Name of the UV layer to create in Blender.
        """
        if not lagrange_mesh.has_attribute(attr_name):
            logger.warning(f"UV attribute '{attr_name}' not found in mesh")
            return

        # Create UV layer
        if uv_layer_name not in bpy_mesh.uv_layers:
            bpy_mesh.uv_layers.new(name=uv_layer_name)
        uv_layer = bpy_mesh.uv_layers[uv_layer_name]

        # Handle indexed attributes (common after compilation/finalization)
        if lagrange_mesh.is_attribute_indexed(attr_name):
            indexed_attr = lagrange_mesh.indexed_attribute(attr_name)
            uv_values = np.asarray(indexed_attr.values.data)
            uv_indices = np.asarray(indexed_attr.indices.data)

            # UV values should be 2D (N x 2)
            if uv_values.ndim != 2 or uv_values.shape[1] < 2:
                logger.warning(
                    f"UV attribute '{attr_name}' has invalid shape: {uv_values.shape}"
                )
                return

            # Indexed UVs: expand using indices to corner UVs
            for i, idx in enumerate(uv_indices.flat):
                if i < len(uv_layer.data) and idx < len(uv_values):
                    uv_layer.data[i].uv = (
                        float(uv_values[idx, 0]),
                        float(uv_values[idx, 1]),
                    )
        else:
            # Handle non-indexed attributes
            attr = lagrange_mesh.attribute(attr_name)
            uv_data = np.asarray(attr.data)

            # UV data should be 2D (N x 2)
            if uv_data.ndim != 2 or uv_data.shape[1] < 2:
                logger.warning(
                    f"UV attribute '{attr_name}' has invalid shape: {uv_data.shape}"
                )
                return

            element_type = attr.element_type

            if element_type == lagrange.AttributeElement.Vertex:
                # Vertex UVs: expand to corner UVs
                for poly in bpy_mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        loop = bpy_mesh.loops[loop_idx]
                        vertex_idx = loop.vertex_index
                        if vertex_idx < len(uv_data):
                            uv_layer.data[loop_idx].uv = (
                                float(uv_data[vertex_idx, 0]),
                                float(uv_data[vertex_idx, 1]),
                            )
            elif element_type == lagrange.AttributeElement.Corner:
                # Corner/loop UVs: direct mapping
                for i, uv in enumerate(uv_data):
                    if i < len(uv_layer.data):
                        uv_layer.data[i].uv = (float(uv[0]), float(uv[1]))
            else:
                logger.warning(
                    f"Blender backend: unsupported UV element type {element_type}"
                )

    def _setup_camera(self, config: Config):
        """Setup Blender camera from config.

        Args:
            config: Rendering configuration.
        """
        from ...setup.sensor import Orthographic, ThinLens

        # Create camera
        camera_data = bpy.data.cameras.new(name="Camera")
        camera_obj = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera_obj)
        bpy.context.scene.camera = camera_obj

        sensor = config.sensor
        location = np.asarray(sensor.location, dtype=float)
        target = np.asarray(getattr(sensor, "target", [0.0, 0.0, 0.0]), dtype=float)
        up = np.asarray(getattr(sensor, "up", [0.0, 1.0, 0.0]), dtype=float)
        camera_obj.location = mathutils.Vector(location)

        # Build the orientation from an explicit look-at basis so the sensor's
        # ``up`` vector is honored (Blender's to_track_quat only takes an up-axis
        # hint and silently ignores arbitrary roll, e.g. for z-up scenes).
        # Blender cameras look down -Z with +Y up and +X right.
        z_axis = location - target  # camera +Z points away from the target
        nz = np.linalg.norm(z_axis)
        z_axis = z_axis / nz if nz > 1e-12 else np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(up, z_axis)
        nx = np.linalg.norm(x_axis)
        if nx < 1e-9:
            # up is (anti)parallel to the view direction; pick any orthogonal axis.
            fallback = (
                np.array([1.0, 0.0, 0.0])
                if abs(z_axis[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            x_axis = np.cross(fallback, z_axis)
            nx = np.linalg.norm(x_axis)
        x_axis = x_axis / nx
        y_axis = np.cross(z_axis, x_axis)
        rot = mathutils.Matrix(
            (
                (x_axis[0], y_axis[0], z_axis[0]),
                (x_axis[1], y_axis[1], z_axis[1]),
                (x_axis[2], y_axis[2], z_axis[2]),
            )
        )
        camera_obj.rotation_euler = rot.to_euler()

        # Clipping planes.
        camera_data.clip_start = float(sensor.near_clip)
        camera_data.clip_end = float(sensor.far_clip)

        width = config.film.width
        height = config.film.height

        if isinstance(sensor, Orthographic):
            camera_data.type = "ORTHO"
            # The scene is normalized to fit a [-1, 1] box, which is also what
            # Mitsuba's orthographic camera frames; an ortho scale of 2 (the full
            # extent) matches that framing.
            camera_data.ortho_scale = 2.0
        else:
            # Perspective / ThinLens.
            camera_data.type = "PERSP"
            camera_data.lens_unit = "FOV"
            fov = getattr(sensor, "fov", 28.8415)
            fov_axis = getattr(sensor, "fov_axis", "smaller")
            fit = self._resolve_sensor_fit(fov_axis, width, height)
            if fit is None:
                # Blender has no diagonal fit; AUTO applies the angle to the
                # larger dimension as an approximation.
                logger.debug(
                    f"Blender backend: fov_axis '{fov_axis}' not directly "
                    "supported; approximating with AUTO (larger axis)."
                )
                camera_data.sensor_fit = "AUTO"
            else:
                camera_data.sensor_fit = fit
            camera_data.angle = np.radians(fov)

            if isinstance(sensor, ThinLens):
                # Depth of field. Mitsuba's ``aperture_radius`` is a world-space
                # lens radius while Blender models DOF via an f-stop, so this is a
                # best-effort mapping: larger aperture -> smaller f-stop -> more
                # blur (radius 0.1 -> f/2.8).
                camera_data.dof.use_dof = True
                focus = float(sensor.focus_distance)
                if focus <= 0.0:
                    focus = float(np.linalg.norm(location - target))
                camera_data.dof.focus_distance = focus
                radius = max(float(sensor.aperture_radius), 1e-4)
                camera_data.dof.aperture_fstop = 0.28 / radius

        logger.debug(f"Camera set at {location}, target {target}, up {up}")

    @staticmethod
    def _resolve_sensor_fit(fov_axis: str, width: int, height: int) -> str | None:
        """Map a hakowan ``fov_axis`` to a Blender camera ``sensor_fit``.

        Returns ``"HORIZONTAL"`` / ``"VERTICAL"``, or ``None`` for axes Blender
        cannot express directly (``"diagonal"`` or unknown values).
        """
        if fov_axis == "x":
            return "HORIZONTAL"
        if fov_axis == "y":
            return "VERTICAL"
        if fov_axis == "smaller":
            return "HORIZONTAL" if width <= height else "VERTICAL"
        if fov_axis == "larger":
            return "HORIZONTAL" if width >= height else "VERTICAL"
        return None

    def _setup_lighting(self, config: Config, **kwargs):
        """Setup Blender lighting from config.

        Args:
            config: Rendering configuration.
            **kwargs: Additional options (currently unused).
        """
        from ...setup.emitter import Envmap, Point

        if not config.emitters:
            # No emitters specified, add default sun light
            light_data = bpy.data.lights.new(name="Sun", type="SUN")
            light_data.energy = 1.0
            light_obj = bpy.data.objects.new("Sun", light_data)
            bpy.context.collection.objects.link(light_obj)
            light_obj.location = (0, 0, 10)
            logger.debug("Added default sun light")
            return

        # Process emitters from config
        for i, emitter in enumerate(config.emitters):
            if isinstance(emitter, Envmap):
                self._setup_environment_light(emitter)
            elif isinstance(emitter, Point):
                self._setup_point_light(emitter, i)
            else:
                logger.warning(
                    f"Emitter type {type(emitter)} not supported in Blender backend"
                )

    def _setup_environment_light(self, envmap):
        """Setup environment lighting using world shader.

        Args:
            envmap: Envmap emitter configuration.
        """
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        _ensure_nodes(world)
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create Environment Texture node
        env_tex = nodes.new(type="ShaderNodeTexEnvironment")
        env_tex.location = (-300, 300)

        # Load environment map image
        if envmap.filename.exists():
            env_tex.image = bpy.data.images.load(str(envmap.filename))
            logger.info(f"Loaded environment map: {envmap.filename}")
        else:
            logger.warning(f"Environment map not found: {envmap.filename}")

        # Create Background shader
        background = nodes.new(type="ShaderNodeBackground")
        background.location = (0, 300)
        background.inputs["Strength"].default_value = envmap.scale

        # Create Mapping node for rotation
        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.location = (-600, 300)

        # Apply rotation around up vector
        # Convert rotation from degrees to radians
        rotation_rad = np.radians(envmap.rotation + 180)

        # Determine rotation axis based on up vector
        up = np.array(envmap.up)
        if np.allclose(up, [0, 1, 0]):
            # Y-up: rotate around Y axis
            mapping.inputs["Rotation"].default_value = (0, rotation_rad, 0)
        elif np.allclose(up, [0, 0, 1]):
            # Z-up: rotate around Z axis
            mapping.inputs["Rotation"].default_value = (0, 0, rotation_rad)
        elif np.allclose(up, [1, 0, 0]):
            # X-up: rotate around X axis
            mapping.inputs["Rotation"].default_value = (rotation_rad, 0, 0)
        else:
            logger.warning(
                f"Non-standard up vector {up}, defaulting to Y-axis rotation"
            )
            mapping.inputs["Rotation"].default_value = (0, rotation_rad, 0)

        # Create Texture Coordinate node
        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        tex_coord.location = (-900, 300)

        # Create World Output node
        world_output = nodes.new(type="ShaderNodeOutputWorld")
        world_output.location = (300, 300)

        # Connect nodes
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
        links.new(env_tex.outputs["Color"], background.inputs["Color"])
        links.new(background.outputs["Background"], world_output.inputs["Surface"])

        logger.debug(
            f"Environment light configured with scale={envmap.scale}, rotation={envmap.rotation}°"
        )

    def _setup_point_light(self, point_light, index: int):
        """Setup a point light source.

        Args:
            point_light: Point emitter configuration.
            index: Light index for naming.
        """
        light_data = bpy.data.lights.new(name=f"Point_{index:03d}", type="POINT")

        # Set intensity
        if isinstance(point_light.intensity, (int, float)):
            light_data.energy = float(point_light.intensity)
        else:
            # If intensity is a color, use its average as energy
            light_data.energy = 1.0
            logger.warning(
                "Color intensity for point lights not fully supported, using default energy"
            )

        # Create light object
        light_obj = bpy.data.objects.new(f"Point_{index:03d}", light_data)
        bpy.context.collection.objects.link(light_obj)

        # Set position
        light_obj.location = point_light.position

        logger.debug(f"Point light {index} added at {point_light.position}")

    def _setup_render_settings(self, config: Config, **kwargs):
        """Configure Blender render settings for the *main* render pass.

        Sets resolution, render engine, sample count, output format, and
        background transparency based on *config* and *kwargs*.  These
        settings apply to the main image render only; the facet-ID pass
        manages its own settings inside :meth:`_render_facet_id_pass`.

        Args:
            config: Rendering configuration (film size, sampler, etc.).
            **kwargs: Additional backend options:
                - ``engine`` (str): Blender render engine to use;
                  ``"CYCLES"`` (default) or ``"BLENDER_EEVEE"``.
        """
        scene = bpy.context.scene

        # Resolution
        scene.render.resolution_x = config.film.width
        scene.render.resolution_y = config.film.height
        scene.render.resolution_percentage = 100

        # Render engine
        engine = kwargs.get("engine", "CYCLES")
        scene.render.engine = engine

        # Samples
        if engine == "CYCLES":
            scene.cycles.samples = config.sampler.sample_count
        elif engine == "BLENDER_EEVEE":
            scene.eevee.taa_render_samples = config.sampler.sample_count

        # File format
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"

        # Transparent background
        scene.render.film_transparent = True

        logger.debug(
            f"Render settings: {scene.render.resolution_x}x{scene.render.resolution_y}, engine={engine}"
        )

    def _setup_compositor_passes(
        self, config: Config, filename: Path
    ) -> list[tuple[str, Path]]:
        """Set up compositor nodes for albedo, depth, and normal passes.

        Enables the relevant view layer passes and routes each through a
        File Output node.  Returns a list of (temp_path, final_path) pairs
        that must be renamed after the render completes (Blender appends a
        4-digit frame number to File Output paths).

        Args:
            config: Rendering configuration.
            filename: Main output image path (used to derive pass filenames).

        Returns:
            List of (temp_path, final_path) rename pairs.
        """
        renames: list[tuple[str, Path]] = []
        if not (config.albedo or config.depth or config.normal):
            return renames

        scene = bpy.context.scene

        # Enable passes on the view layer first so the Render Layers
        # node exposes the right output sockets when added below.
        view_layer = scene.view_layers[0]
        if config.albedo:
            view_layer.use_pass_diffuse_color = True
        if config.depth:
            view_layer.use_pass_z = True
        if config.normal:
            view_layer.use_pass_normal = True

        # Blender 5.0: compositor node tree is now an independent data block.
        # Create it via bpy.data.node_groups and assign to scene.compositing_node_group.
        tree = bpy.data.node_groups.new("hakowan_compositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
        nodes = tree.nodes
        links = tree.links

        rl_node = nodes.new(type="CompositorNodeRLayers")
        rl_node.location = (-400, 200)

        # Blender 5.0: CompositorNodeComposite was removed; use NodeGroupOutput
        # with an interface socket for the main image output.
        output = nodes.new(type="NodeGroupOutput")
        output.location = (400, 200)
        tree.interface.new_socket(
            name="Image", in_out="OUTPUT", socket_type="NodeSocketColor"
        )
        links.new(rl_node.outputs["Image"], output.inputs["Image"])

        stem = filename.stem
        suffix = filename.suffix.lower()

        # Map the user's output suffix to a Blender file-format token.
        fmt_map = {".exr": "OPEN_EXR", ".png": "PNG", ".jpg": "JPEG"}
        file_format = fmt_map.get(suffix, "PNG")

        y = -100

        # '//' resolves to filename.parent because we saved a temp .blend there
        # in render() before calling this method.

        if config.albedo:
            fo = nodes.new(type="CompositorNodeOutputFile")
            # Use '//' so Blender resolves to output_dir (the temp .blend location).
            # Clear file_name to avoid a spurious prefix in the output path.
            fo.directory = "//"
            fo.file_name = ""
            # Blender 5.0: media_type must be set to "IMAGE" before file_format.
            fo.format.media_type = "IMAGE"
            fo.format.file_format = file_format
            fo.format.color_mode = "RGB"
            fo.file_output_items.new("RGBA", stem + "_albedo")
            fo.location = (400, y)
            # Blender 5.0: "DiffCol" renamed to "Diffuse Color"
            links.new(rl_node.outputs["Diffuse Color"], fo.inputs[0])
            pass_name = f"{stem}_albedo{suffix}"
            renames.append((pass_name, filename.with_name(pass_name)))
            y -= 200

        if config.depth:
            # Normalize the floating-point depth to [0, 1] before saving.
            normalize = nodes.new(type="CompositorNodeNormalize")
            normalize.location = (0, y)
            links.new(rl_node.outputs["Depth"], normalize.inputs[0])
            fo = nodes.new(type="CompositorNodeOutputFile")
            fo.directory = "//"
            fo.file_name = ""
            fo.format.media_type = "IMAGE"
            fo.format.file_format = file_format
            fo.format.color_mode = "BW"
            fo.file_output_items.new("FLOAT", stem + "_depth")
            fo.location = (400, y)
            links.new(normalize.outputs[0], fo.inputs[0])
            pass_name = f"{stem}_depth{suffix}"
            renames.append((pass_name, filename.with_name(pass_name)))
            y -= 200

        if config.normal:
            # Remap world-space normals from [-1, 1] to [0, 1]: out = N * 0.5 + 0.5
            # CompositorNodeColorBalance was redesigned in Blender 5.0 (LGG mode removed).
            # Use VectorMath nodes instead since the Normal pass is a Vector type.
            mul_node = nodes.new(type="ShaderNodeVectorMath")
            mul_node.operation = "MULTIPLY"
            mul_node.inputs[1].default_value = (0.5, 0.5, 0.5)
            mul_node.location = (-200, y)
            links.new(rl_node.outputs["Normal"], mul_node.inputs[0])
            add_node = nodes.new(type="ShaderNodeVectorMath")
            add_node.operation = "ADD"
            add_node.inputs[1].default_value = (0.5, 0.5, 0.5)
            add_node.location = (0, y)
            links.new(mul_node.outputs["Vector"], add_node.inputs[0])
            fo = nodes.new(type="CompositorNodeOutputFile")
            fo.directory = "//"
            fo.file_name = ""
            fo.format.media_type = "IMAGE"
            fo.format.file_format = file_format
            fo.format.color_mode = "RGB"
            fo.file_output_items.new("RGBA", stem + "_normal")
            fo.location = (400, y)
            links.new(add_node.outputs["Vector"], fo.inputs[0])
            pass_name = f"{stem}_normal{suffix}"
            renames.append((pass_name, filename.with_name(pass_name)))
            y -= 200

        return renames

    def _save_blend_file(self, filepath: Path):
        """Save the current Blender scene to a .blend file for debugging.

        Args:
            filepath: Path to save the .blend file.
        """
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save the file
        bpy.ops.wm.save_as_mainfile(filepath=str(filepath))
        logger.info(f"Blender scene saved to {filepath} for debugging")
