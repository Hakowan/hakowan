"""Blender rendering backend implementation."""

from ...common import logger
from ...compiler import Scene, View
from ...setup import Config
from ...grammar import mark
from ...grammar.channel.material import (
    Diffuse,
    Plastic,
    Principled,
    RoughPlastic,
)
from ...grammar.scale import Attribute
from ...grammar.texture import ScalarField
from .. import RenderBackend

from pathlib import Path
from typing import Any
import numpy as np
import lagrange

try:
    import bpy
    import mathutils

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    # Don't warn at import time, only when trying to use the backend


class BlenderBackend(RenderBackend):
    """Blender rendering backend with minimal features.

    Currently supports:
    - Surface marks (meshes)
    - Basic materials: Diffuse, Plastic, RoughPlastic, Principled
    - Camera setup
    - Basic lighting
    """

    def __init__(self):
        if not BLENDER_AVAILABLE:
            raise ImportError(
                "Blender (bpy) is not available. "
                "Install with: pip install bpy or use Blender's Python."
            )

    def render(
        self,
        scene: Scene,
        config: Config,
        filename: Path | str | None = None,
        **kwargs,
    ) -> Any:
        """Render scene using Blender.

        Args:
            scene: Compiled scene.
            config: Rendering configuration.
            filename: Output image filename.
            **kwargs: Additional options:
                - engine (str): Render engine ("CYCLES" or "BLENDER_EEVEE")
                - blend_file (str|Path): Optional .blend file path for debugging

        Returns:
            None (Blender renders to file).
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
        """Replace all mesh object materials with flat facet-ID colors.

        Each face is tinted with the RGB encoding of its zero-based index:
            R = (fid >> 16) & 0xFF
            G = (fid >>  8) & 0xFF
            B =  fid        & 0xFF
        All values are stored as linear floats in [0, 1] (no gamma) so that
        the pixel values in the "Raw" output image can be decoded directly.
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
            mat.use_nodes = True
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
        """Run a second Blender render that outputs per-facet ID colors.

        The output file is placed next to *filename* with a ``_facet_id``
        suffix, e.g. ``bust.png`` → ``bust_facet_id.png``.

        Settings applied for this pass only:
        - EEVEE engine (deterministic rasterization, no noise)
        - 1 TAA sample (no temporal blending)
        - Pixel filter disabled (``filter_size = 0``)
        - Raw view transform (no gamma / tone-mapping)
        """
        scene = bpy.context.scene

        # Override materials with flat facet-ID emission shaders.
        self._setup_facet_id_mode()

        # Save engine / filter / color-management state so nothing leaks back.
        prev_engine = scene.render.engine
        prev_filter = scene.render.filter_size
        prev_transform = scene.view_settings.view_transform

        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1
        scene.render.filter_size = 0.0
        scene.view_settings.view_transform = "Raw"

        # Derive output path: <stem>_facet_id<suffix>
        facet_id_path = filename.parent / (filename.stem + "_facet_id" + filename.suffix)
        scene.render.filepath = str(facet_id_path.resolve())

        logger.info("Rendering facet-ID pass...")
        bpy.ops.render.render(write_still=True)
        logger.info(f"Facet-ID pass saved to {facet_id_path}")

        # Restore render settings for any subsequent operations.
        scene.render.engine = prev_engine
        scene.render.filter_size = prev_filter
        scene.view_settings.view_transform = prev_transform

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
            sizes = [float(sizes)] * mesh.num_vertices

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

        # Scalar field texture: copy vertex/face color attribute to Blender mesh
        color_layer_name = None
        scalar_info = self._get_scalar_field_color_attr(view)
        if scalar_info is not None:
            attr_name, element_type = scalar_info
            self._copy_mesh_color_to_blender(
                mesh_data, mesh, attr_name, element_type, color_layer_name="Color"
            )
            color_layer_name = "Color"

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
            mat = self._create_material(view, index, color_layer_name=color_layer_name)
            if mat:
                obj.data.materials.append(mat)

        logger.debug(f"Created surface object {index} with {len(vertices)} vertices")

    def _create_point_object(self, view: View, index: int):
        """Create Blender point cloud (spheres at each vertex) from a view.

        Args:
            view: Point view.
            index: Object index.
        """
        if view.data_frame is None:
            return

        mesh_data = view.data_frame.mesh
        vertices = mesh_data.vertices
        n_points = mesh_data.num_vertices

        radii = self._extract_size(view, default_size=0.01)
        if np.isscalar(radii):
            radii = [float(radii)] * n_points
        radii = np.atleast_1d(radii)
        assert len(radii) == n_points

        # Base icosphere mesh (subdivisions=1 for lightweight spheres)
        ico_name = f"ico_sphere_{index:03d}"
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=0, radius=1.0)
        base_sphere = bpy.context.active_object
        base_sphere.name = "base_ico"
        base_mesh = base_sphere.data
        base_mesh.name = ico_name

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

        # Create one sphere per point, parent to empty
        for i in range(n_points):
            obj = bpy.data.objects.new(f"point_{index:03d}_{i:06d}", base_mesh.copy())
            bpy.context.collection.objects.link(obj)
            obj.location = mathutils.Vector(vertices[i].tolist())
            r = float(radii[i])
            obj.scale = (r, r, r)
            obj.parent = empty

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

        # Remove the temporary base sphere object (mesh is still used by instances)
        bpy.data.objects.remove(base_sphere, do_unlink=True)

        logger.debug(f"Created point object {index} with {n_points} points")

    def _create_curve_object(self, view: View, index: int):
        """Create Blender curve object from view (mesh edges as linear segments).

        Args:
            view: Curve view.
            index: Object index.
        """
        if view.data_frame is None:
            return

        base, tip, base_size, tip_size, base_idx, tip_idx = self._extract_edges(view)
        n_edges = len(base)
        mesh_data = view.data_frame.mesh

        curve_data = bpy.data.curves.new(name=f"curve_{index:03d}", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.bevel_mode = "ROUND"
        curve_data.bevel_depth = 1.0
        curve_data.fill_mode = "FULL"

        for i in range(n_edges):
            spline = curve_data.splines.new("POLY")
            spline.points.add(1)  # POLY starts with 1 point, add(1) gives 2
            spline.points[0].co = (base[i][0], base[i][1], base[i][2], 1.0)
            spline.points[1].co = (tip[i][0], tip[i][1], tip[i][2], 1.0)
            spline.points[0].radius = float(base_size[i])
            spline.points[1].radius = float(tip_size[i])

        # Scalar field: set per-point colors on curve (2 points per segment)
        color_layer_name = None
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
                            for i in range(n_edges):
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

        logger.debug(f"Created curve object {index} with {n_edges} segments")

    def _create_material(
        self,
        view: View,
        index: int,
        *,
        color_layer_name: str | None = None,
        override_color: tuple[float, float, float, float] | None = None,
        material_suffix: str | None = None,
    ):
        """Create Blender material from view's material channel.

        Args:
            view: View with material channel.
            index: Material index.
            color_layer_name: If set, use mesh/curve color attribute (ScalarField).
            override_color: If set, use this RGBA as base color (e.g. per-point).
            material_suffix: Optional suffix for material name (e.g. point index).

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
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create Principled BSDF
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)

        # Material output
        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        # Base color: scalar field attribute, override, or from material
        if color_layer_name is not None:
            attr_node = nodes.new(type="ShaderNodeAttribute")
            attr_node.location = (-200, 0)
            attr_node.attribute_name = color_layer_name
            links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
        elif override_color is not None:
            bsdf.inputs["Base Color"].default_value = override_color
        else:
            color = None
            match mat_data:
                case Diffuse():
                    color = self._extract_color(mat_data.reflectance)
                case Plastic() | RoughPlastic():
                    color = self._extract_color(mat_data.diffuse_reflectance)
                case Principled():
                    color = self._extract_color(mat_data.color)
                case _:
                    pass
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
                    bsdf.inputs["Coat Roughness"].default_value = 1
                else:
                    bsdf.inputs["Coat Roughness"].default_value = 0

            case Principled():
                bsdf.inputs["Roughness"].default_value = mat_data.roughness
                bsdf.inputs["Metallic"].default_value = mat_data.metallic

            case _:
                if color_layer_name is None and override_color is None:
                    logger.warning(
                        f"Material type {type(mat_data)} not fully supported, using default"
                    )

        return mat

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
            if not isinstance(tex, ScalarField):
                return None
            if not isinstance(tex.data, Attribute) or not getattr(
                tex.data, "_internal_color_field", None
            ):
                return None
            name = tex.data._internal_color_field
            if not mesh.has_attribute(name):
                return None
            attr = mesh.attribute(name)
            return (name, attr.element_type)

        match mat_data:
            case Diffuse():
                out = check(mat_data.reflectance)
            case Plastic() | RoughPlastic():
                out = check(mat_data.diffuse_reflectance)
            case Principled():
                out = check(mat_data.color)
            case _:
                out = None
        return out

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
            for i, c in enumerate(colors):
                if i < len(layer.data):
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

    def _setup_camera(self, config: Config):
        """Setup Blender camera from config.

        Args:
            config: Rendering configuration.
        """
        # Create camera
        camera_data = bpy.data.cameras.new(name="Camera")
        camera_obj = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera_obj)
        bpy.context.scene.camera = camera_obj

        # Set camera location and orientation
        sensor = config.sensor
        location = sensor.location
        camera_obj.location = mathutils.Vector(location)

        # Look at target
        target = getattr(sensor, "target", np.array([0.0, 0.0, 0.0]))
        direction = mathutils.Vector(target) - mathutils.Vector(location)
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera_obj.rotation_euler = rot_quat.to_euler()

        # Set up vector if available
        if hasattr(sensor, "up"):
            # TODO: Properly handle up vector
            pass

        # Set focal length / FOV (sensor.fov is the shorter dimension)
        if hasattr(sensor, "fov"):
            camera_data.lens_unit = "FOV"
            width = config.film.width
            height = config.film.height
            aspect = width / height
            if aspect >= 1:
                # Shorter is height
                camera_data.sensor_fit = "VERTICAL"
                camera_data.angle = np.radians(sensor.fov)
            else:
                # Shorter is width
                camera_data.sensor_fit = "HORIZONTAL"
                camera_data.angle = np.radians(sensor.fov)

        logger.debug(f"Camera set at {location}")

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
        world.use_nodes = True
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
        """Setup Blender render settings.

        Args:
            config: Rendering configuration.
            **kwargs: Additional options:
                - engine (str): Render engine
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
    ) -> list[tuple[Path, Path]]:
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
        renames: list[tuple[Path, Path]] = []
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
        parent = filename.parent

        # Map the user's output suffix to a Blender file-format token.
        fmt_map = {".exr": "OPEN_EXR", ".png": "PNG", ".jpg": "JPEG"}
        file_format = fmt_map.get(suffix, "PNG")

        y = -100

        # '//' resolves to filename.parent because we saved a temp .blend there
        # in render() before calling this method.
        output_dir = filename.parent

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
