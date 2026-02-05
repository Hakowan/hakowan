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
from .. import RenderBackend

from pathlib import Path
from typing import Any
import numpy as np

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
            **kwargs: Additional options (e.g., samples, engine).

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
        self._setup_lighting(config)

        # Setup render settings
        self._setup_render_settings(config, **kwargs)

        # Render
        if filename is not None:
            if isinstance(filename, str):
                filename = Path(filename)
            bpy.context.scene.render.filepath = str(filename)

        logger.info("Rendering with Blender...")
        bpy.ops.render.render(write_still=True)
        logger.info(f"Rendering saved to {filename}")

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
                logger.warning("Point mark not yet supported in Blender backend")
            case mark.Curve:
                logger.warning("Curve mark not yet supported in Blender backend")
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

        # Create object
        obj = bpy.data.objects.new(f"object_{index:03d}", mesh)
        bpy.context.collection.objects.link(obj)

        # Apply material
        if view.material_channel is not None:
            mat = self._create_material(view, index)
            if mat:
                obj.data.materials.append(mat)

        logger.debug(f"Created surface object {index} with {len(vertices)} vertices")

    def _create_material(self, view: View, index: int):
        """Create Blender material from view's material channel.

        Args:
            view: View with material channel.
            index: Material index.

        Returns:
            Blender material or None.
        """
        if view.material_channel is None:
            return None

        mat_data = view.material_channel.data
        mat = bpy.data.materials.new(name=f"material_{index:03d}")
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

        # Configure based on material type
        match mat_data:
            case Diffuse():
                # Diffuse: zero roughness and metallic
                bsdf.inputs["Roughness"].default_value = 1.0
                bsdf.inputs["Metallic"].default_value = 0.0
                color = self._extract_color(mat_data.reflectance)
                if color:
                    bsdf.inputs["Base Color"].default_value = color

            case Plastic() | RoughPlastic():
                # Plastic: some roughness, no metallic
                if isinstance(mat_data, RoughPlastic):
                    roughness = getattr(mat_data, "alpha", 0.1)
                else:
                    roughness = 0.01
                bsdf.inputs["Roughness"].default_value = roughness
                bsdf.inputs["Metallic"].default_value = 0.0
                color = self._extract_color(mat_data.diffuse_reflectance)
                if color:
                    bsdf.inputs["Base Color"].default_value = color

            case Principled():
                # Principled material
                bsdf.inputs["Roughness"].default_value = mat_data.roughness
                bsdf.inputs["Metallic"].default_value = mat_data.metallic
                color = self._extract_color(mat_data.color)
                if color:
                    bsdf.inputs["Base Color"].default_value = color

            case _:
                logger.warning(f"Material type {type(mat_data)} not fully supported, using default")
                bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

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
        # TODO: Handle texture types
        return None

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

        # Set focal length / FOV
        if hasattr(sensor, "fov"):
            camera_data.lens_unit = "FOV"
            camera_data.angle = np.radians(sensor.fov)

        logger.debug(f"Camera set at {location}")

    def _setup_lighting(self, config: Config):
        """Setup Blender lighting from config.

        Args:
            config: Rendering configuration.
        """
        # For minimal implementation, add a simple sun light
        light_data = bpy.data.lights.new(name="Sun", type="SUN")
        light_data.energy = 1.0
        light_obj = bpy.data.objects.new("Sun", light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = (0, 0, 10)

        # TODO: Handle config.emitters for more sophisticated lighting
        logger.debug("Added default sun light")

    def _setup_render_settings(self, config: Config, **kwargs):
        """Setup Blender render settings.

        Args:
            config: Rendering configuration.
            **kwargs: Additional options (samples, engine, etc.).
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
            scene.cycles.samples = kwargs.get("samples", 128)
        elif engine == "BLENDER_EEVEE":
            scene.eevee.taa_render_samples = kwargs.get("samples", 64)

        # File format
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"

        logger.debug(f"Render settings: {scene.render.resolution_x}x{scene.render.resolution_y}, engine={engine}")
