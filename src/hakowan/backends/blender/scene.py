"""Camera, lighting, render-settings, compositor and render-pass setup."""

from typing import Literal

from ...common import logger
from ...common.output import manage_native_output
from ...setup import Config
from ...setup.render_pass import FACET_ID, aov_path

from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np

import bpy
import mathutils
from ._common import _ensure_nodes


class _SceneMixin:
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

    @contextmanager
    def _lossless_render_state(self):
        """Temporarily force deterministic, lossless rasterisation.

        Discrete integer-ID passes (:attr:`RenderPass.discrete`) encode data in
        exact pixel values, so they must render without any process that
        perturbs those values.  This context manager disables them all and
        restores every touched setting on exit:

        - **Engine**: ``BLENDER_EEVEE`` — deterministic rasterisation, no
          path-tracing noise.
        - **TAA samples**: 1 — disables temporal anti-aliasing / blending.
        - **Pixel filter** (``filter_size``): 0.0 — no Gaussian spread across
          pixel boundaries.
        - **View transform**: ``"Raw"`` — bypasses all gamma and tone-mapping
          so pixel channel values equal the stored linear float values.
        - **Dither intensity**: 0.0 — prevents ±1 byte noise on 8-bit output.

        Keeping this state in one place means any future discrete pass renders
        losslessly by construction — the bug class (e.g. dithering corrupting
        IDs) cannot silently return.
        """
        scene = bpy.context.scene
        r = scene.render
        vs = scene.view_settings
        saved = (
            r.engine,
            r.filter_size,
            r.dither_intensity,
            vs.view_transform,
            scene.eevee.taa_render_samples,
        )
        r.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1
        r.filter_size = 0.0
        vs.view_transform = "Raw"
        r.dither_intensity = 0.0
        try:
            yield
        finally:
            (
                r.engine,
                r.filter_size,
                r.dither_intensity,
                vs.view_transform,
                scene.eevee.taa_render_samples,
            ) = saved

    def _render_facet_id_pass(self, filename: Path):
        """Perform a second Blender render that writes per-facet ID colors.

        Called automatically by :meth:`render` when ``config.facet_id`` is
        ``True``, *after* the main render has completed.

        The output file is placed next to *filename* with a ``_facet_id``
        suffix, e.g. ``bust.png`` → ``bust_facet_id.png``.

        ``facet_id`` is a discrete (integer-ID) pass, so the render runs inside
        :meth:`_lossless_render_state` — the lossless settings are a property of
        :attr:`RenderPass.discrete`, not hardcoded here.  The compositor is also
        disconnected so the main render's albedo/depth/normal file-output nodes
        are not re-triggered by this second render.

        Args:
            filename: Main output image path used to derive the facet-ID
                output path (``<stem>_facet_id<ext>``).
        """
        scene = bpy.context.scene

        # Override materials with flat facet-ID emission shaders.
        self._setup_facet_id_mode()

        facet_id_path = aov_path(filename, FACET_ID)
        scene.render.filepath = str(facet_id_path.resolve())

        # Disconnect the compositor so the main render's albedo/depth/normal
        # file-output nodes are not re-triggered by this second render.
        prev_compositor = scene.compositing_node_group
        scene.compositing_node_group = None
        try:
            logger.info("Rendering facet-ID pass...")
            # Lossless render iff the pass is discrete (always true for facet_id).
            ctx = self._lossless_render_state() if FACET_ID.discrete else nullcontext()
            with ctx, manage_native_output(logger, prefix="blender"):
                bpy.ops.render.render(write_still=True)
            logger.info(f"Facet-ID pass saved to {facet_id_path}")
        finally:
            scene.compositing_node_group = prev_compositor

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
    def _resolve_sensor_fit(fov_axis: Literal["x", "y", "diagonal", "smaller", "larger"], width: int, height: int) -> str | None:
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
