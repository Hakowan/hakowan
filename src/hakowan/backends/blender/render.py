"""Blender rendering backend implementation."""

from ...common import logger
from ...compiler import Scene
from ...setup import Config
from .. import RenderBackend

from pathlib import Path
from typing import Any

import bpy
from .geometry import _GeometryMixin
from .materials import _MaterialMixin
from .scene import _SceneMixin


class BlenderBackend(_GeometryMixin, _MaterialMixin, _SceneMixin, RenderBackend):
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
