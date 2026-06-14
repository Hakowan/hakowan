"""Blender rendering backend implementation."""

from ...common import logger
from ...common.image_io import check_supported_suffix, convert_image, is_hdr_suffix
from ...common.output import manage_native_output
from ...compiler import Scene
from ...setup import Config
from ...setup.render_pass import ALBEDO, DEPTH, NORMAL, FACET_ID, aov_path
from .. import RenderBackend

import tempfile
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

    SUPPORTED_PASSES = frozenset({ALBEDO, DEPTH, NORMAL, FACET_ID})

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
          Emission shaders and EEVEE, rendered losslessly (no gamma, AA,
          pixel filtering, or dithering).  Written to ``<stem>_facet_id<ext>``.

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
        render_filename: Path | None = None
        out_suffix: str | None = None
        _tmp_render_dir: tempfile.TemporaryDirectory | None = None
        if filename is not None:
            if isinstance(filename, str):
                filename = Path(filename)
            # Reject unwritable formats up front, before spending a render.
            out_suffix = check_supported_suffix(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)

            # Blender encodes LDR output as PNG and HDR as OpenEXR. Any other
            # requested format (e.g. .webp/.jpg/.tif) is produced afterwards by
            # re-encoding the PNG with Pillow (see _finalize_outputs).
            render_suffix = out_suffix if is_hdr_suffix(out_suffix) else ".png"
            if render_suffix == out_suffix:
                # No conversion needed: write straight to the destination.
                render_dir = filename.parent
            else:
                # Render the PNG intermediate(s) into a private temp dir so they
                # never overwrite the user's own files (e.g. a pre-existing
                # ``foo.png`` while rendering ``foo.webp``); convert into place
                # afterwards.
                _tmp_render_dir = tempfile.TemporaryDirectory(prefix="hakowan_blender_")
                render_dir = Path(_tmp_render_dir.name)
            render_filename = render_dir / (filename.stem + render_suffix)

            bpy.context.scene.render.filepath = str(render_filename.resolve())
            bpy.context.scene.render.image_settings.file_format = (
                "OPEN_EXR" if render_suffix == ".exr" else "PNG"
            )
            # In headless bpy, bpy.data.filepath is empty so Blender has no
            # "home directory" and the compositor File Output node writes bare
            # filenames with no directory.  Save a temporary .blend file into
            # the render directory so that '//' resolves to render_dir.
            _temp_blend = render_dir / ".hakowan_compositor.blend"
            with manage_native_output(logger, prefix="blender"):
                bpy.ops.wm.save_as_mainfile(filepath=str(_temp_blend), compress=False)
            renames = self._setup_compositor_passes(config, render_filename)

        logger.info("Rendering with Blender...")
        with manage_native_output(logger, prefix="blender"):
            bpy.ops.render.render(write_still=True)
        logger.info(f"Rendering saved to {render_filename}")

        # Move pass files if needed (src == final means already in the right place).
        if renames:
            import shutil

            assert render_filename is not None
            output_dir = render_filename.parent
            for pass_name, final in renames:
                src = output_dir / pass_name
                if src.exists():
                    if src != final:
                        shutil.move(str(src), str(final))
                    logger.info(f"Pass saved to {final}")
                else:
                    logger.warning(f"Pass file not found: {pass_name}")

        # Facet-ID pass: second render with flat ID-color materials.
        if config.facet_id and render_filename is not None:
            self._render_facet_id_pass(render_filename)

        # Remove the temporary .blend file used to anchor '//' path resolution.
        if _temp_blend is not None and _temp_blend.exists():
            _temp_blend.unlink()

        # Convert Blender's native PNG output into the user's requested format
        # when it differs (e.g. .webp/.jpg). EXR and PNG need no conversion.
        if (
            filename is not None
            and render_filename is not None
            and render_filename.suffix.lower() != out_suffix
        ):
            self._finalize_outputs(config, render_filename, filename)

        # Drop the private intermediate render directory, if one was used.
        if _tmp_render_dir is not None:
            _tmp_render_dir.cleanup()

        return None

    def _finalize_outputs(
        self, config: Config, render_filename: Path, filename: Path
    ) -> None:
        """Re-encode Blender's native PNG output into the user's image format.

        Blender renders LDR passes as PNG (``render_filename``); when the user
        requested another Pillow-writable format (``filename``), each produced
        file — the main image and every enabled sidecar pass — is converted with
        Pillow and the PNG intermediate removed.

        Args:
            config: Rendering configuration (which passes were produced).
            render_filename: Path Blender actually wrote (``<stem>.png``).
            filename: User-requested output path (``<stem>.<ext>``).
        """
        pairs: list[tuple[Path, Path]] = [(render_filename, filename)]
        for enabled, render_pass in (
            (config.albedo, ALBEDO),
            (config.depth, DEPTH),
            (config.normal, NORMAL),
            (config.facet_id, FACET_ID),
        ):
            if enabled:
                pairs.append(
                    (
                        aov_path(render_filename, render_pass),
                        aov_path(filename, render_pass),
                    )
                )

        for src, dst in pairs:
            if not src.exists():
                logger.warning(f"Expected render output missing: {src}")
                continue
            convert_image(src, dst)
            if src != dst:
                src.unlink()
            logger.info(f"Saved to {dst}")

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
        # Geometry-nodes groups (point instancing) are not tied to objects, so
        # remove them explicitly or they accumulate across renders.
        for node_group in bpy.data.node_groups:
            bpy.data.node_groups.remove(node_group)
