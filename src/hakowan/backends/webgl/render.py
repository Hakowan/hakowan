"""WebGL backend top-level orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...common import logger
from ...compiler import Scene
from ...grammar import mark as mark_module
from ...setup import Config
from ...setup.render_pass import ALBEDO, DEPTH, NORMAL
from .. import RenderBackend

from ...setup.emitter import Point as PointEmitter

from .builder import GLTFBuilder
from .camera import add_camera
from .curve import add_curve_view
from .envmap import envmap_descriptor
from .material_translate import translate_material
from .mesh_extract import extract_surface_arrays
from .point_cloud import add_point_view
from .template import render_html
from .utils import glb_to_data_uri


_DEFAULT_THREE_VERSION = "0.170.0"
_DEFAULT_TITLE = "hakowan"

# Beauty-pass background presets. Each is a soft "studio" radial gradient with a
# bright spot in the centre falling off towards the edges: ``(center, edge)``
# colours as (r, g, b) in [0, 1]. Selected via the ``background`` option.
_BACKGROUND_PRESETS: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    "light": ((0.97, 0.97, 0.98), (0.62, 0.63, 0.66)),
    "dark": ((0.30, 0.31, 0.34), (0.05, 0.05, 0.06)),
}
_DEFAULT_BACKGROUND: Literal["light", "dark"] = "dark"


def _validate_background(name: Literal["light", "dark"]) -> None:
    """Raise ``ValueError`` if ``name`` is not a known background preset."""
    if name not in _BACKGROUND_PRESETS:
        raise ValueError(
            f"Unknown background {name!r}; choose from {sorted(_BACKGROUND_PRESETS)}."
        )


class WebGLBackend(RenderBackend):
    """Render a hakowan ``Scene`` as a self-contained three.js HTML viewer."""

    # The interactive viewer always exposes albedo/depth/normal as live,
    # client-side toggle passes (rendered by three.js, not written to files),
    # so they are "supported" regardless of config.render_passes. facet_id has
    # no viewer pass and is therefore unsupported.
    SUPPORTED_PASSES = frozenset({ALBEDO, DEPTH, NORMAL})

    # Passes are live toggles in the HTML viewer, not separate image files.
    PASS_DELIVERY = "interactive"

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def render(
        self,
        scene: Scene,
        config: Config,
        filename: Path | str | None = None,
        *,
        three_version: str = _DEFAULT_THREE_VERSION,
        background: Literal["light", "dark"] = _DEFAULT_BACKGROUND,
        title: str = _DEFAULT_TITLE,
        envmap_background: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Write an interactive HTML viewer and return the output path."""
        if kwargs:
            raise TypeError(
                f"render() got unexpected keyword argument(s): {list(kwargs)}"
            )

        _validate_background(background)
        out_path = _resolve_output_path(filename)
        glb_bytes, envmap, initial_view = self._build_scene_artifacts(
            scene, config, envmap_background
        )

        html = render_html(
            glb_uri=glb_to_data_uri(glb_bytes),
            three_version=three_version,
            backgrounds=_BACKGROUND_PRESETS,
            background=background,
            initial_view=initial_view,
            title=title,
            envmap=envmap,
        )

        out_path.write_bytes(html.encode("utf-8"))
        logger.info(f"WebGL viewer saved to {out_path}")
        return out_path

    def html_string(
        self,
        scene: Scene,
        config: Config,
        *,
        three_version: str = _DEFAULT_THREE_VERSION,
        background: Literal["light", "dark"] = _DEFAULT_BACKGROUND,
        title: str = _DEFAULT_TITLE,
        envmap_background: bool = False,
    ) -> str:
        """Build and return the viewer HTML as a string without writing any files.

        Args:
            scene: Compiled scene to render.
            config: Rendering configuration.
            three_version: Three.js version string to pull from unpkg CDN.
            background: Background preset — ``"light"`` (default) or ``"dark"``.
                Both are soft studio radial gradients with a bright centre spot.
            title: HTML page title.
            envmap_background: Whether to show the environment map as background.

        Returns:
            Complete HTML page as a string.
        """
        _validate_background(background)
        glb_bytes, envmap, initial_view = self._build_scene_artifacts(
            scene, config, envmap_background
        )
        return render_html(
            glb_uri=glb_to_data_uri(glb_bytes),
            three_version=three_version,
            backgrounds=_BACKGROUND_PRESETS,
            background=background,
            initial_view=initial_view,
            title=title,
            envmap=envmap,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_scene_artifacts(
        self,
        scene: Scene,
        config: Config,
        envmap_background: bool = False,
    ) -> tuple[bytes, dict | None, dict]:
        """Compile *scene* into GLB bytes, an envmap descriptor, and camera view.

        Args:
            scene: Compiled scene.
            config: Rendering configuration.
            envmap_background: Whether the envmap is visible as the background.

        Returns:
            ``(glb_bytes, envmap, initial_view)`` where *envmap* may be ``None``.
        """
        builder = GLTFBuilder()
        for index, view in enumerate(scene):
            # Tag every node produced for this view with its juxtaposition cell
            # so the interactive viewer can rotate each comparison cell about its
            # own centre. ``None`` (no `|` in the layer tree) leaves nodes untagged.
            builder._current_cell = _cell_tag(view)
            if view.mark is mark_module.Surface:
                _add_surface_view(builder, view)
            elif view.mark is mark_module.Point:
                add_point_view(builder, view)
            elif view.mark is mark_module.Curve:
                add_curve_view(builder, view)
            else:
                logger.warning(
                    f"WebGL backend: view {index} has unsupported mark "
                    f"{view.mark!r} — skipping."
                )
        _, initial_view = add_camera(builder, config)
        _add_point_lights(builder, config)
        glb_bytes = builder.finalize()
        envmap = envmap_descriptor(config)
        if envmap is not None:
            envmap["background"] = bool(envmap_background)
        return glb_bytes, envmap, initial_view


# ---------------------------------------------------------------------- #
# Helpers                                                                  #
# ---------------------------------------------------------------------- #


def _cell_tag(view) -> str | None:
    """Serialise a view's juxtaposition cell key to a stable per-scene string.

    Returns ``None`` when the view is not part of any juxtaposition (so its
    nodes are left untagged and the viewer treats them as a single group).
    """
    cell = view._layout_cell
    if not cell:
        return None
    return "/".join(f"{node_id}.{branch}" for node_id, branch in cell)


def _resolve_output_path(filename: Path | str | None) -> Path:
    if filename is None:
        return Path("out.html").resolve()
    path = Path(filename)
    if path.suffix.lower() != ".html":
        new_path = path.with_suffix(".html")
        if path.suffix:
            logger.warning(
                f"WebGL backend: rewriting output suffix "
                f"'{path.suffix}' → '.html' ({new_path})."
            )
        path = new_path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Return the path in the form the user supplied it (do not ``resolve()``):
    # this is surfaced as ``RenderResult.path`` and the only correction is the
    # ``.html`` suffix the viewer always writes.
    return path


def _add_point_lights(builder: GLTFBuilder, config: Config) -> None:
    """Emit one KHR_lights_punctual entry per Point emitter."""
    from ...common.color import Color

    for emitter in config.emitters:
        if not isinstance(emitter, PointEmitter):
            continue
        intensity = emitter.intensity
        if isinstance(intensity, Color):
            color = (
                float(intensity.red),
                float(intensity.green),
                float(intensity.blue),
            )
            mag = max(color) if max(color) > 1.0 else 1.0
            color = (color[0] / mag, color[1] / mag, color[2] / mag)
            strength = float(max(intensity.red, intensity.green, intensity.blue))
        else:
            color = (1.0, 1.0, 1.0)
            strength = float(intensity)
        builder.add_point_light(
            position=list(emitter.position), color=color, intensity=strength
        )


def _add_surface_view(builder: GLTFBuilder, view) -> None:
    # Translate material first so per-vertex custom attributes (e.g.
    # ``_scalar_0`` for isocontour) flow into the extractor and follow the
    # same de-indexing path as positions/normals when facet normals are used.
    result = translate_material(view, builder)
    arrays = extract_surface_arrays(view, custom_attrs=result.custom_attrs)
    pbr = result.pbr
    if arrays["colors"] is not None:
        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
    needs_uvs = "baseColorTextureIndex" in pbr or "normalTextureIndex" in pbr
    uvs = arrays["uvs"] if needs_uvs else None
    if result.extras is not None:
        pbr["extras"] = result.extras
    material_idx = builder.add_material(pbr, double_sided=result.double_sided)

    transform = np.asarray(view.global_transform, dtype=np.float64)
    builder.add_mesh_node(
        positions=arrays["positions"],
        indices=arrays["indices"],
        normals=arrays["normals"],
        colors=arrays["colors"],
        uvs=uvs,
        custom_attributes=arrays.get("custom_attributes"),
        material_idx=material_idx,
        transform_4x4=transform,
    )
