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

from ...setup.emitter import Directional as DirectionalEmitter
from ...setup.emitter import Point as PointEmitter
from ...common.to_color import to_color

from .builder import GLTFBuilder
from .camera import add_camera
from .curve import add_curve_view
from .envmap import envmap_descriptor
from .material_translate import translate_material
from .mesh_extract import extract_surface_arrays
from .point_cloud import add_point_view
from .template import render_html
from .utils import glb_to_data_uri
from .assets import copy_three_assets


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
    """Render a Hakowan scene as an embedded-data Three.js HTML viewer."""

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
        background: Literal["light", "dark"] | None = None,
        title: str = _DEFAULT_TITLE,
        envmap_background: bool | None = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Write an interactive HTML viewer and optional offline asset bundle."""
        if kwargs:
            raise TypeError(
                f"render() got unexpected keyword argument(s): {list(kwargs)}"
            )
        if envmap_background is None:
            envmap_background = config.environment_visible
        if background is None:
            background = config.background or _DEFAULT_BACKGROUND

        _validate_background(background)
        out_path = _resolve_output_path(filename)
        glb_bytes, envmap, initial_view, layers = self._build_scene_artifacts(
            scene, config, envmap_background
        )

        three_module_url = None
        three_addons_url = None
        if offline:
            assets = out_path.with_name(f"{out_path.stem}_assets")
            copy_three_assets(three_version, assets)
            three_module_url = f"./{assets.name}/build/three.module.js"
            three_addons_url = f"./{assets.name}/examples/jsm/"
        html = render_html(
            glb_uri=glb_to_data_uri(glb_bytes),
            three_version=three_version,
            backgrounds=_BACKGROUND_PRESETS,
            background=background,
            initial_view=initial_view,
            title=title,
            envmap=envmap,
            layers=layers,
            three_module_url=three_module_url,
            three_addons_url=three_addons_url,
            legends=[legend.to_dict() for legend in scene.legends],
            annotations=[annotation.to_dict() for annotation in scene.annotations],
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
        background: Literal["light", "dark"] | None = None,
        title: str = _DEFAULT_TITLE,
        envmap_background: bool | None = None,
        three_module_url: str | None = None,
        three_addons_url: str | None = None,
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
        if envmap_background is None:
            envmap_background = config.environment_visible
        if background is None:
            background = config.background or _DEFAULT_BACKGROUND
        _validate_background(background)
        glb_bytes, envmap, initial_view, layers = self._build_scene_artifacts(
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
            layers=layers,
            three_module_url=three_module_url,
            three_addons_url=three_addons_url,
            legends=[legend.to_dict() for legend in scene.legends],
            annotations=[annotation.to_dict() for annotation in scene.annotations],
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_scene_artifacts(
        self,
        scene: Scene,
        config: Config,
        envmap_background: bool = False,
    ) -> tuple[bytes, dict | None, dict, list[dict]]:
        """Compile *scene* into GLB bytes, an envmap descriptor, and camera view.

        Args:
            scene: Compiled scene.
            config: Rendering configuration.
            envmap_background: Whether the envmap is visible as the background.

        Returns:
            ``(glb_bytes, envmap, initial_view, layers)`` where *envmap* may be
            ``None`` and *layers* is one ``{"index", "label"}`` entry per rendered
            view, in view order, for the viewer's per-layer visibility checkboxes.
        """
        builder = GLTFBuilder()
        layers: list[dict] = []
        for index, view in enumerate(scene):
            # Tag every node produced for this view with its juxtaposition cell
            # (so the viewer can rotate each comparison cell about its own centre;
            # ``None`` leaves nodes untagged) and its layer index (so the viewer
            # can toggle per-layer visibility).
            builder._current_cell = _cell_tag(view)
            builder._current_layer = index
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
                continue
            layers.append({"index": index, "label": view.name or f"Layer {index + 1}"})
        _, initial_view = add_camera(builder, config)
        _add_lights(builder, config)
        glb_bytes = builder.finalize()
        envmap = envmap_descriptor(config)
        if envmap is not None:
            envmap["background"] = bool(envmap_background)
        return glb_bytes, envmap, initial_view, layers


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


def _light_color_intensity(emitter) -> tuple[tuple[float, float, float], float]:
    if getattr(emitter, "color", None) is not None:
        color = to_color(emitter.color)
        return (
            (float(color.red), float(color.green), float(color.blue)),
            float(emitter.intensity),
        )
    if isinstance(emitter.intensity, (int, float)):
        return (1.0, 1.0, 1.0), float(emitter.intensity)
    color = to_color(emitter.intensity)
    values = (float(color.red), float(color.green), float(color.blue))
    strength = max(values)
    normalized = (
        tuple(value / strength for value in values) if strength > 1.0 else values
    )
    return (
        float(normalized[0]),
        float(normalized[1]),
        float(normalized[2]),
    ), strength if strength > 0.0 else 1.0


def _add_lights(builder: GLTFBuilder, config: Config) -> None:
    """Emit supported KHR_lights_punctual lights."""
    for emitter in config.emitters:
        if not isinstance(emitter, (PointEmitter, DirectionalEmitter)):
            continue
        color, intensity = _light_color_intensity(emitter)
        if isinstance(emitter, PointEmitter):
            builder.add_point_light(
                position=list(emitter.position), color=color, intensity=intensity
            )
        else:
            builder.add_directional_light(
                direction=list(emitter.direction), color=color, intensity=intensity
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
