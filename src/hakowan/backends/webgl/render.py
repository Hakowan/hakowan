"""WebGL backend top-level orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...common import logger
from ...compiler import Scene
from ...grammar import mark as mark_module
from ...setup import Config
from .. import RenderBackend

from .builder import GLTFBuilder
from .camera import add_camera
from .material_translate import translate_material
from .mesh_extract import extract_surface_arrays
from .template import render_html
from .utils import glb_to_data_uri


_DEFAULT_THREE_VERSION = "0.160.0"
_EMBED_SIZE_LIMIT_BYTES = 50 * 1024 * 1024  # 50 MB


class WebGLBackend(RenderBackend):
    """Render a hakowan ``Scene`` as a self-contained three.js HTML viewer."""

    def render(
        self,
        scene: Scene,
        config: Config,
        filename: Path | str | None = None,
        *,
        three_version: str = _DEFAULT_THREE_VERSION,
        embed: bool | Literal["auto"] = "auto",
        bg_color: tuple[float, float, float] = (0.1, 0.1, 0.1),
        title: str = "hakowan",
        **kwargs: Any,
    ) -> Path:
        """Write an interactive HTML viewer and return the output path.

        Args:
            scene: Compiled hakowan scene.
            config: Rendering configuration (camera, film).
            filename: Output path. Defaults to ``out.html`` in cwd. Non-``.html``
                suffixes are rewritten to ``.html`` with a warning.
            three_version: three.js version pinned in the CDN URL.
            embed: ``True`` always embeds the GLB as a base64 data URI;
                ``False`` always writes a sidecar ``.glb`` and references it
                by URL (requires a local HTTP server in some browsers);
                ``"auto"`` embeds if GLB ≤ 50 MB and writes a sidecar otherwise.
            bg_color: Viewer canvas clear color (linear RGB in [0, 1]).
            title: ``<title>`` element of the HTML page.
            **kwargs: Reserved for future use.

        Returns:
            Absolute ``Path`` to the written HTML file.
        """
        if kwargs:
            logger.debug(f"WebGL backend ignoring unknown kwargs: {list(kwargs)}")

        out_path = _resolve_output_path(filename)
        builder = GLTFBuilder()

        for index, view in enumerate(scene):
            if view.mark is mark_module.Surface:
                _add_surface_view(builder, view)
            elif view.mark is mark_module.Point:
                logger.warning(
                    f"WebGL backend: view {index} uses Point mark, which "
                    "lands in Phase 2 — skipping."
                )
            elif view.mark is mark_module.Curve:
                logger.warning(
                    f"WebGL backend: view {index} uses Curve mark, which "
                    "lands in Phase 2 — skipping."
                )
            else:
                logger.warning(
                    f"WebGL backend: view {index} has unsupported mark "
                    f"{view.mark!r} — skipping."
                )

        _, initial_view = add_camera(builder, config)
        glb_bytes = builder.finalize()

        if _decide_embed(embed, glb_bytes):
            glb_uri = glb_to_data_uri(glb_bytes)
        else:
            sidecar = out_path.with_suffix(".glb")
            sidecar.write_bytes(glb_bytes)
            logger.info(
                f"WebGL backend: writing sidecar GLB to {sidecar} "
                "(use a local HTTP server to load it under file://)."
            )
            glb_uri = sidecar.name

        html = render_html(
            glb_uri=glb_uri,
            three_version=three_version,
            bg_color=bg_color,
            initial_view=initial_view,
            title=title,
        )

        out_path.write_bytes(html.encode("utf-8"))
        logger.info(f"WebGL viewer saved to {out_path}")
        return out_path


# ---------------------------------------------------------------------- #
# Helpers                                                                  #
# ---------------------------------------------------------------------- #


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
    return path.resolve()


def _decide_embed(mode: bool | Literal["auto"], glb_bytes: bytes) -> bool:
    if mode is True:
        return True
    if mode is False:
        return False
    return len(glb_bytes) <= _EMBED_SIZE_LIMIT_BYTES


def _add_surface_view(builder: GLTFBuilder, view) -> None:
    arrays = extract_surface_arrays(view)
    pbr, double_sided = translate_material(view)
    material_idx = builder.add_material(pbr, double_sided=double_sided)

    transform = np.asarray(view.global_transform, dtype=np.float64)
    builder.add_mesh_node(
        positions=arrays["positions"],
        indices=arrays["indices"],
        normals=arrays["normals"],
        material_idx=material_idx,
        transform_4x4=transform,
    )
