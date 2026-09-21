"""Render the viewer HTML by substituting placeholders in the bundled template."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def _load_template() -> str:
    pkg = resources.files("hakowan.backends.webgl") / "templates" / "viewer.html"
    return pkg.read_text(encoding="utf-8")


def render_html(
    glb_uri: str,
    three_version: str,
    backgrounds: dict[
        str, tuple[tuple[float, float, float], tuple[float, float, float]]
    ],
    background: str,
    initial_view: dict[str, Any],
    title: str,
    envmap: dict | None = None,
    layers: list[dict] | None = None,
    three_module_url: str | None = None,
    three_addons_url: str | None = None,
    legends: list[dict] | None = None,
    annotations: list[dict] | None = None,
) -> str:
    """Substitute placeholders in the bundled viewer template.

    ``glb_uri`` may be either a ``data:`` URI (embedded GLB) or a relative
    URL pointing to a sidecar ``.glb`` file. The template's JS sniffs the
    prefix and picks ``GLTFLoader.parse`` vs ``GLTFLoader.load`` accordingly.

    ``envmap`` is an ``envmap_descriptor()`` dict or None.

    The viewer always exposes ``beauty | albedo | depth | normal | save png``
    in the top-right corner — passes are not gated by ``config.render_passes``.
    """
    template = _load_template()
    # Expose every preset to the viewer (so the light/dark toggle can switch
    # client-side); the page-letterbox CSS uses the selected preset's edge.
    presets_js = {
        name: {"center": list(center), "edge": list(edge)}
        for name, (center, edge) in backgrounds.items()
    }
    r, g, b = backgrounds[background][1]  # selected preset's edge colour
    if three_module_url is None:
        three_module_url = (
            f"https://unpkg.com/three@{three_version}/build/three.module.js"
        )
    if three_addons_url is None:
        three_addons_url = f"https://unpkg.com/three@{three_version}/examples/jsm/"
    replacements = {
        "{{TITLE}}": title,
        "{{GLB_DATA_URI}}": glb_uri,
        "{{THREE_MODULE_URL}}": three_module_url,
        "{{THREE_ADDONS_URL}}": three_addons_url,
        "{{BG_PRESETS_JSON}}": json.dumps(presets_js),
        "{{BG_MODE}}": background,
        "{{BG_R_255}}": str(int(round(r * 255))),
        "{{BG_G_255}}": str(int(round(g * 255))),
        "{{BG_B_255}}": str(int(round(b * 255))),
        "{{INITIAL_EYE}}": json.dumps(initial_view["eye"]),
        "{{INITIAL_TARGET}}": json.dumps(initial_view["target"]),
        "{{INITIAL_UP}}": json.dumps(initial_view["up"]),
        "{{INITIAL_CAMERA_MODE}}": json.dumps(initial_view.get("mode", "perspective")),
        "{{INITIAL_ORTHO_SCALE}}": json.dumps(initial_view.get("scale")),
        "{{ENVMAP_JSON}}": json.dumps(envmap) if envmap is not None else "null",
        "{{LAYERS_JSON}}": json.dumps(layers if layers is not None else []),
        "{{LEGENDS_JSON}}": json.dumps(legends if legends is not None else []),
        "{{ANNOTATIONS_JSON}}": json.dumps(
            annotations if annotations is not None else []
        ),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template
