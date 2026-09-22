"""Render the viewer HTML by substituting placeholders in the bundled template."""

from __future__ import annotations

import json
from html import escape
from importlib import resources
from typing import Any


def _load_template() -> str:
    pkg = resources.files("hakowan.backends.webgl") / "templates" / "viewer.html"
    return pkg.read_text(encoding="utf-8")


def _script_json(value: Any) -> str:
    """Encode JSON for an inline script without permitting tag termination."""
    return (
        json.dumps(value)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


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
        "{{TITLE}}": escape(title),
        "{{GLB_DATA_URI}}": _script_json(glb_uri),
        "{{THREE_MODULE_URL}}": _script_json(three_module_url),
        "{{THREE_ADDONS_URL}}": _script_json(three_addons_url),
        "{{BG_PRESETS_JSON}}": _script_json(presets_js),
        "{{BG_MODE}}": _script_json(background),
        "{{BG_R_255}}": str(int(round(r * 255))),
        "{{BG_G_255}}": str(int(round(g * 255))),
        "{{BG_B_255}}": str(int(round(b * 255))),
        "{{INITIAL_EYE}}": _script_json(initial_view["eye"]),
        "{{INITIAL_TARGET}}": _script_json(initial_view["target"]),
        "{{INITIAL_UP}}": _script_json(initial_view["up"]),
        "{{INITIAL_CAMERA_MODE}}": _script_json(
            initial_view.get("mode", "perspective")
        ),
        "{{INITIAL_ORTHO_SCALE}}": _script_json(initial_view.get("scale")),
        "{{ENVMAP_JSON}}": _script_json(envmap),
        "{{LAYERS_JSON}}": _script_json(layers if layers is not None else []),
        "{{LEGENDS_JSON}}": _script_json(legends if legends is not None else []),
        "{{ANNOTATIONS_JSON}}": _script_json(
            annotations if annotations is not None else []
        ),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template
