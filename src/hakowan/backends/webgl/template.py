"""Render the viewer HTML by substituting placeholders in the bundled template."""

from __future__ import annotations

import json
from importlib import resources


def _load_template() -> str:
    pkg = resources.files("hakowan.backends.webgl") / "templates" / "viewer.html"
    return pkg.read_text(encoding="utf-8")


def render_html(
    glb_uri: str,
    three_version: str,
    backgrounds: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
    background: str,
    initial_view: dict[str, list[float]],
    title: str,
    envmap: dict | None = None,
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
    replacements = {
        "{{TITLE}}": title,
        "{{THREE_VERSION}}": three_version,
        "{{GLB_DATA_URI}}": glb_uri,
        "{{BG_PRESETS_JSON}}": json.dumps(presets_js),
        "{{BG_MODE}}": background,
        "{{BG_R_255}}": str(int(round(r * 255))),
        "{{BG_G_255}}": str(int(round(g * 255))),
        "{{BG_B_255}}": str(int(round(b * 255))),
        "{{INITIAL_EYE}}": json.dumps(initial_view["eye"]),
        "{{INITIAL_TARGET}}": json.dumps(initial_view["target"]),
        "{{INITIAL_UP}}": json.dumps(initial_view["up"]),
        "{{ENVMAP_JSON}}": json.dumps(envmap) if envmap is not None else "null",
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template
