""" Render functions for Hakowan """

from ..common.exception import InvalidSetting
from ..grammar.layer import Layer
from ..scene.scene_utils import generate_scene
from .mitsuba_render import render_with_mitsuba
from .render_config import RenderConfig


def render(root: Layer, config: RenderConfig):
    """Render layer tree rooted at `root`."""
    scene = generate_scene(root)

    if config.backend == "mitsuba":
        render_with_mitsuba(scene, config)
    else:
        raise InvalidSetting(f"Unsupported rendering backend: {config.backend}")
