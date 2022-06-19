""" Render functions for Hakowan """

from ..common.exception import InvalidSetting
from ..grammar.layer import Layer
from ..scene.scene_utils import generate_scene
from ..scene.scene import Scene
from .mitsuba_utils import generate_mitsuba_config

import pathlib
import subprocess


def render_mitsuba(scene: Scene, filename: str):
    xml_doc = generate_mitsuba_config(scene)
    xml_str = xml_doc.toxml()

    filename = pathlib.Path(filename)
    xml_file = filename.with_suffix(".scene")
    with open(xml_file, "w") as fin:
        xml_doc.writexml(fin, indent="", addindent="    ", newl="\n")


def render(root: Layer, backend: str, filename: str):
    """Render layer tree rooted at `root`."""
    scene = generate_scene(root)

    if backend == "mitsuba":
        render_mitsuba(scene, filename)
    else:
        raise InvalidSetting(f"Unsupported rendering backend: {backend}")
