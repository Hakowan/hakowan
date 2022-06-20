from .mitsuba_utils import (
    generate_scene,
    generate_integrator,
    generate_camera,
    generate_front_light,
    generate_side_light,
    generate_back_light,
    generate_fill_light,
    generate_sphere,
    generate_bsdf_plastic,
    generate_cylinder,
)

from .render_config import RenderConfig
from ..scene.scene import Scene

import numpy as np
import pathlib
import subprocess
from xml.dom import minidom


def generate_mitsuba_config(scene: Scene, config: RenderConfig):
    """Convert layer tree into mitsuba xml input."""
    xml_doc = minidom.Document()
    scene_xml = generate_scene(xml_doc)
    scene_xml.appendChild(generate_integrator(xml_doc, "path"))

    scene_xml.appendChild(
        generate_camera(
            xml_doc,
            config.width,
            config.height,
            config.fov,
            config.focus_distance,
            config.num_samples,
        )
    )
    scene_xml.appendChild(generate_front_light(xml_doc))
    scene_xml.appendChild(generate_side_light(xml_doc))
    scene_xml.appendChild(generate_back_light(xml_doc))
    scene_xml.appendChild(generate_fill_light(xml_doc))

    # Compute global transform to [-1, 1]^3.
    bbox_min, bbox_max = scene.bbox

    # Gather points.
    for p in scene.points:
        sphere = generate_sphere(xml_doc, p.center, p.radius, config.transform)
        material = generate_bsdf_plastic(xml_doc, p.color)
        sphere.appendChild(material)
        scene_xml.appendChild(sphere)

    for s in scene.segments:
        segment = generate_cylinder(
            xml_doc, s.vertices[0], s.vertices[1], np.mean(s.radii), config.transform
        )
        material = generate_bsdf_plastic(xml_doc, np.mean(s.colors, axis=0))
        segment.appendChild(material)
        scene_xml.appendChild(segment)

    xml_doc.appendChild(scene_xml)
    return xml_doc


def render_with_mitsuba(scene: Scene, config: RenderConfig):
    """Render using Mitsuba backend."""
    xml_doc = generate_mitsuba_config(scene, config)
    xml_str = xml_doc.toxml()

    filename = config.filename
    if not isinstance(filename, pathlib.Path):
        filename = pathlib.Path(filename)
    xml_file = filename.with_suffix(".xml")
    with open(xml_file, "w") as fin:
        xml_doc.writexml(fin, indent="", addindent="    ", newl="\n")
