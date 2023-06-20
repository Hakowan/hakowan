import numpy as np
from numpy.linalg import norm
import numpy.typing as npt
import pathlib
from xml.dom import minidom
import mitsuba as mi
from typing import TypedDict, Union

from .mitsuba_utils import (
    generate_bsdf_conductor,
    generate_bsdf_dielectric,
    generate_bsdf_diffuse,
    generate_bsdf_plastic,
    generate_bsdf_principled,
    generate_bsdf_rough_conductor,
    generate_bsdf_rough_plastic,
    generate_camera,
    generate_cylinder,
    generate_envmap,
    generate_integrator,
    generate_mesh,
    generate_scene,
    generate_sphere,
)
from .render_config import RenderConfig
from ..scene.scene import Scene
from ..common.color import Color


def generate_mitsuba_config(scene: Scene, config: RenderConfig):
    """Convert layer tree into mitsuba xml input."""
    xml_doc = minidom.Document()
    scene_xml = generate_scene(xml_doc)
    scene_xml.appendChild(generate_integrator(xml_doc, "path"))

    scene_xml.appendChild(generate_camera(xml_doc, config))
    scene_xml.appendChild(generate_envmap(xml_doc, config.envmap, config.envmap_scale))

    # Compute global transform to [-1, 1]^3.
    bbox_min, bbox_max = scene.bbox
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_radius = norm(bbox_max - bbox_min) / 2

    translate_transform = np.identity(4)
    translate_transform[:3, 3] = -bbox_center
    scale_transform = np.identity(4)
    scale_transform[:3, :3] /= bbox_radius
    global_transform = np.dot(
        config.transform, np.dot(scale_transform, translate_transform)
    )

    # Gather points.
    for p in scene.points:
        sphere = generate_sphere(xml_doc, p.center, p.radius, global_transform)
        material = generate_bsdf_principled(
            xml_doc,
            base_color=p.color,
            roughness=p.roughness,
            metallic=p.metallic,
        )
        sphere.appendChild(material)
        scene_xml.appendChild(sphere)

    ## Gather segments.
    # for s in scene.segments:
    #    segment = generate_cylinder(
    #        xml_doc, s.vertices[0], s.vertices[1], np.mean(s.radii), global_transform
    #    )
    #    material = generate_bsdf_plastic(xml_doc, np.mean(s.colors, axis=0))
    #    segment.appendChild(material)
    #    scene_xml.appendChild(segment)

    # Gather surfaces.
    for m in scene.surfaces:
        mesh_attributes = {}
        material_color: Union[str, Color, pathlib.Path]
        material_roughness: Union[str, float, pathlib.Path]
        material_metallic: Union[str, float]

        prefix = lambda data: "vertex" if len(data) == len(m.vertices) else "face"

        if isinstance(m.normals, np.ndarray):
            mesh_attributes["normal"] = m.normals

        if isinstance(m.uvs, np.ndarray):
            mesh_attributes["uv"] = m.uvs

        if isinstance(m.color, np.ndarray):
            mesh_attributes["color"] = m.color
            material_color = f"{prefix(m.color)}_color"
        else:
            material_color = m.color

        if isinstance(m.roughness, np.ndarray):
            mesh_attributes["roughness"] = m.roughness
            material_roughness = f"{prefix(m.roughness)}_roughness"
        else:
            material_roughness = m.roughness

        if isinstance(m.metallic, np.ndarray):
            mesh_attributes["metallic"] = m.metallic
            material_metallic = f"{prefix(m.metallic)}_metallic"
        else:
            assert isinstance(m.metallic, float)
            material_metallic = m.metallic

        # TODO: double check transform here.
        mesh = generate_mesh(
            xml_doc, m.vertices, m.triangles, global_transform, **mesh_attributes
        )

        material = generate_bsdf_principled(
            xml_doc,
            base_color=material_color,
            roughness=material_roughness,
            metallic=material_metallic,
        )

        mesh.appendChild(material)
        scene_xml.appendChild(mesh)

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

    if config.dry_run:
        # Mission accomplished!
        return

    # mi.set_variant("llvm_ad_rgb")
    mi.set_variant("scalar_rgb")
    mi_scene = mi.load_file(str(xml_file))
    image = mi.render(mi_scene)
    # mi.Bitmap(image).write(str(config.filename))
    mi.util.write_bitmap(config.filename, image)
