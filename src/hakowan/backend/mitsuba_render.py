import numpy as np
from numpy.linalg import norm
import pathlib
from xml.dom import minidom
import mitsuba as mi

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
    generate_bsdf_diffuse,
    generate_bsdf_rough_plastic,
    generate_bsdf_rough_conductor,
    generate_cylinder,
    generate_mesh,
)
from .render_config import RenderConfig
from ..scene.scene import Scene


def generate_mitsuba_config(scene: Scene, config: RenderConfig):
    """Convert layer tree into mitsuba xml input."""
    xml_doc = minidom.Document()
    scene_xml = generate_scene(xml_doc)
    scene_xml.appendChild(generate_integrator(xml_doc, "path"))

    scene_xml.appendChild(
        generate_camera(xml_doc, config)
    )
    scene_xml.appendChild(generate_front_light(xml_doc))
    scene_xml.appendChild(generate_side_light(xml_doc))
    scene_xml.appendChild(generate_back_light(xml_doc))
    #scene_xml.appendChild(generate_fill_light(xml_doc))

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
        material = generate_bsdf_plastic(xml_doc, p.color)
        sphere.appendChild(material)
        scene_xml.appendChild(sphere)

    # Gather segments.
    for s in scene.segments:
        segment = generate_cylinder(
            xml_doc, s.vertices[0], s.vertices[1], np.mean(s.radii), global_transform
        )
        material = generate_bsdf_plastic(xml_doc, np.mean(s.colors, axis=0))
        segment.appendChild(material)
        scene_xml.appendChild(segment)

    # Gather surfaces.
    for m in scene.surfaces:
        if m.colors is None:
            mesh = generate_mesh(
                xml_doc,
                m.vertices,
                m.triangles,
                m.normals,
                m.uvs,
                None,
                global_transform,
            )
            material = generate_bsdf_rough_plastic(xml_doc)
        elif len(m.colors) == 1:
            mesh = generate_mesh(
                xml_doc,
                m.vertices,
                m.triangles,
                m.normals,
                m.uvs,
                None,
                global_transform,
            )
            material = generate_bsdf_rough_plastic(
                xml_doc, diffuse_reflectance=m.colors[0]
            )
        else:
            mesh = generate_mesh(
                xml_doc,
                m.vertices,
                m.triangles,
                m.normals,
                m.uvs,
                m.colors,
                global_transform,
            )
            # TODO: update to use vertex color
            #material = generate_bsdf_rough_plastic(
            #    xml_doc, diffuse_reflectance=m.colors[0], nonlinear=True
            #)
            # material = generate_bsdf_plastic(xml_doc,
            #        diffuse_reflectance=m.colors[0], int_ior=1.9, nonlinear=True)
            material = generate_bsdf_rough_conductor(xml_doc, "Au")
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

    mi.set_variant("llvm_ad_rgb")
    # mi.set_variant("scalar_rgb")
    mi_scene = mi.load_file(str(xml_file))
    image = mi.render(mi_scene)
    # mi.Bitmap(image).write(str(config.filename))
    mi.util.write_bitmap(config.filename, image)
