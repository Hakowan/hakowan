import numpy as np
from numpy.linalg import norm
import pathlib
from xml.dom import minidom
import mitsuba as mi

from .mitsuba_utils import (
    generate_back_light,
    generate_bsdf_conductor,
    generate_bsdf_dielectric,
    generate_bsdf_diffuse,
    generate_bsdf_plastic,
    generate_bsdf_principled,
    generate_bsdf_rough_conductor,
    generate_bsdf_rough_plastic,
    generate_camera,
    generate_cylinder,
    generate_fill_light,
    generate_front_light,
    generate_integrator,
    generate_mesh,
    generate_scene,
    generate_side_light,
    generate_sphere,
)
from .render_config import RenderConfig
from ..scene.scene import Scene
from ..common.color import Color


def generate_material(xml_doc, material, color, material_preset):
    """Generate material specific xml node."""
    if material == "plastic":
        return generate_bsdf_plastic(xml_doc, diffuse_reflectance=color)
    elif material == "roughplastic":
        return generate_bsdf_rough_plastic(xml_doc, diffuse_reflectance=color)
    elif material == "conductor":
        return generate_bsdf_conductor(xml_doc, material=material_preset)
    elif material == "roughconductor":
        return generate_bsdf_rough_conductor(xml_doc, material=material_preset)
    elif material == "diffuse":
        return generate_bsdf_diffuse(xml_doc, reflectance=color)
    elif material == "dielectric":
        return generate_bsdf_dielectric(xml_doc)
    elif material == "principled":
        return generate_bsdf_principled(
            xml_doc, color, roughness=0.5, metallic=0.9, specular=0.1
        )
    else:
        raise ValueError(f"Unknown material {material}")


def generate_mitsuba_config(scene: Scene, config: RenderConfig):
    """Convert layer tree into mitsuba xml input."""
    xml_doc = minidom.Document()
    scene_xml = generate_scene(xml_doc)
    scene_xml.appendChild(generate_integrator(xml_doc, "path"))

    scene_xml.appendChild(generate_camera(xml_doc, config))
    scene_xml.appendChild(generate_front_light(xml_doc))
    scene_xml.appendChild(generate_side_light(xml_doc))
    scene_xml.appendChild(generate_back_light(xml_doc))
    # scene_xml.appendChild(generate_fill_light(xml_doc))

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
        color = p.color.data[:3]
        sphere = generate_sphere(xml_doc, p.center, p.radius, global_transform)
        material = generate_material(xml_doc, p.material, color, p.material_preset)
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
        elif isinstance(m.colors, Color):
            color = m.colors.data[:3]
            mesh = generate_mesh(
                xml_doc,
                m.vertices,
                m.triangles,
                m.normals,
                m.uvs,
                None,
                global_transform,
            )
            material = generate_material(xml_doc, m.material, color, m.material_preset)
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
            if len(m.colors) == len(m.vertices):
                material = generate_bsdf_diffuse(xml_doc, "vertex_color")
            elif len(m.colors) == len(m.triangles):
                material = generate_bsdf_diffuse(xml_doc, "face_color")
            else:
                # Invalid color setting.
                material = generate_bsdf_rough_plastic(xml_doc)
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
