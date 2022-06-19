""" Mitsuba-related utility functions. """

from xml.dom import minidom

import numpy as np
import numpy.typing as npt
from typing import Union

from ..grammar.layer import Layer
from ..grammar.layer_data import LayerData, Mark
from ..common.exception import InvalidSetting
from ..scene.scene import Scene


def generate_tag(xml_doc, tag, name, val):
    """Generate xml element <tag></tag>"""
    tag_xml = xml_doc.createElement(tag)
    tag_xml.setAttribute("name", name)
    tag_xml.setAttribute("value", str(val))
    return tag_xml


def generate_boolean(xml_doc, name, val):
    """Generate xml element <boolean></boolean>"""
    return generate_tag(xml_doc, "boolean", name, val)


def generate_integer(xml_doc, name, val):
    """Generate xml element <integer></integer>"""
    return generate_tag(xml_doc, "integer", name, val)


def generate_float(xml_doc, name, val):
    """Generate xml element <float></float>"""
    return generate_tag(xml_doc, "float", name, val)


def generate_string(xml_doc, name, val):
    """Generate xml element <string></string>"""
    return generate_tag(xml_doc, "string", name, val)


def generate_point(xml_doc, name, point):
    """Generate xml element <point></point>"""
    point_xml = xml_doc.createElement("point")
    point_xml.setAttribute("name", name)
    point_xml.setAttribute(
        "value", f"{point[0]}, {point[1]}, {point[2] if len(point) > 2 else 0}"
    )
    return point_xml


def generate_matrix(xml_doc: minidom.Document, matrix: np.ndarray):
    """Generate xml element <matrix></matrix>"""
    matrix = matrix.flatten(order="C")
    matrix_xml = xml_doc.createElement("matrix")
    matrix_xml.setAttribute("value", " ".join(map(str, matrix)))
    return matrix_xml


def generate_lookat(
    xml_doc: minidom.Document,
    target: npt.ArrayLike,
    origin: npt.ArrayLike,
    up: npt.ArrayLike,
):
    """Generate lookat settings."""
    lookat = xml_doc.createElement("lookat")
    lookat.setAttribute("target", f"{target[0]}, {target[1]}, {target[2]}")
    lookat.setAttribute("origin", f"{origin[0]}, {origin[1]}, {origin[2]}")
    lookat.setAttribute("up", f"{up[0]}, {up[1]}, {up[2]}")
    return lookat


def generate_transform(xml_doc: minidom.Document, name, transform):
    """Generate xml element <transform></transform>"""
    transform_xml = xml_doc.createElement("transform")
    transform_xml.setAttribute("name", name)
    transform_xml.appendChild(generate_matrix(xml_doc, transform))
    return transform_xml


def generate_transform_lookat(
    xml_doc: minidom.Document,
    name: str,
    target: npt.ArrayLike,
    origin: npt.ArrayLike,
    up: npt.ArrayLike,
):
    transform_xml = xml_doc.createElement("transform")
    transform_xml.setAttribute("name", name)
    transform_xml.appendChild(generate_lookat(xml_doc, target, origin, up))
    return transform_xml


def generate_rgb(
    xml_doc: minidom.Document, name: str, color: Union[npt.NDArray, float]
):
    """Generate xml element <rgb></rgb>"""
    rgb_xml = xml_doc.createElement("rgb")
    rgb_xml.setAttribute("name", name)
    if isinstance(color, float):
        rgb_xml.setAttribute("value", color)
    else:
        rgb_xml.setAttribute("value", ",".join([str(v) for v in color]))
    return rgb_xml


def generate_bsdf_plastic(
    xml_doc: minidom.Document,
    diffuse_reflectance: Union[npt.NDArray, float] = 0.5,
    # specular_reflectance=1.0,
    int_ior: float = 1.49,
    # ext_ior=1.000277,
    # nonlinear=False,
):
    """Generate xml element <bsdf></bsdf>"""
    bsdf_xml = xml_doc.createElement("bsdf")
    bsdf_xml.setAttribute("type", "plastic")

    bsdf_xml.appendChild(
        generate_rgb(xml_doc, "diffuse_reflectance", diffuse_reflectance)
    )
    bsdf_xml.appendChild(generate_float(xml_doc, "int_ior", int_ior))

    return bsdf_xml


def generate_sphere(xml_doc, center, radius):
    """Generate xml element <shape type="sphere"></shape>"""
    shape_xml = xml_doc.createElement("shape")
    shape_xml.setAttribute("type", "sphere")

    center_xml = generate_point(xml_doc, "center", center)
    radius_xml = generate_float(xml_doc, "radius", radius)

    shape_xml.appendChild(center_xml)
    shape_xml.appendChild(radius_xml)

    return shape_xml


def generate_sampler(xml_doc: minidom.Document, sample_count: int):
    """Generate xml element <sampler></sampler>"""
    sampler = xml_doc.createElement("sampler")
    sampler.setAttribute("type", "ldsampler")
    sampler.appendChild(generate_integer(xml_doc, "sample_count", sample_count))
    return sampler


def generate_rfilter(xml_doc: minidom.Document, filter_type: str):
    """Generate xml element <rfilter></rfilter>"""
    rfilter = xml_doc.createElement("rfilter")
    rfilter.setAttribute("type", filter_type)
    return rfilter


def generate_film(xml_doc: minidom.Document, width: int, height: int):
    """Generate xml element <film></film>"""
    film = xml_doc.createElement("film")
    film.setAttribute("type", "hdrfilm")
    film.appendChild(generate_integer(xml_doc, "width", width))
    film.appendChild(generate_integer(xml_doc, "height", height))
    film.appendChild(generate_string(xml_doc, "pixel_format", "rgba"))
    film.appendChild(generate_boolean(xml_doc, "banner", "false"))
    film.appendChild(generate_rfilter(xml_doc, "gaussian"))
    return film


def generate_camera(xml_doc: minidom.Document, width: int = 2048, height: int = 1080):
    """Generate camera setting"""
    sensor = xml_doc.createElement("sensor")
    sensor.setAttribute("type", "perspective")
    sensor.appendChild(generate_string(xml_doc, "fov_axis", "smaller"))
    sensor.appendChild(generate_float(xml_doc, "focus_distance", "3.0"))
    sensor.appendChild(generate_float(xml_doc, "fov", "28.8415"))
    sensor.appendChild(
        generate_transform_lookat(
            xml_doc, "to_world", target=[0, 0, 0], origin=[0, 0, 3], up=[0, 1, 0]
        )
    )
    sensor.appendChild(generate_sampler(xml_doc, 64))
    sensor.appendChild(generate_film(xml_doc, width, height))
    return sensor


def generate_emitter(xml_doc: minidom.Document, emitter_type: str):
    """Generate xml element <emitter></emitter>."""
    emitter = xml_doc.createElement("emitter")
    emitter.setAttribute("type", emitter_type)
    return emitter


def generate_front_light(xml_doc: minidom.Document):
    """Generate front light."""
    light = generate_sphere(xml_doc, [-1, 1, 3], 0.5)
    emitter = generate_emitter(xml_doc, "area")
    emitter.appendChild(generate_tag(xml_doc, "spectrum", "radiance", 50))
    light.appendChild(emitter)
    return light


def generate_side_light(xml_doc: minidom.Document):
    """Generate side light."""
    light = generate_sphere(xml_doc, [3, 0, 0], 0.2)
    emitter = generate_emitter(xml_doc, "area")
    emitter.appendChild(generate_tag(xml_doc, "spectrum", "radiance", 15))
    light.appendChild(emitter)
    return light


def generate_back_light(xml_doc: minidom.Document):
    """Generate back light."""
    emitter = generate_emitter(xml_doc, "point")
    emitter.appendChild(generate_tag(xml_doc, "spectrum", "intensity", 10))
    emitter.appendChild(generate_point(xml_doc, "position", [0, 0, -3]))
    return emitter


def generate_fill_light(xml_doc: minidom.Document):
    """Generate fill light"""
    emitter = generate_emitter(xml_doc, "constant")
    emitter.appendChild(generate_tag(xml_doc, "spectrum", "radiance", 0.1))
    return emitter


def generate_scene(xml_doc: minidom.Document):
    """Generate xml element <scene></scene>"""
    scene_xml = xml_doc.createElement("scene")
    scene_xml.setAttribute("version", "2.2.1")
    return scene_xml


def generate_integrator(xml_doc: minidom.Document, integrator_type):
    """Generate xml element <integrator></integrator>"""
    integrator = xml_doc.createElement("integrator")
    integrator.setAttribute("type", integrator_type)
    return integrator


def generate_mitsuba_config(scene: Scene):
    """Convert layer tree into mitsuba xml input."""
    xml_doc = minidom.Document()
    scene_xml = generate_scene(xml_doc)
    scene_xml.appendChild(generate_integrator(xml_doc, "path"))

    scene_xml.appendChild(generate_camera(xml_doc))
    scene_xml.appendChild(generate_front_light(xml_doc))
    scene_xml.appendChild(generate_side_light(xml_doc))
    scene_xml.appendChild(generate_back_light(xml_doc))
    scene_xml.appendChild(generate_fill_light(xml_doc))

    # Gather points.
    for p in scene.points:
        sphere = generate_sphere(xml_doc, p.center, p.radius)
        material = generate_bsdf_plastic(xml_doc, p.color)
        sphere.appendChild(material)
        scene_xml.appendChild(sphere)

    xml_doc.appendChild(scene_xml)
    return xml_doc
