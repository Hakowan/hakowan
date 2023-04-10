""" Mitsuba-related utility functions. """

from xml.dom import minidom
from pathlib import Path
import tempfile
import datetime

import math
import numpy as np
import numpy.typing as npt
from numpy.linalg import norm
from typing import Union

from ..common.exception import InvalidSetting
from ..scene.scene import Scene
from .serialization import serialize_mesh
from .render_config import RenderConfig


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
    target: npt.NDArray,
    origin: npt.NDArray,
    up: npt.NDArray,
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
    target: npt.NDArray,
    origin: npt.NDArray,
    up: npt.NDArray,
):
    """Generate xml element <transform></transform> based on lookat transform."""
    transform_xml = xml_doc.createElement("transform")
    transform_xml.setAttribute("name", name)
    transform_xml.appendChild(generate_lookat(xml_doc, target, origin, up))
    return transform_xml


def generate_scale(xml_doc: minidom.Document, scales: npt.NDArray):
    """Generate xml element <scale></scale>"""
    scale_xml = xml_doc.createElement("scale")
    scale_xml.setAttribute("value", ",".join([str(v) for v in scales]))
    return scale_xml


def generate_rotate_axis_angle(
    xml_doc: minidom.Document, axis: npt.NDArray, angle: float
):
    """Generate xml element <rotate></rotate>"""
    rotate_xml = xml_doc.createElement("rotate")
    rotate_xml.setAttribute("value", ",".join([str(v) for v in axis]))
    rotate_xml.setAttribute("angle", str(angle))
    return rotate_xml


def generate_translate(xml_doc: minidom.Document, offset: npt.NDArray):
    """Generate xml element <translate></translate>"""
    translate_xml = xml_doc.createElement("translate")
    translate_xml.setAttribute("value", ",".join([str(v) for v in offset]))
    return translate_xml


def generate_cylinder_transform(
    xml_doc: minidom.Document, p0: npt.NDArray, p1: npt.NDArray
):
    """Compute the transform needed to convert cononical cylinder to a cylinder
    with end points p0 and p1."""
    l = norm(p1 - p0)
    z_axis = np.array([0, 0, 1])
    c_axis = (p1 - p0) / l
    r_axis = np.cross(z_axis, c_axis)
    n = norm(r_axis)
    if n < 1e-12:
        r_axis = np.array([0, 1, 0], dtype=float)
    else:
        r_axis /= n
    angle = math.degrees(math.atan2(n, np.dot(z_axis, c_axis)))

    transform_xml = xml_doc.createElement("transform")
    transform_xml.setAttribute("name", "to_world")
    transform_xml.appendChild(generate_scale(xml_doc, np.array([1, 1, l])))
    transform_xml.appendChild(generate_rotate_axis_angle(xml_doc, r_axis, angle))
    transform_xml.appendChild(generate_translate(xml_doc, p0))
    return transform_xml


def generate_rgb(
    xml_doc: minidom.Document, name: str, color: Union[npt.NDArray, float]
):
    """Generate xml element <rgb></rgb>"""
    rgb_xml = xml_doc.createElement("rgb")
    rgb_xml.setAttribute("name", name)
    if isinstance(color, float):
        rgb_xml.setAttribute("value", str(color))
    else:
        rgb_xml.setAttribute("value", ",".join([str(v) for v in color]))
    return rgb_xml


def generate_bsdf_plastic(
    xml_doc: minidom.Document,
    diffuse_reflectance: Union[npt.NDArray, float] = 0.5,
    int_ior: float = 1.49,
    ext_ior: float = 1.000277,
    nonlinear: bool = False,
):
    """Generate xml element <bsdf></bsdf>"""
    bsdf_xml = xml_doc.createElement("bsdf")
    bsdf_xml.setAttribute("type", "plastic")

    bsdf_xml.appendChild(
        generate_rgb(xml_doc, "diffuse_reflectance", diffuse_reflectance)
    )
    bsdf_xml.appendChild(generate_float(xml_doc, "int_ior", int_ior))
    bsdf_xml.appendChild(generate_float(xml_doc, "ext_ior", ext_ior))
    bsdf_xml.appendChild(generate_boolean(xml_doc, "nonlinear", nonlinear))

    return bsdf_xml


def generate_bsdf_rough_plastic(
    xml_doc: minidom.Document,
    distribution: str = "beckmann",
    diffuse_reflectance: Union[npt.NDArray, float] = 0.5,
    int_ior: float = 1.5,
    ext_ior: float = 1.000277,
    alpha: float = 0.3,
    nonlinear: bool = False,
):
    """Generate bsdf for rough plastic."""
    bsdf_xml = xml_doc.createElement("bsdf")
    bsdf_xml.setAttribute("type", "roughplastic")
    bsdf_xml.appendChild(generate_string(xml_doc, "distribution", distribution))
    bsdf_xml.appendChild(generate_float(xml_doc, "int_ior", int_ior))
    bsdf_xml.appendChild(generate_float(xml_doc, "ext_ior", ext_ior))
    bsdf_xml.appendChild(
        generate_rgb(xml_doc, "diffuse_reflectance", diffuse_reflectance)
    )
    bsdf_xml.appendChild(generate_float(xml_doc, "alpha", alpha))
    bsdf_xml.appendChild(generate_boolean(xml_doc, "nonlinear", nonlinear))

    return bsdf_xml


def generate_bsdf_rough_conductor(xml_doc: minidom.Document, material: str = "Cu"):
    """Generate bsdf for rough conductor."""
    bsdf_xml = xml_doc.createElement("bsdf")
    bsdf_xml.setAttribute("type", "roughconductor")
    bsdf_xml.appendChild(generate_string(xml_doc, "material", material))
    bsdf_xml.appendChild(generate_string(xml_doc, "distribution", "ggx"))

    return bsdf_xml


def generate_bsdf_diffuse(
    xml_doc: minidom.Document, reflectance: Union[npt.NDArray, float] = 0.5
):
    bsdf_xml = xml_doc.createElement("bsdf")
    bsdf_xml.setAttribute("type", "diffuse")

    bsdf_xml.appendChild(generate_rgb(xml_doc, "reflectance", reflectance))

    return bsdf_xml


def generate_sphere(
    xml_doc, center, radius, transform: Union[npt.NDArray, None] = None
):
    """Generate xml element <shape type="sphere"></shape>"""
    shape_xml = xml_doc.createElement("shape")
    shape_xml.setAttribute("type", "sphere")

    center_xml = generate_point(xml_doc, "center", center)
    radius_xml = generate_float(xml_doc, "radius", radius)

    shape_xml.appendChild(center_xml)
    shape_xml.appendChild(radius_xml)

    if transform is not None:
        shape_xml.appendChild(generate_transform(xml_doc, "to_world", transform))

    return shape_xml


def generate_cylinder(
    xml_doc: minidom.Document,
    p0: npt.NDArray,
    p1: npt.NDArray,
    radius: float,
    transform: Union[npt.NDArray, None] = None,
):
    """Generate xml element <shape type="cylinder"></shape>"""
    shape_xml = xml_doc.createElement("shape")
    shape_xml.setAttribute("type", "cylinder")
    shape_xml.appendChild(generate_float(xml_doc, "radius", radius))

    if transform is None:
        transform = np.identity(4)

    p0 = np.dot(transform[:3, :3], p0) + transform[:3, 3]
    p1 = np.dot(transform[:3, :3], p1) + transform[:3, 3]

    if norm(p1 - p0) < 1e-6:
        return

    shape_xml.appendChild(generate_cylinder_transform(xml_doc, p0, p1))
    return shape_xml


def generate_mesh(
    xml_doc: minidom.Document,
    vertices: npt.NDArray,
    faces: npt.NDArray,
    normals: Union[npt.NDArray, None] = None,
    uvs: Union[npt.NDArray, None] = None,
    colors: Union[npt.NDArray, None] = None,
    transform: Union[npt.NDArray, None] = None,
):
    """Generate xml element <shape type="serialized"></shape>"""
    data = serialize_mesh(vertices, faces, normals, colors, uvs)
    timestamp = datetime.datetime.now().isoformat()
    tmp_file = Path(tempfile.gettempdir()) / f"{timestamp}.scene"
    with open(tmp_file, "wb") as fout:
        fout.write(data)

    shape_xml = xml_doc.createElement("shape")
    shape_xml.setAttribute("type", "serialized")
    shape_xml.appendChild(generate_string(xml_doc, "filename", str(tmp_file)))
    shape_xml.appendChild(generate_boolean(xml_doc, "face_normals", False))

    if transform is not None:
        shape_xml.appendChild(generate_transform(xml_doc, "to_world", transform))

    return shape_xml


def generate_sampler(xml_doc: minidom.Document, sampler_type: str, sample_count: int):
    """Generate xml element <sampler></sampler>"""
    sampler = xml_doc.createElement("sampler")
    sampler.setAttribute("type", sampler_type)
    sampler.appendChild(generate_integer(xml_doc, "sample_count", sample_count))
    return sampler


def generate_rfilter(xml_doc: minidom.Document, filter_type: str):
    """Generate xml element <rfilter></rfilter>"""
    rfilter = xml_doc.createElement("rfilter")
    rfilter.setAttribute("type", filter_type)
    # rfilter.appendChild(generate_float(xml_doc, "stddev", 0.25))
    return rfilter


def generate_film(xml_doc: minidom.Document, width: int, height: int):
    """Generate xml element <film></film>"""
    film = xml_doc.createElement("film")
    film.setAttribute("type", "hdrfilm")
    film.appendChild(generate_integer(xml_doc, "width", width))
    film.appendChild(generate_integer(xml_doc, "height", height))
    film.appendChild(generate_string(xml_doc, "pixel_format", "rgba"))
    # film.appendChild(generate_boolean(xml_doc, "banner", "false"))
    film.appendChild(generate_rfilter(xml_doc, "gaussian"))
    return film


def generate_camera(
    xml_doc: minidom.Document,
    config: RenderConfig,
):
    """Generate camera setting"""
    sensor = xml_doc.createElement("sensor")
    sensor.setAttribute("type", "perspective")
    sensor.appendChild(generate_string(xml_doc, "fov_axis", "smaller"))
    # sensor.appendChild(generate_float(xml_doc, "focus_distance", "3.0"))
    sensor.appendChild(generate_float(xml_doc, "fov", config.fov))
    sensor.appendChild(
        generate_transform_lookat(
            xml_doc,
            "to_world",
            target=np.array([0, 0, 0]),
            origin=np.array([0, 0, 5]),
            up=np.array([0, 1, 0]),
        )
    )
    sensor.appendChild(
        generate_sampler(xml_doc, config.sampler_type, config.num_samples)
    )
    sensor.appendChild(generate_film(xml_doc, config.width, config.height))
    return sensor


def generate_emitter(xml_doc: minidom.Document, emitter_type: str):
    """Generate xml element <emitter></emitter>."""
    emitter = xml_doc.createElement("emitter")
    emitter.setAttribute("type", emitter_type)
    return emitter


def generate_front_light(xml_doc: minidom.Document):
    """Generate front light."""
    light = generate_sphere(xml_doc, [-2, 6, 6], 0.5)
    emitter = generate_emitter(xml_doc, "area")
    emitter.appendChild(generate_tag(xml_doc, "spectrum", "radiance", 250))
    light.appendChild(emitter)
    return light


def generate_side_light(xml_doc: minidom.Document):
    """Generate side light."""
    light = generate_sphere(xml_doc, [6, 0, 0], 0.2)
    emitter = generate_emitter(xml_doc, "area")
    emitter.appendChild(generate_tag(xml_doc, "spectrum", "radiance", 30))
    light.appendChild(emitter)
    return light


def generate_back_light(xml_doc: minidom.Document):
    """Generate back light."""
    emitter = generate_emitter(xml_doc, "point")
    emitter.appendChild(generate_tag(xml_doc, "spectrum", "intensity", 10))
    emitter.appendChild(generate_point(xml_doc, "position", [0, 0, -6]))
    return emitter


def generate_fill_light(xml_doc: minidom.Document):
    """Generate fill light"""
    emitter = generate_emitter(xml_doc, "constant")
    emitter.appendChild(generate_tag(xml_doc, "spectrum", "radiance", 0.1))
    return emitter


def generate_scene(xml_doc: minidom.Document):
    """Generate xml element <scene></scene>"""
    scene_xml = xml_doc.createElement("scene")
    scene_xml.setAttribute("version", "3.0.0")
    return scene_xml


def generate_integrator(xml_doc: minidom.Document, integrator_type):
    """Generate xml element <integrator></integrator>"""
    integrator = xml_doc.createElement("integrator")
    integrator.setAttribute("type", integrator_type)
    # integrator.appendChild(generate_boolean(xml_doc, "hide_emitters", "false"))
    # integrator.appendChild(generate_integer(xml_doc, "rr_depth", 100))
    return integrator
