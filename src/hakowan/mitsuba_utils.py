""" Mitsuba-related utility functions. """

from xml.dom import minidom

import numpy as np

from .layer import Layer
from .layer_data import LayerData, Mark
from .exception import InvalidSetting
from .render_utils import process_position_channel, process_color_channel


def generate_float(xml_doc, name, val):
    """Generate xml element <float></float>"""
    float_xml = xml_doc.createElement("float")
    float_xml.setAttribute("name", name)
    float_xml.setAttribute("value", val)
    return float_xml


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
    matrix_xml.addAttribute("value", " ".join(map(str, matrix)))
    return matrix_xml


def generate_transform(xml_doc: minidom.Document, name, transform):
    """Generate xml element <transform></transform>"""
    transform_xml = xml_doc.createElement("transform")
    transform_xml.setAttribute("name", name)
    transform_xml.appendChild(generate_matrix(xml_doc, transform))
    return transform_xml


def generate_rgb(xml_doc: minidom.Document, name, color):
    """Generate xml element <rgb></rgb>"""
    # TODO


def generate_bsdf_plastic(
    xml_doc: minidom.Document,
    diffuse_reflectance=0.5,
    # specular_reflectance=1.0,
    int_ior=1.49,
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


def __generate_point_view(layer_data: LayerData, xml_doc: minidom.Document):
    positions = process_position_channel(layer_data)
    colors = process_color_channel(layer_data)


def __generate_mitsuba_config(
    node: Layer, data_stack: list[LayerData], xml_doc: minidom.Document
):
    # Generated consolidated layer data for the current layer.
    layer_data = LayerData()
    if len(data_stack) > 0:
        layer_data = data_stack[-1]
    layer_data = layer_data | node.layer_data

    if len(node.children) == 0:
        if layer_data.mark == Mark.Point:
            pass
        elif layer_data.mark == Mark.Curve:
            pass
        elif layer_data.mark == Mark.Surface:
            pass
        else:
            raise InvalidSetting(f"Unsupported mark: {layer_data.mark}")
    else:
        configs = []
        for child in node.children:
            data_stack.append(layer_data)
            configs += generate_mitsuba_config(node.children)
            data_stack.pop()
        return configs


def generate_mitsuba_config(root: Layer):
    """Convert layer tree into mitsuba xml input."""
    xml_doc = minidom.Document()
    data_stack = []

    elements = __generate_mitsuba_config(root, data_stack, xml_doc)
    for elem in elements:
        xml_doc.appendChild(elements)

    return xml_doc
