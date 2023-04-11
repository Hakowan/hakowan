""" Utility functions to generate scenes. """
import numpy as np
from numpy.linalg import norm
import numpy.typing as npt
import numbers
from typing import Union

from ..common.color import Color
from ..common.named_colors import css_colors
from ..common.colormap.named_colormaps import named_colormaps
from ..common.exception import InvalidSetting
from ..common.default import (
    DEFAULT_COLOR,
    DEFAULT_CURVE_COLOR,
    DEFAULT_POINT_COLOR,
    DEFAULT_SIZE,
    DEFAULT_POSITION,
)
from ..grammar.layer import Layer
from ..grammar.layer_data import LayerData, Mark, ChannelSetting, DataFrame
from .scene import Scene, Surface, Point, Segment


def extract_position_channel(layer_data: LayerData) -> npt.NDArray:
    """Extract position channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        The positions.
    """
    assert layer_data.data is not None

    nodes = layer_data.data.vertices

    if layer_data.channel_setting.position_map is not None:
        position_map = layer_data.channel_setting.position_map
        assert callable(position_map)
        nodes = np.array([position_map(p) for p in nodes])

    return nodes


def extract_index_channel(layer_data: LayerData) -> npt.NDArray:
    """Extract index channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        The indices.
    """
    assert layer_data.data is not None
    return layer_data.data.facets


def extract_normal_channel(layer_data: LayerData) -> npt.NDArray:
    """Extract normal channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        The normals.
    """
    assert layer_data.data is not None

    normals = layer_data.data.normals

    if layer_data.channel_setting.normal_map is not None:
        normal_map = layer_data.channel_setting.normal_map
        assert callable(normal_map)
        normals = np.array([normal_map(n) for n in normals])

    return normals


def extract_color_channel(
    layer_data: LayerData, default_color: str
) -> Union[Color, npt.NDArray]:
    """Extract color channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        Color: If the encoded color is uniform.
        np.ndarray: If the encoded color is changing.
    """

    assert layer_data.data is not None
    data = layer_data.data
    num_entries = data.mesh.num_vertices

    attr_name = layer_data.channel_setting.color
    if attr_name is None:
        # Use default color.
        c = css_colors[default_color]
        return c
    if attr_name.startswith("#"):
        # Color hex value.
        c = Color.from_hex(attr_name)
        return c
    if attr_name in css_colors:
        # Color name.
        c = css_colors[attr_name]
        return c
    if isinstance(attr_name, str):
        mesh = layer_data.data.mesh
        assert mesh.has_attribute(attr_name)

        # Convert attribute to color using color map
        colormap = layer_data.channel_setting.color_map
        if colormap is None:
            colormap = "viridis"
        if not callable(colormap):
            assert isinstance(colormap, str)
            if colormap not in named_colormaps:
                raise InvalidSetting(f"Unknown colormap: {colormap}")
            colormap = named_colormaps[colormap]

        attr = mesh.attribute(attr_name).data
        assert len(attr) == num_entries

        # Normalize attribute values.
        if attr.ndim == 2:
            attr = norm(attr, axis=1)
        if (attr.max() - attr.min()) > 0:
            attr = (attr - attr.min()) / (attr.max() - attr.min())
        else:
            # All values are the same.
            attr = np.zeros_like(attr)

        return np.array([colormap(v).data for v in attr.values])

    raise InvalidSetting(f"Unable to interpret 'color' setting: {attr_name}")


def extract_size_channel(layer_data: LayerData) -> npt.NDArray:
    """Extract color channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        np.ndarray: The encoded size.
    """
    assert layer_data.data is not None

    data = layer_data.data
    num_entries = data.mesh.num_vertices

    attr_name = layer_data.channel_setting.size
    if attr_name is None:
        # Default size.
        return np.repeat([DEFAULT_SIZE], num_entries, axis=0)
    if isinstance(attr_name, numbers.Number):
        # Constant size.
        return np.repeat([attr_name], num_entries, axis=0)
    if isinstance(attr_name, str):
        # Convert attribute to size field.
        mesh = layer_data.data.mesh
        assert mesh.has_attribute(attr_name)

        size_data = mesh.attribute(attr_name).data
        if size_data.ndim == 2:
            size_data = norm(size_data, axis=1)

        size_map = layer_data.channel_setting.size_map
        if size_map is None or size_map == "identity":
            return size_data
        elif size_map == "normalized":
            if (size_data.max() - size_data.min()) > 0:
                return (size_data - size_data.min()) / (
                    size_data.max() - size_data.min()
                )
            else:
                return np.zeros_like(size_data)
        elif callable(size_map):
            return np.array([size_map(v) for v in size_data])

    raise InvalidSetting(f"Unable to interpret 'size' setting: {attr_name}")


def update_points(layer_data: LayerData, scene: Scene):
    """Update points based on `layer_data`.

    Args:
        layer_data (LayerData): The input layer data.
        scene (Scene): The output scene object.
    """

    nodes = extract_position_channel(layer_data)
    assert nodes.shape[1] == 3
    colors = extract_color_channel(layer_data, DEFAULT_POINT_COLOR)
    sizes = extract_size_channel(layer_data)
    num_nodes = len(nodes)
    if isinstance(colors, Color):
        colors = np.repeat([colors], num_nodes, axis=0)

    for p, r, c in zip(nodes, sizes, colors):
        point = Point(center=p, radius=r, color=c)
        if layer_data.channel_setting.material is not None:
            point.material = layer_data.channel_setting.material
        if layer_data.channel_setting.material_preset is not None:
            point.material_preset = layer_data.channel_setting.material_preset
        scene.points.append(point)


def update_surfaces(layer_data: LayerData, scene: Scene):
    """Update surfaces based on `layer_data`.

    Args:
        layer_data (LayerData): The input layer data.
        scene (Scene): The output scene object.
    """

    nodes = extract_position_channel(layer_data)
    elements = extract_index_channel(layer_data)
    if len(elements) == 0:
        return

    assert nodes.shape[1] == 3
    assert elements.shape[1] == 3

    colors = extract_color_channel(layer_data, DEFAULT_COLOR)
    normals = extract_normal_channel(layer_data)
    # TODO: uvs

    surface = Surface(
        vertices=nodes, triangles=elements, colors=colors, normals=normals
    )
    if layer_data.channel_setting.material is not None:
        surface.material = layer_data.channel_setting.material
    if layer_data.channel_setting.material_preset is not None:
        surface.material_preset = layer_data.channel_setting.material_preset
    scene.surfaces.append(surface)


def update_scene(layer_data: LayerData, scene: Scene):
    """Update scene based on `layer_data`.

    Args:
        layer_data (LayerData): The input layer data.
        scene (Scene): The output scene object.
    """
    assert layer_data.mark is not None
    assert layer_data.data is not None

    if layer_data.mark == Mark.POINT:
        update_points(layer_data, scene)
    elif layer_data.mark == Mark.CURVE:
        raise NotImplementedError("Curve mark is not yet supported!")
    elif layer_data.mark == Mark.SURFACE:
        update_surfaces(layer_data, scene)
    else:
        raise InvalidSetting(f"Unsupported mark: {layer_data.mark}")


def process_layer(curr_layer: Layer, data_stack: list[LayerData], scene: Scene) -> None:
    """Update scene based on layer graph."""
    prev_data = data_stack[-1] if len(data_stack) > 0 else LayerData()
    curr_data = curr_layer.layer_data | prev_data

    if len(curr_layer.children) == 0:
        update_scene(curr_data, scene)
    else:
        data_stack.append(curr_data)
        for child_layer in curr_layer.children:
            process_layer(child_layer, data_stack, scene)
        data_stack.pop()


def generate_scene(root: Layer) -> Scene:
    """Generate scene based on the grammar specified in the layer graph rooted
    at `root`.

    Args:
        root (Layer): The root layer of a layer graph.

    Returns:
        Scene: The generate scene object.
    """

    scene = Scene([], [], [])
    process_layer(root, [], scene)
    return scene
