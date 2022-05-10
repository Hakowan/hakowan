""" Utility functions to generate scenes. """
import numpy as np

from ..common.color import Color
from ..common.named_colors import css_colors
from ..common.colormap.named_colormaps import named_colormaps
from ..common.exception import InvalidSetting
from ..grammar.layer import Layer
from ..grammar.layer_data import LayerData, Mark, ChannelSetting
from .scene import Scene, Surface, Point, Segment


default_color = css_colors["ivory"]


def extract_position_channel(layer_data: LayerData):
    """Extract position channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        (np.ndarray, np.ndarray): The position encoded as nodes and elements.
    """
    data = layer_data.data

    position_name = layer_data.channel_setting.position
    if position_name is None:
        position_name = "@geometry"  # TODO: document default name somewhere.
    position_attr = data.attributes.get(position_name, None)
    assert position_attr is not None

    nodes = position_attr.values
    elements = position_attr.indices

    if layer_data.channel_setting.position_map is not None:
        position_map = layer_data.channel_setting.position_map
        assert callable(position_map)
        nodes = np.array([position_map(p) for p in nodes])

    return nodes, elements


def extract_color_channel(layer_data: LayerData, shape: tuple[int, int]):
    """Extract color channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.
        shape (tuple[int, int]): The element array shape.

    Returns:
        (np.ndarray, np.ndarray): The encoded color and color indices.
    """

    data = layer_data.data

    attr_name = layer_data.channel_setting.color
    if attr_name is None:
        # Use default color.
        c = default_color
        color_values = np.array([[c[0], c[1], c[2]]])
        return color_values, np.zeros(shape, dtype=int)
    if attr_name.startswith("#"):
        # Color hex value.
        c = Color.from_hex(attr_name)
        color_values = np.array([[c[0], c[1], c[2]]])
        return color_values, np.zeros(shape, dtype=int)
    if attr_name in css_colors:
        # Color name.
        c = css_colors[attr_name]
        color_values = np.array([[c[0], c[1], c[2]]])
        return color_values, np.zeros(shape, dtype=int)
    if attr_name in data.attributes:
        # Convert attribute to color using color map
        colormap = layer_data.channel_setting.color_map
        if colormap is None:
            colormap = "viridis"
        if not callable(colormap):
            assert isinstance(colormap, str)
            if colormap not in named_colormaps:
                raise InvalidSetting(f"Unknown colormap: {colormap}")
            colormap = named_colormaps[colormap]

        attr = data.attribues[attr_name]
        color_values = np.array([colormap(v).data for v in attr.values])
        color_indices = attr.indices
        return color_values, color_indices

    raise InvalidSetting(f"Unable to interpret 'color' setting: {attr_name}")


def extract_size_channel(layer_data: LayerData):
    """Extract color channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        (np.ndarray, np.ndarray): The encoded size and indices.
    """
    raise NotImplementedError("Not supported yet")


def update_points(layer_data: LayerData, scene: Scene):
    """Update points based on `layer_data`.

    Args:
        layer_data (LayerData): The input layer data.
        scene (Scene): The output scene object.
    """

    nodes, _ = extract_position_channel(layer_data)
    raise NotImplementedError("Not supported yet")


def update_segments(layer_data: LayerData, scene: Scene):
    """Update segments based on `layer_data`.

    Args:
        layer_data (LayerData): The input layer data.
        scene (Scene): The output scene object.
    """
    raise NotImplementedError("Not supported yet")


def update_surfaces(layer_data: LayerData, scene: Scene):
    """Update surfaces based on `layer_data`.

    Args:
        layer_data (LayerData): The input layer data.
        scene (Scene): The output scene object.
    """

    nodes, elements = extract_position_channel(layer_data)
    if len(elements) == 0:
        return

    assert nodes.shape[1] == 3
    assert elements.shape[1] == 3

    color_values, color_indices = extract_color_channel(layer_data, elements.shape)
    # TODO: normals, uvs

    # TODO: need unify index buffer capability.
    num_elements = len(elements)
    vertices = np.zeros((num_elements * 3, 3))
    colors = np.zeros((num_elements * 3, 3))
    triangles = np.arange(num_elements * 3).reshape((num_elements, 3))
    for i in range(num_elements):
        vertices[i * 3] = nodes[elements[i, 0]]
        vertices[i * 3 + 1] = nodes[elements[i, 1]]
        vertices[i * 3 + 2] = nodes[elements[i, 2]]

        colors[i * 3] = color_values[color_indices[i, 0]]
        colors[i * 3 + 1] = color_values[color_indices[i, 1]]
        colors[i * 3 + 2] = color_values[color_indices[i, 2]]

    surface = Surface(vertices=vertices, triangles=triangles, colors=colors)
    scene.surfaces.append(surface)


def update_scene(layer_data: LayerData, scene: Scene):
    """Update scene based on `layer_data`.

    Args:
        layer_data (LayerData): The input layer data.
        scene (Scene): The output scene object.
    """
    assert layer_data.mark is not None
    assert layer_data.data is not None
    if layer_data.channel_setting is None:
        layer_data.channel_setting = ChannelSetting()

    if layer_data.mark == Mark.POINT:
        update_points(layer_data, scene)
    elif layer_data.mark == Mark.CURVE:
        update_segments(layer_data, scene)
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
