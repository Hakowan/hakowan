""" Utility functions to generate scenes. """
import numpy as np
import numbers

from ..common.color import Color
from ..common.named_colors import css_colors
from ..common.colormap.named_colormaps import named_colormaps
from ..common.exception import InvalidSetting
from ..common.default import DEFAULT_COLOR, DEFAULT_SIZE, DEFAULT_POSITION
from ..grammar.layer import Layer
from ..grammar.layer_data import LayerData, Mark, ChannelSetting, DataFrame
from .scene import Scene, Surface, Point, Segment


def extract_position_channel(layer_data: LayerData):
    """Extract position channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        (np.ndarray, np.ndarray): The position encoded as nodes and elements.
    """
    assert layer_data.data is not None

    data = layer_data.data

    position_name = layer_data.channel_setting.position
    if position_name is None:
        position_name = DEFAULT_POSITION
    position_attr = data.attributes.get(position_name, None)

    assert position_attr is not None
    assert position_attr.values is not None
    assert position_attr.indices is not None

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

    assert layer_data.data is not None

    data = layer_data.data

    attr_name = layer_data.channel_setting.color
    if attr_name is None:
        # Use default color.
        c = css_colors[DEFAULT_COLOR]
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

        attr = data.attributes[attr_name]
        assert attr.values is not None
        assert attr.indices is not None
        color_values = np.array([colormap(v).data for v in attr.values])
        color_indices = attr.indices
        return color_values, color_indices

    raise InvalidSetting(f"Unable to interpret 'color' setting: {attr_name}")


def extract_size_channel(layer_data: LayerData, shape: tuple[int, int]):
    """Extract color channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        (np.ndarray, np.ndarray): The encoded size and indices.
    """
    assert layer_data.data is not None

    data = layer_data.data

    attr_name = layer_data.channel_setting.size
    if attr_name is None:
        # Default size.
        sizes = np.array([DEFAULT_SIZE])
        return sizes, np.zeros(shape, dtype=int)
    if isinstance(attr_name, numbers.Number):
        # Constant size.
        sizes = np.array([attr_name])
        return sizes, np.zeros(shape, dtype=int)
    if isinstance(attr_name, str) and attr_name in data.attributes:
        # Convert attribute to size field.
        size_map = layer_data.channel_setting.size_map
        if size_map is None:
            size_map = lambda x: x
        elif size_map == "identity":
            size_map = lambda x: x
        elif size_map == "normalized":
            raise NotImplementedError("Not supported yet")
        elif not callable(size_map):
            raise InvalidSetting("Unsupported size_map!")
        attr = data.attributes[attr_name]
        assert attr.values is not None
        assert attr.indices is not None
        size_values = np.array([size_map(v) for v in attr.values])
        size_indices = attr.indices
        return size_values, size_indices

    raise InvalidSetting(f"Unable to interpret 'size' setting: {attr_name}")


def update_points(layer_data: LayerData, scene: Scene):
    """Update points based on `layer_data`.

    Args:
        layer_data (LayerData): The input layer data.
        scene (Scene): The output scene object.
    """

    nodes, _ = extract_position_channel(layer_data)
    assert nodes.shape[1] == 3
    num_nodes = len(nodes)
    color_values, color_indices = extract_color_channel(layer_data, (num_nodes, 1))
    size_values, size_indices = extract_size_channel(layer_data, (num_nodes, 1))

    assert color_indices.size == num_nodes 
    color_indices = color_indices.ravel()
    assert size_indices.size == num_nodes
    size_indices = size_indices.ravel()

    for i in range(num_nodes):
        p = Point(
            center=nodes[i],
            radius=size_values[size_indices[i]],
            color=color_values[color_indices[i]],
        )
        scene.points.append(p)


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
