""" Utility functions to generate scenes. """
import numpy as np
from numpy.linalg import norm
import numpy.typing as npt
import numbers
from typing import Union, Optional
import pathlib

from ..common.color import Color
from ..common.named_colors import css_colors
from ..common.colormap.named_colormaps import named_colormaps
from ..common.exception import InvalidSetting
from ..common.default import (
    DEFAULT_COLOR,
    DEFAULT_CURVE_COLOR,
    DEFAULT_POINT_COLOR,
    DEFAULT_ROUGHNESS,
    DEFAULT_METALLIC,
    DEFAULT_SIZE,
    DEFAULT_POSITION,
)
from ..grammar.layer import Layer
from ..grammar.layer_data import LayerData, Mark, ChannelSetting, DataFrame
from .scene import Scene, Surface, Point, Segment
import lagrange


def extract_position_channel(layer_data: LayerData) -> npt.NDArray:
    """Extract position channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        The positions.
    """
    assert layer_data.data is not None

    # TODO: Check layer_data.channel_setting.position
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

    # TODO: check layer_data.channel-setting.normal
    normals = layer_data.data.normals

    return normals


def extract_uv_channel(layer_data: LayerData) -> Optional[npt.NDArray]:
    """Extract uv channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        The uvs.
    """
    assert layer_data.data is not None

    uv_attr_name = layer_data.channel_setting.uv
    if uv_attr_name is None:
        # Use the first attribute with UV usage
        ids = layer_data.data.mesh.get_matching_attribute_ids(
            usage=lagrange.AttributeUsage.UV
        )
        if len(ids) == 0:
            return None
        uvs = layer_data.data.mesh.attribute(ids[0]).data
    else:
        assert layer_data.data.mesh.has_attribute(uv_attr_name)
        uvs = layer_data.data.mesh.attribute(uv_attr_name).data
    assert uvs.ndim == 2 and uvs.shape[1] == 2

    if layer_data.channel_setting.uv_map is not None:
        assert isinstance(layer_data.channel_setting.uv_map, float)
        scale = layer_data.channel_setting.uv_map
        uvs = uvs * scale
    return uvs


def extract_color_channel(
    layer_data: LayerData, default_color: str
) -> Union[Color, npt.NDArray, str, pathlib.Path]:
    """Extract color channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        Color: If the encoded color is uniform.
        np.ndarray: If the encoded color is changing.
    """

    assert layer_data.data is not None
    data = layer_data.data

    color = layer_data.channel_setting.color
    if color is None:
        # Use default color.
        return css_colors[default_color]
    if color.startswith("#"):
        # Color hex value.
        return Color.from_hex(color)
    if color in css_colors:
        # Color name.
        return css_colors[color]
    if color == "checkerboard":
        return color
    if pathlib.Path(color).exists():
        return pathlib.Path(color)

    assert isinstance(color, str)

    mesh = layer_data.data.mesh
    assert mesh.has_attribute(color)

    # Convert attribute to color using color map
    colormap = layer_data.channel_setting.color_map

    attr = mesh.attribute(color).data
    num_entries = data.mesh.num_vertices
    assert len(attr) == data.mesh.num_vertices or len(attr) == data.mesh.num_facets

    if not callable(colormap):
        if colormap is None:
            colormap = "viridis"
        assert isinstance(colormap, str)
        if colormap not in named_colormaps:
            raise InvalidSetting(f"Unknown colormap: {colormap}")
        colormap = named_colormaps[colormap]

        # Normalize attribute values.
        if attr.ndim == 2:
            attr = norm(attr, axis=1)
        if (attr.max() - attr.min()) > 0:
            attr = (attr - attr.min()) / (attr.max() - attr.min())

    return np.array([colormap(v).data for v in attr])


def extract_roughness_channel(
    layer_data: LayerData, default_roughness: float
) -> Union[float, npt.NDArray, str, pathlib.Path]:
    assert layer_data.data is not None
    data = layer_data.data

    roughness = layer_data.channel_setting.roughness
    if roughness is None:
        return default_roughness
    elif isinstance(roughness, numbers.Number):
        return float(roughness)
    elif roughness == "checkerboard":
        return roughness
    elif isinstance(roughness, str) and pathlib.Path(roughness).exists():
        return pathlib.Path(roughness)

    assert isinstance(roughness, str)
    mesh = data.mesh
    assert mesh.has_attribute(roughness)
    roughness_data = mesh.attribute(roughness).data

    assert (
        len(roughness_data) == mesh.num_vertices
        or len(roughness_data) == mesh.num_facets
    )

    if layer_data.channel_setting.roughness_map is not None:
        assert callable(layer_data.channel_setting.roughness_map)
        roughness_map = layer_data.channel_setting.roughness_map
        roughness_data = np.array([roughness_map(v) for v in roughness_data])
    else:
        # Normalize to [0, 1].
        if roughness_data.ndim == 2:
            roughness_data = norm(roughness_data, axis=1)
        if (roughness_data.max() - roughness_data.min()) > 0:
            roughness_data = (roughness_data - roughness_data.min()) / (
                roughness_data.max() - roughness_data.min()
            )

    return roughness_data


def extract_metallic_channel(
    layer_data: LayerData, default_metallic: float
) -> Union[float, npt.NDArray]:
    assert layer_data.data is not None
    data = layer_data.data

    metallic = layer_data.channel_setting.metallic
    if metallic is None:
        return default_metallic
    elif isinstance(metallic, numbers.Number):
        return float(metallic)

    assert isinstance(metallic, str)
    mesh = data.mesh
    assert mesh.has_attribute(metallic)
    metallic_data = mesh.attribute(metallic).data

    assert (
        len(metallic_data) == mesh.num_vertices or len(metallic_data) == mesh.num_facets
    )

    if layer_data.channel_setting.metallic_map is not None:
        assert callable(layer_data.channel_setting.metallic_map)
        metallic_map = layer_data.channel_setting.metallic_map
        metallic_data = np.array([metallic_map(v) for v in metallic_data])
    else:
        # Normalize to [0, 1].
        if metallic_data.ndim == 2:
            metallic_data = norm(metallic_data, axis=1)
        if (metallic_data.max() - metallic_data.min()) > 0:
            metallic_data = (metallic_data - metallic_data.min()) / (
                metallic_data.max() - metallic_data.min()
            )

    return metallic_data


def extract_size_channel(
    layer_data: LayerData, default_size: float
) -> Union[float, npt.NDArray]:
    """Extract color channel from layer data.

    Args:
        layer_data (LayerData): Input layer data.

    Returns:
        np.ndarray: The encoded size.
    """
    assert layer_data.data is not None

    data = layer_data.data

    size = layer_data.channel_setting.size
    if size is None:
        return default_size
    if isinstance(size, numbers.Number):
        return float(size)

    assert isinstance(size, str)
    mesh = layer_data.data.mesh
    assert mesh.has_attribute(size)

    size_data = mesh.attribute(size).data
    assert len(size_data) == mesh.num_vertices or len(size_data) == mesh.num_facets

    if layer_data.channel_setting.size_map is not None:
        assert callable(layer_data.channel_setting.size_map)
        size_map = layer_data.channel_setting.size_map
        size_data = np.array([size_map(v) for v in size_data])
    else:
        if size_data.ndim == 2:
            size_data = norm(size_data, axis=1)

        if (size_data.max() - size_data.min()) > 0:
            size_data = (size_data - size_data.min()) / (
                size_data.max() - size_data.min()
            )
    return size_data


def update_points(layer_data: LayerData, scene: Scene):
    """Update points based on `layer_data`.

    Args:
        layer_data (LayerData): The input layer data.
        scene (Scene): The output scene object.
    """

    nodes = extract_position_channel(layer_data)
    num_nodes = len(nodes)
    assert nodes.shape[1] == 3

    colors = extract_color_channel(layer_data, DEFAULT_POINT_COLOR)
    if isinstance(colors, Color):
        colors = np.repeat([colors], num_nodes, axis=0)
    elif isinstance(colors, np.ndarray):
        assert len(colors) == num_nodes
    else:
        raise ValueError("Unsupported color type for point cloud.")

    roughnesses = extract_roughness_channel(layer_data, DEFAULT_ROUGHNESS)
    if isinstance(roughnesses, float):
        roughnesses = np.repeat([roughnesses], num_nodes, axis=0)
    elif isinstance(roughnesses, np.ndarray):
        assert len(roughnesses) == num_nodes
    else:
        raise ValueError("Unsupported roughness type for point cloud.")

    metallics = extract_metallic_channel(layer_data, DEFAULT_METALLIC)
    if isinstance(metallics, float):
        metallics = np.repeat([metallics], num_nodes, axis=0)
    assert len(metallics) == num_nodes

    sizes = extract_size_channel(layer_data, DEFAULT_SIZE)
    if isinstance(sizes, float):
        sizes = np.repeat([sizes], num_nodes, axis=0)
    assert len(sizes) == num_nodes

    for p, c, r, m, s in zip(nodes, colors, roughnesses, metallics, sizes):
        point = Point(center=p, radius=s, color=c, roughness=r, metallic=m, alpha=1.0)
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

    color = extract_color_channel(layer_data, DEFAULT_COLOR)
    normals = extract_normal_channel(layer_data)
    uvs = extract_uv_channel(layer_data)
    roughness = extract_roughness_channel(layer_data, DEFAULT_ROUGHNESS)
    metallic = extract_metallic_channel(layer_data, DEFAULT_METALLIC)

    surface = Surface(
        vertices=nodes,
        triangles=elements,
        normals=normals,
        uvs=uvs,
        color=color,
        roughness=roughness,
        metallic=metallic,
        alpha=1.0,  # TODO
    )
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
