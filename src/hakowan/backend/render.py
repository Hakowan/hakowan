""" Render functions for Hakowan """

from .layer_data import LayerData, Attribute, Mark
from .layer import Layer


def get_consolidated_layer_data(node: Layer):
    """Conslidate layer data from `node` all the way to the root.

    Args:
        node (Layer): The current (leaf) layer.

    Returns:
        LayerData: The combined layer data from `node` to the root.
    """
    data = LayerData()
    data = data | node.layer_data

    while node.parent is not None:
        node = node.parent
        data = data | node.layer_data

    return data


def render(root: Layer):
    """Render layer tree rooted at `root`."""
    layer_stack = [root]
    data_stack = [root.layer_data]
