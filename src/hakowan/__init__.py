""" Hakowan: A 3D data visualization grammer """

from . import grammar

__version__ = "0.0.1"

# Add handy alias
POINT = grammar.layer_data.Mark.POINT
CURVE = grammar.layer_data.Mark.CURVE
SURFACE = grammar.layer_data.Mark.SURFACE


def layer(**kwargs):
    """Create a layer."""
    return grammar.layer.Layer(**kwargs)
