"""Hakowan: A 3D data visualization grammar"""

__version__ = "0.4.0"

from .common import logger
from .setup import Config as config
from .grammar import dataframe, mark, channel, scale, texture, transform
from .grammar.layer import Layer as layer
from .grammar.scale import Attribute as attribute
from .grammar.channel import material
from .compiler import compile
from .render import render, set_default_backend, list_backends

# Register backends
from .backends import register_backend

# Try to register Mitsuba backend
try:
    from .backends.mitsuba import MitsubaBackend

    register_backend("mitsuba", MitsubaBackend)
except ImportError as e:
    logger.info(f"Mitsuba backend not available: {e}")

# Try to register Blender backend
try:
    from .backends.blender import BlenderBackend

    register_backend("blender", BlenderBackend)
except ImportError:
    logger.info("Blender backend not available (bpy not installed)")

__all__ = [
    "logger",
    "config",
    "dataframe",
    "mark",
    "channel",
    "scale",
    "texture",
    "transform",
    "layer",
    "material",
    "compile",
    "render",
    "set_default_backend",
    "list_backends",
]
