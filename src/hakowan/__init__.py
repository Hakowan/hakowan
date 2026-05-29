"""Hakowan: A 3D data visualization grammar"""

__version__ = "0.4.4"

from .common import logger
from .setup import Config as config
from .grammar import dataframe, mark, channel, scale, texture, transform
from .grammar.layer import Layer as layer
from .grammar.scale import Attribute as attribute
from .grammar.channel import material
from .compiler import compile
from .render import render, set_default_backend, list_backends

# Register backends lazily: the loader (and thus the heavy import — Mitsuba/LLVM,
# bpy, pygltflib) only runs when that backend is first requested. ``requires`` is
# probed without importing, so a backend whose dependency is missing simply
# doesn't appear in ``list_backends()``.
from .backends import register_backend_loader


def _load_mitsuba_backend():
    from .backends.mitsuba import MitsubaBackend

    return MitsubaBackend


def _load_blender_backend():
    from .backends.blender import BlenderBackend

    return BlenderBackend


def _load_webgl_backend():
    from .backends.webgl import WebGLBackend

    return WebGLBackend


register_backend_loader("mitsuba", _load_mitsuba_backend, requires="mitsuba")
register_backend_loader("blender", _load_blender_backend, requires="bpy")
register_backend_loader("webgl", _load_webgl_backend, requires="pygltflib")

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
