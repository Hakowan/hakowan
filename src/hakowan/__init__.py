"""Hakowan: A 3D data visualization grammar"""

__version__ = "0.5.3"

from .common import logger
from .setup import Config as config
from .grammar import dataframe, mark, channel, scale, texture, transform
from .grammar.layer import Layer as layer
from .grammar.scale import Attribute as attribute
from .grammar.scale import norm
from .grammar.channel import material
from .grammar.overlay import Annotation, Legend
from .compiler import compile
from .inspection import AttributeSummary, DataSummary, inspect
from .validation import Diagnostic, ValidationError, ValidationReport, validate
from .spec import (
    FigureSpec,
    SpecConversionError,
    from_json,
    from_spec,
    json_schema as schema,
    load_layer,
    load_spec,
    to_spec,
)
from .render import (
    render,
    RenderResult,
    set_default_backend,
    list_backends,
)
from .observation import (
    CameraState,
    Observation,
    ObservationError,
    PixelHit,
    SceneSummary,
    Snapshot,
    observe,
    snapshot,
)

# Register backends lazily: the loader (and thus the heavy import — Mitsuba/LLVM,
# bpy, pygltflib) only runs when that backend is first requested. ``requires`` is
# probed without importing, so a backend whose dependency is missing simply
# doesn't appear in ``list_backends()``.
from .backends import (
    BLENDER_CAPABILITIES,
    MITSUBA_CAPABILITIES,
    WEBGL_CAPABILITIES,
    BackendCapabilities,
    get_backend_capabilities as backend_capabilities,
    list_backend_capabilities,
    register_backend_loader,
)


def _load_mitsuba_backend():
    import sys

    if "bpy" in sys.modules and "mitsuba" not in sys.modules:
        raise RuntimeError(
            "Cannot load the Mitsuba backend: 'bpy' (Blender) is already imported "
            "and Mitsuba has not been loaded yet. Blender's bundled LLVM conflicts "
            "with Dr.Jit/Mitsuba when loaded first. Use the Mitsuba backend before "
            "the Blender backend in the same process."
        )
    from .backends.mitsuba import MitsubaBackend

    return MitsubaBackend


def _load_blender_backend():
    from .backends.blender import BlenderBackend

    return BlenderBackend


def _load_webgl_backend():
    from .backends.webgl import WebGLBackend

    return WebGLBackend


register_backend_loader(
    "mitsuba",
    _load_mitsuba_backend,
    requires="mitsuba",
    capabilities=MITSUBA_CAPABILITIES,
)
register_backend_loader(
    "blender",
    _load_blender_backend,
    requires="bpy",
    capabilities=BLENDER_CAPABILITIES,
)
register_backend_loader(
    "webgl",
    _load_webgl_backend,
    requires="pygltflib",
    capabilities=WEBGL_CAPABILITIES,
)

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
    "attribute",
    "material",
    "norm",
    "compile",
    "render",
    "RenderResult",
    "set_default_backend",
    "Annotation",
    "Legend",
    "list_backends",
    "AttributeSummary",
    "DataSummary",
    "inspect",
    "Diagnostic",
    "ValidationError",
    "ValidationReport",
    "validate",
    "BackendCapabilities",
    "backend_capabilities",
    "list_backend_capabilities",
    "FigureSpec",
    "SpecConversionError",
    "schema",
    "to_spec",
    "from_spec",
    "from_json",
    "load_layer",
    "load_spec",
    "CameraState",
    "Observation",
    "ObservationError",
    "PixelHit",
    "SceneSummary",
    "Snapshot",
    "observe",
    "snapshot",
]
