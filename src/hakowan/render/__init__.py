"""Rendering module - provides unified render interface."""

from ..backends import get_backend, set_default_backend, list_backends
from ..compiler import compile
from ..grammar import layer
from ..setup import Config
from ..common import logger
from pathlib import Path
from typing import Any

# NOTE: Mitsuba is intentionally NOT imported here. Variant selection (which
# initializes the Mitsuba runtime and its LLVM backend) happens lazily inside
# the Mitsuba backend, so importing hakowan or using a non-Mitsuba backend
# (Blender/WebGL) never loads Mitsuba/LLVM.


def render(
    root: layer.Layer,
    config: Config | None = None,
    filename: Path | str | None = None,
    backend: str | None = None,
    **kwargs: Any,
) -> Any:
    """Render a layer using the specified backend.

    Args:
        root: Root layer to render.
        config: Rendering configuration. If None, uses default.
        filename: Output filename.
        backend: Backend name ('mitsuba' or 'blender'). If None, uses default.
        **kwargs: Backend-specific options.

    Returns:
        Rendered image (format depends on backend).

    Examples:
        >>> import hakowan as hkw
        >>> layer = hkw.layer(mesh)
        >>> # Use default backend (mitsuba)
        >>> hkw.render(layer, filename="output.png")
        >>> # Use blender backend
        >>> hkw.render(layer, filename="output.png", backend="blender")
    """
    # Compile the layer tree into a scene
    scene = compile(root)
    logger.info("Compilation done")

    # Get config
    if config is None:
        config = Config()

    # Get backend and render
    logger.info(f"Using backend: {backend or 'default'}")
    backend_impl = get_backend(backend)
    return backend_impl.render(scene, config, filename, **kwargs)


__all__ = ["render", "set_default_backend", "list_backends"]
