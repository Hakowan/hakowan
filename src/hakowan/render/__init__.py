"""Rendering module - provides unified render interface."""

from ..backends import (
    get_backend,
    resolve_backend_name,
    set_default_backend,
    list_backends,
)
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
    backend_name = resolve_backend_name(backend)
    logger.info(f"Using backend: {backend_name}")

    # The facet-ID pass is implemented only by the Blender backend; other
    # backends silently ignore config.facet_id, so warn rather than mislead.
    if config.facet_id and backend_name != "blender":
        logger.warning(
            f"config.facet_id is only supported by the 'blender' backend; "
            f"the '{backend_name}' backend will ignore it and produce no "
            f"facet-ID image."
        )

    backend_impl = get_backend(backend_name)
    return backend_impl.render(scene, config, filename, **kwargs)


__all__ = ["render", "set_default_backend", "list_backends"]
