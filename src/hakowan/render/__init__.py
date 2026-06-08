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
    backend_impl = get_backend(backend_name)

    # Warn about any requested render pass the chosen backend cannot honor,
    # rather than silently dropping it. Capability is declared per backend via
    # RenderBackend.SUPPORTED_PASSES.
    supported = {p.name for p in backend_impl.SUPPORTED_PASSES}
    unsupported = config.render_passes - supported
    if unsupported:
        logger.warning(
            f"The '{backend_name}' backend does not support render pass(es) "
            f"{sorted(unsupported)}; they will be ignored. "
            f"Supported passes: {sorted(supported)}."
        )

    return backend_impl.render(scene, config, filename, **kwargs)


__all__ = ["render", "set_default_backend", "list_backends"]
