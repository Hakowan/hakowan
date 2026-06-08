"""Rendering module - provides unified render interface."""

from ..backends import (
    RenderBackend,
    get_backend,
    resolve_backend_name,
    set_default_backend,
    list_backends,
)
from ..compiler import compile
from ..grammar import layer
from ..setup import Config
from ..setup.render_pass import aov_path
from ..common import logger
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# NOTE: Mitsuba is intentionally NOT imported here. Variant selection (which
# initializes the Mitsuba runtime and its LLVM backend) happens lazily inside
# the Mitsuba backend, so importing hakowan or using a non-Mitsuba backend
# (Blender/WebGL) never loads Mitsuba/LLVM.


@dataclass
class RenderResult:
    """The outcome of a :func:`render` call.

    Attributes:
        backend: Name of the backend that produced this result.
        outputs: Manifest mapping ``"main"`` and each *honored* render pass to
            its artifact — a :class:`~pathlib.Path` for file backends, or the
            string ``"interactive"`` for viewer backends (WebGL). Empty when
            nothing was written to disk (``filename`` was ``None``).
        image: The in-memory rendered image when the backend produces one
            (Mitsuba); ``None`` otherwise. Use this for notebook display.
        path: The main output path when written to disk; ``None`` otherwise.

    The object is also a :pep:`519` path-like (``__fspath__``), so it can be
    passed straight to ``open()``, :class:`~pathlib.Path`, etc. when a main
    output file was written.
    """

    backend: str
    outputs: dict[str, Path | str] = field(default_factory=dict)
    image: Any = field(default=None, repr=False)
    path: Path | None = None

    def __fspath__(self) -> str:
        if self.path is None:
            raise TypeError(
                "RenderResult has no output path "
                "(render() was called without a filename)."
            )
        return str(self.path)


def render(
    root: layer.Layer,
    config: Config | None = None,
    filename: Path | str | None = None,
    backend: str | None = None,
    **kwargs: Any,
) -> RenderResult:
    """Render a layer using the specified backend.

    Args:
        root: Root layer to render.
        config: Rendering configuration. If None, uses default.
        filename: Output filename.
        backend: Backend name ('mitsuba' or 'blender'). If None, uses default.
        **kwargs: Backend-specific options.

    Returns:
        A :class:`RenderResult` bundling the in-memory ``image`` (Mitsuba), the
        main output ``path``, and the ``outputs`` manifest (per-pass sidecar
        files, or ``"interactive"`` for the WebGL viewer).

    Examples:
        >>> import hakowan as hkw
        >>> layer = hkw.layer(mesh)
        >>> result = hkw.render(layer, filename="output.png")
        >>> result.path            # PosixPath('output.png')
        >>> result.outputs         # {'main': PosixPath('output.png'), ...}
        >>> # Mitsuba: display the rendered image in a notebook
        >>> result.image
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

    raw = backend_impl.render(scene, config, filename, **kwargs)

    # Surface the produced artifacts so the user does not have to guess which
    # sidecar files appear (or that passes are live in the viewer instead).
    manifest = _manifest_for(backend_impl, config, filename)
    if manifest:
        logger.info(
            "Render outputs: "
            + ", ".join(f"{k}={v}" for k, v in manifest.items())
        )

    # Normalize the backend's primary return into a RenderResult. Mitsuba
    # returns an in-memory image; file backends (Blender/WebGL) return a path
    # or None. The main output path is the user's filename when one was given,
    # else any path the backend reported (e.g. WebGL's default output).
    image = raw if not isinstance(raw, (str, Path)) else None
    if filename is not None:
        path: Path | None = Path(filename)
    elif isinstance(raw, (str, Path)):
        path = Path(raw)
    else:
        path = None
    return RenderResult(
        backend=backend_name, outputs=manifest, image=image, path=path
    )


def _manifest_for(
    backend_impl: RenderBackend, config: Config, filename: Path | str | None
) -> dict[str, Path | str]:
    """Build the output manifest for a render given the resolved backend.

    Maps ``"main"`` and each *honored* render pass to its artifact: a
    :class:`~pathlib.Path` for file-writing backends, or the string
    ``"interactive"`` for backends whose passes live inside a viewer. Passes
    the backend cannot honor are omitted (the caller is warned separately).
    Empty when *filename* is ``None`` (nothing is written to disk).
    """
    if filename is None:
        return {}
    main = Path(filename)
    manifest: dict[str, Path | str] = {"main": main}
    supported = {p.name for p in backend_impl.SUPPORTED_PASSES}
    interactive = backend_impl.PASS_DELIVERY == "interactive"
    for name in sorted(config.render_passes & supported):
        manifest[name] = "interactive" if interactive else aov_path(main, name)
    return manifest


__all__ = [
    "render",
    "RenderResult",
    "set_default_backend",
    "list_backends",
]
