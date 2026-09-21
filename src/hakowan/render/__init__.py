"""Rendering module - provides unified render interface."""

from ..backends import (
    BackendName,
    RenderBackend,
    get_backend,
    resolve_backend_name,
    set_default_backend,
    list_backends,
)
from ..compiler import compile
from ..grammar import layer
from ..grammar.figure import Figure
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

    backend: BackendName
    outputs: dict[str, Path | str] = field(default_factory=dict)
    image: Any = field(default=None, repr=False)
    path: Path | None = None

    def __fspath__(self) -> str:
        """Return the main output path for os.PathLike consumers.

        Raises:
            TypeError: If rendering produced no file path.

        """
        if self.path is None:
            raise TypeError(
                "RenderResult has no output path "
                "(render() was called without a filename)."
            )
        return str(self.path)


def render(
    root: layer.Layer | Figure,
    config: Config | None = None,
    filename: Path | str | None = None,
    backend: BackendName | None = None,
    **kwargs: Any,
) -> RenderResult:
    """Compile and render a Layer or Figure with the selected backend.

    Figure scene settings supply camera, lighting, environment, and output
    intent only when ``config`` is omitted. An explicit Config overrides the
    complete Figure-derived Config; backend keyword arguments override their
    corresponding backend-facing settings.

    Args:
        root: Layer tree or declarative Figure to render.
        config: Explicit invocation configuration, or Figure/default settings.
        filename: Main output path; WebGL rewrites non-HTML suffixes to HTML.
        backend: ``webgl``, ``mitsuba``, ``blender``, or the configured default.
        **kwargs: Backend-specific options such as WebGL ``background`` and
            ``offline``, Mitsuba ``yaml_file``, or Blender ``blender_engine``.

    Returns:
        RenderResult with the backend, primary path/image, and output manifest.

    Raises:
        TypeError: If backend-specific keyword arguments are unknown.

    Examples:
        >>> import hakowan as hkw
        >>> result = hkw.render(hkw.layer("mesh.obj"), filename="viewer.html")
        >>> result.path
        PosixPath('viewer.html')

    """
    runtime_layer = root.layer if isinstance(root, Figure) else root
    use_figure_settings = isinstance(root, Figure) and config is None
    if isinstance(root, Figure) and config is None:
        config = root.to_config()
    elif config is None:
        config = Config()

    # Compile the layer tree after resolving the figure wrapper.
    scene = compile(runtime_layer)
    logger.info("Compilation done")
    # Get backend and render
    backend_name = resolve_backend_name(backend)
    if use_figure_settings:
        assert isinstance(root, Figure)
        output = root.scene.output
        environment = root.scene.environment
        if output is not None and backend_name == "webgl":
            kwargs.setdefault("background", output.background)
        if environment is not None:
            if backend_name == "webgl":
                kwargs.setdefault("envmap_background", environment.visible)
            elif backend_name == "blender":
                kwargs.setdefault("environment_visible", environment.visible)
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

    # Normalize the backend's primary return into a RenderResult. Mitsuba
    # returns an in-memory image; file backends (Blender/WebGL) return the path
    # they actually wrote, or None. Prefer the backend-reported path — the
    # WebGL backend rewrites the suffix to ``.html``, so its returned path is
    # the truthful main output, not the user's *filename*.
    image = raw if not isinstance(raw, (str, Path)) else None
    if isinstance(raw, (str, Path)):
        path: Path | None = Path(raw)
    elif filename is not None:
        path = Path(filename)
    else:
        path = None

    # Surface the produced artifacts so the user does not have to guess which
    # sidecar files appear (or that passes are live in the viewer instead).
    manifest = _manifest_for(backend_impl, config, path)
    if manifest:
        logger.info(
            "Render outputs: " + ", ".join(f"{k}={v}" for k, v in manifest.items())
        )

    return RenderResult(backend=backend_name, outputs=manifest, image=image, path=path)


def _manifest_for(
    backend_impl: RenderBackend, config: Config, main_path: Path | str | None
) -> dict[str, Path | str]:
    """Build the output manifest for a render given the resolved backend.

    Maps ``"main"`` and each *honored* render pass to its artifact: a
    :class:`~pathlib.Path` for file-writing backends, or the string
    ``"interactive"`` for backends whose passes live inside a viewer. Passes
    the backend cannot honor are omitted (the caller is warned separately).
    Empty when *main_path* is ``None`` (nothing is written to disk).

    *main_path* is the *actual* primary output (the backend's reported path,
    not necessarily the user's requested filename), so sidecar paths are
    derived from the format that was really written.
    """
    if main_path is None:
        return {}
    main = Path(main_path)
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
