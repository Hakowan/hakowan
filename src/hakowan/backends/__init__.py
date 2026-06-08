"""Backend abstraction for different rendering engines."""

import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Callable
from ..compiler import Scene
from ..setup import Config
from ..setup.render_pass import RenderPass
from pathlib import Path
from typing import Any


class RenderBackend(ABC):
    """Abstract base class for rendering backends."""

    #: Render passes (AOVs) this backend can honor. Subclasses override with the
    #: subset of :mod:`hakowan.setup.render_pass` descriptors they implement. The
    #: render dispatcher warns about any requested pass not in this set, so a
    #: pass is never silently dropped. Empty by default (supports no passes).
    SUPPORTED_PASSES: frozenset[RenderPass] = frozenset()

    #: How render passes are delivered. ``"file"`` (default) writes one sidecar
    #: image per pass next to the main output; ``"interactive"`` exposes passes
    #: live inside a viewer (no per-pass files). Used to build the output
    #: manifest reported in :class:`~hakowan.render.RenderResult.outputs`.
    PASS_DELIVERY: str = "file"

    @abstractmethod
    def render(
        self,
        scene: Scene,
        config: Config,
        filename: Path | str | None = None,
        **kwargs,
    ) -> Any:
        """Render the scene and return the image.

        Args:
            scene: Compiled scene to render.
            config: Rendering configuration.
            filename: Optional output filename.
            **kwargs: Backend-specific options.

        Returns:
            Rendered image (format depends on backend).
        """
        pass


# Backend registry.
#
# Backends register a *lazy loader* plus the name of the module whose presence
# indicates availability. The loader (which imports the heavy backend module —
# and for Mitsuba, initializes Dr.Jit/LLVM) only runs when that backend is
# actually requested via ``get_backend``. This keeps ``import hakowan`` light
# and ensures using a non-Mitsuba backend never loads Mitsuba/LLVM.
_BackendLoader = Callable[[], type[RenderBackend]]
_backend_loaders: dict[str, tuple[_BackendLoader, str | None]] = {}
_backends: dict[str, type[RenderBackend]] = {}  # eager registrations + load cache

# The default backend. ``None`` means "auto": at render time, resolve to the
# first *available* backend in registration order. Backends register in
# preference order (see ``hakowan/__init__.py``: Mitsuba, then Blender, then the
# always-present WebGL), so this keeps the historical Mitsuba-first behavior when
# Mitsuba is installed and degrades gracefully when it is not — with no hardcoded
# list to keep in sync as backends are added or removed.
_default_backend: str | None = None


def _resolve_default() -> str:
    # Iterate in registration order. Lazy loaders first (the declared backends,
    # in declaration order), then any eagerly-registered extras. ``dict.fromkeys``
    # dedups while preserving order, so a lazily-loaded backend that has since
    # been cached into ``_backends`` does not jump the queue.
    for name in dict.fromkeys((*_backend_loaders, *_backends)):
        if _is_available(name):
            return name
    raise ValueError(
        "No rendering backend is available. "
        "Install at least one with e.g. 'pip install hakowan[webgl]', "
        "'pip install hakowan[mitsuba]', or 'pip install hakowan[blender]'."
    )


def register_backend(name: str, backend_class: type[RenderBackend]):
    """Register a rendering backend class directly (eager).

    Args:
        name: Backend name (e.g., 'mitsuba', 'blender').
        backend_class: Backend class implementing RenderBackend.
    """
    _backends[name] = backend_class


def register_backend_loader(
    name: str, loader: _BackendLoader, *, requires: str | None = None
):
    """Register a rendering backend behind a lazy loader.

    Args:
        name: Backend name.
        loader: Zero-arg callable that imports and returns the backend class.
        requires: Optional module name probed (without importing it) to decide
            whether the backend is available. ``None`` means always available.
    """
    _backend_loaders[name] = (loader, requires)


def _is_available(name: str) -> bool:
    if name in _backends:
        return True
    entry = _backend_loaders.get(name)
    if entry is None:
        return False
    _, requires = entry
    if requires is None:
        return True
    try:
        return importlib.util.find_spec(requires) is not None
    except (ImportError, ValueError):
        return False


def _resolve_class(name: str) -> type[RenderBackend]:
    if name in _backends:
        return _backends[name]
    loader, _ = _backend_loaders[name]
    backend_class = loader()
    _backends[name] = backend_class  # cache so the import happens at most once
    return backend_class


def set_default_backend(name: str):
    """Set the default rendering backend.

    Args:
        name: Backend name.

    Raises:
        ValueError: If backend is not registered.
    """
    global _default_backend
    if name not in _backends and name not in _backend_loaders:
        raise ValueError(f"Unknown backend: {name}. Available: {list_backends()}")
    _default_backend = name


def resolve_backend_name(name: str | None = None) -> str:
    """Resolve the effective backend name.

    Applies the same resolution as :func:`get_backend` (explicit name, else the
    configured default, else the first available backend) without importing the
    backend module. Useful for backend-name-dependent decisions before render.

    Args:
        name: Backend name. If None, uses the default / auto-resolved backend.

    Returns:
        The resolved backend name.
    """
    return name or _default_backend or _resolve_default()


def get_backend(name: str | None = None) -> RenderBackend:
    """Get a rendering backend instance, importing it on first use.

    Args:
        name: Backend name. If None, uses default backend.

    Returns:
        Backend instance.

    Raises:
        ValueError: If the backend is unknown or its dependencies are missing.
    """
    backend_name = resolve_backend_name(name)
    if backend_name not in _backends and backend_name not in _backend_loaders:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available: {list_backends()}"
        )
    try:
        backend_class = _resolve_class(backend_name)
    except ImportError as e:
        raise ValueError(
            f"Backend '{backend_name}' is unavailable (missing dependency): {e}"
        ) from e
    return backend_class()


def list_backends() -> list[str]:
    """List the rendering backends whose dependencies are installed.

    Availability is probed without importing the heavy backend modules.
    """
    names = set(_backends) | set(_backend_loaders)
    return sorted(n for n in names if _is_available(n))


__all__ = [
    "RenderBackend",
    "register_backend",
    "register_backend_loader",
    "set_default_backend",
    "get_backend",
    "resolve_backend_name",
    "list_backends",
]
