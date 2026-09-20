"""Backend abstraction for different rendering engines."""

import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Callable
from ..compiler import Scene
from ..setup import Config
from ..setup.render_pass import RenderPass
from pathlib import Path
from typing import Any, Literal
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Machine-readable rendering capabilities available without loading a backend."""

    name: str
    marks: frozenset[str]
    render_passes: frozenset[str]
    pass_delivery: Literal["file", "interactive"]
    features: frozenset[str] = frozenset()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe capability description."""
        result = asdict(self)
        result["marks"] = sorted(self.marks)
        result["render_passes"] = sorted(self.render_passes)
        result["features"] = sorted(self.features)
        result["limitations"] = list(self.limitations)
        return result


_COMMON_MARKS = frozenset({"point", "curve", "surface"})

WEBGL_CAPABILITIES = BackendCapabilities(
    name="webgl",
    marks=_COMMON_MARKS,
    render_passes=frozenset({"albedo", "depth", "normal"}),
    pass_delivery="interactive",
    features=frozenset(
        {
            "interactive",
            "layer_visibility",
            "interactive_clip",
            "image_texture",
            "image_bump_map",
            "image_normal_map",
        }
    ),
    limitations=(
        "Hair shading is approximated; melanin, gradients, and data-driven hair color are unsupported.",
        "Fur child hairs are ignored.",
        "Thin-lens depth of field is rendered as perspective.",
    ),
)

MITSUBA_CAPABILITIES = BackendCapabilities(
    name="mitsuba",
    marks=_COMMON_MARKS,
    render_passes=frozenset({"albedo", "depth", "normal"}),
    pass_delivery="file",
    features=frozenset(
        {"in_memory_image", "image_texture", "bump_map", "normal_map", "hair"}
    ),
    limitations=(
        "Hair root/tip gradients collapse to one average color.",
        "Data-driven hair color and fur child hairs are unsupported.",
    ),
)

BLENDER_CAPABILITIES = BackendCapabilities(
    name="blender",
    marks=_COMMON_MARKS,
    render_passes=frozenset({"albedo", "depth", "normal", "facet_id"}),
    pass_delivery="file",
    features=frozenset(
        {
            "image_texture",
            "image_bump_map",
            "image_normal_map",
            "hair",
            "hair_gradient",
            "fur_children",
            "thin_lens",
        }
    ),
    limitations=(
        "Data-driven hair color is unsupported.",
        "Textured back-face materials fall back to a uniform color.",
    ),
)

BackendName = Literal["webgl", "mitsuba", "blender"]


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
    PASS_DELIVERY: Literal["file", "interactive"] = "file"

    @abstractmethod
    def render(
        self,
        scene: Scene,
        config: Config,
        filename: Path | str | None = None,
        **kwargs,
    ) -> Any:
        """Render the scene and return the backend's primary artifact.

        Returns whatever the backend produces natively: an in-memory image
        (Mitsuba), an output path (WebGL), or ``None`` (Blender). The public
        :func:`hakowan.render` wraps this into a
        :class:`~hakowan.render.RenderResult` together with the output
        manifest, so end users see a uniform return type.

        Args:
            scene: Compiled scene to render.
            config: Rendering configuration.
            filename: Optional output filename.
            **kwargs: Backend-specific options.

        Returns:
            The backend's primary artifact (image, path, or ``None``).
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
_backend_capabilities: dict[str, BackendCapabilities] = {
    item.name: item
    for item in (WEBGL_CAPABILITIES, MITSUBA_CAPABILITIES, BLENDER_CAPABILITIES)
}
_backends: dict[str, type[RenderBackend]] = {}  # eager registrations + load cache

# The default backend. WebGL is the default because its dependency (pygltflib)
# ships with the base install, so it is always available — the heavier Mitsuba
# and Blender backends must be requested explicitly (per render via the
# ``backend=`` argument, or process-wide via :func:`set_default_backend`).
# ``None`` means "auto": resolve at render time to the first *available* backend
# in registration order. ``_resolve_default`` is only reached if the default is
# cleared; it degrades gracefully with no hardcoded list to keep in sync.
_default_backend: str | None = "webgl"


def _resolve_default() -> str:
    # Iterate in registration order. Lazy loaders first (the declared backends,
    # in declaration order), then any eagerly-registered extras. ``dict.fromkeys``
    # dedups while preserving order, so a lazily-loaded backend that has since
    # been cached into ``_backends`` does not jump the queue.
    for name in dict.fromkeys((*_backend_loaders, *_backends)):
        if _is_available(name):
            return name
    raise ValueError(
        "No rendering backend is available. The WebGL backend ships by default; "
        "reinstall hakowan, or add a heavier backend with "
        "'pip install hakowan[mitsuba]' or 'pip install hakowan[blender]'."
    )


def register_backend(
    name: str,
    backend_class: type[RenderBackend],
    *,
    capabilities: BackendCapabilities | None = None,
):
    """Register a rendering backend class directly (eager)."""
    _backends[name] = backend_class
    if capabilities is not None:
        if capabilities.name != name:
            raise ValueError("Backend capability name must match its registry name.")
        _backend_capabilities[name] = capabilities


def register_backend_loader(
    name: str,
    loader: _BackendLoader,
    *,
    requires: str | None = None,
    capabilities: BackendCapabilities | None = None,
):
    """Register a backend behind a lazy loader and optional capability record."""
    _backend_loaders[name] = (loader, requires)
    if capabilities is not None:
        if capabilities.name != name:
            raise ValueError("Backend capability name must match its registry name.")
        _backend_capabilities[name] = capabilities


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


def set_default_backend(name: BackendName):
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


def resolve_backend_name(name: BackendName | None = None) -> BackendName:
    """Resolve the effective backend name.

    Applies the same resolution as :func:`get_backend` (explicit name, else the
    configured default, else the first available backend) without importing the
    backend module. Useful for backend-name-dependent decisions before render.

    Args:
        name: Backend name. If None, uses the default / auto-resolved backend.

    Returns:
        The resolved backend name.
    """
    return name or _default_backend or _resolve_default()  # type: ignore[return-value]


def get_backend(name: BackendName | None = None) -> RenderBackend:
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


def get_backend_capabilities(name: BackendName | None = None) -> BackendCapabilities:
    """Return declared capabilities without importing the backend implementation."""
    backend_name = resolve_backend_name(name)
    try:
        return _backend_capabilities[backend_name]
    except KeyError as exc:
        raise ValueError(f"Backend '{backend_name}' has no capability declaration.") from exc


def list_backend_capabilities() -> dict[str, BackendCapabilities]:
    """Return all declared backend capabilities, including optional backends."""
    return dict(sorted(_backend_capabilities.items()))


__all__ = [
    "BackendName",
    "BackendCapabilities",
    "WEBGL_CAPABILITIES",
    "MITSUBA_CAPABILITIES",
    "BLENDER_CAPABILITIES",
    "RenderBackend",
    "register_backend",
    "register_backend_loader",
    "set_default_backend",
    "get_backend",
    "get_backend_capabilities",
    "list_backend_capabilities",
    "resolve_backend_name",
    "list_backends",
]
