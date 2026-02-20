"""Backend abstraction for different rendering engines."""

from abc import ABC, abstractmethod
from ..compiler import Scene
from ..setup import Config
from ..grammar import layer
from pathlib import Path
from typing import Any


class RenderBackend(ABC):
    """Abstract base class for rendering backends."""

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


# Backend registry
_backends: dict[str, type[RenderBackend]] = {}
_default_backend = "mitsuba"


def register_backend(name: str, backend_class: type[RenderBackend]):
    """Register a rendering backend.

    Args:
        name: Backend name (e.g., 'mitsuba', 'blender').
        backend_class: Backend class implementing RenderBackend.
    """
    _backends[name] = backend_class


def set_default_backend(name: str):
    """Set the default rendering backend.

    Args:
        name: Backend name.

    Raises:
        ValueError: If backend is not registered.
    """
    global _default_backend
    if name not in _backends:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(_backends.keys())}"
        )
    _default_backend = name


def get_backend(name: str | None = None) -> RenderBackend:
    """Get a rendering backend instance.

    Args:
        name: Backend name. If None, uses default backend.

    Returns:
        Backend instance.

    Raises:
        ValueError: If backend is not registered.
    """
    backend_name = name or _default_backend
    if backend_name not in _backends:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available: {list(_backends.keys())}"
        )
    return _backends[backend_name]()


def list_backends() -> list[str]:
    """List all registered backends.

    Returns:
        List of backend names.
    """
    return list(_backends.keys())


__all__ = [
    "RenderBackend",
    "register_backend",
    "set_default_backend",
    "get_backend",
    "list_backends",
]
