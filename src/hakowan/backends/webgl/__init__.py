"""WebGL/three.js rendering backend.

Translates a compiled hakowan ``Scene`` into a self-contained HTML file
containing an interactive three.js viewer with the geometry embedded as a
glTF 2.0 (GLB) data URI.
"""

from .render import WebGLBackend

__all__ = ["WebGLBackend"]
