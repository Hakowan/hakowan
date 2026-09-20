"""Download-once Three.js assets for offline WebGL viewer bundles."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


_ASSETS = {
    "build/three.module.js": "build/three.module.js",
    "examples/jsm/controls/OrbitControls.js": "examples/jsm/controls/OrbitControls.js",
    "examples/jsm/loaders/GLTFLoader.js": "examples/jsm/loaders/GLTFLoader.js",
    "examples/jsm/utils/BufferGeometryUtils.js": "examples/jsm/utils/BufferGeometryUtils.js",
}


class WebGLAssetError(RuntimeError):
    """Raised when offline Three.js assets cannot be prepared."""


def _cache_root(version: str) -> Path:
    configured = os.environ.get("HAKOWAN_CACHE_DIR")
    base = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "hakowan"
    )
    return base / "three" / version


def ensure_three_assets(version: str) -> Path:
    """Return a complete local Three.js module tree, downloading missing files once."""
    root = _cache_root(version)
    for relative, remote in _ASSETS.items():
        target = root / relative
        if target.is_file() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://unpkg.com/three@{version}/{remote}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
        except (OSError, urllib.error.URLError) as exc:
            raise WebGLAssetError(
                f"Cannot prepare offline Three.js asset {url}: {exc}. "
                "Generate the offline bundle once while network access is available."
            ) from exc
        if not payload:
            raise WebGLAssetError(f"Downloaded empty Three.js asset: {url}")
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            output.write(payload)
            temporary = Path(output.name)
        temporary.replace(target)
    return root


def copy_three_assets(version: str, output: str | Path) -> Path:
    """Copy the cached module tree into an offline viewer directory."""
    source = ensure_three_assets(version)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for relative in _ASSETS:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    return destination


__all__ = ["WebGLAssetError", "copy_three_assets", "ensure_three_assets"]
