"""Download-once Three.js assets for offline WebGL viewer bundles."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


_ASSET_MANIFEST = {
    "0.170.0": {
        "build/three.module.js": "ce1fa418de16a19495a9f72495580e3015d7745c296d3ce0485897f902ddedfb",
        "examples/jsm/controls/OrbitControls.js": "80efaadea4f8a636a65fb0bd08bfef62f3d93a0bb94e2e7500f23176c5c07f4e",
        "examples/jsm/loaders/GLTFLoader.js": "45139faddd5aaf48ed2d62203d976e5cbd703db1a592de40527f9f6cf58abd44",
        "examples/jsm/loaders/EXRLoader.js": "9ab29edbe812c6f48fd70910643d15433d5efcd0e4e2a2ca24b2bcb9a7eefe6b",
        "examples/jsm/loaders/RGBELoader.js": "f0e87d0008d9484d31358b32befd1bf80e4301f77573cc9a7cf7d871cc3f64b4",
        "examples/jsm/libs/fflate.module.js": "209a4412eb48ce609edb4391992a792ffcc3983d30ee7e2b0b89a8c470f3cd8a",
        "examples/jsm/utils/BufferGeometryUtils.js": "c25b7930e570e9ec56173cd3b866ec8d2e10016630db3937efb439daf1cedbf6",
    }
}
_MAX_ASSET_BYTES = 8 * 1024 * 1024


class WebGLAssetError(RuntimeError):
    """Raised when offline Three.js assets cannot be prepared."""


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cache_root(version: str) -> Path:
    configured = os.environ.get("HAKOWAN_CACHE_DIR")
    base = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "hakowan"
    )
    return base / "three" / version


def ensure_three_assets(version: str) -> Path:
    """Return a complete, integrity-checked local Three.js module tree."""
    manifest = _ASSET_MANIFEST.get(version)
    if manifest is None:
        raise WebGLAssetError(
            f"Offline bundles support Three.js versions {sorted(_ASSET_MANIFEST)}; "
            f"got {version!r}."
        )
    root = _cache_root(version)
    for relative, expected_digest in manifest.items():
        target = root / relative
        if target.is_file() and _file_sha256(target) == expected_digest:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://unpkg.com/three@{version}/{relative}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read(_MAX_ASSET_BYTES + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise WebGLAssetError(
                f"Cannot prepare offline Three.js asset {url}: {exc}. "
                "Generate the offline bundle once while network access is available."
            ) from exc
        if len(payload) > _MAX_ASSET_BYTES:
            raise WebGLAssetError(f"Downloaded Three.js asset is too large: {url}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_digest:
            raise WebGLAssetError(
                f"Downloaded Three.js asset failed integrity verification: {url}"
            )
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
    for relative in _ASSET_MANIFEST[version]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    return destination


__all__ = ["WebGLAssetError", "copy_three_assets", "ensure_three_assets"]
