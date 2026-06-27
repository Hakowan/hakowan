"""Shared image I/O for the rendering backends.

Pillow is the single low-dynamic-range image encoder, so every backend writes
the same set of formats (PNG, JPEG, WebP, BMP, TIFF, GIF, ...).  High-dynamic-
range formats (currently ``.exr``) carry float data and bypass Pillow; each
backend writes those with its own float-capable encoder (Mitsuba's
``write_bitmap``, Blender's ``OPEN_EXR``).

Backends call :func:`check_supported_suffix` to reject unknown extensions with a
clear error *before* rendering, and either :func:`save_array` (write an in-memory
8-bit buffer) or :func:`convert_image` (re-encode a file they rendered natively
as PNG).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from . import logger

# HDR formats carry float data and never go through Pillow's 8-bit path.
HDR_SUFFIXES = frozenset({".exr"})

# Pillow formats that cannot store an alpha channel.  RGBA/LA input is
# flattened to RGB before saving so the write does not fail.
_NO_ALPHA_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".bmp", ".dib", ".ppm", ".pgm", ".pbm", ".pcx"}
)


@lru_cache(maxsize=1)
def _pillow_writable_suffixes() -> frozenset[str]:
    """File suffixes Pillow can *write* in this environment (lowercased).

    ``Image.registered_extensions()`` maps ``".ext" -> "FORMAT"``; a format is
    writable only when it also appears in ``Image.SAVE``.
    """
    Image.init()  # ensure all plugins (and their extensions) are registered
    return frozenset(
        ext.lower()
        for ext, fmt in Image.registered_extensions().items()
        if fmt in Image.SAVE
    )


def supported_suffixes() -> frozenset[str]:
    """All writable image suffixes: Pillow's LDR set plus the HDR formats."""
    return _pillow_writable_suffixes() | HDR_SUFFIXES


def is_hdr_suffix(suffix: str) -> bool:
    """Whether *suffix* names a high-dynamic-range format written natively."""
    return suffix.lower() in HDR_SUFFIXES


def is_supported_suffix(suffix: str) -> bool:
    """Whether *suffix* is a writable image format (Pillow LDR or HDR)."""
    return suffix.lower() in supported_suffixes()


def check_supported_suffix(filename: Path) -> str:
    """Validate *filename*'s extension and return it lowercased.

    Raises:
        ValueError: when the suffix is missing or not writable in this
            environment, listing the formats that are.
    """
    suffix = filename.suffix.lower()
    if not is_supported_suffix(suffix):
        known = ", ".join(sorted(supported_suffixes()))
        raise ValueError(
            f"Unsupported output image format {suffix or '(no extension)'!r} "
            f"for '{filename}'. Supported formats: {known}."
        )
    return suffix


def _save_options(suffix: str) -> dict:
    """High-fidelity Pillow save options per format (renders, not photos)."""
    if suffix in (".jpg", ".jpeg"):
        return {"quality": 95, "subsampling": 0}
    if suffix == ".webp":
        # method=6 is the slowest/highest-quality encode; quality=95 keeps the
        # lossy artefacts negligible for synthetic renders.
        return {"quality": 95, "method": 6}
    return {}


def _prepare_for_suffix(image: Image.Image, suffix: str) -> Image.Image:
    """Coerce *image* into a mode the target format can encode."""
    if suffix in _NO_ALPHA_SUFFIXES:
        if image.mode not in ("RGB", "L"):
            return image.convert("RGB")
        return image
    if suffix == ".webp":
        # WebP only encodes RGB/RGBA; grayscale/palette must be promoted.
        if image.mode not in ("RGB", "RGBA"):
            return image.convert("RGBA" if "A" in image.mode else "RGB")
        return image
    return image


def _write(image: Image.Image, filename: Path) -> None:
    """Encode *image* to *filename* via Pillow, validating the suffix first."""
    suffix = check_supported_suffix(filename)
    if suffix in HDR_SUFFIXES:
        raise ValueError(
            f"'{suffix}' is a high-dynamic-range format and must be written by "
            "the backend's float encoder, not Pillow."
        )
    image = _prepare_for_suffix(image, suffix)
    filename.parent.mkdir(parents=True, exist_ok=True)
    image.save(filename, **_save_options(suffix))
    logger.debug(f"Image written to {filename}")


def save_array(array: np.ndarray, filename: Path) -> None:
    """Save an 8-bit ``H×W×{1,3,4}`` (or ``H×W``) *array* to *filename*.

    The output format is inferred from *filename*'s suffix.  Formats without an
    alpha channel (e.g. JPEG) drop it; WebP promotes grayscale to RGB.

    Raises:
        ValueError: if the suffix is not a Pillow-writable format.
    """
    array = np.ascontiguousarray(array)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    _write(Image.fromarray(array), filename)


def convert_image(src: Path, dst: Path) -> None:
    """Re-encode the image file *src* into *dst*'s format via Pillow.

    Used by backends (Blender) that render natively as PNG and then convert to
    the user's requested format.  Skip the call when ``src == dst``.

    Raises:
        ValueError: if *dst*'s suffix is not a Pillow-writable format.
    """
    with Image.open(src) as im:
        im.load()
        _write(im, dst)


__all__ = [
    "HDR_SUFFIXES",
    "supported_suffixes",
    "is_hdr_suffix",
    "is_supported_suffix",
    "check_supported_suffix",
    "save_array",
    "convert_image",
]
