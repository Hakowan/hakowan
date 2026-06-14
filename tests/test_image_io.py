"""Unit tests for the shared Pillow-based image I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hakowan.common import image_io as iio


class TestSuffixSupport:
    def test_common_formats_supported(self):
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"):
            assert iio.is_supported_suffix(ext), ext

    def test_exr_is_hdr(self):
        assert iio.is_hdr_suffix(".exr")
        assert ".exr" in iio.supported_suffixes()
        assert not iio.is_hdr_suffix(".png")

    def test_case_insensitive(self):
        assert iio.is_supported_suffix(".PNG")
        assert iio.is_hdr_suffix(".EXR")

    def test_check_returns_lowercased_suffix(self):
        assert iio.check_supported_suffix(Path("a/b/c.WEBP")) == ".webp"

    def test_unsupported_suffix_raises(self):
        with pytest.raises(ValueError, match="Unsupported output image format"):
            iio.check_supported_suffix(Path("out.xyz"))

    def test_missing_suffix_raises(self):
        with pytest.raises(ValueError, match="no extension"):
            iio.check_supported_suffix(Path("out"))


class TestSaveArray:
    @pytest.fixture
    def rgba(self):
        rng = np.random.default_rng(0)
        return (rng.random((8, 8, 4)) * 255).astype(np.uint8)

    @pytest.mark.parametrize("ext", [".png", ".jpg", ".webp", ".bmp", ".tif", ".gif"])
    def test_roundtrip_format(self, tmp_path, rgba, ext):
        out = tmp_path / f"img{ext}"
        iio.save_array(rgba, out)
        assert out.exists() and out.stat().st_size > 0
        with Image.open(out) as im:
            assert im.size == (8, 8)

    def test_alpha_dropped_for_jpeg(self, tmp_path, rgba):
        out = tmp_path / "img.jpg"
        iio.save_array(rgba, out)
        with Image.open(out) as im:
            assert im.mode == "RGB"  # JPEG cannot hold alpha

    def test_grayscale_to_webp(self, tmp_path):
        gray = (np.random.default_rng(1).random((8, 8)) * 255).astype(np.uint8)
        out = tmp_path / "g.webp"
        iio.save_array(gray, out)  # WebP needs RGB; must be promoted, not crash
        assert out.exists() and out.stat().st_size > 0

    def test_single_channel_3d_squeezed(self, tmp_path):
        arr = (np.random.default_rng(2).random((8, 8, 1)) * 255).astype(np.uint8)
        out = tmp_path / "s.png"
        iio.save_array(arr, out)
        with Image.open(out) as im:
            assert im.size == (8, 8)

    def test_unsupported_suffix_raises(self, tmp_path, rgba):
        with pytest.raises(ValueError, match="Unsupported output image format"):
            iio.save_array(rgba, tmp_path / "x.xyz")

    def test_hdr_suffix_rejected(self, tmp_path, rgba):
        # Pillow must not be asked to write HDR; backends route .exr elsewhere.
        with pytest.raises(ValueError, match="high-dynamic-range"):
            iio.save_array(rgba, tmp_path / "x.exr")


class TestConvertImage:
    def test_png_to_webp_and_jpg(self, tmp_path):
        rgba = (np.random.default_rng(3).random((8, 8, 4)) * 255).astype(np.uint8)
        src = tmp_path / "src.png"
        iio.save_array(rgba, src)
        for ext, expected in ((".webp", "WEBP"), (".jpg", "JPEG")):
            dst = tmp_path / f"dst{ext}"
            iio.convert_image(src, dst)
            with Image.open(dst) as im:
                assert im.format == expected
        assert src.exists()  # convert does not delete the source
