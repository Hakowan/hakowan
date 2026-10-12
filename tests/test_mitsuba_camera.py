"""Mitsuba camera coverage across square and non-square films."""

import os
import sys

import numpy as np
import pytest

if sys.platform == "win32" and os.environ.get("CI") == "true":
    pytest.skip(
        "mitsuba/Dr.Jit causes process crash on Windows CI during Python shutdown",
        allow_module_level=True,
    )

mi = pytest.importorskip("mitsuba", reason="mitsuba not installed")

from hakowan.backends.mitsuba.render import _apply_film_crop, generate_base_config
from hakowan.setup import Config
from hakowan.setup.sensor import Orthographic


@pytest.mark.parametrize("width,height", [(1024, 800), (800, 1024), (800, 800)])
def test_orthographic_rays_cover_full_vertical_scale(width, height):
    mi.set_variant("scalar_rgb")
    config = Config()
    config.sensor = Orthographic(scale=4.0)
    config.film.width = width
    config.film.height = height
    sensor = mi.load_dict(generate_base_config(config)["camera"])

    def origin(x, y):
        ray, _ = sensor.sample_ray(0.0, 0.0, mi.Point2f(x, y), mi.Point2f(0.5, 0.5))
        return np.asarray(ray.o)

    center = origin(0.5, 0.5)
    left_top = origin(0.0, 0.0)
    right_bottom = origin(1.0, 1.0)
    np.testing.assert_allclose(center[:2], [0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(left_top[2], center[2], atol=1e-5)
    np.testing.assert_allclose(right_bottom[2], center[2], atol=1e-5)
    # WebGL uses ymag=scale/2 and xmag=ymag*width/height. Check
    # both world-space spans, catching aspect inversion and swapped axes.
    np.testing.assert_allclose(
        np.abs(left_top[:2] - center[:2]),
        [2.0 * width / height, 2.0],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        right_bottom[:2] - center[:2], center[:2] - left_top[:2], atol=1e-5
    )


def test_cropped_sensor_rays_match_full_film_crop_coordinates():
    mi.set_variant("scalar_rgb")
    config = Config()
    config.sensor = Orthographic(scale=4.0)
    config.film.width, config.film.height = 240, 160
    reference = mi.load_dict(generate_base_config(config)["camera"])
    cropped = mi.load_dict(generate_base_config(config)["camera"])

    crop_offset = (20, 10)
    crop_size = (160, 100)
    # Exercise the exact late-crop path used by the backend, including the
    # Mitsuba 3.9 camera projection refresh after film parameters are updated.
    _apply_film_crop(cropped, crop_offset, crop_size)

    for u, v in ((0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (1.0, 1.0)):
        full_u = (crop_offset[0] + u * crop_size[0]) / config.film.width
        full_v = (crop_offset[1] + v * crop_size[1]) / config.film.height
        cropped_ray, _ = cropped.sample_ray(
            0.0, 0.0, mi.Point2f(u, v), mi.Point2f(0.5, 0.5)
        )
        reference_ray, _ = reference.sample_ray(
            0.0, 0.0, mi.Point2f(full_u, full_v), mi.Point2f(0.5, 0.5)
        )
        np.testing.assert_allclose(
            np.asarray(cropped_ray.o), np.asarray(reference_ray.o), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(cropped_ray.d), np.asarray(reference_ray.d), atol=1e-5
        )
