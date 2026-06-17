from .emitter import generate_emitter_config
from .film import generate_film_config
from .integrator import generate_integrator_config
from .sampler import generate_sampler_config
from .sensor import generate_sensor_config
from .shape import generate_point_config, generate_curve_config, generate_surface_config

from ...common import logger
from ...common.image_io import check_supported_suffix, is_hdr_suffix, save_array
from ...compiler import Scene, View
from ...setup import Config
from ...setup.render_pass import ALBEDO, DEPTH, NORMAL, aov_path
from ...grammar import mark
from .. import RenderBackend

import datetime
import mitsuba as mi
import drjit
from typing import Any
from pathlib import Path
import numpy as np
import yaml


def _camera_axis_cosine(shape: tuple[int, ...], sensor: Any) -> np.ndarray:
    """Per-pixel cos(theta) between each camera ray and the optical axis.

    Multiplying Mitsuba's ray-distance ``depth`` AOV by this converts it to
    planar Z-depth (distance measured along the optical axis), which matches
    Blender's Z pass: a plane facing the camera then reads a constant depth
    instead of bowing outward.

    Args:
        shape: ``(height, width, ...)`` of the depth array.
        sensor: The configured sensor (uses ``fov``/``fov_axis`` for perspective
            cameras; orthographic cameras have no field of view).

    Returns:
        ``(height, width)`` float32 array of cos(theta), all ones for cameras
        without a field of view.
    """
    h, w = int(shape[0]), int(shape[1])
    fov = getattr(sensor, "fov", None)
    if fov is None:
        return np.ones((h, w), dtype=np.float32)

    half_tan = float(np.tan(np.radians(float(fov)) * 0.5))
    fov_axis = getattr(sensor, "fov_axis", "smaller")
    ref = {
        "x": w,
        "y": h,
        "larger": max(w, h),
        "diagonal": float(np.hypot(w, h)),
    }.get(fov_axis, min(w, h))  # default: "smaller" (Mitsuba's default)

    # Pixel-center tangent offset from the optical axis; the fov axis spans
    # [-half_tan, half_tan], other axes scale with the same per-pixel step.
    step = 2.0 * half_tan / ref
    xs = ((np.arange(w, dtype=np.float64) + 0.5) - w * 0.5) * step
    ys = ((np.arange(h, dtype=np.float64) + 0.5) - h * 0.5) * step
    cos = 1.0 / np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2 + 1.0)
    return cos.astype(np.float32)


def generate_base_config(config: Config) -> dict:
    """Generate a Mitsuba base config dict from a Config."""
    sensor_config = generate_sensor_config(config.sensor)
    sensor_config["film"] = generate_film_config(config.film)
    sensor_config["sampler"] = generate_sampler_config(config.sampler)
    integrator_config = generate_integrator_config(config.integrator)

    mi_config = {
        "type": "scene",
        "camera": sensor_config,
        "integrator": integrator_config,
    }

    for i, emitter in enumerate(config.emitters):
        mi_config[f"emitter_{i:03}"] = generate_emitter_config(config.emitters[i])

    return mi_config


def generate_view_config(view: View, stamp: str, index: int) -> dict:
    """Generate a Mitsuba shape description dict from a View."""

    # Generate shape.
    match view.mark:
        case mark.Point:
            mi_config = generate_point_config(view, stamp, index)
        case mark.Curve:
            mi_config = generate_curve_config(view, stamp, index)
        case mark.Surface:
            mi_config = generate_surface_config(view, stamp, index)

    return mi_config


def generate_scene_config(scene: Scene) -> dict:
    """Generate a mitsuba scene description dict from a Scene."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scene_config: dict[str, Any] = {}
    for i, view in enumerate(scene):
        scene_config |= generate_view_config(view, stamp, i)
    return scene_config


def _mi_config_to_serializable(obj: Any) -> Any:
    """Recursively convert mi_config to YAML-serializable types (e.g. convert mi.ScalarTransform4f)."""
    if isinstance(obj, dict):
        return {k: _mi_config_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_mi_config_to_serializable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Mitsuba ScalarTransform4f and similar: convert 4x4 matrix to list of lists
    if isinstance(obj, mi.ScalarTransform4f):
        m = getattr(obj, "matrix", None)
        assert m is not None
        return m.numpy().tolist()

    try:
        arr = np.array(obj)
        if arr.ndim <= 2 and arr.size <= 16:
            return arr.tolist()
    except (TypeError, ValueError):
        pass
    return obj


def save_image(image: drjit.ArrayBase, filename: Path, srgb_gamma: bool = True) -> None:
    """Write an image to disk.

    HDR formats (``.exr``) are written by Mitsuba's float encoder; every other
    format is converted to 8-bit RGBA and encoded by Pillow, so any Pillow-
    writable format (PNG, JPEG, WebP, BMP, TIFF, ...) is supported.

    ``srgb_gamma`` controls whether the 8-bit conversion applies the sRGB
    transfer function. Use ``True`` for radiance/colour output (beauty, albedo)
    and ``False`` for data passes (e.g. packed normals) that must stay linear.
    """
    suffix = check_supported_suffix(filename)  # clear error for unknown formats
    if is_hdr_suffix(suffix):
        mi.util.write_bitmap(str(filename), image)  # type: ignore
    else:
        bitmap = mi.Bitmap(image).convert(  # type: ignore
            pixel_format=mi.Bitmap.PixelFormat.RGBA,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=srgb_gamma,
        )
        save_array(np.array(bitmap), filename)
    logger.info(f"Rendering saved to {filename}")


def ensure_variant() -> None:
    """Select a Mitsuba variant if none is active yet.

    Lives in the Mitsuba backend (rather than at ``import hakowan`` time) so the
    Mitsuba runtime — and its LLVM backend — is only initialized when the
    Mitsuba backend is actually used. Non-Mitsuba backends (Blender/WebGL) never
    trigger this.
    """
    if mi.variant() is not None:
        return
    for variant in ["scalar_rgb", "cuda_ad_rgb", "llvm_ad_rgb"]:
        if variant in mi.variants():
            try:
                mi.set_variant(variant)
                break
            except Exception:
                pass
    if mi.variant() is None:
        logger.warning("Could not initialize any Mitsuba variant")


class MitsubaBackend(RenderBackend):
    """Mitsuba rendering backend."""

    # facet_id has no Mitsuba AOV counterpart; the other passes ride the AOV
    # integrator (see Config.__sync_aovs and the channel slicing in render()).
    SUPPORTED_PASSES = frozenset({ALBEDO, DEPTH, NORMAL})

    def render(
        self,
        scene: Scene,
        config: Config,
        filename: Path | str | None = None,
        yaml_file: Path | str | None = None,
    ):
        """Render scene using Mitsuba.

        Args:
            scene: Compiled scene.
            config: Rendering configuration.
            filename: Output image filename.
            yaml_file: Optional YAML scene export filename (mi_config).

        Returns:
            Rendered image as Mitsuba tensor.
        """
        ensure_variant()
        logger.info(f"Using Mitsuba variant '{mi.variant()}'.")

        mi_config = generate_base_config(config)
        mi_config |= generate_scene_config(scene)

        if yaml_file is not None:
            if isinstance(yaml_file, str):
                yaml_file = Path(yaml_file)
            yaml_file.parent.mkdir(parents=True, exist_ok=True)
            serializable = _mi_config_to_serializable(mi_config)
            with open(yaml_file, "w") as f:
                yaml.dump(serializable, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Scene saved to {yaml_file}")

        mi_scene = mi.load_dict(mi_config)
        image = mi.render(scene=mi_scene)  # type: ignore
        logger.info("Rendering done")
        image_layers = image

        # Always extract the main RGBA image from the first 4 channels.
        # When an AOV integrator is active the remaining channels hold pass
        # data; without one, image_layers already has exactly 4 channels.
        image = image_layers[:, :, mi.ArrayXi([0, 1, 2, 3])]  # type: ignore

        # Compute per-pass channel offsets by walking the AOV list in order.
        # Mitsuba lays out AOV channels sequentially after the 4 RGBA channels.
        # Known widths: RGB AOVs contribute 3 channels, scalar AOVs 1 channel.
        _aov_width = {
            "albedo:albedo": 3,
            "depth:depth": 1,
            "sh_normal:sh_normal": 3,
        }
        albedo_offset: int | None = None
        depth_offset: int | None = None
        normal_offset: int | None = None

        from ...setup.integrator import AOV as AOVIntegrator

        if isinstance(config.integrator, AOVIntegrator):
            _offset = 4
            for aov_str in config.integrator.aovs:
                if aov_str == "albedo:albedo":
                    albedo_offset = _offset
                elif aov_str == "depth:depth":
                    depth_offset = _offset
                elif aov_str == "sh_normal:sh_normal":
                    normal_offset = _offset
                _offset += _aov_width.get(aov_str, 1)

        if config.albedo:
            if albedo_offset is None:
                logger.warning(
                    "Albedo pass requested but no albedo AOV found in integrator"
                )
            else:
                o = albedo_offset
                # Mitsuba's ``albedo`` AOV is ``BSDF::eval_diffuse_reflectance``,
                # which is only meaningful for diffuse-style BSDFs. For conductors
                # it overshoots [0, 1] (e.g. ~24x the copper tint) and darkens at
                # grazing angles, so clamp to a valid reflectance range. (A clean
                # conductor albedo isn't available from this AOV — the Blender
                # backend reads the Glossy Color pass for that.)
                albedo = np.array(image_layers[:, :, mi.ArrayXi([o, o + 1, o + 2])])
                fg = np.array(image_layers[:, :, 3]) > 0.5
                if fg.any() and float(albedo[fg].max()) > 1.5:
                    logger.warning(
                        "Albedo pass: BSDF reflectance exceeds [0, 1] (max "
                        f"{float(albedo[fg].max()):.1f}); Mitsuba's albedo AOV is "
                        "unreliable for non-diffuse BSDFs (e.g. conductors) and "
                        "has been clamped. Use the Blender backend for a faithful "
                        "albedo of such materials."
                    )
                albedo = np.clip(albedo, 0.0, 1.0)
                alpha = np.array(image_layers[:, :, 3])
                albedo_image = mi.TensorXf(
                    np.concatenate([albedo, alpha[:, :, None]], axis=2)
                )

        if config.depth:
            if depth_offset is None:
                logger.warning(
                    "Depth pass requested but no depth AOV found in integrator"
                )
            else:
                alpha = np.array(image_layers[:, :, 3])
                depth = np.array(image_layers[:, :, depth_offset])
                # Mitsuba's ``depth`` AOV is the ray distance (camera → hit),
                # which carries a radial/perspective falloff: even a flat plane
                # facing the camera reads farther toward the edges. That swamps an
                # object's fine relief. Convert to planar Z-depth (distance along
                # the optical axis) by multiplying by cos(theta) per pixel, so a
                # plane reads constant and only relief remains — matching the
                # Blender Z pass.
                depth = depth * _camera_axis_cosine(depth.shape, config.sensor)
                # Normalize over the foreground only. The background depth AOV is
                # 0 (no hit); including it (and anti-aliased silhouette pixels,
                # whose depth blends the object with the 0 background) collapses
                # the object's depth range. Use near-opaque pixels for the range.
                mask = alpha > 0.99
                vis = np.ones_like(depth)  # background → white
                if mask.any():
                    fg = depth[mask]
                    min_depth = float(fg.min())
                    max_depth = float(fg.max())
                    if max_depth > min_depth:
                        norm = np.clip(
                            (depth - min_depth) / (max_depth - min_depth), 0.0, 1.0
                        )
                    else:
                        norm = np.zeros_like(depth)
                    # Nearest (smallest depth) → white, farthest → black, matching
                    # the Blender backend's depth pass.
                    vis[mask] = 1.0 - norm[mask]
                # Opaque (alpha = 1) so the white background renders flat, not
                # composited-transparent, again matching the Blender backend.
                depth_image = mi.TensorXf(
                    np.stack([vis, vis, vis, np.ones_like(vis)], axis=2)
                )

        if config.normal:
            if normal_offset is None:
                logger.warning(
                    "Normal pass requested but no normal AOV found in integrator"
                )
            else:
                o = normal_offset
                # ``sh_normal`` holds signed components in [-1, 1]. Remap to
                # [0, 1] (out = N * 0.5 + 0.5) so an 8-bit image keeps the
                # negative half instead of clamping it to black — matching the
                # Blender backend's normal pass.
                alpha = np.array(image_layers[:, :, 3])
                nx = np.array(image_layers[:, :, o])
                ny = np.array(image_layers[:, :, o + 1])
                nz = np.array(image_layers[:, :, o + 2])
                normal = np.clip(np.stack([nx, ny, nz], axis=2) * 0.5 + 0.5, 0.0, 1.0)
                normal_image = mi.TensorXf(
                    np.concatenate([normal, alpha[:, :, None]], axis=2)
                )

        if filename is not None:
            if isinstance(filename, str):
                filename = Path(filename)
            # Create the output directory up front; Mitsuba's file writer does
            # not, and a missing parent fails silently (deferred I/O error).
            filename.parent.mkdir(parents=True, exist_ok=True)
            save_image(image, filename)

            if config.albedo and albedo_offset is not None:
                save_image(albedo_image, aov_path(filename, ALBEDO))

            if config.depth and depth_offset is not None:
                save_image(depth_image, aov_path(filename, DEPTH))

            if config.normal and normal_offset is not None:
                # Packed normals are linear data, not colour — save without gamma.
                save_image(normal_image, aov_path(filename, NORMAL), srgb_gamma=False)

        return image
