from .emitter import generate_emitter_config
from .film import generate_film_config
from .integrator import generate_integrator_config
from .sampler import generate_sampler_config
from .sensor import generate_sensor_config
from .shape import (
    extract_edges,
    generate_point_config,
    generate_curve_config,
    generate_surface_config,
)

from ...common import logger
from ...common.image_io import check_supported_suffix, is_hdr_suffix, save_array
from ...common.overlay import _project_world_annotations, composite_overlay_file
from ...compiler import Scene, View
from ...compiler.overlay import CompiledWorldAnnotation
from ...setup import Config
from ...setup.integrator import AOV, Integrator
from ...setup.render_pass import ALBEDO, DEPTH, NORMAL, RENDER_PASSES, aov_path
from ...grammar import mark
from .. import RenderBackend

import datetime
import mitsuba as mi
import drjit
from typing import Any, Callable
from pathlib import Path
import numpy as np
import yaml


def _camera_axis_cosine(
    shape: tuple[int, ...],
    sensor: Any,
    *,
    film_size: tuple[int, int] | None = None,
    crop_offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Per-pixel cos(theta) between each camera ray and the optical axis.

    Cropped arrays retain their coordinates in the full sensor film so depth
    normalization stays identical to the corresponding uncropped pixels.
    """
    height, width = int(shape[0]), int(shape[1])
    film_width, film_height = film_size or (width, height)
    crop_x, crop_y = crop_offset
    fov = getattr(sensor, "fov", None)
    if fov is None:
        return np.ones((height, width), dtype=np.float32)

    half_tan = float(np.tan(np.radians(float(fov)) * 0.5))
    fov_axis = getattr(sensor, "fov_axis", "smaller")
    reference = {
        "x": film_width,
        "y": film_height,
        "larger": max(film_width, film_height),
        "diagonal": float(np.hypot(film_width, film_height)),
    }.get(fov_axis, min(film_width, film_height))
    step = 2.0 * half_tan / reference
    xs = ((np.arange(width, dtype=np.float64) + crop_x + 0.5) - film_width * 0.5) * step
    ys = (
        (np.arange(height, dtype=np.float64) + crop_y + 0.5) - film_height * 0.5
    ) * step
    cos = 1.0 / np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2 + 1.0)
    return cos.astype(np.float32)


def _film_render_window(sensor: Any):
    """Return full film size, active crop size, and crop offset in pixels."""
    film = sensor.film()
    return (
        tuple(int(value) for value in film.size()),
        tuple(int(value) for value in film.crop_size()),
        tuple(int(value) for value in film.crop_offset()),
    )


def _apply_film_crop(
    sensor: Any, crop_offset: tuple[int, int], crop_size: tuple[int, int]
) -> None:
    """Apply a late film crop and refresh the sensor's cached projection."""
    film_parameters = mi.traverse(sensor.film())
    film_parameters["crop_offset"] = mi.Point2u(*(int(value) for value in crop_offset))
    film_parameters["crop_size"] = mi.Vector2u(*(int(value) for value in crop_size))
    film_parameters.update()
    # Mitsuba 3.9 does not refresh the camera projection when only the film
    # parameters are updated after sensor construction.
    sensor.parameters_changed([])


def _show_emitters(integrator: dict[str, Any]) -> None:
    """Make emitters visible through nested Mitsuba integrators."""
    current = integrator
    while True:
        current["hide_emitters"] = False
        nested = current.get("integrator")
        if not isinstance(nested, dict):
            return
        current = nested


def _effective_aovs(config: Config) -> list[str]:
    """Merge explicit Mitsuba AOVs with requested semantic render passes."""
    result = list(config.integrator.aovs) if isinstance(config.integrator, AOV) else []
    for render_pass in RENDER_PASSES.values():
        aov = render_pass.mitsuba_aov
        if (
            render_pass.name in config.render_passes
            and aov is not None
            and aov not in result
        ):
            result.append(aov)
    return result


def _effective_integrator(config: Config) -> Integrator:
    """Build the render-local integrator without mutating Config."""
    aovs = _effective_aovs(config)
    if isinstance(config.integrator, AOV):
        if aovs == config.integrator.aovs:
            return config.integrator
        return AOV(aovs=aovs, integrator=config.integrator.integrator)
    if aovs:
        return AOV(aovs=aovs, integrator=config.integrator)
    return config.integrator


_BEAUTY_CHANNEL_NAMES = ("integrator.R", "integrator.G", "integrator.B", "integrator.A")


def _beauty_rgba_indexes(
    channel_names: list[str], channel_count: int
) -> tuple[int, int, int, int]:
    """Return image indexes of the RGBA beauty block.

    ``aov_names()`` lists every image channel in order. A nested beauty
    integrator contributes ``integrator.R/G/B/A`` at those indexes; they are
    already absolute, so do not add another four-channel prefix. A plain render
    has an empty name list and four RGBA channels. Auxiliary names with no
    beauty block are not a color image.
    """
    if not channel_names:
        if channel_count < 4:
            raise RuntimeError(
                f"Mitsuba beauty image has {channel_count} channels; expected RGBA."
            )
        return (0, 1, 2, 3)
    missing = [name for name in _BEAUTY_CHANNEL_NAMES if name not in channel_names]
    if missing:
        raise RuntimeError(
            "Mitsuba AOV output has no beauty channels "
            f"({', '.join(missing)}). Set AOV.integrator to a beauty integrator; "
            "None selects a path tracer."
        )
    red, green, blue, alpha = (
        channel_names.index(name) for name in _BEAUTY_CHANNEL_NAMES
    )
    return (red, green, blue, alpha)


def _point_glyph_owns_anchor(
    hit: Any, compiled_scene: Scene | None, annotation: CompiledWorldAnnotation
) -> bool:
    """Whether a Mitsuba point-mark hit belongs to this annotation anchor."""
    if compiled_scene is None or not hit.is_valid():
        return False
    shape_id = hit.shape.id()
    for owner in annotation.layers or (annotation.layer,):
        prefix = f"view_{owner:03}_shape_"
        if not shape_id.startswith(prefix):
            continue
        view = compiled_scene[owner]
        if view.mark is not mark.Point or view.data_frame is None:
            continue
        try:
            vertex_index = int(shape_id[len(prefix) :])
        except ValueError:
            continue
        vertices = view.data_frame.mesh.vertices
        if vertex_index >= len(vertices):
            continue
        point = view.global_transform @ np.array((*vertices[vertex_index], 1.0))
        if np.allclose(point[:3], annotation.position, rtol=0.0, atol=1e-7):
            return True
    return False


def _curve_glyph_owns_anchor(
    hit: Any, compiled_scene: Scene | None, annotation: CompiledWorldAnnotation
) -> bool:
    """Whether a curve-tube hit belongs to the curve containing this anchor."""
    if compiled_scene is None or not hit.is_valid():
        return False
    shape_id = hit.shape.id()
    anchor = np.asarray(annotation.position, dtype=np.float64)
    for owner in annotation.layers or (annotation.layer,):
        if not shape_id.startswith(f"view_{owner:03}_shape_"):
            continue
        view = compiled_scene[owner]
        if (
            view.mark is not mark.Curve
            or view.vector_field_channel is not None
            or view.data_frame is None
        ):
            continue
        base, tip, base_size, tip_size = extract_edges(view)
        transform = np.asarray(view.global_transform, dtype=np.float64)
        radius_scale = abs(np.cbrt(np.linalg.det(transform[:3, :3])))
        for p0, p1, r0, r1 in zip(base, tip, base_size, tip_size, strict=True):
            start = (transform @ np.array((*p0, 1.0)))[:3]
            end = (transform @ np.array((*p1, 1.0)))[:3]
            segment = end - start
            length_squared = float(np.dot(segment, segment))
            fraction = (
                float(np.clip(np.dot(anchor - start, segment) / length_squared, 0, 1))
                if length_squared > 1e-20
                else 0.0
            )
            closest = start + fraction * segment
            radius = (
                (1.0 - fraction) * float(r0) + fraction * float(r1)
            ) * radius_scale
            if np.linalg.norm(anchor - closest) <= radius + 5e-5:
                return True
    return False


def _visible_world_annotations(
    scene: "mi.Scene",
    camera: Any,
    annotations: list[CompiledWorldAnnotation],
    compiled_scene: Scene | None = None,
) -> list[CompiledWorldAnnotation]:
    """Keep anchors whose camera ray first hits their surface or empty space.

    Cast one ray through each projected anchor using the loaded Mitsuba sensor,
    so orthographic rays originate on the corresponding part of the film rather
    than converging on the eye. The same scene BVH handles all shapes/layers.
    """
    if not annotations:
        return annotations

    sensor = scene.sensors()[0]
    film_size, (width, height), crop_offset = _film_render_window(sensor)
    projected = _project_world_annotations(
        annotations,
        (width, height),
        sensor=camera,
        world_to_camera=None,
        projection=None,
        film_size=film_size,
        crop_offset=crop_offset,
    )

    scene_scale = 1.0
    if compiled_scene is not None:
        for view in compiled_scene:
            if view.bbox is not None:
                scene_scale = max(
                    scene_scale,
                    float(np.linalg.norm(view.bbox[1] - view.bbox[0])),
                )
    visible = []
    for annotation, x, y in projected:
        ray, _ = sensor.sample_ray(
            0.0, 0.0, mi.Point2f(x / width, y / height), mi.Point2f(0.5, 0.5)
        )
        anchor_distance = float(
            np.dot(
                np.asarray(annotation.position, dtype=np.float64) - np.array(ray.o),
                np.array(ray.d),
            )
        )
        hit = scene.ray_intersect(ray)
        tolerance = max(5e-5, scene_scale * 1e-6)
        if (
            not hit.is_valid()
            or _point_glyph_owns_anchor(hit, compiled_scene, annotation)
            or _curve_glyph_owns_anchor(hit, compiled_scene, annotation)
            or float(hit.t) >= anchor_distance - tolerance
        ):
            visible.append(annotation)
    return visible


def generate_base_config(config: Config) -> dict:
    """Generate a Mitsuba base config dict from a Config."""
    sensor_config = generate_sensor_config(
        config.sensor, film_width=config.film.width, film_height=config.film.height
    )
    sensor_config["film"] = generate_film_config(config.film)
    sensor_config["sampler"] = generate_sampler_config(config.sampler)
    integrator_config = generate_integrator_config(_effective_integrator(config))
    if config.environment_visible:
        _show_emitters(integrator_config)

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

    # facet_id has no Mitsuba AOV counterpart; other requested passes are
    # derived into a render-local AOV integrator.
    SUPPORTED_PASSES = frozenset({ALBEDO, DEPTH, NORMAL})

    def render(
        self,
        scene: Scene,
        config: Config,
        filename: Path | str | None = None,
        yaml_file: Path | str | None = None,
        **kwargs: Any,
    ):
        """Render scene using Mitsuba.

        Args:
            scene: Compiled scene.
            config: Rendering configuration.
            filename: Output image filename.
            yaml_file: Optional YAML scene export filename (mi_config).
            **kwargs: Rejected compatibility catch-all for unknown options.

        Returns:
            Rendered image as Mitsuba tensor.
        """
        if kwargs:
            raise TypeError(
                f"render() got unexpected keyword argument(s): {list(kwargs)}"
            )
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

        film_settings = mi_config["camera"]["film"]
        crop_offset_value = film_settings.pop("crop_offset", None)
        crop_size_value = film_settings.pop("crop_size", None)
        if crop_offset_value is None or crop_size_value is None:
            crop_offset_value = crop_size_value = None

        mi_scene = mi.load_dict(mi_config)
        if crop_offset_value is not None:
            _apply_film_crop(
                mi_scene.sensors()[0],
                tuple(crop_offset_value),
                tuple(crop_size_value),
            )
        film_size, crop_size, crop_offset = _film_render_window(mi_scene.sensors()[0])

        image = mi.render(scene=mi_scene)  # type: ignore
        logger.info("Rendering done")
        image_layers = image

        # Beauty channels are named by Mitsuba. A nested integrator contributes
        # integrator.R/G/B/A at absolute image indexes; do not add another
        # four-channel prefix. Auxiliary names with no beauty block are not RGBA.
        channel_names = list(mi_scene.integrator().aov_names())
        beauty_r, beauty_g, beauty_b, beauty_a = _beauty_rgba_indexes(
            channel_names, int(image_layers.shape[-1])
        )
        image = image_layers[:, :, mi.ArrayXi([beauty_r, beauty_g, beauty_b, beauty_a])]
        beauty_alpha = np.array(image_layers[:, :, beauty_a])

        def channel_offset(name: str) -> int | None:
            # Semantic AOVs are appended after explicit AOVs. Search backwards
            # so a caller reusing a reserved label cannot shadow our pass.
            return next(
                (
                    index
                    for index in range(len(channel_names) - 1, -1, -1)
                    if channel_names[index] == name
                ),
                None,
            )

        albedo_offset = channel_offset("albedo.R")
        depth_offset = channel_offset("depth.T")
        normal_offset = channel_offset("sh_normal.X")

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
                fg = beauty_alpha > 0.5
                if fg.any() and float(albedo[fg].max()) > 1.5:
                    logger.warning(
                        "Albedo pass: BSDF reflectance exceeds [0, 1] (max "
                        f"{float(albedo[fg].max()):.1f}); Mitsuba's albedo AOV is "
                        "unreliable for non-diffuse BSDFs (e.g. conductors) and "
                        "has been clamped. Use the Blender backend for a faithful "
                        "albedo of such materials."
                    )
                albedo = np.clip(albedo, 0.0, 1.0)
                alpha = beauty_alpha
                albedo_image = mi.TensorXf(
                    np.concatenate([albedo, alpha[:, :, None]], axis=2)
                )

        if config.depth:
            if depth_offset is None:
                logger.warning(
                    "Depth pass requested but no depth AOV found in integrator"
                )
            else:
                alpha = beauty_alpha
                depth = np.array(image_layers[:, :, depth_offset])
                # Mitsuba's ``depth`` AOV is the ray distance (camera → hit),
                # which carries a radial/perspective falloff: even a flat plane
                # facing the camera reads farther toward the edges. That swamps an
                # object's fine relief. Convert to planar Z-depth (distance along
                # the optical axis) by multiplying by cos(theta) per pixel, so a
                # plane reads constant and only relief remains — matching the
                # Blender Z pass.
                depth = depth * _camera_axis_cosine(
                    depth.shape,
                    config.sensor,
                    film_size=film_size,
                    crop_offset=crop_offset,
                )
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
                alpha = beauty_alpha
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
            world_annotations = scene.world_annotations
            geometry_at: Callable[[float, float], bool] | None = None
            if world_annotations and not is_hdr_suffix(filename.suffix):
                world_annotations = _visible_world_annotations(
                    mi_scene, config.sensor, world_annotations, scene
                )
                sensor = mi_scene.sensors()[0]
                width, height = crop_size

                def world_geometry_at(x: float, y: float) -> bool:
                    ray, _ = sensor.sample_ray(
                        0.0,
                        0.0,
                        mi.Point2f(x / width, y / height),
                        mi.Point2f(0.5, 0.5),
                    )
                    return bool(mi_scene.ray_test(ray))

                geometry_at = world_geometry_at
            if not composite_overlay_file(
                filename,
                scene.legends,
                scene.annotations,
                config.background,
                world_annotations=world_annotations,
                sensor=config.sensor,
                geometry_at=geometry_at,
                film_size=film_size,
                crop_offset=crop_offset,
            ):
                logger.warning(
                    "Legends and annotations (including world annotations) are not "
                    "composited into HDR output; use PNG or another "
                    "Pillow-supported format."
                )

            if config.albedo and albedo_offset is not None:
                save_image(albedo_image, aov_path(filename, ALBEDO))

            if config.depth and depth_offset is not None:
                save_image(depth_image, aov_path(filename, DEPTH))

            if config.normal and normal_offset is not None:
                # Packed normals are linear data, not colour — save without gamma.
                save_image(normal_image, aov_path(filename, NORMAL), srgb_gamma=False)

        return image
