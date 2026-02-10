from .emitter import generate_emitter_config
from .film import generate_film_config
from .integrator import generate_integrator_config
from .sampler import generate_sampler_config
from .sensor import generate_sensor_config
from .shape import generate_point_config, generate_curve_config, generate_surface_config

from ...common import logger
from ...compiler import Scene, View
from ...setup import Config
from ...grammar import mark
from .. import RenderBackend

import datetime
import mitsuba as mi
import drjit
from typing import Any
from pathlib import Path
import numpy as np
import yaml


def generate_base_config(config: Config):
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


def generate_view_config(view: View, stamp: str, index: int):
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
        return m.numpy().tolist()

    try:
        arr = np.array(obj)
        if arr.ndim <= 2 and arr.size <= 16:
            return arr.tolist()
    except (TypeError, ValueError):
        pass
    return obj


def save_image(image: drjit.ArrayBase, filename: Path):
    if filename.suffix == ".exr":
        mi.util.write_bitmap(str(filename), image)  # type: ignore
    else:
        mi.Bitmap(image).convert(  # type: ignore
            pixel_format=mi.Bitmap.PixelFormat.RGBA,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        ).write(str(filename), quality=-1)  # type: ignore
    logger.info(f"Rendering saved to {filename}")


class MitsubaBackend(RenderBackend):
    """Mitsuba rendering backend."""

    def render(
        self,
        scene: Scene,
        config: Config,
        filename: Path | str | None = None,
        scene_file: Path | str | None = None,
        **kwargs,
    ):
        """Render scene using Mitsuba.

        Args:
            scene: Compiled scene.
            config: Rendering configuration.
            filename: Output image filename.
            scene_file: Optional YAML scene export filename (mi_config).
            **kwargs: Additional backend-specific options.

        Returns:
            Rendered image as Mitsuba tensor.
        """
        logger.info(f"Using Mitsuba variant '{mi.variant()}'.")

        mi_config = generate_base_config(config)
        mi_config |= generate_scene_config(scene)

        if scene_file is not None:
            if isinstance(scene_file, str):
                scene_file = Path(scene_file)
            scene_file.parent.mkdir(parents=True, exist_ok=True)
            serializable = _mi_config_to_serializable(mi_config)
            with open(scene_file, "w") as f:
                yaml.dump(serializable, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Scene saved to {scene_file}")

        mi_scene = mi.load_dict(mi_config)
        image = mi.render(scene=mi_scene)  # type: ignore
        logger.info("Rendering done")
        image_layers = image

        if config.albedo:
            # Select the albedo channels.
            image = image_layers[:, :, mi.ArrayXi([0, 1, 2, 3])]  # type: ignore
            albedo_image = image_layers[:, :, mi.ArrayXi([4, 5, 6, 3])]  # type: ignore

        if config.depth:
            image = image_layers[:, :, mi.ArrayXi([0, 1, 2, 3])]  # type: ignore
            alpha = image_layers[:, :, 3]
            depth = image_layers[:, :, 4]
            min_depth = drjit.min(depth)
            max_depth = drjit.max(depth)
            depth = (depth - min_depth) / (max_depth - min_depth)
            depth_image = mi.TensorXf(np.stack([depth, depth, depth, alpha], axis=2))

        if config.normal:
            image = image_layers[:, :, mi.ArrayXi([0, 1, 2, 3])]
            normal_image = image_layers[:, :, mi.ArrayXi([4, 5, 6, 3])]

        if filename is not None:
            if isinstance(filename, str):
                filename = Path(filename)
            save_image(image, filename)

            if config.albedo:
                albedo_filename = filename.with_name(
                    filename.stem + "_albedo" + filename.suffix
                )
                save_image(albedo_image, albedo_filename)

            if config.depth:
                depth_filename = filename.with_name(
                    filename.stem + "_depth" + filename.suffix
                )
                save_image(depth_image, depth_filename)

            if config.normal:
                normal_filename = filename.with_name(
                    filename.stem + "_normal" + filename.suffix
                )
                save_image(normal_image, normal_filename)

        return image
