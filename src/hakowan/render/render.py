from .emitter import generate_emitter_config
from .film import generate_film_config
from .integrator import generate_integrator_config
from .sampler import generate_sampler_config
from .sensor import generate_sensor_config
from .shape import generate_point_config, generate_curve_config, generate_surface_config

from ..common import logger
from ..compiler import Scene, View, compile
from ..setup import Config
from ..grammar import mark
from ..grammar import layer

import datetime
import mitsuba as mi
from typing import Any
from pathlib import Path


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
    shapes = []

    # Generate shape.
    match view.mark:
        case mark.Point:
            shapes = generate_point_config(view)
        case mark.Curve:
            shapes.append(generate_curve_config(view, stamp, index))
        case mark.Surface:
            shapes.append(generate_surface_config(view, stamp, index))

    mi_config = {
        f"view_{index:03}_shape_{i:06}": shape for i, shape in enumerate(shapes)
    }
    return mi_config


def generate_scene_config(scene: Scene) -> dict:
    """Generate a mitsuba scene description dict from a Scene."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scene_config: dict[str, Any] = {}
    for i, view in enumerate(scene):
        scene_config |= generate_view_config(view, stamp, i)
    return scene_config


def dump_dict(data: dict, indent: int = 0):
    lines = []
    for key, value in data.items():
        lines.append(" " * indent + f"{key}:")
        if isinstance(value, dict):
            lines.append(" " * indent + "{")
            lines += dump_dict(value, indent + 4)
            lines.append(" " * indent + "}")
        else:
            sublines = value.__repr__().split("\n")
            if len(sublines) == 1:
                lines[-1] += f" {sublines[0]}"
            else:
                sublines = [" " * indent + line for line in sublines]
                lines += sublines
    return lines


def render(
    root: layer.Layer,
    config: Config | None = None,
    filename: Path | str | None = None,
    xml_filename: Path | None = None,
):
    logger.info(f"Using Mitsuba variant '{mi.variant()}'.")
    scene = compile(root)

    if config is None:
        config = Config()

    mi_config = generate_base_config(config)
    mi_config |= generate_scene_config(scene)

    if xml_filename is not None:
        mi.xml.dict_to_xml(mi_config, xml_filename)

    mi_scene = mi.load_dict(mi_config)
    image = mi.render(scene=mi_scene)  # type: ignore

    if config.albedo_only:
        # Select the albedo channels.
        image = image[:, :, [4, 5, 6, 3]]

    if filename is not None:
        if isinstance(filename, str):
            filename = Path(filename)

        if filename.suffix == ".exr":
            mi.util.write_bitmap(str(filename), image) # type: ignore
        else:
            mi.Bitmap(image).convert( # type: ignore
                pixel_format=mi.Bitmap.PixelFormat.RGBA,
                component_format=mi.Struct.Type.UInt8,
                srgb_gamma=True
            ).write(str(filename), quality=-1) # type: ignore

    return image
