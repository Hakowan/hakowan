from .emitter import generate_emitter_config
from .film import generate_film_config
from .integrator import generate_integrator_config
from .sampler import generate_sampler_config
from .sensor import generate_sensor_config
from .shape import generate_point_cloud_config, generate_mesh_config

from ..common import logger
from ..compiler import Scene, View, compile
from ..config import Config
from ..grammar import mark
from ..grammar import layer

import datetime
import mitsuba as mi
from typing import Any


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
            shapes = generate_point_cloud_config(view)
        case mark.Curve:
            raise NotImplementedError("Curve rendering is not yet supported.")
        case mark.Surface:
            shapes.append(generate_mesh_config(view, stamp, index))

    mi_config = {f"shape_{i:06}": shape for i, shape in enumerate(shapes)}
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


def render(root: layer.Layer, config: Config):
    scene = compile(root)

    mi.set_variant("scalar_rgb")
    mi_config = generate_base_config(config)
    mi_config |= generate_scene_config(scene)
    mi.xml.dict_to_xml(mi_config, "tmp.xml")

    mi_scene = mi.load_dict(mi_config)
    image = mi.render(scene=mi_scene)  # type: ignore
    mi.util.write_bitmap("tmp.exr", image)
