from ..setup.sensor import Sensor, Perspective, Orthographic, ThinLens

import mitsuba as mi
import numpy as np


def generate_sensor_config(sensor: Sensor) -> dict:
    """Generate a Mitsuba sensor description dict from a Sensor."""

    mi_config = {
        "to_world": mi.ScalarTransform4f.look_at(  # type: ignore
            origin=sensor.location, target=sensor.target, up=sensor.up
        ),
        "near_clip": sensor.near_clip,
        "far_clip": sensor.far_clip,
    }

    match sensor:
        case Perspective():
            mi_config["type"] = "perspective"
            mi_config["fov"] = sensor.fov
            mi_config["fov_axis"] = sensor.fov_axis
        case Orthographic():
            mi_config["type"] = "orthographic"
        case ThinLens():
            mi_config["type"] = "thinlens"
            mi_config["aperture_radius"] = sensor.aperture_radius
            mi_config["focus_distance"] = sensor.focus_distance
        case _:
            raise NotImplementedError(f"Sensor {sensor} not implemented.")

    return mi_config
