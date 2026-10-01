from ...setup.sensor import Sensor, Perspective, Orthographic, ThinLens

import mitsuba as mi


def generate_sensor_config(
    sensor: Sensor, *, film_width: int, film_height: int
) -> dict:
    """Generate a Mitsuba sensor description for the given film dimensions."""

    mi_config = {
        "to_world": mi.ScalarTransform4f().look_at(  # type: ignore
            origin=sensor.location,
            target=sensor.target,
            up=sensor.up,
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
            # Mitsuba's orthographic film already applies the inverse aspect ratio
            # to its vertical rays. Scale both axes by the aspect ratio so the
            # full vertical extent stays at sensor.scale for any film shape.
            half_width = sensor.scale * film_width / film_height * 0.5
            mi_config["to_world"] = mi_config[
                "to_world"
            ] @ mi.ScalarTransform4f().scale(  # type: ignore
                [half_width, half_width, 1.0]
            )
        case ThinLens():
            mi_config["type"] = "thinlens"
            mi_config["aperture_radius"] = sensor.aperture_radius
            mi_config["focus_distance"] = sensor.focus_distance
        case _:
            raise NotImplementedError(f"Sensor {sensor} not implemented.")

    return mi_config
