from ..config.film import Film

import mitsuba as mi


def generate_film_config(film: Film) -> dict:
    """Generate a Mitsuba film description dict from a Film."""

    mi_config = {
        "type": "hdrfilm",
        "width": film.width,
        "height": film.height,
        "file_format": film.file_format,
        "pixel_format": film.pixel_format,
        "component_format": film.component_format,
    }
    if film.crop_offset is not None:
        mi_config["crop_offset"] = film.crop_offset.tolist()
    if film.crop_size is not None:
        mi_config["crop_size"] = film.crop_size.tolist()

    return mi_config
