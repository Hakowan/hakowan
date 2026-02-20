from dataclasses import dataclass, field

import numpy.typing as npt


@dataclass(kw_only=True, slots=True)
class Film:
    """Film dataclass stores specifications of the output image.

    Attributes:
        width: Width of the output image in pixels.
        height: Height of the output image in pixels.
        file_format: File format of the output image.
        pixel_format: Pixel format of the output image.
        component_format: Component format of the output image.
        crop_offset: Offset of the crop window in pixels.
        crop_size: Size of the crop window in pixels.

    Together, `width` and `height` specify the output image resolution.
    `crop_offset` and `crop_size` defines a crop region. If either is `None`, no cropping is performed.
    `file_format`, `pixel_format` and `component_format` are for advanced user only. The default
    values should work in most cases.
    """

    width: int = 1024
    height: int = 800
    file_format: str = "openexr"
    pixel_format: str = "rgba"
    component_format: str = "float16"
    crop_offset: npt.NDArray | None = None
    crop_size: npt.NDArray | None = None
