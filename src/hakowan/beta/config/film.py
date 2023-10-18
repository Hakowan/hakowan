from dataclasses import dataclass, field

import numpy.typing as npt

@dataclass(kw_only=True, slots=True)
class Film:
    width: int = 1024
    height: int = 800
    file_format: str = "openexr"
    pixel_format: str = "rgba"
    component_format: str = "float16"
    crop_offset: npt.NDArray | None = None
    crop_size: npt.NDArray | None = None

