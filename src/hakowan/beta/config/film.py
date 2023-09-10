from dataclasses import dataclass, field

import numpy.typing as npt

@dataclass(kw_only=True)
class Film:
    width: int = 1024
    height: int = 800
    file_format: str = "openexr"
    pixel_format: str = "rgba"
    crop_offset: npt.NDArray | = None
    crop_size: npt.NDArray | None = None

