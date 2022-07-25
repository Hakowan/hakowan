""" Redner configurations """
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt


@dataclass
class RenderConfig:
    """Render configurations."""

    filename: Path = Path("scene.exr")
    backend: str = "mitsuba"
    width: int = 2048
    height: int = 1800
    fov: float = 28.8415
    num_samples: int = 64
    transform: npt.NDArray = np.identity(4)
    dry_run = False


    def __post_init__(self):
        if not isinstance(self.filename, Path):
            self.filename = Path(self.filename)
