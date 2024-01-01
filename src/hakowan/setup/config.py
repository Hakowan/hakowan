from .sensor import Sensor, Perspective
from .film import Film
from .sampler import Sampler, Independent
from .emitter import Emitter, Envmap
from .integrator import Integrator, Path, AOV
from ..common import logger

import numpy as np
from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class Config:
    """Configuration for rendering.

    Attributes:
        sensor: Sensor settings.
        film: Film settings.
        sampler: Sampler settings.
        emitters: Emitter settings.
        integrator: Integrator settings.
        albedo_only: Whether to render albedo only (i.e. without shading).
    """
    sensor: Sensor = field(default_factory=Perspective)
    film: Film = field(default_factory=Film)
    sampler: Sampler = field(default_factory=Independent)
    emitters: list[Emitter] = field(default_factory=lambda: [Envmap()])
    integrator: Integrator = field(default_factory=Path)
    _albedo_only: bool = False

    def z_up(self):
        self.sensor.location = np.array([0, -5, 0])
        self.sensor.up = np.array([0, 0, 1])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 0, 1])
                emitter.rotation = 180.0

    def z_down(self):
        self.sensor.location = np.array([0, 5, 0])
        self.sensor.up = np.array([0, 0, -1])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 0, -1])
                emitter.rotation = 180.0

    def y_up(self):
        self.sensor.location = np.array([0, 0, 5])
        self.sensor.up = np.array([0, 1, 0])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 1, 0])
                emitter.rotation = 180

    def y_down(self):
        self.sensor.location = np.array([0, 0, -5])
        self.sensor.up = np.array([0, -1, 0])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, -1, 0])
                emitter.rotation = 180

    @property
    def albedo_only(self) -> bool:
        """Whether to render albedo only (i.e. without shading).
        """
        return self._albedo_only

    @albedo_only.setter
    def albedo_only(self, value: bool):
        """Whether to render albedo only (i.e. without shading).

        Note that this setting will modify Config.integrator property.
        """
        self._albedo_only = value
        if self._albedo_only:
            if not isinstance(self.integrator, AOV):
                self.integrator = AOV(
                    aovs=["albedo:albedo"], integrator=self.integrator
                )
            else:
                logger.warning("Albedo only is already enabled!")
        else:
            if isinstance(self.integrator, AOV):
                if self.integrator.integrator is not None:
                    self.integrator = self.integrator.integrator
                else:
                    self.integrator = Path()
            else:
                logger.warning("Albedo only is already disabled!")
