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
        albedo: Whether to render albedo (i.e. without shading).
        depth: Whether to render depth.
    """
    sensor: Sensor = field(default_factory=Perspective)
    film: Film = field(default_factory=Film)
    sampler: Sampler = field(default_factory=Independent)
    emitters: list[Emitter] = field(default_factory=lambda: [Envmap()])
    integrator: Integrator = field(default_factory=Path)
    _albedo: bool = False
    _depth: bool = False
    _normal: bool = False

    def z_up(self):
        """Update configuration for z-up coordinate system."""
        self.sensor.location = np.array([0, -5, 0])
        self.sensor.up = np.array([0, 0, 1])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 0, 1])
                emitter.rotation = 180.0

    def z_down(self):
        """Update configuration for z-down coordinate system."""
        self.sensor.location = np.array([0, 5, 0])
        self.sensor.up = np.array([0, 0, -1])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 0, -1])
                emitter.rotation = 180.0

    def y_up(self):
        """Update configuration for y-up coordinate system."""
        self.sensor.location = np.array([0, 0, 5])
        self.sensor.up = np.array([0, 1, 0])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 1, 0])
                emitter.rotation = 180

    def y_down(self):
        """Update configuration for y-down coordinate system."""
        self.sensor.location = np.array([0, 0, -5])
        self.sensor.up = np.array([0, -1, 0])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, -1, 0])
                emitter.rotation = 180

    @property
    def albedo(self) -> bool:
        """Whether to render albedo (i.e. without shading)."""
        return self._albedo

    @albedo.setter
    def albedo(self, value: bool):
        """Whether to render albedo (i.e. without shading).

        Note that this setting will modify Config.integrator property.
        """
        self._albedo = value
        if self._albedo:
            if not isinstance(self.integrator, AOV):
                self.integrator = AOV(
                    aovs=["albedo:albedo"], integrator=self.integrator
                )
            else:
                if "albedo:albedo" not in self.integrator.aovs:
                    self.integrator.aovs.append("albedo:albedo")
        else:
            if isinstance(self.integrator, AOV):
                if self.integrator.integrator is not None:
                    self.integrator = self.integrator.integrator
                else:
                    self.integrator = Path()

    @property
    def depth(self) -> bool:
        """Whether to render depth."""
        return self._depth

    @depth.setter
    def depth(self, value: bool):
        """Whether to render depth.

        Note that this setting will modify Config.integrator property.
        """
        self._depth = value
        if self._depth:
            if not isinstance(self.integrator, AOV):
                self.integrator = AOV(aovs=["depth:depth"], integrator=self.integrator)
            elif "depth:depth" not in self.integrator.aovs:
                self.integrator.aovs.append("depth:depth")
        else:
            if isinstance(self.integrator, AOV):
                if self.integrator.integrator is not None:
                    self.integrator = self.integrator.integrator
                else:
                    self.integrator = Path()

    @property
    def normal(self) -> bool:
        """Whether to render normal."""
        return self._normal

    @normal.setter
    def normal(self, value: bool):
        """Whether to render normal.

        Note that this setting will modify Config.integrator property.
        """
        self._normal = value
        if self._normal:
            if not isinstance(self.integrator, AOV):
                self.integrator = AOV(aovs=["sh_normal:sh_normal"], integrator=self.integrator)
            elif "sh_normal:sh_normal" not in self.integrator.aovs:
                self.integrator.aovs.append("sh_normal:sh_normal")
        else:
            if isinstance(self.integrator, AOV):
                if self.integrator.integrator is not None:
                    self.integrator = self.integrator.integrator
                else:
                    self.integrator = Path()
