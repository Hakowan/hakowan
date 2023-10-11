from .sensor import Sensor, Perspective
from .film import Film
from .sampler import Sampler, Independent
from .emitter import Emitter, Envmap
from .integrator import Integrator, Path

import numpy as np
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class Config:
    sensor: Sensor = field(default_factory=Perspective)
    film: Film = field(default_factory=Film)
    sampler: Sampler = field(default_factory=Independent)
    emitters: list[Emitter] = field(default_factory=lambda: [Envmap()])
    integrator: Integrator = field(default_factory=Path)

    def z_up(self):
        self.sensor.location = np.array([0, -5, 0])
        self.sensor.up = np.array([0, 0, 1])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 0, 1])
