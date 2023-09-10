from dataclasses import dataclass, field

from .sensor import Sensor, Perspective
from .film import Film
from .sampler import Sampler, IndependentSampler
from .emitter import Emitter, Envmap
from .integrator import Integrator, Path


@dataclass(kw_only=True)
class Config:
    sensor: Sensor = field(default_factory=Perspective)
    film: Film = field(default_factory=Film)
    sampler: Sampler = field(default_factory=IndependentSampler)
    emitters: list[Emitter] = field(default_factory=lambda: [Envmap()])
    integrator: Integrator = field(default_factory=Path)
