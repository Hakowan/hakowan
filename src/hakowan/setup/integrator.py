from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class Integrator:
    hide_emitters: bool = True


@dataclass(kw_only=True, slots=True)
class Direct(Integrator):
    shading_samples: int | None = None
    emitter_samples: int | None = None
    bsdf_samples: int | None = None


@dataclass(kw_only=True, slots=True)
class Path(Integrator):
    max_depth: int = -1
    rr_depth: int = 5


@dataclass(kw_only=True, slots=True)
class AOV(Integrator):
    aovs: list[str]
    integrator: Integrator | None = None


@dataclass(kw_only=True, slots=True)
class VolPath(Integrator):
    max_depth: int = -1
    rr_depth: int = 5

@dataclass(kw_only=True, slots=True)
class VolPathMIS(Integrator):
    max_depth: int = -1
    rr_depth: int = 5
