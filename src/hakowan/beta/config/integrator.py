from dataclasses import dataclass


@dataclass(kw_only=True)
class Integrator:
    hide_emitters: bool = False


@dataclass(kw_only=True)
class Direct(Integrator):
    shading_samples: int | None = None
    emitter_samples: int | None = None
    bsdf_samples: int | None = None


@dataclass(kw_only=True)
class Path(Integrator):
    max_depth: int = -1
    rr_depth: int = 5


@dataclass(kw_only=True)
class AOV(Integrator):
    aovs: list[str]
    integrator: Integrator | None = None


@dataclass(kw_only=True)
class VolPath(Integrator):
    max_depth: int = -1
    rr_depth: int = 5
