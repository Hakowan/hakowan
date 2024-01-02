from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class Integrator:
    """Integrator dataclass contains parameters of various rendering techniques.

    Attributes:
        hide_emitters: Whether to hide emitters from the camera.
    """
    hide_emitters: bool = True


@dataclass(kw_only=True, slots=True)
class Direct(Integrator):
    """Direct integrator.

    Attributes:
        shading_samples: Number of shading samples.
        emitter_samples: Number of emitter samples.
        bsdf_samples: Number of BSDF samples.

    Note:
        See
        [Mitsuba doc](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#direct-illumination-integrator-direct)
        for more details.
    """
    shading_samples: int | None = None
    emitter_samples: int | None = None
    bsdf_samples: int | None = None


@dataclass(kw_only=True, slots=True)
class Path(Integrator):
    """Path integrator.

    Attributes:
        max_depth: Maximum path depth. (-1 for unlimited)
        rr_depth: Depth at which Russian roulette starts.

    Note:
        This integrator should work well for most surface-based scenes.
        See
        [Mitsuba
        doc](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#path-tracer-path)
        for more details.
    """
    max_depth: int = -1
    rr_depth: int = 5


@dataclass(kw_only=True, slots=True)
class AOV(Integrator):
    """Arbitrary output variable (AOV) integrator.

    Attributes:
        aovs: List of AOVs to render.
        integrator: Integrator to use for rendering AOVs.

    Note:
        See
        [Mitsuba
        doc](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#arbitrary-output-variables-integrator-aov)
        for supported AOV types and other details.
    """
    aovs: list[str]
    integrator: Integrator | None = None


@dataclass(kw_only=True, slots=True)
class VolPath(Integrator):
    """Volumetric path integrator.

    Attributes:
        max_depth: Maximum path depth. (-1 for unlimited)
        rr_depth: Depth at which Russian roulette starts.

    Note:
        This integrator should work well for most volume-based scenes. For example, if dielectric
        material is involved, `VolPath` integrator sometimes produces better results than `Path`
        integrator.

        See
        [Mitsuba
        doc](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#volumetric-path-tracer-volpath)
        for more details.
    """
    max_depth: int = -1
    rr_depth: int = 5

@dataclass(kw_only=True, slots=True)
class VolPathMIS(Integrator):
    """Volumetric path integrator with spectral MIS.

    Attributes:
        max_depth: Maximum path depth. (-1 for unlimited)
        rr_depth: Depth at which Russian roulette starts.

    Note:
        See
        [Mitsuba
        doc](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#volumetric-path-tracer-with-spectral-mis-volpathmis)
        for more details.
    """
    max_depth: int = -1
    rr_depth: int = 5
