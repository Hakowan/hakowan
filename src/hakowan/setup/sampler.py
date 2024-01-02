from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class Sampler:
    """Sampler dataclass contains sampling-related settings.

    Attributes:
        sample_count: Number of samples per pixel.
        seed: Seed for random number generate.
    """
    sample_count: int = 256  # Samples per pixel.
    seed: int = 0


@dataclass(kw_only=True, slots=True)
class Independent(Sampler):
    """Independent sampler.

    Note:
        See
        [Mitsuba
        doc](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#independent-sampler-independent)
        for more details.
    """
    pass


@dataclass(kw_only=True, slots=True)
class Stratified(Sampler):
    """Stratified sampler.

    Attributes:
        jitter: Whether to jitter the samples.

    Note:
        See [Mitsuba
        doc](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#stratified-sampler-stratified)
        for more details.
    """
    jitter: bool = True
