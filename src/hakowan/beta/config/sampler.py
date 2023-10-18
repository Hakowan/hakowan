from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class Sampler:
    sample_count: int = 32  # Samples per pixel.
    seed: int = 0


@dataclass(kw_only=True, slots=True)
class Independent(Sampler):
    pass


@dataclass(kw_only=True, slots=True)
class Stratified(Sampler):
    jitter: bool = True
