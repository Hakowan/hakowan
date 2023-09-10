from dataclasses import dataclass


@dataclass(kw_only=True)
class Sampler:
    sample_count: int = 32  # Samples per pixel.
    seed: int = 0


@dataclass(kw_only=True)
class IndependentSampler(Sampler):
    pass


@dataclass(kw_only=True)
class StratifiedSampler(Sampler):
    jitter: bool = True
