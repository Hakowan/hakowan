from ..config.sampler import Sampler, Independent, Stratified

from typing import Any


def generate_sampler_config(sampler: Sampler) -> dict:
    mi_config: dict[str, Any] = {
        "sample_count": sampler.sample_count,
        "seed": sampler.seed,
    }
    match sampler:
        case Independent():
            mi_config["type"] = "independent"
        case Stratified():
            mi_config["type"] = "stratified"
            mi_config["jitter"] = sampler.jitter
        case _:
            raise NotImplementedError(f"Sampler {sampler} not implemented.")

    return mi_config
