from ..config.integrator import Integrator, Direct, Path, AOV, VolPath

from typing import Any


def generate_integrator_config(integrator: Integrator) -> dict:
    """Generate a Mitsuba integrator description dict from an Integrator."""
    mi_config: dict[str, Any] = {
        "hide_emitters": integrator.hide_emitters,
    }

    match integrator:
        case Direct():
            mi_config["type"] = "direct"
            if integrator.shading_samples is not None:
                mi_config["shading_samples"] = integrator.shading_samples
            else:
                if integrator.emitter_samples is not None:
                    mi_config["emitter_samples"] = integrator.emitter_samples
                if integrator.bsdf_samples is not None:
                    mi_config["bsdf_samples"] = integrator.bsdf_samples
        case Path():
            mi_config["type"] = "path"
            mi_config["max_depth"] = integrator.max_depth
            mi_config["rr_depth"] = integrator.rr_depth
        case AOV():
            mi_config["type"] = "aov"
            mi_config["aovs"] = ",".join(integrator.aovs)
            if integrator.integrator is not None:
                mi_config["aov_image"] = generate_integrator_config(
                    integrator.integrator
                )
        case VolPath():
            mi_config["type"] = "volpath"
            mi_config["max_depth"] = integrator.max_depth
            mi_config["rr_depth"] = integrator.rr_depth
        case _:
            raise NotImplementedError(f"Unknown integrator type: {type(integrator)}")

    return mi_config
