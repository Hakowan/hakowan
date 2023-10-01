import mitsuba as mi

from ..compiler import Scene, View
from ..config import Config

def generate_config(view: View):
    """ Generate a Mitsuba shape description dict from a View."""
    # Gather all involved attributes.
    # Unify index buffer.
    # Save mesh.
    # Generate shape description.
    # Generate bsdf.
    # TODO
    return {}

def render(scene: Scene, config: Config):
    mi_config = {}
    for i, view in enumerate(scene):
        mi_config[f"view_{i:03}"] = generate_config(view)

    mi_scene = mi.load_dict(mi_config)
    image = mi.render(scene = mi_scene) # type: ignore
    mi.util.write_bitmap("tmp.exr", image)
