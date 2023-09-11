import mitsuba as mi

from ..compiler import Scene, View

def generate_config(view: View):
    # TODO
    return {}

def render(scene: Scene):
    mi_config = {}
    for i, view in enumerate(scene):
        mi_config[f"view_{i:03}"] = generate_config(view)

    mi_scene = mi.load_dict(mi_config)
    image = mi.render(scene = mi_scene) # type: ignore
    mi.util.write_bitmap("tmp.exr", image)
