from ..grammar import layer
from .view import View
from .scene import Scene
from .transform import apply_transform
from .channel import preprocess_channels, process_channels

import copy


def condense_layer_tree_to_scene(root: layer.Layer) -> Scene:
    scene = Scene()

    def generate_view(ancestors: list[layer.Layer]) -> View:
        view = View()
        for l in ancestors:
            if view.data_frame is None:
                view.data_frame = copy.deepcopy(l._spec.data)
            if view.mark is None:
                view.mark = l._spec.mark
            if view.transform is None:
                view.transform = copy.deepcopy(l._spec.transform)
            elif l._spec.transform is not None:
                view.transform *= l._spec.transform
            if len(l._spec.channels) > 0:
                view.channels.extend(l._spec.channels)

        view.validate()
        return view

    def traverse(l: layer.Layer, ancestors: list[layer.Layer]) -> None:
        # `ancestors` is a list of layers from the root to the current layer.
        ancestors.append(l)
        if len(l._children) == 0:
            scene.append(generate_view(ancestors))
        else:
            for child in l._children:
                traverse(child, ancestors)
        ancestors.pop()

    traverse(root, [])
    return scene


def compile(root: layer.Layer) -> Scene:
    # Step 1: condense each path from root to leaf in the layer tree into a view.
    scene = condense_layer_tree_to_scene(root)

    # Step 2: carry out transform operations on each view.
    for view in scene:
        apply_transform(view)

    # Step 3: preprocess channels, compute dependent parameters based on data.
    for view in scene:
        preprocess_channels(view)

    # Step 4: process channels, apply scales.
    for view in scene:
        process_channels(view)

    return scene
