from ..common import logger
from ..grammar import layer, mark
from .view import View
from .scene import Scene
from .transform import apply_transform
from .channel import preprocess_channels, process_channels

import copy


def condense_layer_tree_to_scene(
    root: layer.Layer,
) -> tuple[Scene, dict[int, layer.LayoutOptions]]:
    scene = Scene()
    # Layout options for every juxtaposition node, keyed by id(node) so the
    # recursive layout can honour each node's own axis/gap/normalize.
    node_options: dict[int, layer.LayoutOptions] = {}

    def generate_view(ancestors: list[layer.Layer]) -> View:
        """Generate a view from a path in the layer tree.

        :param ancestors: A list of layers from the root layer to a leaf layer. Layers closer to the
                          root have precedence over layers closer to the leaf.

        :return: a view.
        """
        view = View()
        for lyr in ancestors:
            if view.data_frame is None:
                view.data_frame = copy.deepcopy(lyr._spec.data)
            if view.mark is None:
                view.mark = lyr._spec.mark
            if view.transform is None:
                view.transform = copy.deepcopy(lyr._spec.transform)
            elif lyr._spec.transform is not None:
                view.transform *= lyr._spec.transform
            if len(lyr._spec.channels) > 0:
                view.channels.extend(copy.deepcopy(lyr._spec.channels))

        if view.mark is None:
            logger.debug("Apply default surface mark.")
            view.mark = mark.Mark.Surface

        view.validate()
        view.initialize_bbox()
        return view

    def traverse(
        lyr: layer.Layer, ancestors: list[layer.Layer], cell_key: tuple
    ) -> None:
        # `ancestors` is a list of layers from the root to the current layer.
        # `cell_key` identifies which juxtaposition cell this branch belongs to.
        ancestors.append(lyr)
        if lyr._layout is not None and len(lyr._children) > 0:
            # Juxtaposition node: record its options and extend the cell key so
            # each child becomes a distinct cell (or sub-layout).
            node_options[id(lyr)] = lyr._layout
            for i, child in enumerate(lyr._children):
                traverse(child, ancestors, cell_key + ((id(lyr), i),))
        elif len(lyr._children) == 0:
            # Leaf layer: condense the ancestor path into a view.
            view = generate_view(ancestors)
            view._layout_cell = cell_key
            scene.append(view)
        else:
            # Overlay node (created by `+`): children share the same cell.
            for child in lyr._children:
                traverse(child, ancestors, cell_key)
        ancestors.pop()

    traverse(root, [], ())
    return scene, node_options


def compile(root: layer.Layer) -> Scene:
    """Compile a layer tree into a renderable :class:`Scene`.

    Traverses the layer tree, resolves channels, applies transforms and scales,
    and finalises per-view data frames so the result is ready to pass directly
    to a rendering backend.

    Args:
        root: The root :class:`~hakowan.grammar.layer.Layer` of the visualization.

    Returns:
        A compiled :class:`Scene` containing one :class:`View` per leaf path in
        the layer tree.
    """
    # Step 1: condense each path from root to leaf in the layer tree into a view.
    scene, node_options = condense_layer_tree_to_scene(root)
    logger.debug(f"Created scene with {len(scene)} views")

    # Step 2: carry out transform operations on each view.
    for view in scene:
        apply_transform(view)

    # Step 3: preprocess channels.
    for view in scene:
        preprocess_channels(view)

    # Step 4: process channels, apply scales.
    for view in scene:
        process_channels(view)

    # Step 5: finalize the data frame.
    for view in scene:
        view.finalize()

    # Step 5.5: lay out juxtaposition cells (no-op without `|` / `&`).
    if node_options:
        scene.apply_layout(node_options)

    # Step 6: compute the global scene transform
    scene.compute_global_transform()

    return scene
