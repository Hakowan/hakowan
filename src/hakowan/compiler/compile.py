"""Compile declarative layer trees into backend-ready scenes."""

from ..common import logger
from ..grammar import layer, mark
from .view import View
from .scene import Scene
from .transform import apply_transform
from .channel import preprocess_channels, process_channels
from .overlay import collect_overlays
from ..setup import Config

import copy


def condense_layer_tree_to_scene(
    root: layer.Layer,
) -> tuple[Scene, dict[int, layer.LayoutOptions]]:
    """Flatten each leaf path into a resolved View and collect layout options."""
    scene = Scene()
    # Deterministic pre-order IDs identify layout nodes both during recursive
    # packing and in serialized WebGL cell tags.
    node_options: dict[int, layer.LayoutOptions] = {}
    next_layout_id = 0

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
            if view.name is None:
                view.name = lyr._spec.name
            if lyr._spec.annotations:
                view.annotations.extend(copy.deepcopy(lyr._spec.annotations))
            if view.transform is None:
                view.transform = copy.deepcopy(lyr._spec.transform)
            elif lyr._spec.transform is not None:
                view.transform *= lyr._spec.transform
            if len(lyr._spec.channels) > 0:
                view.channels.extend(copy.deepcopy(lyr._spec.channels))

        if view.mark is None:
            assert view.data_frame is not None, "Data component is not specified"
            if view.data_frame.mesh.num_facets == 0:
                logger.debug("Apply default point mark to facet-free data.")
                view.mark = mark.Mark.Point
            else:
                logger.debug("Apply default surface mark.")
                view.mark = mark.Mark.Surface

        view.validate()
        view.initialize_bbox()
        return view

    def traverse(
        lyr: layer.Layer, ancestors: list[layer.Layer], cell_key: tuple
    ) -> None:
        nonlocal next_layout_id
        # `ancestors` is a list of layers from the root to the current layer.
        # `cell_key` identifies which juxtaposition cell this branch belongs to.
        ancestors.append(lyr)
        if lyr._layout is not None and len(lyr._children) > 0:
            # Juxtaposition node: record its options and extend the cell key so
            # each child becomes a distinct cell (or sub-layout).
            node_id = next_layout_id
            next_layout_id += 1
            node_options[node_id] = lyr._layout
            for i, child in enumerate(lyr._children):
                traverse(child, ancestors, cell_key + ((node_id, i),))
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


def compile(root: layer.Layer | object, *, preserve_attributes: bool = False) -> Scene:
    """Compile a layer tree into a resolution-independent scene.

    Traverses the layer tree, resolves channels, applies transforms and scales,
    and finalises per-view data frames. Scene- and screen-space sizes remain in
    their declared units; call :func:`prepare_scene` with the render configuration
    before passing the scene to a backend.

    Args:
        root: Root visualization layer.
        preserve_attributes: Keep inactive source attributes in compiled meshes
            for observation and pixel picking. Defaults to False for lean render
            payloads.

    Returns:
        A resolution-independent scene containing one view per leaf path.

    """
    if not isinstance(root, layer.Layer):
        from ..grammar.figure import Figure

        if not isinstance(root, Figure):
            raise TypeError(f"Expected Layer or Figure, got {type(root)!r}")
        root = root.layer
    # Step 1: condense each path from the root layer tree into a view.
    scene, node_options = condense_layer_tree_to_scene(root)
    logger.debug(f"Created scene with {len(scene)} views")

    # Step 2: carry out transform operations on each view.
    for view in scene:
        apply_transform(view)

    # Step 3: preprocess channels.
    for view in scene:
        preprocess_channels(view)

    # Step 4: process channels and apply scales.
    for view in scene:
        process_channels(view)
    # Collect overlays while processed texture metadata and source attributes
    # are still available.
    collect_overlays(scene)

    # Step 5: finalize the data frame.
    for view in scene:
        view.finalize(preserve_attributes=preserve_attributes)

    # Step 6: lay out juxtaposition cells.
    if node_options:
        scene.apply_layout(node_options)

    # Step 7: compute the global scene transform.
    scene.compute_global_transform()
    return scene


def prepare_scene(scene: Scene, config: Config) -> Scene:
    """Resolve configuration-dependent units and return a backend-ready scene."""
    scene.resolve_size_spaces(config)
    return scene
