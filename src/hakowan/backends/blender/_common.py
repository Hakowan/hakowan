"""Shared low-level helpers for the Blender backend."""


def _ensure_nodes(datablock) -> None:
    """Ensure a material/world has a shader node tree.

    Newer Blender (4.x+) creates the node tree automatically, so the legacy
    ``use_nodes = True`` toggle is deprecated (slated for removal in 6.0).
    Only fall back to it when the node tree is genuinely absent, which keeps
    compatibility with older Blender without emitting deprecation warnings.
    """
    if datablock.node_tree is None:
        datablock.use_nodes = True
