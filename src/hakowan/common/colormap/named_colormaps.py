from functools import lru_cache

import numpy as np

from ..color import Color
from .colormap import ColorMap
from .coolwarm import coolwarm
from .inferno import inferno
from .magma import magma
from .plasma import plasma
from .turbo import turbo
from .viridis import viridis
from .colorbrewer2 import accent, dark2, paired, pastel1, pastel2, set1, set2, set3

named_colormaps = {
    "coolwarm": coolwarm,
    "inferno": inferno,
    "magma": magma,
    "plasma": plasma,
    "turbo": turbo,
    "viridis": viridis,
    "accent": accent,
    "dark2": dark2,
    "paired": paired,
    "pastel1": pastel1,
    "pastel2": pastel2,
    "set1": set1,
    "set2": set2,
    "set3": set3,
}


@lru_cache(maxsize=None)
def _colorcet_colormap(name: str) -> ColorMap | None:
    """Build a :class:`ColorMap` from a ``colorcet`` palette, or ``None``.

    ``colorcet`` exposes ~210 perceptually-uniform palettes as lists of hex
    color strings under ``colorcet.palette``. Results are cached since the same
    colormap is typically reused across a scene.
    """
    try:
        import colorcet
    except ImportError:
        return None
    palette = colorcet.palette.get(name)
    if not palette:
        return None
    samples = np.array([Color.from_hex(h).data for h in palette])
    return ColorMap(samples)


def get_colormap(name: str) -> ColorMap | None:
    """Resolve a colormap name to a :class:`ColorMap`.

    Hakowan's curated built-in colormaps take precedence; any other name falls
    back to the ``colorcet`` package (e.g. ``"fire"``, ``"rainbow"``,
    ``"CET_L16"``, ``"glasbey"``). Returns ``None`` when the name is unknown in
    both sources.
    """
    if name in named_colormaps:
        return named_colormaps[name]
    return _colorcet_colormap(name)
