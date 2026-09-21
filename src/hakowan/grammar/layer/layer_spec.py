"""Properties contributed by one runtime layer-tree node."""

from dataclasses import dataclass, field
from typing import Optional

from ..dataframe import DataFrame
from ..mark import Mark
from ..channel import Channel
from ..transform import Transform
from ..overlay import Annotation


@dataclass(kw_only=True, slots=True)
class LayerSpec:
    """Mutable internal data, mark, channel, transform, and overlay bundle."""

    data: Optional[DataFrame] = None
    mark: Optional[Mark] = None
    channels: list[Channel] = field(default_factory=list)
    transform: Optional[Transform] = None
    name: Optional[str] = None
    annotations: list[Annotation] = field(default_factory=list)
