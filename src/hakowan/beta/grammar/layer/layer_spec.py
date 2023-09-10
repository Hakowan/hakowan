from dataclasses import dataclass, field
from typing import Optional

from ..dataframe import DataFrame
from ..mark import Mark
from ..channel import Channel
from ..transform import Transform

@dataclass(kw_only=True)
class LayerSpec:
    data: Optional[DataFrame] = None
    mark: Optional[Mark] = None
    channels: list[Channel] = field(default_factory=list)
    transform: Optional[Transform] = None
