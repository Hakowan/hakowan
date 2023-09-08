from dataclasses import dataclass
from typing import Optional

from .scale import Scale

@dataclass(kw_only=True)
class Attribute:
    name: str
    scale: Optional[Scale]
