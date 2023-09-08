from dataclasses import dataclass

from .scale import Scale

@dataclass(kw_only=True)
class Attribute:
    name: str
    scale: Scale | None = None
