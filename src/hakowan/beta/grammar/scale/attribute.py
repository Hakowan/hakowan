from dataclasses import dataclass

from .scale import Scale


@dataclass(kw_only=True)
class Attribute:
    name: str
    scale: Scale | None = None

    # (internal) The name of the attribute with scale applied.
    _internal_name: str | None = None

    # (internal) The name of the attribute representing the color field mapped from the scaled attribute.
    _internal_color_field: str | None = None
