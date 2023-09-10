from dataclasses import dataclass, field
from ..grammar import dataframe as df, mark as mk, channel as ch, transform as tf


@dataclass(kw_only=True)
class View:
    data: df.DataFrame | None = None
    mark: mk.Mark | None = None
    channels: list[ch.Channel] = field(default_factory=list)
    transform: tf.Transform | None = None
