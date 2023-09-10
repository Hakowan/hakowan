from dataclasses import dataclass, field
from ..grammar import dataframe as df, mark as mk, channel as ch, transform as tf


@dataclass(kw_only=True)
class View:
    data: df.DataFrame | None = None
    mark: mk.Mark | None = None
    channels: list[ch.Channel] = field(default_factory=list)
    transform: tf.Transform | None = None

    def validate(self):
        """ Validate the currvent view is complete.
        A view is complete if data and mark are both not None
        """
        assert self.data is not None, "View must have data"
        assert self.mark is not None, "View must have mark"

