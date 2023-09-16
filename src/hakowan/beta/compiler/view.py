from dataclasses import dataclass, field
from ..grammar import dataframe as df, mark as mk, channel as ch, transform as tf


@dataclass(kw_only=True)
class View:
    data_frame: df.DataFrame | None = None
    mark: mk.Mark | None = None
    channels: list[ch.Channel] = field(default_factory=list)
    transform: tf.Transform | None = None

    _position_channel: ch.Channel | None = None
    _normal_channel: ch.Channel | None = None
    _size_channel: ch.Channel | None = None
    _uv_channel: ch.Channel | None = None
    _material_channel: ch.Channel | None = None

    def validate(self):
        """Validate the currvent view is complete.
        A view is complete if data_frame and mark are both not None
        """
        assert self.data_frame is not None, "View must have data_frame"
        assert self.mark is not None, "View must have mark"
