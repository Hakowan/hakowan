from ..grammar.channel import Channel
from ..grammar.dataframe import DataFrame
from ..grammar.mark import Mark
from ..grammar.scale import Attribute
from ..grammar.transform import Transform
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class View:
    data_frame: DataFrame | None = None
    mark: Mark | None = None
    channels: list[Channel] = field(default_factory=list)
    transform: Transform | None = None

    _position_channel: Channel | None = None
    _normal_channel: Channel | None = None
    _size_channel: Channel | None = None
    _material_channel: Channel | None = None
    _uv_attribute: Attribute | None = None

    def validate(self):
        """Validate the currvent view is complete.
        A view is complete if data_frame and mark are both not None
        """
        assert self.data_frame is not None, "View must have data_frame"
        assert self.mark is not None, "View must have mark"
