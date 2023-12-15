from ..grammar.channel import Channel, Position, Normal, Size, VectorField
from ..grammar.channel.material import Material
from ..grammar.dataframe import DataFrame
from ..grammar.mark import Mark
from ..grammar.scale import Attribute
from ..grammar.transform import Transform
from ..common import logger

from dataclasses import dataclass, field
import lagrange
import numpy as np
import numpy.typing as npt


@dataclass(kw_only=True)
class View:
    data_frame: DataFrame | None = None
    mark: Mark | None = None
    channels: list[Channel] = field(default_factory=list)
    transform: Transform | None = None
    global_transform: npt.NDArray = field(default_factory=lambda: np.eye(4))

    _position_channel: Position | None = None
    _normal_channel: Normal | None = None
    _size_channel: Size | None = None
    _vector_field_channel: VectorField | None = None
    _material_channel: Material | None = None
    _uv_attribute: Attribute | None = None

    _active_attributes: list[Attribute] = field(default_factory=list)
    _bbox: npt.NDArray | None = None

    def initialize_bbox(self):
        assert self.data_frame is not None
        mesh = self.data_frame.mesh
        if mesh.num_vertices == 0:
            return

        bbox_min = np.amin(mesh.vertices, axis=0)
        bbox_max = np.amax(mesh.vertices, axis=0)
        self.bbox = np.stack([bbox_min, bbox_max])

    def validate(self):
        """Validate the currvent view is complete.
        A view is complete if data_frame and mark are both not None
        """
        assert self.data_frame is not None, "View must have data_frame"
        assert self.mark is not None, "View must have mark"

    def finalize(self):
        """Finalize the view by updating the data frame.

        This function will ensure all attributes are either vertex or facet attribute.
        """
        mesh = self.data_frame.mesh
        active_attribute_names = [
            attr._internal_name for attr in self._active_attributes
        ]
        active_attribute_names += [
            attr._internal_color_field
            for attr in self._active_attributes
            if attr._internal_color_field is not None
        ]

        # Drop all non-active attributes
        for attr_id in mesh.get_matching_attribute_ids():
            attr_name = mesh.get_attribute_name(attr_id)
            if attr_name not in active_attribute_names and not attr_name.startswith(
                "_hakowan"
            ):
                mesh.delete_attribute(attr_name)

        # Convert all corner attributes to indexed attributes
        for attr in self._active_attributes:
            attr_name = attr._internal_name
            assert attr_name is not None
            if mesh.is_attribute_indexed(attr_name):
                continue

            mesh_attr = mesh.attribute(attr_name)
            if mesh_attr.element_type == lagrange.AttributeElement.Corner:
                attr_id = lagrange.map_attribute_in_place(
                    mesh, attr_name, lagrange.AttributeElement.Indexed
                )
                lagrange.weld_indexed_attribute(mesh, attr_id)

        # Gather all indexed attributes
        indexed_attr_names = []
        for attr in self._active_attributes:
            attr_name = attr._internal_name
            if mesh.is_attribute_indexed(attr_name):
                indexed_attr_names.append(attr_name)
            if attr._internal_color_field is not None and mesh.is_attribute_indexed(
                attr._internal_color_field
            ):
                indexed_attr_names.append(attr._internal_color_field)

        # Unify all active index buffers.
        if len(indexed_attr_names) > 0:
            unified_mesh = lagrange.unify_index_buffer(mesh, indexed_attr_names)
        else:
            unified_mesh = mesh

        # Update mesh vertices to the scaled version if needed.
        if (
            self._position_channel is not None
            and self._position_channel.data._internal_name is not None
        ):
            position_attr_name = self._position_channel.data._internal_name
            if position_attr_name != unified_mesh.attr_name_vertex_to_position:
                unified_mesh.vertices = unified_mesh.attribute(
                    position_attr_name
                ).data.copy()
                unified_mesh.delete_attribute(position_attr_name)
                self._position_channel.data._internal_name = (
                    unified_mesh.attr_name_vertex_to_position
                )

        uv_ids = unified_mesh.get_matching_attribute_ids(
            usage=lagrange.AttributeUsage.UV
        )
        assert len(uv_ids) <= 1, "At most one UV attribute is allowed"
        self.data_frame.mesh = unified_mesh

    @property
    def position_channel(self) -> Position | None:
        return self._position_channel

    @position_channel.setter
    def position_channel(self, channel: Position):
        assert self.data_frame is not None
        assert isinstance(channel, Position)
        if isinstance(channel.data, str):
            channel.data = Attribute(name=channel.data)
        assert isinstance(channel.data, Attribute)
        attr = channel.data
        mesh = self.data_frame.mesh
        assert mesh.has_attribute(attr.name)
        if mesh.is_attribute_indexed(attr.name):
            position_attr = mesh.indexed_attribute(attr.name)
        else:
            position_attr = mesh.attribute(attr.name)

        assert position_attr.num_channels == mesh.dimension
        self._position_channel = channel

    @property
    def normal_channel(self) -> Normal | None:
        return self._normal_channel

    @normal_channel.setter
    def normal_channel(self, channel: Normal):
        assert self.data_frame is not None
        assert isinstance(channel, Normal)
        if isinstance(channel.data, str):
            channel.data = Attribute(name=channel.data)
        assert isinstance(channel.data, Attribute)
        attr = channel.data
        mesh = self.data_frame.mesh
        assert mesh.has_attribute(attr.name)
        if mesh.is_attribute_indexed(attr.name):
            normal_attr = mesh.indexed_attribute(attr.name)
        else:
            normal_attr = mesh.attribute(attr.name)

        assert normal_attr.num_channels == mesh.dimension
        self._normal_channel = channel

    @property
    def size_channel(self) -> Size | None:
        return self._size_channel

    @size_channel.setter
    def size_channel(self, channel: Size):
        assert isinstance(channel, Size)

        if isinstance(channel.data, str):
            channel.data = Attribute(name=channel.data)
        assert isinstance(channel.data, (Attribute, float))

        match (channel.data):
            case Attribute():
                assert self.data_frame is not None
                attr = channel.data
                mesh = self.data_frame.mesh
                assert mesh.has_attribute(attr.name)
                if mesh.is_attribute_indexed(attr.name):
                    size_attr = mesh.indexed_attribute(attr.name)
                else:
                    size_attr = mesh.attribute(attr.name)

                assert size_attr.num_channels == 1

        self._size_channel = channel

    @property
    def vector_field_channel(self) -> VectorField | None:
        return self._vector_field_channel

    @vector_field_channel.setter
    def vector_field_channel(self, channel: VectorField):
        assert isinstance(channel, VectorField)
        if isinstance(channel.data, str):
            channel.data = Attribute(name=channel.data)
        assert isinstance(channel.data, Attribute)
        self._vector_field_channel = channel

    @property
    def material_channel(self) -> Material | None:
        return self._material_channel

    @material_channel.setter
    def material_channel(self, channel: Material):
        assert isinstance(channel, Material)
        self._material_channel = channel

    @property
    def uv_attribute(self) -> Attribute | None:
        return self._uv_attribute

    @uv_attribute.setter
    def uv_attribute(self, attribute: Attribute | None):
        if attribute is None:
            return
        assert isinstance(attribute, Attribute)
        if self._uv_attribute is not None and self._uv_attribute != attribute:
            raise ValueError("UV attribute can only be set once.")

        self._uv_attribute = attribute

    @property
    def bbox(self) -> npt.NDArray | None:
        return self._bbox

    @bbox.setter
    def bbox(self, value: npt.NDArray):
        self._bbox = value
