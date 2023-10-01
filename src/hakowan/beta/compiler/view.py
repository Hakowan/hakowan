from ..grammar.channel import Channel
from ..grammar.dataframe import DataFrame
from ..grammar.mark import Mark
from ..grammar.scale import Attribute
from ..grammar.transform import Transform
from dataclasses import dataclass, field

import lagrange


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

    _active_attributes: list[Attribute] = field(default_factory=list)

    def validate(self):
        """Validate the currvent view is complete.
        A view is complete if data_frame and mark are both not None
        """
        assert self.data_frame is not None, "View must have data_frame"
        assert self.mark is not None, "View must have mark"

    def finalize(self):
        """ Finalize the view by updating the data frame.

        This function will ensure all attributes are either vertex or facet attribute.
        """
        mesh = self.data_frame.mesh

        # Convert all corner attributes to indexed attributes
        for attr in self._active_attributes:
            attr_name = attr._internal_name
            if mesh.is_attribute_indexed(attr_name):
                continue

            mesh_attr = mesh.attribute(attr_name)
            if mesh_attr.element_type == lagrange.AttributeElement.Corner:
                attr_id = lagrange.map_attribute_in_plane(
                    mesh, attr_name, lagrange.AttributeElement.Indexed
                )
                lagrange.weld_indexed_attribute(mesh, attr_id)

        # Gather all indexed attributes
        indexed_attr_names = []
        for attr in self._active_attributes:
            attr_name = attr._internal_name
            if mesh.is_attribute_indexed(attr_name):
                indexed_attr_names.append(attr_name)

        # Unify all active index buffers.
        unified_mesh = lagrange.unify_index_buffer(mesh, indexed_attr_names)

        self.data_frame.mesh = unified_mesh
