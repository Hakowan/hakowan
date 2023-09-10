from dataclasses import dataclass, field
from ..grammar import dataframe as df, mark as mk, channel as ch, transform as tf
from ..common import logger

import lagrange
import numpy as np


@dataclass(kw_only=True)
class View:
    data: df.DataFrame | None = None
    mark: mk.Mark | None = None
    channels: list[ch.Channel] = field(default_factory=list)
    transform: tf.Transform | None = None

    def validate(self):
        """Validate the currvent view is complete.
        A view is complete if data and mark are both not None
        """
        assert self.data is not None, "View must have data"
        assert self.mark is not None, "View must have mark"

    def apply_transform(self):
        self.__apply(self.transform)

    def __apply(self, t: tf.Transform | None):
        if t is None:
            return
        match (t):
            case tf.Filter():
                self.__filter(t)
            case _:
                raise TypeError(f"Unsupported transform: {type(t)}!")

        self.__apply(t._child)

    def __filter(self, transform: tf.Filter):
        assert self.data is not None, "View must have data"
        mesh = self.data.mesh
        attr_name = transform.data.name
        if transform.data.scale is not None:
            logger.warning("Attribute scale is ignored for applying transform.")
        assert mesh.has_attribute(
            attr_name
        ), f"Attribute {attr_name} does not exist in data"
        attr = mesh.attribute(attr_name)
        keep = [transform.condition(value) for value in attr.data]

        match (attr.element_type):
            case lagrange.AttributeElement.Facet:
                selected_facets = np.arange(mesh.num_facets, dtype=np.uint32)[keep]
                self.data.mesh = lagrange.extract_submesh(
                    mesh,
                    selected_facets=selected_facets,
                    map_attributes=True,
                )
            case _:
                raise RuntimeError(f"Unsupported element type: {attr.element_type}!")
