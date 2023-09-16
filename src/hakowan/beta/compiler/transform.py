from .view import View
from ..grammar.dataframe import DataFrame
from ..grammar.transform import Transform, Filter
from ..common import logger

import lagrange
import numpy as np


def _apply_filter_transform(data: DataFrame, transform: Filter):
    """Filter the data based on attribute value and condition specified in the transform."""
    assert data is not None
    assert transform is not None
    mesh = data.mesh
    attr_name = transform.data.name
    if transform.data.scale is not None:
        logger.warning("Attribute scale is ignored when applying transform.")
    assert mesh.has_attribute(
        attr_name
    ), f"Attribute {attr_name} does not exist in data"
    attr = mesh.attribute(attr_name)
    keep = [transform.condition(value) for value in attr.data]

    match (attr.element_type):
        case lagrange.AttributeElement.Facet:
            selected_facets = np.arange(mesh.num_facets, dtype=np.uint32)[keep]
            data.mesh = lagrange.extract_submesh(
                mesh,
                selected_facets=selected_facets,
                map_attributes=True,
            )
        case _:
            raise RuntimeError(f"Unsupported element type: {attr.element_type}!")


def apply_transform(view: View):
    """Apply a chain of transforms specified by view.transform to view.data.
    Transforms are applied in the order specified by the chain.
    """

    def _apply(t: Transform | None):
        if t is None:
            return
        match (t):
            case Filter():
                assert view.data is not None
                _apply_filter_transform(view.data, t)
            case _:
                raise NotImplementedError(f"Unsupported transform: {type(t)}!")

        _apply(t._child)

    _apply(view.transform)
