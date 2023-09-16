from ..grammar.dataframe import DataFrame
from ..grammar.scale import (
    Attribute,
    Normalize,
    Log,
    Uniform,
    Custom,
    Affine,
    Offset,
    Scale,
)

import lagrange
import numpy as np
import numpy.typing as npt
import numbers


### Public API


def update_scale(data: DataFrame, attr_name: str, attr_scale: Scale):
    """Update scale with attribute data

    Some scale has parameters that depends on the data. This method compute such data.

    :param data:       The input data frame.
    :param attr_name:  Target attribute name.
    :param attr_scale: Scale to be updated in place.
    """
    _update_scale(data, attr_name, attr_scale)


def apply_scale(data: DataFrame, attr_name: str, attr_scale: Scale):
    """Apply scale to attribute data

    :param data:       The data frame, which will be modified in place.
    :param attr_name:  Target attribute name
    :param attr_scale: Scale to apply
    """
    _apply_scale(data, attr_name, attr_scale)


### Private API


def _update_normalize_scale(data: DataFrame, attr_name: str, attr_scale: Normalize):
    mesh = data.mesh
    assert mesh is not None
    assert mesh.has_attribute(attr_name)

    if mesh.is_attribute_indexed(attr_name):
        pass
    else:
        attr_data = mesh.attribute(attr_name).data
        attr_scale._value_min = (
            np.amin(attr_data, axis=0)
            if attr_scale._value_min is None
            else np.minimum(attr_scale._value_min, np.amin(attr_data, axis=0))
        )
        attr_scale._value_max = (
            np.amax(attr_data, axis=0)
            if attr_scale._value_max is None
            else np.maximum(attr_scale._value_max, np.amax(attr_data, axis=0))
        )


def _update_scale(data: DataFrame, attr_name: str, attr_scale: Scale):
    match attr_scale:
        case Normalize():
            _update_normalize_scale(data, attr_name, attr_scale)

    if attr_scale._child is not None:
        update_scale(data, attr_name, attr_scale._child)


def _apply_normalize(data: npt.NDArray, scale: Normalize):
    if data.ndim == 2:
        dim = data.shape[1]
    else:
        assert data.ndim == 1
        dim = 1
    assert np.all(np.isfinite(data))
    assert scale._value_min is not None
    assert scale._value_max is not None
    if dim > 1:
        assert isinstance(scale.bbox_min, npt.NDArray)
        assert isinstance(scale.bbox_max, npt.NDArray)
        assert isinstance(scale._value_min, npt.NDArray)
        assert isinstance(scale._value_max, npt.NDArray)
        assert dim == len(scale.bbox_min)
        assert dim == len(scale.bbox_max)
        assert dim == len(scale._value_min)
        assert dim == len(scale._value_max)
        assert np.all(scale.bbox_max >= scale.bbox_min)
        assert np.all(scale._value_max > scale._value_min)
        domain_size = scale._value_max - scale._value_min
        range_size = scale.bbox_max - scale.bbox_min
    else:
        assert dim == 1
        assert isinstance(scale.bbox_min, numbers.Number)
        assert isinstance(scale.bbox_max, numbers.Number)
        assert isinstance(scale._value_min, numbers.Number)
        assert isinstance(scale._value_max, numbers.Number)
        assert scale.bbox_max >= scale.bbox_min
        assert scale._value_max > scale._value_min
        domain_size = scale._value_max - scale._value_min
        range_size = scale.bbox_max - scale.bbox_min

    data[:] = (data - scale._value_min) / domain_size * range_size + scale.bbox_min
    assert np.all(np.isfinite(data))


def _apply_log(data: npt.NDArray, scale: Log):
    assert not np.issubdtype(
        data.dtype, np.integer
    ), "Log scale cannot be applied to integer data"
    assert data.ndim == 1 or data.shape[1] == 1
    match scale.base:
        case np.e:
            data[:] = np.log(data)
        case 2:
            data[:] = np.log2(data)
        case 10:
            data[:] = np.log10(data)
        case _:
            assert scale.base > 1
            data[:] = np.log2(data) / np.log2(scale.base)


def _apply_uniform(data: npt.NDArray, scale: Uniform):
    data *= np.cast[data.dtype](scale.factor)


def _apply_custom(data: npt.NDArray, scale: Custom):
    for i, entry in enumerate(data):
        data[i] = scale.function(entry)


def _apply_offset(
    target_attr: lagrange.Attribute | lagrange.IndexedAttribute,
    scale: Offset,
    df: DataFrame,
):
    if scale.offset.scale is not None:
        apply_scale(df, scale.offset.name, scale.offset.scale)

    mesh = df.mesh
    assert mesh.has_attribute(scale.offset.name)

    target_is_indexed = target_attr.element_type == lagrange.AttributeElement.Indexed
    offset_is_indexed = mesh.is_attribute_indexed(scale.offset.name)

    if not target_is_indexed and not offset_is_indexed:
        assert isinstance(target_attr, lagrange.Attribute)
        offset_attr = mesh.attribute(scale.offset.name)
        assert target_attr.element_type == offset_attr.element_type
        assert target_attr.num_channels == offset_attr.num_channels
        target_attr.data += offset_attr.data
    elif target_is_indexed and offset_is_indexed:
        assert isinstance(target_attr, lagrange.IndexedAttribute)
        offset_attr = mesh.attribute(scale.offset.name)
        target_attr.values[target_attr.indices] += offset_attr.values[
            offset_attr.indices
        ]
    elif target_is_indexed:
        pass
    else:
        pass


def _apply_scale_attribute(df: DataFrame, attr_name: str, attr_scale: Scale):
    mesh = df.mesh
    mesh_attr = mesh.attribute(attr_name)

    match attr_scale:
        case Normalize():
            _apply_normalize(mesh_attr.data, attr_scale)
        case Log():
            _apply_log(mesh_attr.data, attr_scale)
        case Uniform():
            _apply_uniform(mesh_attr.data, attr_scale)
        case Custom():
            _apply_custom(mesh_attr.data, attr_scale)
        case Affine():
            raise NotImplementedError("Affine scale is not supported")
            pass
        case Offset():
            _apply_offset(mesh_attr, attr_scale, df)
        case _:
            raise NotImplementedError(f"Scale type {type(attr_scale)} is not supported")


def _apply_scale_indexed_attribute(df: DataFrame, attr_name: str, attr_scale: Scale):
    mesh = df.mesh
    indexed_attr = mesh.indexed_attribute(attr_name)

    match attr_scale:
        case Normalize():
            _apply_normalize(indexed_attr.values.data, attr_scale)
        case Log():
            _apply_log(indexed_attr.values.data, attr_scale)
        case Uniform():
            _apply_uniform(indexed_attr.values.data, attr_scale)
        case Custom():
            _apply_custom(indexed_attr.values.data, attr_scale)
        case Affine():
            raise NotImplementedError("Affine scale is not supported")
            pass
        case Offset():
            _apply_offset(indexed_attr, attr_scale, df)
        case _:
            raise NotImplementedError(f"Scale type {type(attr_scale)} is not supported")


def _apply_scale(df: DataFrame, attr_name: str, attr_scale: Scale):
    mesh = df.mesh
    assert mesh.has_attribute(
        attr_name
    ), f"Attribute {attr_name} does not exist in the mesh"

    if mesh.is_attribute_indexed(attr_name):
        _apply_scale_indexed_attribute(df, attr_name, attr_scale)
    else:
        _apply_scale_attribute(df, attr_name, attr_scale)

    if attr_scale._child is not None:
        apply_scale(df, attr_name, attr_scale._child)
