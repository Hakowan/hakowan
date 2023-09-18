from ..grammar.dataframe import DataFrame
from ..grammar.scale import (
    Affine,
    Attribute,
    Clip,
    Custom,
    Log,
    Normalize,
    Offset,
    Scale,
    Uniform,
)

import lagrange
import numpy as np
import numpy.typing as npt
import numbers

### Public API


def apply_scale(df: DataFrame, attr_name: str, attr_scale: Scale):
    """Apply scale to attribute data

    :param df:         The data frame, which will be modified in place.
    :param attr_name:  Target attribute name
    :param attr_scale: Scale to apply
    """
    _apply_scale(df, attr_name, attr_scale)


def compute_scaled_attribute(df: DataFrame, attr: Attribute):
    """Compute a new attribute which is the scaled version of the original attribute.

    The scaled attribute is stored in the data frame the name `attr._internal_name`.

    :param df:   The input data frame.
    :param attr: The attribute to be scaled.
    """
    if attr.scale is not None:
        if attr._internal_name is None:
            attr._internal_name = f"_hakowan_{attr.name}"
            df.mesh.duplicate_attribute(attr.name, attr._internal_name)
            apply_scale(df, attr._internal_name, attr.scale)
    else:
        # No scale.
        attr._internal_name = attr.name


def compute_attribute_minmax(df: DataFrame, attr_name: str):
    """Compute the column-wise min and max value of an attribute.

    :param df:        The input data frame.
    :param attr_name: Target attribute name.

    :return: A tuple of (min, max) value of the attribute.
    """
    mesh = df.mesh
    assert mesh is not None
    assert mesh.has_attribute(attr_name)

    if mesh.is_attribute_indexed(attr_name):
        attr = mesh.indexed_attribute(attr_name)
        values = attr.values.data
    else:
        values = mesh.attribute(attr_name).data

    return np.amin(values, axis=0), np.amax(values, axis=0)


### Private API


def _apply_normalize(data: npt.NDArray, scale: Normalize):
    if data.ndim == 2:
        dim = data.shape[1]
    else:
        assert data.ndim == 1
        dim = 1
    assert np.all(np.isfinite(data))
    domain_min: npt.NDArray | float = (
        scale.domain_min if scale.domain_min is not None else np.amin(data, axis=0)
    )
    domain_max: npt.NDArray | float = (
        scale.domain_max if scale.domain_max is not None else np.amax(data, axis=0)
    )
    if dim > 1:
        assert isinstance(scale.bbox_min, npt.NDArray)
        assert isinstance(scale.bbox_max, npt.NDArray)
        assert isinstance(domain_min, npt.NDArray)
        assert isinstance(domain_max, npt.NDArray)
        assert dim == len(scale.bbox_min)
        assert dim == len(scale.bbox_max)
        assert dim == len(domain_min)
        assert dim == len(domain_max)
        assert np.all(scale.bbox_max >= scale.bbox_min)
        assert np.all(domain_max > domain_min)
        domain_size = domain_max - domain_min
        range_size = scale.bbox_max - scale.bbox_min
    else:
        assert dim == 1
        assert isinstance(scale.bbox_min, numbers.Number)
        assert isinstance(scale.bbox_max, numbers.Number)
        assert isinstance(domain_min, numbers.Number)
        assert isinstance(domain_max, numbers.Number)
        assert scale.bbox_max >= scale.bbox_min
        assert domain_max >= domain_min
        domain_size = domain_max - domain_min
        range_size = scale.bbox_max - scale.bbox_min

    data[:] = (data - domain_min) / domain_size * range_size + scale.bbox_min
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


def _apply_affine(data: npt.NDArray, scale: Affine):
    assert data.ndim == 2
    assert scale.matrix.ndim == 2
    dim = data.shape[1]
    M = scale.matrix

    if dim == M.shape[1]:
        data[:] = data @ M.T
    elif dim + 1 == M.shape[1]:
        data[:] = data @ M[0:dim, 0:dim].T + M[0:dim, dim].T


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
        # TODO
        pass
    else:
        # TODO
        pass


def _apply_clip(data: npt.NDArray, scale: Clip):
    assert scale.domain[0] <= scale.domain[1]
    np.clip(data, scale.domain[0], scale.domain[1], out=data)


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
            _apply_affine(mesh_attr.data, attr_scale)
        case Offset():
            _apply_offset(mesh_attr, attr_scale, df)
        case Clip():
            _apply_clip(mesh_attr.data, attr_scale)
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
            _apply_affine(indexed_attr.values.data, attr_scale)
        case Offset():
            _apply_offset(indexed_attr, attr_scale, df)
        case Clip():
            _apply_clip(indexed_attr.values.data, attr_scale)
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
