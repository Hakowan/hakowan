from ..grammar.dataframe import DataFrame
from ..grammar.scale import Attribute, Normalize, Log, Uniform, Custom, Affine, Offset

import lagrange
import numpy as np
import numpy.typing as npt


def _apply_normalize(data: npt.NDArray, scale: Normalize):
    assert data.ndim == 2
    dim = data.shape[1]
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
        assert isinstance(scale.bbox_min, float)
        assert isinstance(scale.bbox_max, float)
        assert isinstance(scale._value_min, float)
        assert isinstance(scale._value_max, float)
        assert scale.bbox_max >= scale.bbox_min
        assert scale._value_max > scale._value_min
        domain_size = scale._value_max - scale._value_min
        range_size = scale.bbox_max - scale.bbox_min

    data = (data - scale._value_min) / domain_size * range_size + scale.bbox_min
    assert np.all(np.isfinite(data))


def _apply_log(data: npt.NDArray, scale: Log):
    assert data.ndim == 1 or data.shape[1] == 1
    match scale.base:
        case np.e:
            data = np.log(data)
        case 2:
            data = np.log2(data)
        case 10:
            data = np.log10(data)
        case _:
            assert scale.base > 1
            data = np.log2(data) / np.log2(scale.base)


def _apply_uniform(data: npt.NDArray, scale: Uniform):
    data *= scale.factor


def _apply_custom(data: npt.NDArray, scale: Custom):
    for i, entry in enumerate(data):
        data[i] = scale.function(entry)


def _apply_offset(
    target_attr: lagrange.Attribute | lagrange.IndexedAttribute,
    scale: Offset,
    mesh: lagrange.SurfaceMesh,
):
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


def _apply_scale_attribute(mesh, attr: Attribute):
    mesh_attr = mesh.attribute(attr.name)
    scale = attr.scale

    match scale:
        case Normalize():
            _apply_normalize(mesh_attr.data, scale)
        case Log():
            _apply_log(mesh_attr.data, scale)
        case Uniform():
            _apply_uniform(mesh_attr.data, scale)
        case Custom():
            _apply_custom(mesh_attr.data, scale)
        case Affine():
            raise NotImplementedError("Affine scale is not supported")
            pass
        case Offset():
            _apply_offset(mesh_attr, scale, mesh)
        case _:
            raise NotImplementedError(f"Scale type {type(scale)} is not supported")


def _apply_scale_indexed_attribute(mesh, attr: Attribute):
    indexed_attr = mesh.indexed_attribute(attr.name)
    scale = attr.scale

    match scale:
        case Normalize():
            _apply_normalize(indexed_attr.values.data, scale)
        case Log():
            _apply_log(indexed_attr.values.data, scale)
        case Uniform():
            _apply_uniform(indexed_attr.values.data, scale)
        case Custom():
            _apply_custom(indexed_attr.values.data, scale)
        case Affine():
            raise NotImplementedError("Affine scale is not supported")
            pass
        case Offset():
            _apply_offset(indexed_attr, scale, mesh)
        case _:
            raise NotImplementedError(f"Scale type {type(scale)} is not supported")


def apply_scale(data: DataFrame, attr: Attribute):
    mesh = data.mesh
    assert mesh.has_attribute(
        attr.name
    ), f"Attribute {attr.name} does not exist in the mesh"

    if mesh.is_attribute_indexed(attr.name):
        _apply_scale_indexed_attribute(mesh, attr)
    else:
        _apply_scale_attribute(mesh, attr)
