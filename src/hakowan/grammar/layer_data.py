""" Layer data module """

from __future__ import annotations  # To allow type hint of the enclosing class.
from dataclasses import dataclass, fields, field
from enum import Enum
from typing import Callable, Union, Optional
import numpy as np
import lagrange
from pathlib import Path

from ..common.color import Color


class Mark(Enum):
    """Mark enums."""

    POINT = 1
    CURVE = 2
    SURFACE = 3


@dataclass
class Attribute:
    """An attribute defines a mapping from geometry to values."""

    values: np.ndarray
    """ An array of scalar or vectors """

    indices: np.ndarray
    """ An array of elements, where each element is defined by a set of indices
    into the `values` array.
    """


@dataclass
class ChannelSetting:
    """Visualization channel settings"""

    # Geometry channels
    position: Optional[str] = None
    normal: Optional[str] = None
    uv: Optional[str] = None

    # Material channels
    color: Optional[str] = None
    roughness: Union[float, str, None] = None
    metallic: Union[float, str, None] = None
    alpha: Union[float, str, None] = None

    # Other channels
    size: Union[str, float, None] = None

    # Scale mappings
    position_map: Union[str, Callable[..., np.ndarray], None] = None
    color_map: Union[str, Callable[..., Color], None] = None
    roughness_map: Union[str, Callable[..., float], None] = None
    metallic_map: Union[str, Callable[..., float], None] = None
    alpha_map: Union[str, Callable[..., float], None] = None
    size_map: Union[str, Callable[..., float], None] = None

    def __or__(self, other: ChannelSetting) -> ChannelSetting:
        """Merge settings defined in `self` and `other`.

        If both defines the same setting, use the one from `other`.

        Args:
            other (ChannelSetting): The other channel setting.

        Returns:
            ChannelSetting: The merged channel setting.
        """

        result = ChannelSetting()

        for field in fields(result):
            if getattr(other, field.name) is None:
                setattr(result, field.name, getattr(self, field.name))
            else:
                setattr(result, field.name, getattr(other, field.name))

        return result


@dataclass
class DataFrame:
    """3D geometry data frame."""

    mesh: lagrange.SurfaceMesh = field(default_factory=lagrange.SurfaceMesh)

    def __or__(self, other: DataFrame) -> DataFrame:
        """Merge two data frames.

        Policy: other's mesh will overwrite self's mesh.

        Args:
            other (DataFrame): The other data frame.

        Returns:
            DataFrame: The combined data frame.
        """
        result = DataFrame(mesh=other.mesh)
        return result

    def finalize(self):
        if not self.mesh.is_triangle_mesh:
            lagrange.triangulate_polygonal_facets(self.mesh)

        normal_attr_ids = self.mesh.get_matching_attribute_ids(
            element=lagrange.AttributeElement.Indexed,
            usage=lagrange.AttributeUsage.Normal
        )
        if len(normal_attr_ids) == 0:
            lagrange.compute_normal(self.mesh)

        # Clear edges to avoid warning.
        self.mesh.clear_edges()

        # This must be the last operation, so normal, uv, color and other attributes share the same
        # index buffer.
        self.mesh = lagrange.unify_index_buffer(self.mesh, [])

    @property
    def vertices(self):
        return self.mesh.vertices

    @property
    def facets(self):
        return self.mesh.facets

    @property
    def normals(self):
        normal_attr_ids = self.mesh.get_matching_attribute_ids(
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Normal
        )
        if len(normal_attr_ids) == 0:
            raise RuntimeError("Mesh does not have normal attribute.")
        else:
            return self.mesh.attribute(normal_attr_ids[0]).data


@dataclass
class Transform:
    """3D rigid body transform matrix."""

    matrix: np.ndarray = field(default_factory=lambda: np.identity(4))
    overwrite: bool = False

    def __post_init__(self):
        if self.matrix.shape == (3, 3):
            matrix = np.identity(4)
            matrix[:3, :3] = self.matrix
            self.matrix = matrix
        else:
            assert self.matrix.shape == (4, 4)

    @property
    def rotation(self):
        return self.matrix[:3, :3]

    @rotation.setter
    def rotation(self, matrix: np.ndarray):
        assert matrix.shape == (3, 3)
        self.matrix[:3, :3] = matrix

    @property
    def translation(self):
        return self.matrix[:3, 3]

    @translation.setter
    def translation(self, vector: np.ndarray):
        self.matrix[:3, 3] = vector

    def __or__(self, other: Transform) -> Transform:
        """Combine `matrix` from self with `matrix` from `other`.

        If `other.overwrite` is True, use the transform from other.
        Otherwise, the output matrix is `other.matrix * self.matrix`, i.e. other
        is left-multiplied to self.

        Args:
            other (Transform): The other transformation.
        """
        result = Transform()
        if other.overwrite:
            result.matrix = other.matrix
            result.overwrite = True
        else:
            result.matrix = np.dot(other.matrix, self.matrix)
            result.overwrite = self.overwrite
        return result


@dataclass
class LayerData:
    """Data and settings associated with each layer."""

    mark: Optional[Mark] = None
    """ The base type of visualization to use """

    channel_setting: ChannelSetting = field(default_factory=ChannelSetting)
    """ Channel setting specificiations."""

    data: Optional[DataFrame] = None
    """ A set of named data attributes, each attribute encodes a 3D geometric
    variable."""

    transform: Optional[Transform] = None
    """ Coordinate system transformation associated with this layer """

    def __or__(self, other: LayerData) -> LayerData:
        """Combine layer data in self with other.

        If a field is defined by both layers, use the one from `other`.

        Args:
            other (LayerData): The other layer data to be merged.

        Returns:
            LayerData: The merged layer data.
        """
        result = LayerData()

        # Mark (policy: overwrite)
        if other.mark is not None:
            result.mark = other.mark
        elif self.mark is not None:
            result.mark = self.mark

        # channel_setting (policy: merge, and overwrite when necessary)
        result.channel_setting = self.channel_setting | other.channel_setting

        # Data (policy: merge, and overwrite when necessary)
        if self.data is None:
            result.data = other.data
        elif other.data is None:
            result.data = self.data
        else:
            result.data = self.data | other.data

        # Transform (policy: left multiply if not overwrite)
        result.transform = None
        if self.transform is None:
            result.transform = other.transform
        elif other.transform is None:
            result.transform = self.transform
        else:
            result.transform = self.transform | other.transform

        return result
