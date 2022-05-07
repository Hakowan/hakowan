""" Layer data module """

from __future__ import annotations  # To allow type hint of the enclosing class.
from dataclasses import dataclass, fields
from enum import Enum
from typing import Callable, Union
import numpy as np
from csscolor.color import Color


class Mark(Enum):
    """Mark enums."""

    POINT = 1
    CURVE = 2
    SURFACE = 3


@dataclass
class Attribute:
    """An attribute defines a mapping from geometry to values."""

    values: np.ndarray = None
    """ An array of scalar or vectors """

    indices: np.ndarray = None
    """ An array of elements, where each element is defined by a set of indices
    into the `values` array.
    """


@dataclass
class ChannelSetting:
    """Visualization channel settings"""

    # Channal source data.
    position: str = None
    color: str = None
    size: Union[str, float] = None
    alpha: Union[str, float] = None

    # Channel-specific mapping.
    position_map: Union[str, Callable[..., np.ndarray]] = None
    color_map: Union[str, Callable[..., Color]] = None
    size_map: Union[str, Callable[..., float]] = None
    alpha_map: Union[str, Callable[..., float]] = None

    # Material
    material: str = None

    def __or__(self, other: ChannelSetting):
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


@dataclass
class DataFrame:
    """3D geometry data frame."""

    attributes: dict[str, Attribute] = None

    def __post_init__(self):
        # Initialize to default dict to avoid attributes being shared by
        # multiple data frames.
        if self.attributes is None:
            self.attributes = {}

    def __or__(self, other: DataFrame):
        """ Merge two data frames.

        If a field is defined by both, use the one from `other`.

        Args:
            other (DataFrame): The other data frame.

        Returns:
            DataFrame: The combined data frame.
        """
        result = DataFrame()
        result.attributes = self.attributes | other.attributes
        return result

    # The following are some of the common attributes with reserved attribute
    # names.

    @property
    def geometry(self):
        """Indexed vertex positions."""
        return self.attributes.get("@geometry", None)

    @property
    def uv(self):
        """Indexed UV coordinates. (optional)"""
        return self.attributes.get("@uv", None)

    @property
    def normal(self):
        """Indexed normal attribute. (optional)"""
        return self.attributes.get("@normal", None)

    @geometry.setter
    def geometry(self, attr: Attribute):
        self.attributes["@geometry"] = attr

    @uv.setter
    def uv(self, attr: Attribute):
        self.attributes["@uv"] = attr

    @normal.setter
    def normal(self, attr: Attribute):
        self.attributes["@normal"] = attr


@dataclass
class Transform:
    """3D rigid body transform matrix."""

    matrix: np.ndarray = None
    overwrite: bool = False

    def __post_init__(self):
        if self.matrix is None:
            self.matrix = np.identity(4, dtype=float)
        elif self.matrix.shape == (3, 3):
            matrix = np.identity(4)
            matrix[:3, :3] = self.matrix
            self.matrix = matrix
        else:
            assert self.matrix.shape == (4, 4)

    @property
    def rotation(self):
        return self.matrix[:3, :3]

    @property
    def translation(self):
        return self.matrix[:3, 3]

    @rotation.setter
    def rotation(self, matrix: np.ndarray):
        assert matrix.shape == (3, 3)
        self.matrix[:3, :3] = matrix

    @translation.setter
    def translation(self, vector: np.ndarray):
        self.matrix[:3, 3] = vector

    def __or__(self, other: Transform):
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

    mark: Mark = None
    """ The base type of visualization to use """

    channel_setting: ChannelSetting = None
    """ Channel setting specificiations."""

    data: DataFrame = None
    """ A set of named data attributes, each attribute encodes a 3D geometric
    variable."""

    transform: Transform = None
    """ Coordinate system transformation associated with this layer """

    def __or__(self, other: LayerData):
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
        if self.channel_setting is None:
            result.channel_setting = other.channel_setting
        elif other.channel_setting is None:
            result.channel_setting = self.channel_setting
        else:
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
