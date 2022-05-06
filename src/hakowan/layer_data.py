""" Layer data module
"""
from __future__ import annotations  # To allow type hint of the enclosing class.
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np


class AttributeProperty(Enum):
    """Attribute Property enums."""

    INTRINSIC = 1
    """ Intrinsic properties are coordinate independent.  Thus, its value will
    not be affected by coordinate transformations. """

    EXTRINSIC = 2
    """ Extrinsic properties are coordiantes dependent.  Thus, its value will be
    affected by coordinate transformtions.
    """


class Mark(Enum):
    """Mark enums."""

    POINT = 1
    CURVE = 2
    SURFACE = 3


@dataclass
class Attribute:
    """An attribute is a mapping from geometry to values.  Its values are an
    list of raw values, and its indices defines how values are mapped to each
    node in each element.
    """

    property: AttributeProperty = None
    values: np.ndarray = None
    indices: np.ndarray = None


@dataclass
class Channel: # Rename to Encoding
    """Settings of a single visual channel."""

    source: Any = None
    """ Raw data field name or value

    If `source` is a valid name, the raw data will be the attribute with
    the same name.  Otherwise, `source` is assumed to contain a constant value.
    """

    mapping: Any = "identity"
    """ Mapping between raw data and channel specific values.

    If `mapping` is a name (i.e. string), a predefined mapping of the same name
    is used.  Otherwise, `mapping` is assumed to be a callable that convert each
    raw data item into a channel value or `None` (i.e. dropped).

    Default mapping is the identity map.
    """

    normalize: bool = False
    """ Whether raw data should be normalized to [0, 1] range before applying
    the mapping.  Default is NO.
    """


@dataclass
class Channels: # Rename to Encodings
    """A set of channels"""

    position: Channel = None
    color: Channel = None
    alpha: Channel = None
    size: Channel = None

    def __or__(self, other: Channels):
        """Combine current channels with other channels.

        If a channel is defined by both, use the channel defined by `other`.

        Args:
            other (Channels): The other channels to be mserged.

        Returns:
            Channels: The merged channels.
        """
        result = Channels()
        for name in vars(self).keys:
            if name.startswith("_"):
                continue
            if getattr(other, name) is None:
                setattr(result, name, getattr(self, name))
            else:
                setattr(result, name, getattr(other, name))
        return result


@dataclass
class LayerData:
    """Data and settings associated with each layer."""

    mark: Mark = None
    channels: Channels = None
    data: dict[str, Attribute] = None
    transform: np.ndarray = None

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

        # Channels (policy: merge, and overwrite when necessary)
        if self.channels is None:
            result.channels = other.channels
        elif other.channels is None:
            result.channels = self.channels
        else:
            result.channels = self.channels | other.channels

        # Data (policy: merge, and overwrite when necessary)
        if self.data is None:
            result.data = other.data
        elif other.data is None:
            result.data = self.data
        else:
            result.data = self.data | other.data

        # Transform (policy: right multiply)
        result.transform = np.identity(4)
        if self.transform is not None:
            result.transform = np.dot(result.transform, self.transform)
        if other.transform is not None:
            result.transform = np.dot(result.transform, other.transform)

        return result
