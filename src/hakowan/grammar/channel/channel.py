from dataclasses import dataclass

from typing import Optional, Union

from .curvestyle import CurveStyle
from ..scale import Attribute, AttributeLike


@dataclass(kw_only=True, slots=True)
class Channel:
    """Channel base class."""
    pass


@dataclass(slots=True)
class Position(Channel):
    """Position channel

    This class is used to specify the mapping from an attribute to the position channel.
    Note that, by default, the vertex coordinates of the data frame is used as the position
    channel. Thus, this class is mainly useful when we want to use non-vertex-coordinates as the
    position channel. For example, this method can be used for visualizing a deformed shape when
    the deformed position is stored as a vertex attribute in the data frame.

    Attributes:
        data (AttributeLike): The attribute used to encode the position field.
    """

    data: AttributeLike


@dataclass(slots=True)
class Normal(Channel):
    """Normal channel

    This class is used to specify the mapping from an attribute to the normal channel.
    By default, Hakowan will automatically compute the normal field from the geometry if normal
    channel is not specified. This class is useful for ensure the visualization uses a pre-defined
    normal field.

    Attributes:
        data (AttributeLike): The attribute used to encode the normal field.
    """

    data: AttributeLike


@dataclass(slots=True)
class Size(Channel):
    """Size channel

    This class is used to specify the mapping from an attribute or value to the size channel. If a
    value is used, all elements will have the same size. Note that size is defined in the same unit
    as the input geometry.

    Attributes:
        data (AttributeLike | float): The attribute or value used to encode the size field.
    """

    data: AttributeLike | float


@dataclass(slots=True)
class VectorField(Channel):
    """Vector field channel

    This class is used to specify the mapping from an attribute to the vector field channel.

    A vector field can be define over the vertices or facets of the geometry. The vector field must
    have the same dimension as the geometry.

    Attributes:
        data (AttributeLike): The attribute used to encode the vector field.
        refinement_level (int): The refinement level of the vector field. This parameter is used to
            control the density of the vector field. The default value is 0.
        style (CurveStyle | None): The style of the vector field. If None, the default style will
            be used.
        end_type (str): The type of the vector field end. The default value is "point".
    """

    data: AttributeLike
    refinement_level: int = 0
    style: CurveStyle | None = None
    end_type: str = "point"
