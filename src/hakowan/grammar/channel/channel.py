from dataclasses import dataclass

from typing import Optional, Union

from .curvestyle import CurveStyle
from ..scale import Attribute, AttributeLike
from ..texture import TextureLike


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


@dataclass(slots=True)
class Covariance(Channel):
    """ Covariance channel

    This class is used to specify the mapping from an attribute to the covariance matrix channel.
    The covariance channel only applies to point mark. It is represented as a per-vertex 3x3
    symmetric matrix, which defines the stretch and rotation of the point marks.

    Attributes:
        data (AttributeLike): The attribute used to encode the covariance matrix.
        full: (bool): If True, the full covariance matrix is stored in the attribute.
            If False, its "square root", M, is stored. The full covariance matrix is ∑ := M @ M^T.
            The matrix M represenst the stretch and rotation transform applied on each mark.
        base_shape (str): The base shape of the covariance matrix. Options include "sphere" and
            "cube". The default value is "sphere".
    """

    data: AttributeLike
    full: bool = False
    base_shape: str = "sphere"


@dataclass(slots=True)
class BumpMap(Channel):
    """Bump map channel

    This class specifies the bump map channel.

    Attributes:
        texture (TextureLike | None): The texture used to encode the bump map.
        scale (float): The scale of the bump map. The default value is 1.0.
    """

    texture: TextureLike
    scale: float = 1.0

@dataclass(slots=True)
class NormalMap(Channel):
    """Normal map channel

    This class specifies the normal map channel.

    Attributes:
        texture (TextureLike | None): The texture used to encode the normal map.
    """

    texture: TextureLike
