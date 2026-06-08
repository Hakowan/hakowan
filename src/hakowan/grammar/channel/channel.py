from dataclasses import dataclass

from typing import Literal, Optional

from .curvestyle import CurveStyle
from ..scale import AttributeLike
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
        end_type (Literal["point", "arrow", "flat"]): The type of the vector field end.
            ``"point"`` tapers the tip to zero (cone/spike); ``"arrow"`` renders a
            flared arrowhead; ``"flat"`` keeps a constant radius at both ends
            (cylinder). The default value is ``"point"``.
        normalize (bool): If True, every vector is rescaled to unit length so
            that all arrows have the same length and only encode direction.
            The magnitude is freed up to be mapped to another channel (e.g.
            ``size`` or color via ``hakowan.norm()``). Normalization is
            applied *before* any scale attached to ``data``, so a uniform scale
            on ``data`` controls the common arrow length. By default (False),
            arrow length is proportional to the vector magnitude. The default
            value is ``False``.
    """

    data: AttributeLike
    refinement_level: int = 0
    style: CurveStyle | None = None
    end_type: Literal["point", "arrow", "flat"] = "point"
    normalize: bool = False


@dataclass(slots=True)
class Covariance(Channel):
    """Covariance channel

    This class is used to specify the mapping from an attribute to the covariance matrix channel.
    The covariance channel only applies to point mark. It is represented as a per-vertex 3x3
    symmetric matrix, which defines the stretch and rotation of the point marks.

    Attributes:
        data (AttributeLike): The attribute used to encode the covariance matrix.
        full: (bool): If True, the full covariance matrix is stored in the attribute.
            If False, its "square root", M, is stored. The full covariance matrix is ∑ := M @ M^T.
            The matrix M represenst the stretch and rotation transform applied on each mark.
    """

    data: AttributeLike
    full: bool = False


@dataclass(slots=True)
class Shape(Channel):
    """Shape channel

    This class is used to specify the mapping from an attribute to the shape channel.
    This channel is only used for point mark.

    Attributes:
        base_shape (Literal["sphere", "disk", "cube"]): The base shape used to represent a point.
            The default value is ``"sphere"``.
        orientation (AttributeLike | None): The attribute used to encode the normal orientation
            of the shape. If None, orientation will be identity (i.e. normal along z-axis).
    """

    base_shape: Literal["sphere", "disk", "cube"] = "sphere"
    orientation: Optional[AttributeLike] = None


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
