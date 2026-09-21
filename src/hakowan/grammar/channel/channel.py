"""Visual channel models for positions, glyphs, fields, and maps."""

from dataclasses import dataclass

from typing import Literal, Optional

from .curvestyle import CurveStyle
from ..scale import AttributeLike
from ..texture import TextureLike


# Fallback sizes (in geometry units) used by every rendering backend when a
# :class:`Size` channel is not specified. Shared so all backends agree on the
# default glyph size for point/curve marks.
DEFAULT_MARK_SIZE = 0.01
DEFAULT_COVARIANCE_SIZE = 1.0


@dataclass(kw_only=True, slots=True)
class Channel:
    """Channel base class."""

    pass


@dataclass(slots=True)
class Position(Channel):
    """Map a geometry-dimensional attribute to rendered positions.

    Without this channel, source mesh vertices supply positions. Use it for
    deformed or alternate position attributes.

    Attributes:
        data (AttributeLike): The attribute used to encode the position field.

    """

    data: AttributeLike


@dataclass(slots=True)
class Normal(Channel):
    """Map a geometry-dimensional attribute to surface normals.

    Without this channel, Hakowan computes normals from geometry.

    Attributes:
        data (AttributeLike): The attribute used to encode the normal field.

    """

    data: AttributeLike


@dataclass(slots=True)
class Size(Channel):
    """Map a scalar attribute or constant to point/curve glyph size.

    Sizes use the input geometry's units.

    Attributes:
        data (AttributeLike | float): The attribute or value used to encode the size field.

    """

    data: AttributeLike | float


@dataclass(slots=True)
class VectorField(Channel):
    """Render a vertex- or facet-domain vector attribute as curve glyphs.

    The vector must have the same dimension as the source geometry.

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
    """Map per-point 3x3 covariance data to anisotropic point glyphs.

    ``full=True`` interprets the attribute as covariance; otherwise values are
    its square-root transform ``M`` where covariance is ``M @ M.T``.

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
    """Select and optionally orient sphere, disk, or cube point glyphs.

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
    """Perturb surface shading normals with a scalar bump texture.

    Attributes:
        texture (TextureLike): The texture used to encode the bump map.
        scale (float): The scale of the bump map. The default value is 1.0.

    """

    texture: TextureLike
    scale: float = 1.0


@dataclass(slots=True)
class NormalMap(Channel):
    """Perturb surface shading normals with a tangent-space normal texture.

    Attributes:
        texture (TextureLike): The texture used to encode the normal map.

    """

    texture: TextureLike
