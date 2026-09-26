"""Declarative surface and hair material channels."""

from dataclasses import dataclass
from typing import Literal

from .medium import Medium
from ..channel import Channel
from ...texture import ScalarTextureLike, TextureLike
from ....common.color import ColorLike


@dataclass(kw_only=True, slots=True)
class Material(Channel):
    """Material base class.

    Attributes:
        two_sided: Whether to render both sides of the surface (default: False).
        back_side: An independent material for the back face of each facet
            (default: None). When set, the front face uses this material's
            owning material and the back face uses ``back_side``; this implies
            two-sided rendering regardless of ``two_sided``. Only meaningful for
            surface marks. A nested ``back_side`` on the back material itself is
            ignored.

    """

    two_sided: bool = False
    back_side: "Material | None" = None


@dataclass(slots=True)
class Diffuse(Material):
    """Diffuse material.

    Attributes:
        reflectance: Diffuse reflectance (i.e. base color) texture (default: 0.5).

    """

    reflectance: TextureLike = 0.5


@dataclass(slots=True)
class Conductor(Material):
    """Conductor material.

    Attributes:
        material: Conductor material name based on [Mitsuba preset](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#conductor-ior-list).

    """

    material: str


@dataclass(slots=True)
class RoughConductor(Conductor):
    """Rough conductor material.

    Attributes:
        distribution (Literal["beckmann", "ggx", "phong"]): Microfacet distribution
            (default: ``"beckmann"``).
        alpha: Roughness value (default: 0.1).

    """

    distribution: Literal["beckmann", "ggx", "phong"] = "beckmann"
    alpha: ScalarTextureLike = 0.1


@dataclass(slots=True)
class Plastic(Material):
    """Plastic material.

    Attributes:
        diffuse_reflectance: Diffuse reflectance (i.e. base color) texture (default: 0.5).
        specular_reflectance: Specular reflectance texture (default: 1.0).

    """

    diffuse_reflectance: TextureLike = 0.5
    specular_reflectance: ScalarTextureLike = 1.0


@dataclass(slots=True)
class RoughPlastic(Plastic):
    """Rough plastic material.

    Attributes:
        distribution (Literal["beckmann", "ggx", "phong"]): Microfacet distribution
            (default: ``"beckmann"``).
        alpha: Roughness value (default: 0.1).

    """

    distribution: Literal["beckmann", "ggx", "phong"] = "beckmann"
    alpha: float = 0.1


@dataclass(slots=True)
class Principled(Material):
    """Principled material.

    Attributes:
        color: Base color texture (default: 0.5).
        roughness: Roughness texture (default: 0.5).
        metallic: Metallic texture (default: 0.0).
        anisotropic: Anisotropy amount (default: 0.0).
        spec_trans: Specular transmission / transparency (default: 0.0).
        eta: Interior index of refraction (default: 1.5).
        spec_tint: Specular tint towards base color (default: 0.0).
        sheen: Sheen amount (default: 0.0).
        sheen_tint: Sheen tint towards base color (default: 0.0).
        flatness: Blend between thin and volumetric subsurface scattering (default: 0.0).

    """

    color: TextureLike = 0.5
    roughness: ScalarTextureLike = 0.5
    metallic: ScalarTextureLike = 0.0
    anisotropic: float = 0.0
    spec_trans: float = 0.0
    eta: float = 1.5
    spec_tint: float = 0.0
    sheen: float = 0.0
    sheen_tint: float = 0.0
    flatness: float = 0.0


@dataclass(slots=True)
class ThinPrincipled(Principled):
    """Thin Principled material.

    Attributes:
        diff_trans: Diffuse transmission amount (default: 0.0).

    """

    diff_trans: float = 0.0


@dataclass(slots=True)
class Dielectric(Material):
    """Dielectric material.

    Attributes:
        int_ior: Interior index of refraction (default: "bk7").
        ext_ior: Exterior index of refraction (default: "air").
        medium: Medium (default: None).
        specular_reflectance: Specular reflectance (default: 1.0).
        specular_transmittance: Specular transmittance (default: 1.0).

    """

    int_ior: str | float = "bk7"
    ext_ior: str | float = "air"
    medium: Medium | None = None
    specular_reflectance: float = 1.0
    specular_transmittance: float = 1.0


@dataclass(slots=True)
class ThinDielectric(Dielectric):
    """Thin dielectric material."""

    pass


@dataclass(slots=True)
class RoughDielectric(Dielectric):
    """Rough dielectric material.

    Attributes:
        distribution (Literal["beckmann", "ggx", "phong"]): Microfacet distribution
            (default: ``"beckmann"``).
        alpha: Roughness value (default: 0.1).

    """

    distribution: Literal["beckmann", "ggx", "phong"] = "beckmann"
    alpha: ScalarTextureLike = 0.1


@dataclass(slots=True)
class Hair(Material):
    """Hair material.

    Color is controlled either physically, via the two melanin pigment
    concentrations (natural hair palette: black → brown → red → blonde), or
    directly via ``color`` with an RGB / named color for uniform fur of any
    hue.  ``color`` overrides the melanin parametrization when set.

    For richer fur, colors can be mixed procedurally along and across strands:
    ``root_color`` / ``tip_color`` give a root-to-tip gradient (dark undercoat →
    lighter tips), and ``color_variation`` adds per-strand random brightness
    jitter.  These are evaluated in the hair shader (Blender), so they compose
    with the guide/child hairs for free.

    Note: a *data-driven* color (e.g. ``ScalarField``) is not supported by the
    hair renderers and falls back to melanin with a warning — the hair BSDF
    darkens/tints colors, which is poor for reading data anyway.  To color fur
    by an attribute, use a non-Hair material (e.g. ``Diffuse``) on the fur
    strands.

    Attributes:
        eumelanin: Eumelanin (dark/brown pigment) concentration (default: 1.3).
            Ignored when ``color`` or a root/tip gradient is set.
        pheomelanin: Pheomelanin (reddish-yellow pigment) concentration
            (default: 0.2).  Ignored when ``color`` or a gradient is set.
        color: Direct RGB / named hair color (overrides melanin).  ``None``
            (default) uses the melanin parametrization.
        root_color: Strand-root color of a root-to-tip gradient.  Setting
            ``root_color`` and/or ``tip_color`` overrides ``color`` and melanin.
        tip_color: Strand-tip color of the root-to-tip gradient.
        color_variation: Per-strand random brightness jitter in ``[0, 1]``
            (0 = uniform).  Applied on top of the melanin / color / gradient.

    """

    eumelanin: float = 1.3
    pheomelanin: float = 0.2
    color: TextureLike | None = None
    root_color: ColorLike | None = None
    tip_color: ColorLike | None = None
    color_variation: float = 0.0

    def __post_init__(self) -> None:
        if self.eumelanin < 0.0 or self.pheomelanin < 0.0:
            raise ValueError("Hair melanin concentrations must be non-negative.")
        if not 0.0 <= self.color_variation <= 1.0:
            raise ValueError("Hair.color_variation must be in [0, 1].")
