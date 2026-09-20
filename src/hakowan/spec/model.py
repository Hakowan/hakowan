"""Versioned, JSON-safe public specification models for Hakowan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = "1.0"
SCHEMA_URL = "https://hakowan.github.io/hakowan/schema/v1.json"


class SpecModel(BaseModel):
    """Strict immutable base for every public specification node."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, allow_inf_nan=False
    )


Number: TypeAlias = int | float
Vector: TypeAlias = list[Number]
Matrix: TypeAlias = list[list[Number]]
ColorValue: TypeAlias = Number | str | list[Number]


class ExpressionSpec(SpecModel):
    kind: Literal["expression"] = "expression"
    source: str = Field(min_length=1, max_length=1024)

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        from .expression import compile_expression

        compile_expression(value)
        return value


class FunctionRefSpec(SpecModel):
    kind: Literal["function"] = "function"
    id: str = Field(min_length=1)


CallableSpec = Annotated[ExpressionSpec | FunctionRefSpec, Field(discriminator="kind")]


class UniformScaleSpec(SpecModel):
    kind: Literal["uniform"] = "uniform"
    factor: float


class LogScaleSpec(SpecModel):
    kind: Literal["log"] = "log"
    base: float = 10.0


class ClipScaleSpec(SpecModel):
    kind: Literal["clip"] = "clip"
    domain: tuple[float, float]


class NormalizeScaleSpec(SpecModel):
    kind: Literal["normalize"] = "normalize"
    range_min: Number | Vector
    range_max: Number | Vector
    domain_min: Number | Vector | None = None
    domain_max: Number | Vector | None = None


class AffineScaleSpec(SpecModel):
    kind: Literal["affine"] = "affine"
    matrix: Matrix


class NormScaleSpec(SpecModel):
    kind: Literal["norm"] = "norm"
    order: float = 2.0


class CustomScaleSpec(SpecModel):
    kind: Literal["custom"] = "custom"
    function: CallableSpec


class AttributeSpec(SpecModel):
    name: str = Field(min_length=1)
    scales: tuple["ScaleSpec", ...] = ()


class OffsetScaleSpec(SpecModel):
    kind: Literal["offset"] = "offset"
    offset: AttributeSpec


ScaleSpec = Annotated[
    UniformScaleSpec
    | LogScaleSpec
    | ClipScaleSpec
    | NormalizeScaleSpec
    | AffineScaleSpec
    | NormScaleSpec
    | CustomScaleSpec
    | OffsetScaleSpec,
    Field(discriminator="kind"),
]


class UniformTextureSpec(SpecModel):
    kind: Literal["uniform"] = "uniform"
    color: ColorValue


class ImageTextureSpec(SpecModel):
    kind: Literal["image"] = "image"
    path: str = Field(min_length=1)
    uv: AttributeSpec | None = None
    raw: bool = False
    saturation: float = 1.0
    whiteness: float = 0.0


class CheckerboardTextureSpec(SpecModel):
    kind: Literal["checkerboard"] = "checkerboard"
    uv: AttributeSpec | None = None
    texture1: "TextureValue" = 0.8
    texture2: "TextureValue" = 0.2
    size: int = 8


class IsocontourTextureSpec(SpecModel):
    kind: Literal["isocontour"] = "isocontour"
    data: AttributeSpec
    ratio: float = 0.1
    texture1: "TextureValue" = 0.4
    texture2: "TextureValue" = 0.2
    num_contours: int = 8


class ScalarFieldTextureSpec(SpecModel):
    kind: Literal["scalar_field"] = "scalar_field"
    data: AttributeSpec
    colormap: str | list[ColorValue] = "viridis"
    domain: tuple[float, float] | None = None
    range: tuple[float, float] | None = None
    categories: bool = False
    reverse: bool = False


TextureSpec = Annotated[
    UniformTextureSpec
    | ImageTextureSpec
    | CheckerboardTextureSpec
    | IsocontourTextureSpec
    | ScalarFieldTextureSpec,
    Field(discriminator="kind"),
]
TextureValue: TypeAlias = TextureSpec | ColorValue
ScalarTextureValue: TypeAlias = TextureSpec | Number


class MediumSpec(SpecModel):
    albedo: ColorValue = 0.75
    scale: float = 1.0


class MaterialBaseSpec(SpecModel):
    two_sided: bool = False
    back_side: "MaterialSpec | None" = None


class DiffuseMaterialSpec(MaterialBaseSpec):
    kind: Literal["diffuse"] = "diffuse"
    reflectance: TextureValue = 0.5


class ConductorMaterialSpec(MaterialBaseSpec):
    kind: Literal["conductor"] = "conductor"
    material: str


class RoughConductorMaterialSpec(MaterialBaseSpec):
    kind: Literal["rough_conductor"] = "rough_conductor"
    material: str
    distribution: Literal["beckmann", "ggx", "phong"] = "beckmann"
    alpha: ScalarTextureValue = 0.1


class PlasticMaterialSpec(MaterialBaseSpec):
    kind: Literal["plastic"] = "plastic"
    diffuse_reflectance: TextureValue = 0.5
    specular_reflectance: ScalarTextureValue = 1.0


class RoughPlasticMaterialSpec(MaterialBaseSpec):
    kind: Literal["rough_plastic"] = "rough_plastic"
    diffuse_reflectance: TextureValue = 0.5
    specular_reflectance: ScalarTextureValue = 1.0
    distribution: Literal["beckmann", "ggx", "phong"] = "beckmann"
    alpha: float = 0.1


class PrincipledMaterialBaseSpec(MaterialBaseSpec):
    color: TextureValue = 0.5
    roughness: ScalarTextureValue = 0.5
    metallic: ScalarTextureValue = 0.0
    anisotropic: float = 0.0
    spec_trans: float = 0.0
    eta: float = 1.5
    spec_tint: float = 0.0
    sheen: float = 0.0
    sheen_tint: float = 0.0
    flatness: float = 0.0


class PrincipledMaterialSpec(PrincipledMaterialBaseSpec):
    kind: Literal["principled"] = "principled"


class ThinPrincipledMaterialSpec(PrincipledMaterialBaseSpec):
    kind: Literal["thin_principled"] = "thin_principled"
    diff_trans: float = 0.0


class DielectricMaterialBaseSpec(MaterialBaseSpec):
    int_ior: str | float = "bk7"
    ext_ior: str | float = "air"
    medium: MediumSpec | None = None
    specular_reflectance: float = 1.0
    specular_transmittance: float = 1.0


class DielectricMaterialSpec(DielectricMaterialBaseSpec):
    kind: Literal["dielectric"] = "dielectric"


class ThinDielectricMaterialSpec(DielectricMaterialBaseSpec):
    kind: Literal["thin_dielectric"] = "thin_dielectric"


class RoughDielectricMaterialSpec(DielectricMaterialBaseSpec):
    kind: Literal["rough_dielectric"] = "rough_dielectric"
    distribution: Literal["beckmann", "ggx", "phong"] = "beckmann"
    alpha: ScalarTextureValue = 0.1


class HairMaterialSpec(MaterialBaseSpec):
    kind: Literal["hair"] = "hair"
    eumelanin: float = 1.3
    pheomelanin: float = 0.2
    color: TextureValue | None = None
    root_color: ColorValue | None = None
    tip_color: ColorValue | None = None
    color_variation: float = 0.0


MaterialSpec = Annotated[
    DiffuseMaterialSpec
    | ConductorMaterialSpec
    | RoughConductorMaterialSpec
    | PlasticMaterialSpec
    | RoughPlasticMaterialSpec
    | PrincipledMaterialSpec
    | ThinPrincipledMaterialSpec
    | DielectricMaterialSpec
    | ThinDielectricMaterialSpec
    | RoughDielectricMaterialSpec
    | HairMaterialSpec,
    Field(discriminator="kind"),
]


class BendStyleSpec(SpecModel):
    kind: Literal["bend"] = "bend"
    direction: AttributeSpec
    bend_type: Literal["n", "r", "s"] = "n"


class PositionChannelSpec(SpecModel):
    kind: Literal["position"] = "position"
    data: AttributeSpec


class NormalChannelSpec(SpecModel):
    kind: Literal["normal"] = "normal"
    data: AttributeSpec


class SizeChannelSpec(SpecModel):
    kind: Literal["size"] = "size"
    data: float | AttributeSpec


class ShapeChannelSpec(SpecModel):
    kind: Literal["shape"] = "shape"
    base_shape: Literal["sphere", "disk", "cube"] = "sphere"
    orientation: AttributeSpec | None = None


class VectorFieldChannelSpec(SpecModel):
    kind: Literal["vector_field"] = "vector_field"
    data: AttributeSpec
    refinement_level: int = 0
    style: BendStyleSpec | None = None
    end_type: Literal["point", "arrow", "flat"] = "point"
    normalize: bool = False


class CovarianceChannelSpec(SpecModel):
    kind: Literal["covariance"] = "covariance"
    data: AttributeSpec
    full: bool = False


class BumpMapChannelSpec(SpecModel):
    kind: Literal["bump_map"] = "bump_map"
    texture: TextureValue
    scale: float = 1.0


class NormalMapChannelSpec(SpecModel):
    kind: Literal["normal_map"] = "normal_map"
    texture: TextureValue


class ChannelsSpec(SpecModel):
    position: PositionChannelSpec | None = None
    normal: NormalChannelSpec | None = None
    size: SizeChannelSpec | None = None
    shape: ShapeChannelSpec | None = None
    vector_field: VectorFieldChannelSpec | None = None
    covariance: CovarianceChannelSpec | None = None
    material: MaterialSpec | None = None
    bump_map: BumpMapChannelSpec | None = None
    normal_map: NormalMapChannelSpec | None = None


class FilterTransformSpec(SpecModel):
    kind: Literal["filter"] = "filter"
    data: AttributeSpec | None = None
    condition: CallableSpec | None = None


class ClipTransformSpec(SpecModel):
    kind: Literal["clip"] = "clip"
    point: Vector = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    normal: Vector = Field(default_factory=lambda: [1.0, 0.0, 0.0])


class UVMeshTransformSpec(SpecModel):
    kind: Literal["uv_mesh"] = "uv_mesh"
    uv: AttributeSpec | None = None


class AffineTransformSpec(SpecModel):
    kind: Literal["affine"] = "affine"
    matrix: Matrix


class PrincipalAxesTransformSpec(SpecModel):
    kind: Literal["principal_axes"] = "principal_axes"
    frame: Matrix = Field(
        default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    orthonormalize_frame: bool = True


class NormalizeTransformSpec(SpecModel):
    kind: Literal["normalize"] = "normalize"
    normalize_normals: bool = True
    normalize_tangents_bitangents: bool = True


class ComputeTransformSpec(SpecModel):
    kind: Literal["compute"] = "compute"
    x: str | None = None
    y: str | None = None
    z: str | None = None
    normal: str | None = None
    vertex_normal: str | None = None
    facet_normal: str | None = None
    component: str | None = None


class ExplodeTransformSpec(SpecModel):
    kind: Literal["explode"] = "explode"
    pieces: AttributeSpec
    magnitude: float = 1.0


class NormTransformSpec(SpecModel):
    kind: Literal["norm"] = "norm"
    data: AttributeSpec
    norm_attr_name: str
    order: int = 2


class BoundaryTransformSpec(SpecModel):
    kind: Literal["boundary"] = "boundary"
    attributes: tuple[str, ...] = ()


class StreamlineTransformSpec(SpecModel):
    kind: Literal["streamline"] = "streamline"
    vec_field: AttributeSpec
    n: int = 50
    cross_field: bool = True
    length: float | None = None
    seed: int = 0
    min_length: int = 3
    max_steps: int | None = None
    id_attr_name: str = "_hakowan_streamline_id"


class FurTransformSpec(SpecModel):
    kind: Literal["fur"] = "fur"
    vec_field: AttributeSpec
    n: int = 2000
    length: float | None = None
    lift: float = 30.0
    curl: float = 0.35
    segments: int = 6
    root_radius: float | None = None
    tip_radius: float = 0.0
    randomness: float = 0.3
    follow_surface: bool = False
    children: int = 0
    clump: float = 0.6
    spread: float | None = None
    seed: int = 0


TransformSpec = Annotated[
    FilterTransformSpec
    | ClipTransformSpec
    | UVMeshTransformSpec
    | AffineTransformSpec
    | PrincipalAxesTransformSpec
    | NormalizeTransformSpec
    | ComputeTransformSpec
    | ExplodeTransformSpec
    | NormTransformSpec
    | BoundaryTransformSpec
    | StreamlineTransformSpec
    | FurTransformSpec,
    Field(discriminator="kind"),
]


class MeshFileDataSpec(SpecModel):
    kind: Literal["mesh_file"] = "mesh_file"
    path: str = Field(min_length=1)
    roi_box: Matrix | None = None


class ExternalDataSpec(SpecModel):
    kind: Literal["external"] = "external"
    id: str = Field(min_length=1)
    roi_box: Matrix | None = None


DataSpec = Annotated[MeshFileDataSpec | ExternalDataSpec, Field(discriminator="kind")]


class LayerPropertiesSpec(SpecModel):
    data: DataSpec | None = None
    mark: Literal["point", "curve", "surface"] | None = None
    channels: ChannelsSpec = Field(default_factory=ChannelsSpec)
    transforms: tuple[TransformSpec, ...] = ()
    name: str | None = None


class LayerNodeSpec(SpecModel):
    kind: Literal["layer"] = "layer"
    spec: LayerPropertiesSpec = Field(default_factory=LayerPropertiesSpec)


class InheritNodeSpec(SpecModel):
    kind: Literal["inherit"] = "inherit"
    spec: LayerPropertiesSpec = Field(default_factory=LayerPropertiesSpec)
    child: "NodeSpec"


class OverlayNodeSpec(SpecModel):
    kind: Literal["overlay"] = "overlay"
    spec: LayerPropertiesSpec = Field(default_factory=LayerPropertiesSpec)
    children: tuple["NodeSpec", ...] = Field(min_length=2)


class LayoutNodeSpec(SpecModel):
    kind: Literal["layout"] = "layout"
    spec: LayerPropertiesSpec = Field(default_factory=LayerPropertiesSpec)
    axis: Literal["x", "y", "z"] = "x"
    gap: float = 0.05
    normalize: bool = False
    children: tuple["NodeSpec", ...] = Field(min_length=2)


NodeSpec = Annotated[
    LayerNodeSpec | InheritNodeSpec | OverlayNodeSpec | LayoutNodeSpec,
    Field(discriminator="kind"),
]


class FigureSpec(SpecModel):
    """Canonical versioned Hakowan visualization specification."""

    schema_url: Literal["https://hakowan.github.io/hakowan/schema/v1.json"] = Field(
        default="https://hakowan.github.io/hakowan/schema/v1.json", alias="$schema"
    )
    version: Literal["1.0"] = "1.0"

    root: NodeSpec

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=False)

    def to_json(self, *, indent: int | None = 2, canonical: bool = False) -> str:
        payload = self.to_dict()
        return json.dumps(
            payload,
            indent=None if canonical else indent,
            sort_keys=True,
            separators=(",", ":") if canonical else None,
            allow_nan=False,
        )

    def save(self, path: str | Path, *, indent: int | None = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, text: str | bytes) -> "FigureSpec":
        return cls.model_validate_json(text)

    @classmethod
    def load(cls, path: str | Path) -> "FigureSpec":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def json_schema() -> dict[str, Any]:
    """Return the canonical Hakowan JSON Schema."""
    result = FigureSpec.model_json_schema(by_alias=True)
    result["$id"] = SCHEMA_URL
    return result


FigureSpec.model_rebuild()


__all__ = [
    "SCHEMA_URL",
    "SCHEMA_VERSION",
    "AttributeSpec",
    "FigureSpec",
    "FunctionRefSpec",
    "ExpressionSpec",
    "LayerPropertiesSpec",
    "NodeSpec",
    "json_schema",
]
