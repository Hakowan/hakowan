"""Bidirectional conversion between canonical specifications and runtime layers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ..common.color import Color
from ..grammar.channel import (
    BumpMap,
    Covariance,
    Normal,
    NormalMap,
    Position,
    Shape,
    Size,
    VectorField,
)
from ..grammar.channel.curvestyle import Bend
from ..grammar.channel.material import (
    Conductor,
    Dielectric,
    Diffuse,
    Hair,
    Material,
    Medium,
    Plastic,
    Principled,
    RoughConductor,
    RoughDielectric,
    RoughPlastic,
    ThinDielectric,
    ThinPrincipled,
)
from ..grammar.dataframe import DataFrameLike
from ..grammar.layer import Layer, LayoutOptions
from ..grammar.figure import (
    DirectionalLight,
    Environment,
    Figure,
    OrthographicCamera,
    OutputSettings,
    PerspectiveCamera,
    PointLight,
    SceneSettings,
    ThinLensCamera,
)
from ..grammar.overlay import Annotation, Legend
from ..grammar.scale import (
    Affine as AffineScale,
    Attribute,
    Clip as ClipScale,
    Custom,
    Log,
    Norm as NormScale,
    Normalize as NormalizeScale,
    Offset,
    Scale,
    Uniform,
)
from ..grammar.texture import Checkerboard, Image, Isocontour, ScalarField, Texture
from ..grammar.texture import Uniform as UniformTexture
from ..grammar.transform import (
    Affine,
    Boundary,
    Clip,
    Compute,
    Explode,
    Filter,
    Fur,
    Norm,
    Normalize,
    PrincipalAxes,
    Streamline,
    Transform,
    UVMesh,
)
from .expression import compile_expression
from . import model as sm


class SpecConversionError(ValueError):
    """Raised when a runtime object cannot cross the canonical spec boundary."""


FunctionIds = Mapping[Any, str] | Callable[[Callable[..., Any]], str]
FunctionResolver = (
    Mapping[str, Callable[..., Any]] | Callable[[str], Callable[..., Any]]
)
DataIds = Mapping[int, str] | Callable[[Any], str]
DataResolver = Mapping[str, DataFrameLike] | Callable[[str], DataFrameLike]


def _json_value(value: Any, path: str) -> Any:
    if isinstance(value, Color):
        return [float(value.red), float(value.green), float(value.blue)]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_json_value(item, path) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise SpecConversionError(f"{path}: NaN and Infinity are not JSON-safe.")
        return value
    raise SpecConversionError(f"{path}: unsupported JSON value {type(value)!r}.")


def _lookup_function_id(
    function: Callable[..., Any], function_ids: FunctionIds | None, path: str
) -> sm.CallableSpec:
    if getattr(function, "__name__", None) == "hakowan_expression" and function.__doc__:
        return sm.ExpressionSpec(source=function.__doc__)
    if function_ids is None:
        raise SpecConversionError(
            f"{path}: callable {function!r} requires function_ids or an expression spec."
        )
    if callable(function_ids):
        identifier = function_ids(function)
    else:
        try:
            identifier = function_ids[function]
        except (KeyError, TypeError):
            try:
                identifier = function_ids[id(function)]
            except KeyError as exc:
                raise SpecConversionError(
                    f"{path}: callable {function!r} has no external function id."
                ) from exc
    if not identifier:
        raise SpecConversionError(f"{path}: external function id must not be empty.")
    return sm.FunctionRefSpec(id=identifier)


def _resolve_function(
    function: sm.CallableSpec,
    resolver: FunctionResolver | None,
    path: str,
) -> Callable[..., Any]:
    if isinstance(function, sm.ExpressionSpec):
        return compile_expression(function.source)
    if resolver is None:
        raise SpecConversionError(
            f"{path}: external function '{function.id}' requires function_resolver."
        )
    try:
        result = resolver(function.id) if callable(resolver) else resolver[function.id]
    except KeyError as exc:
        raise SpecConversionError(
            f"{path}: function_resolver has no entry for '{function.id}'."
        ) from exc
    if not callable(result):
        raise SpecConversionError(f"{path}: resolved object is not callable.")
    return result


def _attribute_to_spec(
    value: str | Attribute, function_ids: FunctionIds | None, path: str
) -> sm.AttributeSpec:
    if isinstance(value, str):
        return sm.AttributeSpec(name=value)
    scales: list[sm.ScaleSpec] = []
    current = value.scale if isinstance(value.scale, Scale) else None
    index = 0
    while current is not None:
        scales.append(_scale_to_spec(current, function_ids, f"{path}.scales[{index}]"))
        current = current._child
        index += 1
    return sm.AttributeSpec(name=value.name, scales=tuple(scales), unit=value.unit)


def _attribute_from_spec(
    value: sm.AttributeSpec,
    function_resolver: FunctionResolver | None,
    path: str,
) -> Attribute:
    scales = [
        _scale_from_spec(scale, function_resolver, f"{path}.scales[{index}]")
        for index, scale in enumerate(value.scales)
    ]
    chain = _chain_scales(scales)
    return Attribute(name=value.name, scale=chain, unit=value.unit)


def _scale_to_spec(
    scale: Scale, function_ids: FunctionIds | None, path: str
) -> sm.ScaleSpec:
    if isinstance(scale, Uniform):
        return sm.UniformScaleSpec(factor=scale.factor)
    if isinstance(scale, Log):
        return sm.LogScaleSpec(base=scale.base)
    if isinstance(scale, ClipScale):
        return sm.ClipScaleSpec(domain=cast(tuple[float, float], tuple(scale.domain)))
    if isinstance(scale, NormalizeScale):
        return sm.NormalizeScaleSpec(
            range_min=_json_value(scale.range_min, f"{path}.range_min"),
            range_max=_json_value(scale.range_max, f"{path}.range_max"),
            domain_min=_json_value(scale.domain_min, f"{path}.domain_min"),
            domain_max=_json_value(scale.domain_max, f"{path}.domain_max"),
        )
    if isinstance(scale, AffineScale):
        return sm.AffineScaleSpec(matrix=_json_value(scale.matrix, f"{path}.matrix"))
    if isinstance(scale, NormScale):
        return sm.NormScaleSpec(order=scale.order)
    if isinstance(scale, Offset):
        return sm.OffsetScaleSpec(
            offset=_attribute_to_spec(scale.offset, function_ids, f"{path}.offset")
        )
    if isinstance(scale, Custom):
        return sm.CustomScaleSpec(
            function=_lookup_function_id(
                scale.function, function_ids, f"{path}.function"
            )
        )
    raise SpecConversionError(f"{path}: unsupported scale {type(scale).__name__}.")


def _scale_from_spec(
    scale: sm.ScaleSpec, resolver: FunctionResolver | None, path: str
) -> Scale:
    if isinstance(scale, sm.UniformScaleSpec):
        return Uniform(factor=scale.factor)
    if isinstance(scale, sm.LogScaleSpec):
        return Log(base=scale.base)
    if isinstance(scale, sm.ClipScaleSpec):
        return ClipScale(domain=scale.domain)
    if isinstance(scale, sm.NormalizeScaleSpec):
        return NormalizeScale(
            range_min=scale.range_min,
            range_max=scale.range_max,
            domain_min=scale.domain_min,
            domain_max=scale.domain_max,
        )
    if isinstance(scale, sm.AffineScaleSpec):
        return AffineScale(matrix=np.asarray(scale.matrix, dtype=np.float64))
    if isinstance(scale, sm.NormScaleSpec):
        return NormScale(order=scale.order)
    if isinstance(scale, sm.OffsetScaleSpec):
        return Offset(
            offset=_attribute_from_spec(scale.offset, resolver, f"{path}.offset")
        )
    if isinstance(scale, sm.CustomScaleSpec):
        return Custom(
            function=_resolve_function(scale.function, resolver, f"{path}.function")
        )
    raise SpecConversionError(f"{path}: unsupported scale spec {type(scale).__name__}.")


def _chain_scales(scales: list[Scale]) -> Scale | None:
    if not scales:
        return None
    head = scales[0]
    current = head
    for scale in scales[1:]:
        current._child = scale
        current = scale
    return head


def _texture_value_to_spec(
    value: Any, function_ids: FunctionIds | None, path: str
) -> sm.TextureValue:
    if isinstance(value, Texture):
        return _texture_to_spec(value, function_ids, path)
    return _json_value(value, path)


def _legend_to_spec(legend: bool | Legend) -> bool | sm.LegendSpec:
    if isinstance(legend, bool):
        return legend
    return sm.LegendSpec(
        title=legend.title,
        units=legend.units,
        ticks=legend.ticks,
        format=legend.format,
        position=legend.position,
        category_labels=legend.category_labels,
        width=legend.width,
    )


def _legend_from_spec(legend: bool | sm.LegendSpec) -> bool | Legend:
    if isinstance(legend, bool):
        return legend
    return Legend(
        title=legend.title,
        units=legend.units,
        ticks=legend.ticks,
        format=legend.format,
        position=legend.position,
        category_labels=legend.category_labels,
        width=legend.width,
    )


def _texture_to_spec(
    texture: Texture, function_ids: FunctionIds | None, path: str
) -> sm.TextureSpec:
    if isinstance(texture, UniformTexture):
        return sm.UniformTextureSpec(color=_json_value(texture.color, f"{path}.color"))
    if isinstance(texture, Image):
        return sm.ImageTextureSpec(
            path=Path(texture._source or texture.filename).as_posix(),
            uv=_attribute_to_spec(texture.uv, function_ids, f"{path}.uv")
            if texture.uv is not None
            else None,
            raw=texture.raw,
            saturation=texture.saturation,
            whiteness=texture.whiteness,
        )
    if isinstance(texture, Checkerboard):
        return sm.CheckerboardTextureSpec(
            uv=_attribute_to_spec(texture.uv, function_ids, f"{path}.uv")
            if texture.uv is not None
            else None,
            texture1=_texture_value_to_spec(
                texture.texture1, function_ids, f"{path}.texture1"
            ),
            texture2=_texture_value_to_spec(
                texture.texture2, function_ids, f"{path}.texture2"
            ),
            size=texture.size,
        )
    if isinstance(texture, Isocontour):
        return sm.IsocontourTextureSpec(
            data=_attribute_to_spec(texture.data, function_ids, f"{path}.data"),
            ratio=texture.ratio,
            texture1=_texture_value_to_spec(
                texture.texture1, function_ids, f"{path}.texture1"
            ),
            texture2=_texture_value_to_spec(
                texture.texture2, function_ids, f"{path}.texture2"
            ),
            num_contours=texture.num_contours,
        )
    if isinstance(texture, ScalarField):
        return sm.ScalarFieldTextureSpec(
            data=_attribute_to_spec(texture.data, function_ids, f"{path}.data"),
            colormap=cast(Any, _json_value(texture.colormap, f"{path}.colormap")),
            domain=cast(tuple[float, float], tuple(texture.domain))
            if texture.domain is not None
            else None,
            range=cast(tuple[float, float], tuple(texture.range))
            if texture.range is not None
            else None,
            categories=texture.categories,
            reverse=texture.reverse,
            legend=_legend_to_spec(texture.legend),
        )
    raise SpecConversionError(f"{path}: unsupported texture {type(texture).__name__}.")


def _texture_value_from_spec(
    value: sm.TextureValue,
    function_resolver: FunctionResolver | None,
    base_dir: Path | None,
    path: str,
) -> Any:
    if isinstance(value, sm.SpecModel):
        return _texture_from_spec(value, function_resolver, base_dir, path)
    return value


def _texture_from_spec(
    texture: sm.TextureSpec,
    resolver: FunctionResolver | None,
    base_dir: Path | None,
    path: str,
) -> Texture:
    if isinstance(texture, sm.UniformTextureSpec):
        return UniformTexture(color=texture.color)
    if isinstance(texture, sm.ImageTextureSpec):
        filename = Path(texture.path)
        if base_dir is not None and not filename.is_absolute():
            filename = base_dir / filename
        result = Image(
            filename=filename,
            uv=_attribute_from_spec(texture.uv, resolver, f"{path}.uv")
            if texture.uv is not None
            else None,
            raw=texture.raw,
            saturation=texture.saturation,
            whiteness=texture.whiteness,
        )
        result._source = Path(texture.path)
        return result
    if isinstance(texture, sm.CheckerboardTextureSpec):
        return Checkerboard(
            uv=_attribute_from_spec(texture.uv, resolver, f"{path}.uv")
            if texture.uv is not None
            else None,
            texture1=_texture_value_from_spec(
                texture.texture1, resolver, base_dir, f"{path}.texture1"
            ),
            texture2=_texture_value_from_spec(
                texture.texture2, resolver, base_dir, f"{path}.texture2"
            ),
            size=texture.size,
        )
    if isinstance(texture, sm.IsocontourTextureSpec):
        return Isocontour(
            data=_attribute_from_spec(texture.data, resolver, f"{path}.data"),
            ratio=texture.ratio,
            texture1=_texture_value_from_spec(
                texture.texture1, resolver, base_dir, f"{path}.texture1"
            ),
            texture2=_texture_value_from_spec(
                texture.texture2, resolver, base_dir, f"{path}.texture2"
            ),
            num_contours=texture.num_contours,
        )
    if isinstance(texture, sm.ScalarFieldTextureSpec):
        return ScalarField(
            data=_attribute_from_spec(texture.data, resolver, f"{path}.data"),
            colormap=cast(Any, texture.colormap),
            domain=texture.domain,
            range=texture.range,
            categories=texture.categories,
            reverse=texture.reverse,
            legend=_legend_from_spec(texture.legend),
        )
    raise SpecConversionError(
        f"{path}: unsupported texture spec {type(texture).__name__}."
    )


def _medium_to_spec(medium: Medium | None, path: str) -> sm.MediumSpec | None:
    if medium is None:
        return None
    return sm.MediumSpec(
        albedo=_json_value(medium.albedo, f"{path}.albedo"), scale=medium.scale
    )


def _medium_from_spec(medium: sm.MediumSpec | None) -> Medium | None:
    return Medium(albedo=medium.albedo, scale=medium.scale) if medium else None


def _material_common(
    material: Material,
    function_ids: FunctionIds | None,
    path: str,
) -> dict[str, Any]:
    return {
        "two_sided": material.two_sided,
        "back_side": _material_to_spec(
            material.back_side, function_ids, f"{path}.back_side"
        )
        if material.back_side is not None
        else None,
    }


def _material_to_spec(
    material: Material, function_ids: FunctionIds | None, path: str
) -> sm.MaterialSpec:
    common = _material_common(material, function_ids, path)

    def tv(value: Any, field: str) -> sm.TextureValue:
        return _texture_value_to_spec(value, function_ids, f"{path}.{field}")

    def stv(value: Any, field: str) -> sm.ScalarTextureValue:
        return cast(sm.ScalarTextureValue, tv(value, field))

    if isinstance(material, RoughConductor):
        return sm.RoughConductorMaterialSpec(
            material=material.material,
            distribution=material.distribution,
            alpha=stv(material.alpha, "alpha"),
            **common,
        )
    if isinstance(material, Conductor):
        return sm.ConductorMaterialSpec(material=material.material, **common)
    if isinstance(material, RoughPlastic):
        return sm.RoughPlasticMaterialSpec(
            diffuse_reflectance=tv(material.diffuse_reflectance, "diffuse_reflectance"),
            specular_reflectance=stv(
                material.specular_reflectance, "specular_reflectance"
            ),
            distribution=material.distribution,
            alpha=material.alpha,
            **common,
        )
    if isinstance(material, Plastic):
        return sm.PlasticMaterialSpec(
            diffuse_reflectance=tv(material.diffuse_reflectance, "diffuse_reflectance"),
            specular_reflectance=stv(
                material.specular_reflectance, "specular_reflectance"
            ),
            **common,
        )
    if isinstance(material, ThinPrincipled):
        return sm.ThinPrincipledMaterialSpec(
            color=tv(material.color, "color"),
            roughness=stv(material.roughness, "roughness"),
            metallic=stv(material.metallic, "metallic"),
            anisotropic=material.anisotropic,
            spec_trans=material.spec_trans,
            eta=material.eta,
            spec_tint=material.spec_tint,
            sheen=material.sheen,
            sheen_tint=material.sheen_tint,
            flatness=material.flatness,
            diff_trans=material.diff_trans,
            **common,
        )
    if isinstance(material, Principled):
        return sm.PrincipledMaterialSpec(
            color=tv(material.color, "color"),
            roughness=stv(material.roughness, "roughness"),
            metallic=stv(material.metallic, "metallic"),
            anisotropic=material.anisotropic,
            spec_trans=material.spec_trans,
            eta=material.eta,
            spec_tint=material.spec_tint,
            sheen=material.sheen,
            sheen_tint=material.sheen_tint,
            flatness=material.flatness,
            **common,
        )
    if isinstance(material, RoughDielectric):
        return sm.RoughDielectricMaterialSpec(
            int_ior=material.int_ior,
            ext_ior=material.ext_ior,
            medium=_medium_to_spec(material.medium, f"{path}.medium"),
            specular_reflectance=material.specular_reflectance,
            specular_transmittance=material.specular_transmittance,
            distribution=material.distribution,
            alpha=stv(material.alpha, "alpha"),
            **common,
        )
    if isinstance(material, ThinDielectric):
        return sm.ThinDielectricMaterialSpec(
            int_ior=material.int_ior,
            ext_ior=material.ext_ior,
            medium=_medium_to_spec(material.medium, f"{path}.medium"),
            specular_reflectance=material.specular_reflectance,
            specular_transmittance=material.specular_transmittance,
            **common,
        )
    if isinstance(material, Dielectric):
        return sm.DielectricMaterialSpec(
            int_ior=material.int_ior,
            ext_ior=material.ext_ior,
            medium=_medium_to_spec(material.medium, f"{path}.medium"),
            specular_reflectance=material.specular_reflectance,
            specular_transmittance=material.specular_transmittance,
            **common,
        )
    if isinstance(material, Hair):
        return sm.HairMaterialSpec(
            eumelanin=material.eumelanin,
            pheomelanin=material.pheomelanin,
            color=tv(material.color, "color") if material.color is not None else None,
            root_color=_json_value(material.root_color, f"{path}.root_color"),
            tip_color=_json_value(material.tip_color, f"{path}.tip_color"),
            color_variation=material.color_variation,
            **common,
        )
    if isinstance(material, Diffuse):
        return sm.DiffuseMaterialSpec(
            reflectance=tv(material.reflectance, "reflectance"), **common
        )
    raise SpecConversionError(
        f"{path}: unsupported material {type(material).__name__}."
    )


def _material_from_spec(
    material: sm.MaterialSpec,
    resolver: FunctionResolver | None,
    base_dir: Path | None,
    path: str,
) -> Material:
    common: dict[str, Any] = {
        "two_sided": material.two_sided,
        "back_side": _material_from_spec(
            material.back_side, resolver, base_dir, f"{path}.back_side"
        )
        if material.back_side is not None
        else None,
    }

    def tv(value: sm.TextureValue, field: str) -> Any:
        return _texture_value_from_spec(value, resolver, base_dir, f"{path}.{field}")

    if isinstance(material, sm.DiffuseMaterialSpec):
        return Diffuse(reflectance=tv(material.reflectance, "reflectance"), **common)
    if isinstance(material, sm.RoughConductorMaterialSpec):
        return RoughConductor(
            material=material.material,
            distribution=material.distribution,
            alpha=tv(material.alpha, "alpha"),
            **common,
        )
    if isinstance(material, sm.ConductorMaterialSpec):
        return Conductor(material=material.material, **common)
    if isinstance(material, sm.RoughPlasticMaterialSpec):
        return RoughPlastic(
            diffuse_reflectance=tv(material.diffuse_reflectance, "diffuse_reflectance"),
            specular_reflectance=tv(
                material.specular_reflectance, "specular_reflectance"
            ),
            distribution=material.distribution,
            alpha=material.alpha,
            **common,
        )
    if isinstance(material, sm.PlasticMaterialSpec):
        return Plastic(
            diffuse_reflectance=tv(material.diffuse_reflectance, "diffuse_reflectance"),
            specular_reflectance=tv(
                material.specular_reflectance, "specular_reflectance"
            ),
            **common,
        )
    if isinstance(material, sm.PrincipledMaterialBaseSpec):
        principled: dict[str, Any] = {
            "color": tv(material.color, "color"),
            "roughness": tv(material.roughness, "roughness"),
            "metallic": tv(material.metallic, "metallic"),
            "anisotropic": material.anisotropic,
            "spec_trans": material.spec_trans,
            "eta": material.eta,
            "spec_tint": material.spec_tint,
            "sheen": material.sheen,
            "sheen_tint": material.sheen_tint,
            "flatness": material.flatness,
            **common,
        }
        if isinstance(material, sm.ThinPrincipledMaterialSpec):
            return ThinPrincipled(diff_trans=material.diff_trans, **principled)
        return Principled(**principled)
    if isinstance(material, sm.DielectricMaterialBaseSpec):
        dielectric: dict[str, Any] = {
            "int_ior": material.int_ior,
            "ext_ior": material.ext_ior,
            "medium": _medium_from_spec(material.medium),
            "specular_reflectance": material.specular_reflectance,
            "specular_transmittance": material.specular_transmittance,
            **common,
        }
        if isinstance(material, sm.RoughDielectricMaterialSpec):
            return RoughDielectric(
                distribution=material.distribution,
                alpha=tv(material.alpha, "alpha"),
                **dielectric,
            )
        if isinstance(material, sm.ThinDielectricMaterialSpec):
            return ThinDielectric(**dielectric)
        return Dielectric(**dielectric)
    if isinstance(material, sm.HairMaterialSpec):
        return Hair(
            eumelanin=material.eumelanin,
            pheomelanin=material.pheomelanin,
            color=tv(material.color, "color") if material.color is not None else None,
            root_color=material.root_color,
            tip_color=material.tip_color,
            color_variation=material.color_variation,
            **common,
        )
    raise SpecConversionError(
        f"{path}: unsupported material spec {type(material).__name__}."
    )


def _channel_name(channel: Any) -> str:
    if isinstance(channel, Position):
        return "position"
    if isinstance(channel, Normal):
        return "normal"
    if isinstance(channel, Size):
        return "size"
    if isinstance(channel, Shape):
        return "shape"
    if isinstance(channel, VectorField):
        return "vector_field"
    if isinstance(channel, Covariance):
        return "covariance"
    if isinstance(channel, Material):
        return "material"
    if isinstance(channel, BumpMap):
        return "bump_map"
    if isinstance(channel, NormalMap):
        return "normal_map"
    raise SpecConversionError(f"Unsupported channel {type(channel).__name__}.")


def _channels_to_spec(
    channels: list[Any], function_ids: FunctionIds | None, path: str
) -> sm.ChannelsSpec:
    result: dict[str, Any] = {}
    for index, channel in enumerate(channels):
        name = _channel_name(channel)
        channel_path = f"{path}.{name}"
        if name in result:
            raise SpecConversionError(
                f"{channel_path}: duplicate channels cannot be represented canonically."
            )
        if isinstance(channel, Position):
            result[name] = sm.PositionChannelSpec(
                data=_attribute_to_spec(
                    channel.data, function_ids, f"{channel_path}.data"
                )
            )
        elif isinstance(channel, Normal):
            result[name] = sm.NormalChannelSpec(
                data=_attribute_to_spec(
                    channel.data, function_ids, f"{channel_path}.data"
                )
            )
        elif isinstance(channel, Size):
            result[name] = sm.SizeChannelSpec(
                data=float(channel.data)
                if isinstance(channel.data, (int, float))
                else _attribute_to_spec(
                    channel.data, function_ids, f"{channel_path}.data"
                )
            )
        elif isinstance(channel, Shape):
            result[name] = sm.ShapeChannelSpec(
                base_shape=channel.base_shape,
                orientation=_attribute_to_spec(
                    channel.orientation, function_ids, f"{channel_path}.orientation"
                )
                if channel.orientation is not None
                else None,
            )
        elif isinstance(channel, VectorField):
            style = None
            if channel.style is not None:
                if not isinstance(channel.style, Bend):
                    raise SpecConversionError(
                        f"{channel_path}.style: unsupported style {type(channel.style).__name__}."
                    )
                style = sm.BendStyleSpec(
                    direction=_attribute_to_spec(
                        channel.style.direction,
                        function_ids,
                        f"{channel_path}.style.direction",
                    ),
                    bend_type=channel.style.bend_type,
                )
            result[name] = sm.VectorFieldChannelSpec(
                data=_attribute_to_spec(
                    channel.data, function_ids, f"{channel_path}.data"
                ),
                refinement_level=channel.refinement_level,
                style=style,
                end_type=channel.end_type,
                normalize=channel.normalize,
            )
        elif isinstance(channel, Covariance):
            result[name] = sm.CovarianceChannelSpec(
                data=_attribute_to_spec(
                    channel.data, function_ids, f"{channel_path}.data"
                ),
                full=channel.full,
            )
        elif isinstance(channel, Material):
            result[name] = _material_to_spec(channel, function_ids, channel_path)
        elif isinstance(channel, BumpMap):
            result[name] = sm.BumpMapChannelSpec(
                texture=_texture_value_to_spec(
                    channel.texture, function_ids, f"{channel_path}.texture"
                ),
                scale=channel.scale,
            )
        elif isinstance(channel, NormalMap):
            result[name] = sm.NormalMapChannelSpec(
                texture=_texture_value_to_spec(
                    channel.texture, function_ids, f"{channel_path}.texture"
                )
            )
    return sm.ChannelsSpec(**result)


def _channels_from_spec(
    channels: sm.ChannelsSpec,
    resolver: FunctionResolver | None,
    base_dir: Path | None,
    path: str,
) -> list[Any]:
    result: list[Any] = []
    if channels.position:
        result.append(
            Position(
                _attribute_from_spec(
                    channels.position.data, resolver, f"{path}.position.data"
                )
            )
        )
    if channels.normal:
        result.append(
            Normal(
                _attribute_from_spec(
                    channels.normal.data, resolver, f"{path}.normal.data"
                )
            )
        )
    if channels.size:
        data = channels.size.data
        result.append(
            Size(
                float(data)
                if isinstance(data, (int, float))
                else _attribute_from_spec(data, resolver, f"{path}.size.data")
            )
        )
    if channels.shape:
        result.append(
            Shape(
                base_shape=channels.shape.base_shape,
                orientation=_attribute_from_spec(
                    channels.shape.orientation, resolver, f"{path}.shape.orientation"
                )
                if channels.shape.orientation
                else None,
            )
        )
    if channels.vector_field:
        vf = channels.vector_field
        style = (
            Bend(
                direction=_attribute_from_spec(
                    vf.style.direction, resolver, f"{path}.vector_field.style.direction"
                ),
                bend_type=vf.style.bend_type,
            )
            if vf.style
            else None
        )
        result.append(
            VectorField(
                data=_attribute_from_spec(
                    vf.data, resolver, f"{path}.vector_field.data"
                ),
                refinement_level=vf.refinement_level,
                style=style,
                end_type=vf.end_type,
                normalize=vf.normalize,
            )
        )
    if channels.covariance:
        result.append(
            Covariance(
                data=_attribute_from_spec(
                    channels.covariance.data, resolver, f"{path}.covariance.data"
                ),
                full=channels.covariance.full,
            )
        )
    if channels.material:
        result.append(
            _material_from_spec(
                channels.material, resolver, base_dir, f"{path}.material"
            )
        )
    if channels.bump_map:
        result.append(
            BumpMap(
                texture=_texture_value_from_spec(
                    channels.bump_map.texture,
                    resolver,
                    base_dir,
                    f"{path}.bump_map.texture",
                ),
                scale=channels.bump_map.scale,
            )
        )
    if channels.normal_map:
        result.append(
            NormalMap(
                texture=_texture_value_from_spec(
                    channels.normal_map.texture,
                    resolver,
                    base_dir,
                    f"{path}.normal_map.texture",
                )
            )
        )
    return result


def _transform_to_spec(
    transform: Transform, function_ids: FunctionIds | None, path: str
) -> sm.TransformSpec:
    def attr(value: str | Attribute, field: str) -> sm.AttributeSpec:
        return _attribute_to_spec(value, function_ids, f"{path}.{field}")

    if isinstance(transform, Filter):
        condition = None
        if getattr(transform.condition, "__name__", None) != "_default_condition":
            condition = _lookup_function_id(
                transform.condition, function_ids, f"{path}.condition"
            )
        return sm.FilterTransformSpec(
            data=attr(transform.data, "data") if transform.data is not None else None,
            condition=condition,
        )
    if isinstance(transform, Clip):
        return sm.ClipTransformSpec(
            point=_json_value(transform.point, f"{path}.point"),
            normal=_json_value(transform.normal, f"{path}.normal"),
        )
    if isinstance(transform, UVMesh):
        return sm.UVMeshTransformSpec(
            uv=attr(transform.uv, "uv") if transform.uv is not None else None
        )
    if isinstance(transform, Affine):
        return sm.AffineTransformSpec(
            matrix=_json_value(transform.matrix, f"{path}.matrix")
        )
    if isinstance(transform, PrincipalAxes):
        return sm.PrincipalAxesTransformSpec(
            frame=_json_value(transform.frame, f"{path}.frame"),
            orthonormalize_frame=transform.orthonormalize_frame,
        )
    if isinstance(transform, Normalize):
        return sm.NormalizeTransformSpec(
            normalize_normals=transform.normalize_normals,
            normalize_tangents_bitangents=transform.normalize_tangents_bitangents,
        )
    if isinstance(transform, Compute):
        return sm.ComputeTransformSpec(
            x=transform.x,
            y=transform.y,
            z=transform.z,
            normal=transform.normal,
            vertex_normal=transform.vertex_normal,
            facet_normal=transform.facet_normal,
            component=transform.component,
        )
    if isinstance(transform, Explode):
        return sm.ExplodeTransformSpec(
            pieces=attr(transform.pieces, "pieces"), magnitude=transform.magnitude
        )
    if isinstance(transform, Norm):
        return sm.NormTransformSpec(
            data=attr(transform.data, "data"),
            norm_attr_name=transform.norm_attr_name,
            order=transform.order,
        )
    if isinstance(transform, Boundary):
        return sm.BoundaryTransformSpec(attributes=tuple(transform.attributes))
    if isinstance(transform, Streamline):
        return sm.StreamlineTransformSpec(
            vec_field=attr(transform.vec_field, "vec_field"),
            n=transform.n,
            cross_field=transform.cross_field,
            length=transform.length,
            seed=transform.seed,
            min_length=transform.min_length,
            max_steps=transform.max_steps,
            id_attr_name=transform.id_attr_name,
        )
    if isinstance(transform, Fur):
        return sm.FurTransformSpec(
            vec_field=attr(transform.vec_field, "vec_field"),
            n=transform.n,
            length=transform.length,
            lift=transform.lift,
            curl=transform.curl,
            segments=transform.segments,
            root_radius=transform.root_radius,
            tip_radius=transform.tip_radius,
            randomness=transform.randomness,
            follow_surface=transform.follow_surface,
            children=transform.children,
            clump=transform.clump,
            spread=transform.spread,
            seed=transform.seed,
        )
    raise SpecConversionError(
        f"{path}: unsupported transform {type(transform).__name__}."
    )


def _transform_from_spec(
    transform: sm.TransformSpec, resolver: FunctionResolver | None, path: str
) -> Transform:
    def attr(value: sm.AttributeSpec, field: str) -> Attribute:
        return _attribute_from_spec(value, resolver, f"{path}.{field}")

    if isinstance(transform, sm.FilterTransformSpec):
        kwargs: dict[str, Any] = {
            "data": attr(transform.data, "data") if transform.data else None
        }
        if transform.condition is not None:
            kwargs["condition"] = _resolve_function(
                transform.condition, resolver, f"{path}.condition"
            )
        return Filter(**kwargs)
    if isinstance(transform, sm.ClipTransformSpec):
        return Clip(point=transform.point, normal=transform.normal)
    if isinstance(transform, sm.UVMeshTransformSpec):
        return UVMesh(uv=attr(transform.uv, "uv") if transform.uv else None)
    if isinstance(transform, sm.AffineTransformSpec):
        return Affine(matrix=np.asarray(transform.matrix, dtype=float))
    if isinstance(transform, sm.PrincipalAxesTransformSpec):
        return PrincipalAxes(
            frame=np.asarray(transform.frame, dtype=float),
            orthonormalize_frame=transform.orthonormalize_frame,
        )
    if isinstance(transform, sm.NormalizeTransformSpec):
        return Normalize(
            normalize_normals=transform.normalize_normals,
            normalize_tangents_bitangents=transform.normalize_tangents_bitangents,
        )
    if isinstance(transform, sm.ComputeTransformSpec):
        return Compute(
            x=transform.x,
            y=transform.y,
            z=transform.z,
            normal=transform.normal,
            vertex_normal=transform.vertex_normal,
            facet_normal=transform.facet_normal,
            component=transform.component,
        )
    if isinstance(transform, sm.ExplodeTransformSpec):
        return Explode(
            pieces=attr(transform.pieces, "pieces"), magnitude=transform.magnitude
        )
    if isinstance(transform, sm.NormTransformSpec):
        return Norm(
            data=attr(transform.data, "data"),
            norm_attr_name=transform.norm_attr_name,
            order=transform.order,
        )
    if isinstance(transform, sm.BoundaryTransformSpec):
        return Boundary(attributes=list(transform.attributes))
    if isinstance(transform, sm.StreamlineTransformSpec):
        return Streamline(
            vec_field=attr(transform.vec_field, "vec_field"),
            n=transform.n,
            cross_field=transform.cross_field,
            length=transform.length,
            seed=transform.seed,
            min_length=transform.min_length,
            max_steps=transform.max_steps,
            id_attr_name=transform.id_attr_name,
        )
    if isinstance(transform, sm.FurTransformSpec):
        return Fur(
            vec_field=attr(transform.vec_field, "vec_field"),
            n=transform.n,
            length=transform.length,
            lift=transform.lift,
            curl=transform.curl,
            segments=transform.segments,
            root_radius=transform.root_radius,
            tip_radius=transform.tip_radius,
            randomness=transform.randomness,
            follow_surface=transform.follow_surface,
            children=transform.children,
            clump=transform.clump,
            spread=transform.spread,
            seed=transform.seed,
        )
    raise SpecConversionError(
        f"{path}: unsupported transform spec {type(transform).__name__}."
    )


def _transforms_to_spec(
    transform: Transform | None, function_ids: FunctionIds | None, path: str
) -> tuple[sm.TransformSpec, ...]:
    nodes: list[Transform] = []
    current = transform
    while current is not None:
        nodes.append(current)
        current = current._child
    # Runtime chains execute tail-first. Canonical arrays are deliberately
    # application-ordered, so reverse the internal linked representation.
    return tuple(
        _transform_to_spec(item, function_ids, f"{path}[{index}]")
        for index, item in enumerate(reversed(nodes))
    )


def _transforms_from_spec(
    transforms: tuple[sm.TransformSpec, ...],
    resolver: FunctionResolver | None,
    path: str,
) -> Transform | None:
    runtime = [
        _transform_from_spec(item, resolver, f"{path}[{index}]")
        for index, item in enumerate(transforms)
    ]
    if not runtime:
        return None
    # Runtime applies linked transforms tail-first, so reverse the canonical
    # application order when rebuilding the chain.
    runtime.reverse()
    for current, child in zip(runtime, runtime[1:]):
        current._child = child
    return runtime[0]


def _lookup_data_id(mesh: Any, data_ids: DataIds | None, path: str) -> str:
    if data_ids is None:
        raise SpecConversionError(f"{path}: in-memory mesh requires data_ids.")
    if callable(data_ids):
        identifier = data_ids(mesh)
    else:
        try:
            identifier = data_ids[id(mesh)]
        except KeyError as exc:
            raise SpecConversionError(f"{path}: mesh has no external data id.") from exc
    if not identifier:
        raise SpecConversionError(f"{path}: external data id must not be empty.")
    return identifier


def _properties_to_spec(
    layer: Layer, data_ids: DataIds | None, function_ids: FunctionIds | None, path: str
) -> sm.LayerPropertiesSpec:
    runtime = layer._spec
    data: sm.DataSpec | None = None
    if runtime.data is not None:
        roi = _json_value(runtime.data.roi_box, f"{path}.data.roi_box")
        if runtime.data.source is not None:
            data = sm.MeshFileDataSpec(path=runtime.data.source.as_posix(), roi_box=roi)
        else:
            data = sm.ExternalDataSpec(
                id=_lookup_data_id(runtime.data.mesh, data_ids, f"{path}.data"),
                roi_box=roi,
            )
    return sm.LayerPropertiesSpec(
        data=data,
        mark=cast(Literal["point", "curve", "surface"], runtime.mark.name.lower())
        if runtime.mark is not None
        else None,
        channels=_channels_to_spec(runtime.channels, function_ids, f"{path}.channels"),
        transforms=_transforms_to_spec(
            runtime.transform, function_ids, f"{path}.transforms"
        ),
        name=runtime.name,
        annotations=tuple(
            sm.AnnotationSpec(
                text=item.text,
                position=item.position,
                color=cast(Any, _json_value(item.color, f"{path}.annotations.color")),
                font_size=item.font_size,
                anchor=item.anchor,
                background=cast(
                    Any,
                    _json_value(item.background, f"{path}.annotations.background"),
                ),
                padding=item.padding,
            )
            for item in runtime.annotations
        ),
    )


def _node_to_spec(
    layer: Layer, data_ids: DataIds | None, function_ids: FunctionIds | None, path: str
) -> sm.NodeSpec:
    spec = _properties_to_spec(layer, data_ids, function_ids, f"{path}.spec")
    children = tuple(
        _node_to_spec(child, data_ids, function_ids, f"{path}.children[{index}]")
        for index, child in enumerate(layer._children)
    )
    if not children:
        return sm.LayerNodeSpec(spec=spec)
    if layer._layout is not None:
        return sm.LayoutNodeSpec(
            spec=spec,
            axis=("x", "y", "z")[layer._layout.axis],
            gap=layer._layout.gap,
            normalize=layer._layout.normalize,
            children=children,
        )
    if len(children) == 1:
        return sm.InheritNodeSpec(spec=spec, child=children[0])
    return sm.OverlayNodeSpec(spec=spec, children=children)


def _float3(values) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _camera_to_spec(camera):
    common = {
        "eye": _float3(camera.eye),
        "target": _float3(camera.target),
        "up": _float3(camera.up),
        "near": camera.near,
        "far": camera.far,
    }
    if isinstance(camera, ThinLensCamera):
        return sm.ThinLensCameraSpec(
            fov=camera.fov,
            fov_axis=camera.fov_axis,
            aperture_radius=camera.aperture_radius,
            focus_distance=camera.focus_distance,
            **common,
        )
    if isinstance(camera, PerspectiveCamera):
        return sm.PerspectiveCameraSpec(
            fov=camera.fov, fov_axis=camera.fov_axis, **common
        )
    return sm.OrthographicCameraSpec(scale=camera.scale, **common)


def _camera_from_spec(camera):
    common = {
        "eye": tuple(camera.eye),
        "target": tuple(camera.target),
        "up": tuple(camera.up),
        "near": camera.near,
        "far": camera.far,
    }
    if isinstance(camera, sm.ThinLensCameraSpec):
        return ThinLensCamera(
            fov=camera.fov,
            fov_axis=camera.fov_axis,
            aperture_radius=camera.aperture_radius,
            focus_distance=camera.focus_distance,
            **common,
        )
    if isinstance(camera, sm.PerspectiveCameraSpec):
        return PerspectiveCamera(fov=camera.fov, fov_axis=camera.fov_axis, **common)
    return OrthographicCamera(scale=camera.scale, **common)


def _scene_to_spec(scene: SceneSettings | None) -> sm.SceneSettingsSpec | None:
    if scene is None:
        return None
    lights = None
    if scene.lights is not None:
        lights = tuple(
            sm.PointLightSpec(
                position=list(_float3(light.position)),
                color=cast(Any, _json_value(light.color, "scene.lights.color")),
                intensity=light.intensity,
            )
            if isinstance(light, PointLight)
            else sm.DirectionalLightSpec(
                direction=list(_float3(light.direction)),
                color=cast(Any, _json_value(light.color, "scene.lights.color")),
                intensity=light.intensity,
            )
            for light in scene.lights
        )
    environment = None
    if scene.environment is not None:
        logical_path = scene.environment._source or scene.environment.path
        environment = sm.EnvironmentSpec(
            enabled=scene.environment.enabled,
            path=logical_path.as_posix() if logical_path is not None else None,
            scale=scene.environment.scale,
            up=list(_float3(scene.environment.up)),
            rotation=scene.environment.rotation,
            visible=scene.environment.visible,
        )
    output = None
    if scene.output is not None:
        output = sm.OutputSettingsSpec(
            width=scene.output.width,
            height=scene.output.height,
            background=scene.output.background,
            passes=scene.output.passes,
            sampler_seed=scene.output.sampler_seed,
        )
    return sm.SceneSettingsSpec(
        camera=_camera_to_spec(scene.camera) if scene.camera is not None else None,
        lights=lights,
        environment=environment,
        output=output,
    )


def _scene_from_spec(
    scene: sm.SceneSettingsSpec, base_dir: Path | None
) -> SceneSettings:
    lights = None
    if scene.lights is not None:
        lights = tuple(
            PointLight(
                position=_float3(light.position),
                color=light.color,
                intensity=light.intensity,
            )
            if isinstance(light, sm.PointLightSpec)
            else DirectionalLight(
                direction=_float3(light.direction),
                color=light.color,
                intensity=light.intensity,
            )
            for light in scene.lights
        )
    environment = None
    if scene.environment is not None:
        logical = Path(scene.environment.path) if scene.environment.path else None
        resolved = logical
        if logical is not None and base_dir is not None and not logical.is_absolute():
            resolved = base_dir / logical
        environment = Environment(
            path=resolved,
            scale=scene.environment.scale,
            up=_float3(scene.environment.up),
            rotation=scene.environment.rotation,
            visible=scene.environment.visible,
            enabled=scene.environment.enabled,
            _source=logical,
        )
    output = None
    if scene.output is not None:
        output = OutputSettings(
            width=scene.output.width,
            height=scene.output.height,
            background=scene.output.background,
            passes=scene.output.passes,
            sampler_seed=scene.output.sampler_seed,
        )
    return SceneSettings(
        camera=_camera_from_spec(scene.camera) if scene.camera is not None else None,
        lights=lights,
        environment=environment,
        output=output,
    )


def to_spec(
    value: Layer | Figure,
    *,
    data_ids: DataIds | None = None,
    function_ids: FunctionIds | None = None,
) -> sm.FigureSpec:
    """Convert a runtime Layer or Figure into a canonical specification."""
    if isinstance(value, Figure):
        return sm.FigureSpec(
            version="1.1",
            root=_node_to_spec(value.layer, data_ids, function_ids, "root"),
            scene=_scene_to_spec(value.scene),
        )
    if not isinstance(value, Layer):
        raise TypeError(f"Expected Layer or Figure, got {type(value)!r}")
    return sm.FigureSpec(
        version="1.0", root=_node_to_spec(value, data_ids, function_ids, "root")
    )


def _resolve_data(
    data: sm.DataSpec,
    resolver: DataResolver | None,
    base_dir: Path | None,
    path: str,
) -> DataFrameLike:
    if isinstance(data, sm.MeshFileDataSpec):
        filename = Path(data.path)
        return (
            base_dir / filename
            if base_dir is not None and not filename.is_absolute()
            else filename
        )
    if resolver is None:
        raise SpecConversionError(
            f"{path}: external data '{data.id}' requires data_resolver."
        )
    try:
        return resolver(data.id) if callable(resolver) else resolver[data.id]
    except KeyError as exc:
        raise SpecConversionError(
            f"{path}: data_resolver has no entry for '{data.id}'."
        ) from exc


def _properties_from_spec(
    layer: Layer,
    spec: sm.LayerPropertiesSpec,
    data_resolver: DataResolver | None,
    function_resolver: FunctionResolver | None,
    base_dir: Path | None,
    path: str,
) -> None:
    if spec.data is not None:
        data = _resolve_data(spec.data, data_resolver, base_dir, f"{path}.data")
        layer.data(data, roi_box=spec.data.roi_box, in_place=True)
        if isinstance(spec.data, sm.MeshFileDataSpec) and layer._spec.data is not None:
            layer._spec.data.source = Path(spec.data.path)
    if spec.mark is not None:
        layer.mark(spec.mark, in_place=True)
    layer._spec.channels = _channels_from_spec(
        spec.channels, function_resolver, base_dir, f"{path}.channels"
    )
    layer._spec.transform = _transforms_from_spec(
        spec.transforms, function_resolver, f"{path}.transforms"
    )
    layer._spec.name = spec.name
    layer._spec.annotations = [
        Annotation(
            text=item.text,
            position=item.position,
            color=item.color,
            font_size=item.font_size,
            anchor=item.anchor,
            background=item.background,
            padding=item.padding,
        )
        for item in spec.annotations
    ]


def _node_from_spec(
    node: sm.NodeSpec,
    data_resolver: DataResolver | None,
    function_resolver: FunctionResolver | None,
    base_dir: Path | None,
    path: str,
) -> Layer:
    layer = Layer()
    _properties_from_spec(
        layer, node.spec, data_resolver, function_resolver, base_dir, f"{path}.spec"
    )
    if isinstance(node, sm.InheritNodeSpec):
        layer._children = [
            _node_from_spec(
                node.child, data_resolver, function_resolver, base_dir, f"{path}.child"
            )
        ]
    elif isinstance(node, (sm.OverlayNodeSpec, sm.LayoutNodeSpec)):
        layer._children = [
            _node_from_spec(
                child,
                data_resolver,
                function_resolver,
                base_dir,
                f"{path}.children[{index}]",
            )
            for index, child in enumerate(node.children)
        ]
        if isinstance(node, sm.LayoutNodeSpec):
            layer._layout = LayoutOptions(
                axis={"x": 0, "y": 1, "z": 2}[node.axis],
                gap=node.gap,
                normalize=node.normalize,
            )
    return layer


def from_spec(
    spec: sm.FigureSpec | Mapping[str, Any],
    *,
    data_resolver: DataResolver | None = None,
    function_resolver: FunctionResolver | None = None,
    base_dir: str | Path | None = None,
) -> Layer | Figure:
    """Build a runtime Layer or Figure from a canonical specification."""
    parsed = (
        spec if isinstance(spec, sm.FigureSpec) else sm.FigureSpec.model_validate(spec)
    )
    directory = Path(base_dir) if base_dir is not None else None
    layer = _node_from_spec(
        parsed.root, data_resolver, function_resolver, directory, "root"
    )
    if parsed.scene is None:
        return layer
    return Figure(layer=layer, scene=_scene_from_spec(parsed.scene, directory))


def from_json(
    text: str | bytes,
    *,
    data_resolver: DataResolver | None = None,
    function_resolver: FunctionResolver | None = None,
    base_dir: str | Path | None = None,
) -> Layer | Figure:
    """Parse canonical JSON and build a runtime layer tree."""
    return from_spec(
        sm.FigureSpec.from_json(text),
        data_resolver=data_resolver,
        function_resolver=function_resolver,
        base_dir=base_dir,
    )


def load_spec(path: str | Path) -> sm.FigureSpec:
    """Load and validate a canonical specification document."""
    return sm.FigureSpec.load(path)


def load_layer(
    path: str | Path,
    *,
    data_resolver: DataResolver | None = None,
    function_resolver: FunctionResolver | None = None,
) -> Layer | Figure:
    """Load a specification and resolve relative resources beside its file."""
    filename = Path(path)
    return from_spec(
        load_spec(filename),
        data_resolver=data_resolver,
        function_resolver=function_resolver,
        base_dir=filename.parent,
    )


__all__ = [
    "DataIds",
    "DataResolver",
    "FunctionIds",
    "FunctionResolver",
    "SpecConversionError",
    "from_json",
    "from_spec",
    "load_spec",
    "load_layer",
    "to_spec",
]
