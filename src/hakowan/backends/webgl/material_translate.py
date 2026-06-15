"""Translate hakowan ``Material`` → glTF ``pbrMetallicRoughness`` dict.

Coverage:
  * Diffuse / Plastic / RoughPlastic with Uniform, ScalarField (vertex-color
    bake), Image, and Checkerboard reflectance.
  * Principled with float roughness/metallic + Uniform/ScalarField/Image color.
  * Conductor / RoughConductor: mapped to metallic=1 with a built-in IOR-name
    → albedo lookup table covering the common Mitsuba presets (Au/Ag/Cu/Al/
    Cr/Ni/Pt/Ti/W/Fe).
  * Dielectric / ThinDielectric / RoughDielectric: best-effort transmissive
    approximation (loud warning that volumetrics are dropped).
  * Hair: gray-fallback with a warning.

Roughness/metallic factors accept floats and ``ScalarField`` textures: the
latter are baked per-vertex and multiplied into the factor by the viewer's
shader patch (see ``templates/viewer.html``). This covers Principled
roughness/metallic and the ``alpha`` of RoughConductor / RoughDielectric. Other
texture types (Image/Checkerboard) downgrade to the default scalar factor with
a warning. RoughPlastic.alpha is float-only (matching the grammar and the other
backends).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image as PILImage, ImageEnhance

from ...common import logger
from ...common.color import linear_to_srgb, srgb_to_linear, srgb_to_linear_array
from ...common.to_color import to_color
from ...compiler import View
from ...grammar.channel import BumpMap, NormalMap
from ...grammar.channel.material import (
    Conductor,
    Dielectric,
    Diffuse,
    Hair,
    Material,
    Plastic,
    Principled,
    RoughConductor,
    RoughDielectric,
    RoughPlastic,
    ThinDielectric,
)
from ...grammar.texture import Checkerboard, Image, Isocontour, ScalarField, Uniform

from .builder import (
    GLTFBuilder,
    _FILTER_NEAREST,
)

# glTF allows arbitrary underscore-prefixed names, but three.js GLTFLoader maps
# unknown attributes with ``name.toLowerCase()`` (see ``addPrimitiveAttributes``).
# Shader patches must use the lowercased names so attributes bind after load.
_SCALAR_ATTR = "_scalar_0"
_ROUGHNESS_ATTR = "_roughness_0"
_METALLIC_ATTR = "_metallic_0"
_ISO_COLOR1_ATTR = "_iso_color1_0"
_ISO_COLOR2_ATTR = "_iso_color2_0"


@dataclass
class MaterialResult:
    """What ``translate_material`` returns.

    ``custom_attrs`` carries per-vertex arrays that must be plumbed onto the
    mesh node (e.g. ``"_scalar_0"`` driving an isocontour shader).
    ``extras`` carries the dict that lands on the glTF material's ``extras``
    field — the viewer JS reads it to wire up ``onBeforeCompile`` patches.
    """

    pbr: dict[str, Any]
    double_sided: bool
    custom_attrs: dict[str, np.ndarray] = field(default_factory=dict)
    extras: dict[str, Any] | None = None


_DEFAULT_BASE_COLOR: list[float] = [0.5, 0.5, 0.5, 1.0]


# Mitsuba's named IOR presets (subset). Values are absolute refractive indices
# at ~589 nm; for glTF we emit the relative ratio ``int_ior / ext_ior``.
# Source: Mitsuba 3 docs (`dielectric` plugin) + standard references.
_IOR_NAMES: dict[str, float] = {
    "vacuum": 1.0,
    "air": 1.000277,
    "helium": 1.000036,
    "hydrogen": 1.000132,
    "water": 1.333,
    "ethanol": 1.361,
    "acetone": 1.36,
    "carbontetrachloride": 1.461,
    "fusedquartz": 1.458,
    "pyrex": 1.470,
    "acrylicglass": 1.49,
    "polypropylene": 1.49,
    "bk7": 1.5046,
    "glass": 1.5046,
    "sodiumchloride": 1.544,
    "polystyrene": 1.59,
    "polycarbonate": 1.585,
    "amber": 1.55,
    "sapphire": 1.762,
    "diamond": 2.419,
}


def _resolve_ior(ior_like: Any, default: float) -> float:
    if isinstance(ior_like, (float, int)):
        return float(ior_like)
    if isinstance(ior_like, str):
        v = _IOR_NAMES.get(ior_like.lower())
        if v is None:
            logger.warning(
                f"WebGL backend: unknown IOR preset '{ior_like}'; using {default}."
            )
            return default
        return v
    return default


# Approximate sRGB F0 albedo for common metals (Mitsuba conductor presets).
# Sources: Naty Hoffman's PBR Diffuse Lighting course notes; Filament docs.
_CONDUCTOR_PRESETS: dict[str, tuple[float, float, float]] = {
    "Au": (1.000, 0.766, 0.336),  # Gold
    "Ag": (0.972, 0.960, 0.915),  # Silver
    "Cu": (0.955, 0.638, 0.538),  # Copper
    "Al": (0.913, 0.921, 0.925),  # Aluminium
    "Cr": (0.550, 0.556, 0.554),  # Chromium
    "Fe": (0.560, 0.570, 0.580),  # Iron
    "Ni": (0.660, 0.609, 0.526),  # Nickel
    "Pt": (0.672, 0.637, 0.585),  # Platinum
    "Ti": (0.542, 0.497, 0.449),  # Titanium
    "W": (0.560, 0.520, 0.475),  # Tungsten
}


def _color_to_rgba(color_like: Any) -> list[float]:
    color = to_color(color_like)
    return [
        srgb_to_linear(float(color.red)),
        srgb_to_linear(float(color.green)),
        srgb_to_linear(float(color.blue)),
        1.0,
    ]


def _reflectance_to_base_color(reflectance: Any) -> list[float]:
    """Resolve Uniform/raw-color reflectance to a baseColorFactor.

    ScalarField, Image, and Checkerboard inputs return white — their colour
    comes through COLOR_0 / baseColorTexture, applied multiplicatively by
    glTF / three.js.
    """
    if isinstance(reflectance, Uniform):
        return _color_to_rgba(reflectance.color)
    if isinstance(reflectance, (ScalarField, Image, Checkerboard, Isocontour)):
        return [1.0, 1.0, 1.0, 1.0]
    if isinstance(reflectance, (float, int, str, tuple, list)):
        return _color_to_rgba(reflectance)
    logger.warning(
        f"WebGL backend: reflectance type {type(reflectance).__name__} not "
        "supported yet; using gray."
    )
    return list(_DEFAULT_BASE_COLOR)


def _resolve_reflectance(mat: Material) -> Any | None:
    """Return the colour-carrying field of ``mat``, or ``None`` for materials
    whose colour is intrinsic (Conductor) or absent (Dielectric/Hair).
    """
    if isinstance(mat, Diffuse):
        return mat.reflectance
    if isinstance(mat, Principled):
        return mat.color
    if isinstance(mat, Plastic):
        return mat.diffuse_reflectance
    return None


def _back_base_color(mat: Material) -> list[float]:
    """Best-effort linear RGB (3 floats) for a back-face material.

    The viewer renders the back face by overriding only the diffuse base colour
    in the fragment shader (``gl_FrontFacing`` branch — see ``viewer.html``), so
    a back material collapses to a single flat colour here. Materials whose
    appearance isn't a single base colour (Conductor / Dielectric / Hair, or a
    textured reflectance) are approximated with a warning.
    """
    if isinstance(mat, Conductor):
        albedo = _CONDUCTOR_PRESETS.get(mat.material, (0.7, 0.7, 0.7))
        logger.warning(
            "WebGL backend: back_side Conductor approximated as a flat metal color."
        )
        return [srgb_to_linear(c) for c in albedo]
    if isinstance(mat, Dielectric):
        logger.warning(
            "WebGL backend: back_side Dielectric approximated as white "
            "(no back-face transmission)."
        )
        return [1.0, 1.0, 1.0]
    if isinstance(mat, Hair):
        logger.warning("WebGL backend: back_side Hair approximated as brown.")
        return _color_to_rgba("saddlebrown")[:3]
    reflectance = _resolve_reflectance(mat)
    if reflectance is None:
        return list(_DEFAULT_BASE_COLOR[:3])
    if isinstance(reflectance, (ScalarField, Image, Checkerboard, Isocontour)):
        logger.warning(
            "WebGL backend: textured back_side color not supported; using white."
        )
        return [1.0, 1.0, 1.0]
    return _reflectance_to_base_color(reflectance)[:3]


def _load_image_as_png_bytes(image: Image) -> bytes:
    path = Path(image.filename)
    img = PILImage.open(path).convert("RGBA")
    if image.saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(image.saturation)
    if image.whiteness != 0.0:
        white = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
        img = PILImage.blend(img.convert("RGBA"), white, alpha=image.whiteness)
    # hakowan stores UVs with V=0 at the bottom of the image (OBJ
    # convention — Mitsuba compensates via ``to_uv = diag(1, -1, 1)``).
    # glTF/three.js samples with V=0 at the top, so we flip the image
    # vertically here to match without touching the UV buffer.
    img = img.transpose(PILImage.Transpose.FLIP_TOP_BOTTOM)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bake_checkerboard_png(
    tex1_color: list[float],
    tex2_color: list[float],
    size: int,
) -> bytes:
    """Bake a checker into an ``n × n`` PNG, where ``n`` is ``size`` rounded up
    to the nearest even number (min 2).

    Rounding to an even period keeps the checker seamless when the texture tiles
    under ``REPEAT`` wrapping — an odd period puts two same-colored cells side by
    side at the wrap seam. One texel per cell keeps edges sharp under ``NEAREST``
    filtering.  Mitsuba tiles via UV scale; we bake the repeats into the image
    instead of relying on a 2×2 texture plus ``KHR_texture_transform`` (which
    blurs under ``LINEAR`` minification in three.js).
    """

    def _linear_to_srgb_byte(c: float) -> int:
        return max(0, min(255, int(round(linear_to_srgb(c) * 255))))

    c1 = tuple(_linear_to_srgb_byte(c) for c in tex1_color[:3])
    c2 = tuple(_linear_to_srgb_byte(c) for c in tex2_color[:3])
    # Round the cell count up to an even number so the (x + y) % 2 pattern tiles
    # seamlessly under REPEAT wrapping (an odd period leaves a seam where two
    # same-colored cells meet).
    n = max(int(round(size)), 2)
    if n % 2:
        n += 1
    img = PILImage.new("RGB", (n, n))
    for y in range(n):
        for x in range(n):
            img.putpixel((x, y), c1 if (x + y) % 2 == 0 else c2)
    # Match Image textures: hakowan UVs use V=0 at the image bottom (OBJ-style).
    img = img.transpose(PILImage.Transpose.FLIP_TOP_BOTTOM)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _resolve_texture_color(texture_like: Any) -> list[float]:
    """Resolve a TextureLike to a single sRGB→linear color (best effort).
    Used inside Checkerboard cells. Float/scalar → grayscale.
    """
    if isinstance(texture_like, Uniform):
        return _color_to_rgba(texture_like.color)
    if isinstance(texture_like, (float, int)):
        v = float(texture_like)
        return [v, v, v, 1.0]
    if isinstance(texture_like, (str, tuple, list)):
        return _color_to_rgba(texture_like)
    logger.warning(
        f"WebGL backend: checkerboard cell type {type(texture_like).__name__} "
        "not directly supported; using gray."
    )
    return [0.5, 0.5, 0.5, 1.0]


def _apply_image_or_checker(
    pbr: dict[str, Any],
    builder: GLTFBuilder,
    reflectance: Any,
    extras: dict[str, Any] | None = None,
) -> None:
    """Attach a baseColorTexture (and optional UV transform) if applicable."""
    if isinstance(reflectance, Image):
        try:
            png_bytes = _load_image_as_png_bytes(reflectance)
            pbr["baseColorTextureIndex"] = builder.add_image_texture(png_bytes)
        except Exception as e:
            logger.warning(
                f"WebGL backend: failed to embed image texture "
                f"'{reflectance.filename}': {e}"
            )
    elif isinstance(reflectance, Checkerboard):
        c1 = _resolve_texture_color(reflectance.texture1)
        c2 = _resolve_texture_color(reflectance.texture2)
        png_bytes = _bake_checkerboard_png(c1, c2, reflectance.size)
        pbr["baseColorTextureIndex"] = builder.add_image_texture(
            png_bytes,
            mag_filter=_FILTER_NEAREST,
            min_filter=_FILTER_NEAREST,
        )
        if extras is not None:
            extras["checkerboard"] = True


def _apply_bump_map(
    extras: dict[str, Any],
    builder: GLTFBuilder,
    bump_map: BumpMap | None,
) -> None:
    """Embed a bump map image and store its glTF texture index in extras.

    glTF has no standard bump-map slot, so we embed the texture as a
    plain glTF image and record its index in ``extras["hakowan"]["bump"]``.
    The viewer JS loads it via ``gltf.parser.loadTexture`` and assigns it
    to ``material.bumpMap`` / ``material.bumpScale`` directly on the
    three.js material.
    """
    if bump_map is None:
        return
    texture = bump_map.texture
    if not isinstance(texture, Image):
        logger.warning(
            f"WebGL backend: BumpMap.texture type {type(texture).__name__} "
            "not supported; only Image is wired."
        )
        return
    try:
        png_bytes = _load_image_as_png_bytes(texture)
        idx = builder.add_image_texture(png_bytes)
        extras["bump"] = {"texture_idx": idx, "scale": float(bump_map.scale)}
    except Exception as e:
        logger.warning(
            f"WebGL backend: failed to embed bump map '{texture.filename}': {e}"
        )


def _apply_normal_map(
    pbr: dict[str, Any], builder: GLTFBuilder, normal_map: NormalMap | None
) -> None:
    """Attach a normalTexture if the view declares a ``NormalMap`` channel."""
    if normal_map is None:
        return
    texture = normal_map.texture
    if not isinstance(texture, Image):
        logger.warning(
            f"WebGL backend: NormalMap.texture type {type(texture).__name__} "
            "not supported; only Image is wired."
        )
        return
    try:
        png_bytes = _load_image_as_png_bytes(texture)
        pbr["normalTextureIndex"] = builder.add_image_texture(png_bytes)
        pbr["normalScale"] = 1.0
    except Exception as e:
        logger.warning(
            f"WebGL backend: failed to embed normal map '{texture.filename}': {e}"
        )


def _read_scalar_attribute(view: View, attr_like: Any) -> np.ndarray | None:
    """Return a 1D float32 array per source-mesh vertex for a ScalarField or
    Isocontour ``data`` reference, or None if the source attribute can't be
    resolved.
    """
    import lagrange

    if view.data_frame is None:
        return None
    if isinstance(attr_like, (ScalarField, Isocontour)):
        ref = attr_like.data
    else:
        ref = attr_like
    name = getattr(ref, "_internal_name", None)
    if not isinstance(name, str):
        return None
    mesh = view.data_frame.mesh
    if not mesh.has_attribute(name):
        return None

    if mesh.is_attribute_indexed(name):
        indexed = mesh.indexed_attribute(name)
        if indexed.element_type != lagrange.AttributeElement.Vertex:
            return None
        values = np.asarray(indexed.values.data, dtype=np.float32).reshape(-1)
        indices = np.asarray(indexed.indices.data, dtype=np.uint32).reshape(-1)
        corner_vertices = mesh.facets.reshape(-1)
        if indices.shape[0] != corner_vertices.shape[0]:
            return None
        out = np.zeros(mesh.num_vertices, dtype=np.float32)
        out[corner_vertices] = values[indices]
        return out

    attr = mesh.attribute(name)
    if attr.element_type != lagrange.AttributeElement.Vertex:
        return None
    arr = np.asarray(attr.data, dtype=np.float32).reshape(-1)
    if arr.shape[0] != mesh.num_vertices:
        return None
    return arr


def _read_color_field(view: View, scalar_field: ScalarField) -> np.ndarray | None:
    """Return (N, 3) float32 *linear* RGB per source-mesh vertex for a
    ``ScalarField``'s baked colormap, or None if it can't be resolved.

    The compiler evaluates the colormap into a per-vertex Color attribute named
    ``data._internal_color_field`` (see ``compiler/color.py``). hakowan stores
    that field in sRGB; the isocontour shader multiplies it into ``diffuseColor``
    in linear space, so we convert here (matching the COLOR_0 path in
    ``mesh_extract``). Indexed/Vertex handling mirrors
    :func:`_read_scalar_attribute` so the result lines up per-vertex with
    ``_scalar_0`` through the de-indexing path.
    """
    import lagrange

    if view.data_frame is None:
        return None
    name = getattr(scalar_field.data, "_internal_color_field", None)
    if not isinstance(name, str):
        return None
    mesh = view.data_frame.mesh
    if not mesh.has_attribute(name):
        return None

    if mesh.is_attribute_indexed(name):
        indexed = mesh.indexed_attribute(name)
        if indexed.element_type != lagrange.AttributeElement.Vertex:
            return None
        values = np.asarray(indexed.values.data, dtype=np.float32).reshape(
            -1, indexed.values.num_channels
        )
        indices = np.asarray(indexed.indices.data, dtype=np.uint32).reshape(-1)
        corner_vertices = mesh.facets.reshape(-1)
        if indices.shape[0] != corner_vertices.shape[0]:
            return None
        out = np.zeros((mesh.num_vertices, values.shape[1]), dtype=np.float32)
        out[corner_vertices] = values[indices]
    else:
        attr = mesh.attribute(name)
        if attr.element_type != lagrange.AttributeElement.Vertex:
            return None
        out = np.asarray(attr.data, dtype=np.float32).reshape(-1, attr.num_channels)
        if out.shape[0] != mesh.num_vertices:
            return None

    rgb = out[:, :3]
    if rgb.shape[1] < 3:  # grayscale color field — broadcast to RGB
        rgb = np.repeat(rgb[:, :1], 3, axis=1)
    return srgb_to_linear_array(np.ascontiguousarray(rgb))


def _resolve_isocontour_band(
    key: str,
    texture: Any,
    attr_name: str,
    view: View,
    custom_attrs: dict[str, np.ndarray],
    iso: dict[str, Any],
) -> None:
    """Resolve one isocontour band (``texture1``/``texture2``) into ``iso``.

    A ``ScalarField`` band carries a colormap, so its colour varies per vertex:
    we bake the baked colour field as a ``vec3`` custom attribute and record
    ``"<key>_attr"`` so the viewer reads it as a varying. Any other band
    collapses to a single flat linear-RGB colour stored under ``"<key>"``.

    A flat ``"<key>"`` is always written so the shader has a uniform fallback
    (the viewer prefers the attribute when ``"<key>_attr"`` is present).
    """
    if isinstance(texture, ScalarField):
        arr = _read_color_field(view, texture)
        if arr is not None:
            custom_attrs[attr_name] = arr
            iso[f"{key}_attr"] = attr_name
            iso[key] = [1.0, 1.0, 1.0]
            return
        logger.warning(
            f"WebGL backend: Isocontour {key} ScalarField colormap could not be "
            "baked (non-vertex element or missing color field); using gray."
        )
    iso[key] = _resolve_texture_color(texture)[:3]


def _apply_isocontour(
    pbr: dict[str, Any],
    extras: dict[str, Any],
    custom_attrs: dict[str, np.ndarray],
    view: View,
    reflectance: Isocontour,
) -> None:
    """Wire an Isocontour reflectance via the viewer's ``onBeforeCompile``
    shader patch.

    Bakes the (pre-scaled) scalar field as ``_scalar_0`` and stores
    ``num_contours / ratio / color1 / color2`` in ``extras["isocontour"]``;
    the viewer's fragment-shader injection then re-creates the contour
    stripes per-pixel from the scalar value — no UV gymnastics or texture
    sampling needed.
    """
    arr = _read_scalar_attribute(view, reflectance)
    if arr is None:
        logger.warning(
            "WebGL backend: Isocontour scalar attribute could not be "
            "resolved (non-vertex element or missing data); falling back "
            "to flat color1."
        )
        c1 = _resolve_texture_color(reflectance.texture1)
        pbr["baseColorFactor"] = list(c1)
        return
    custom_attrs[_SCALAR_ATTR] = arr
    iso: dict[str, Any] = {
        "num_contours": int(reflectance.num_contours),
        "ratio": float(reflectance.ratio),
    }
    # texture1 / texture2 may each be a flat colour or a ScalarField colormap.
    # A colormap band bakes per-vertex colours (``<key>_attr``) the viewer reads
    # as a varying; otherwise the band collapses to a flat uniform colour.
    _resolve_isocontour_band(
        "color1", reflectance.texture1, _ISO_COLOR1_ATTR, view, custom_attrs, iso
    )
    _resolve_isocontour_band(
        "color2", reflectance.texture2, _ISO_COLOR2_ATTR, view, custom_attrs, iso
    )
    extras["isocontour"] = iso
    # The shader multiplies diffuseColor by the per-pixel isoColor; keep
    # the base white so the multiply produces the chosen colors exactly.
    pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]


def _bake_scalar_pbr_factor(
    kind: Literal["roughness", "metallic"],
    texture: Any,
    pbr: dict[str, Any],
    extras: dict[str, Any],
    custom_attrs: dict[str, np.ndarray],
    view: View,
) -> bool:
    """Bake a ``ScalarField`` roughness/metallic factor into a per-vertex attribute.

    The bundled viewer's ``onBeforeCompile`` patch multiplies ``roughnessFactor``
    / ``metalnessFactor`` by the per-vertex attribute (see ``templates/viewer.html``),
    so this path works for any glTF metallic-roughness material — not just
    Principled. ``kind`` is ``"roughness"`` or ``"metallic"``.

    Returns ``True`` if the field was baked, ``False`` otherwise (caller should
    then fall back to a constant factor).
    """
    if not isinstance(texture, ScalarField):
        return False
    arr = _read_scalar_attribute(view, texture)
    if arr is None:
        return False
    attr_name = _ROUGHNESS_ATTR if kind == "roughness" else _METALLIC_ATTR
    factor_key = "roughnessFactor" if kind == "roughness" else "metallicFactor"
    meta_key = "roughness_attr" if kind == "roughness" else "metallic_attr"
    custom_attrs[attr_name] = arr
    extras.setdefault("principled_attrs", {})[meta_key] = attr_name
    pbr[factor_key] = 1.0  # shader multiplies factor × attr
    return True


def _resolve_roughness_factor(
    alpha: Any,
    default: float,
    label: str,
    pbr: dict[str, Any],
    extras: dict[str, Any],
    custom_attrs: dict[str, np.ndarray],
    view: View,
) -> None:
    """Set ``pbr['roughnessFactor']`` from a float or ``ScalarField`` alpha value.

    Floats map directly; ``ScalarField`` is baked per-vertex (see
    :func:`_bake_scalar_pbr_factor`). Other texture types are not expressible as
    a glTF scalar factor, so they downgrade to ``default`` with a warning.
    """
    if isinstance(alpha, (float, int)):
        pbr["roughnessFactor"] = float(alpha)
        return
    if _bake_scalar_pbr_factor("roughness", alpha, pbr, extras, custom_attrs, view):
        return
    pbr["roughnessFactor"] = default
    logger.warning(
        f"WebGL backend: textured {label} type {type(alpha).__name__} not "
        f"supported (only float and ScalarField); using fallback {default}."
    )


def _apply_textured_pbr_factor(
    pbr: dict[str, Any],
    extras: dict[str, Any],
    custom_attrs: dict[str, np.ndarray],
    view: View,
    mat: Principled,
) -> None:
    """Bake ScalarField roughness/metallic into custom attributes."""
    _bake_scalar_pbr_factor("roughness", mat.roughness, pbr, extras, custom_attrs, view)
    _bake_scalar_pbr_factor("metallic", mat.metallic, pbr, extras, custom_attrs, view)


def translate_material(view: View, builder: GLTFBuilder) -> MaterialResult:
    """Translate the view's material channel into a glTF material plan.

    Returns a :class:`MaterialResult` carrying the PBR dict, whether the
    material is two-sided, any per-vertex attributes to plumb onto the mesh
    node (e.g. for isocontour shader injection), and the ``extras`` dict to
    attach to the glTF material so the viewer JS can wire ``onBeforeCompile``
    patches.
    """
    mat = view.material_channel
    if mat is None:
        return MaterialResult(
            pbr={"baseColorFactor": list(_DEFAULT_BASE_COLOR)},
            double_sided=False,
        )

    # A back-side material is rendered via a ``gl_FrontFacing`` fragment-shader
    # branch, which is only valid under double-sided rendering — so back_side
    # forces double-sided regardless of the ``two_sided`` flag.
    back_mat = getattr(mat, "back_side", None)
    double_sided = bool(getattr(mat, "two_sided", False)) or back_mat is not None
    custom_attrs: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}
    if back_mat is not None:
        extras["back"] = {"color": _back_base_color(back_mat)}

    # Intrinsic-colour materials: Conductor / Dielectric / Hair.
    if isinstance(mat, Conductor):
        albedo = _CONDUCTOR_PRESETS.get(mat.material)
        if albedo is None:
            logger.warning(
                f"WebGL backend: conductor preset '{mat.material}' not in "
                "built-in LUT; falling back to neutral gray metal."
            )
            albedo = (0.7, 0.7, 0.7)
        pbr: dict[str, Any] = {
            "baseColorFactor": [
                srgb_to_linear(albedo[0]),
                srgb_to_linear(albedo[1]),
                srgb_to_linear(albedo[2]),
                1.0,
            ],
            "metallicFactor": 1.0,
            "roughnessFactor": 0.1,
        }
        if isinstance(mat, RoughConductor):
            _resolve_roughness_factor(
                mat.alpha,
                0.2,
                "RoughConductor.alpha",
                pbr,
                extras,
                custom_attrs,
                view,
            )
        _apply_normal_map(pbr, builder, view.normal_map)
        _apply_bump_map(extras, builder, view.bump_map)
        return MaterialResult(
            pbr=pbr,
            double_sided=double_sided,
            custom_attrs=custom_attrs,
            extras={"hakowan": extras} if extras else None,
        )

    if isinstance(mat, Dielectric):
        # Translate to glTF KHR_materials_transmission + _ior (+ _volume when
        # a medium is attached). three.js's GLTFLoader maps these onto
        # MeshPhysicalMaterial → real refraction + Beer-Lambert volume tint.
        int_ior = _resolve_ior(mat.int_ior, 1.5046)
        ext_ior = _resolve_ior(mat.ext_ior, 1.0)
        # glTF stores the relative IOR (assumes air-side); three.js does the
        # same Snell-law math internally.
        relative_ior = int_ior / max(ext_ior, 1e-6)

        pbr = {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.0,
            "transmissionFactor": float(mat.specular_transmittance),
            "ior": relative_ior,
        }
        if isinstance(mat, RoughDielectric):
            _resolve_roughness_factor(
                mat.alpha,
                0.1,
                "RoughDielectric.alpha",
                pbr,
                extras,
                custom_attrs,
                view,
            )

        if isinstance(mat, ThinDielectric):
            # Thin slab: no refraction offset, no volume absorption.
            pbr["thicknessFactor"] = 0.0
        elif mat.medium is not None:
            # Beer-Lambert via KHR_materials_volume. We estimate a sensible
            # thickness from the view bbox so refraction has a length scale
            # to integrate against. ``medium.albedo`` is the per-channel
            # transmitted tint; ``medium.scale`` is an extinction multiplier
            # (Mitsuba treats it as ``scale * bbox_diag``).
            from numpy.linalg import norm as _norm

            bbox = getattr(view, "bbox", None)
            if bbox is not None:
                diag = float(_norm(np.asarray(bbox[0]) - np.asarray(bbox[1])))
            else:
                diag = 1.0
            pbr["thicknessFactor"] = diag
            attn_color = _color_to_rgba(mat.medium.albedo)[:3]
            # Larger ``scale`` → shorter attenuation distance (more absorption).
            scale = float(mat.medium.scale)
            attn_distance = diag / scale if scale > 1e-6 else float("inf")
            pbr["attenuationDistance"] = attn_distance
            pbr["attenuationColor"] = attn_color
        else:
            # Smooth solid dielectric with no volume — still pick a thickness
            # so refraction has an offset, but skip attenuation entirely.
            from numpy.linalg import norm as _norm

            bbox = getattr(view, "bbox", None)
            if bbox is not None:
                pbr["thicknessFactor"] = float(
                    _norm(np.asarray(bbox[0]) - np.asarray(bbox[1]))
                )
            else:
                pbr["thicknessFactor"] = 1.0

        # ``specular_reflectance`` modulates the Fresnel reflection. We don't
        # expose it (would need KHR_materials_specular) when it's the default
        # 1.0 — log only when non-default and ignored.
        if float(mat.specular_reflectance) != 1.0:
            logger.debug(
                "WebGL backend: Dielectric.specular_reflectance != 1.0 not "
                "yet wired (would require KHR_materials_specular)."
            )

        _apply_normal_map(pbr, builder, view.normal_map)
        _apply_bump_map(extras, builder, view.bump_map)
        return MaterialResult(
            pbr=pbr,
            double_sided=double_sided,
            custom_attrs=custom_attrs,
            extras={"hakowan": extras} if extras else None,
        )

    if isinstance(mat, Hair):
        logger.warning(
            "WebGL backend: Hair material not supported; using brown diffuse."
        )
        return MaterialResult(
            pbr={
                "baseColorFactor": _color_to_rgba("saddlebrown"),
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            },
            double_sided=double_sided,
            extras={"hakowan": extras} if extras else None,
        )

    # Reflectance-bearing materials.
    reflectance = _resolve_reflectance(mat)
    if reflectance is None:
        logger.warning(
            f"WebGL backend: material type {type(mat).__name__} is not yet "
            "supported; falling back to gray diffuse."
        )
        return MaterialResult(
            pbr={
                "baseColorFactor": list(_DEFAULT_BASE_COLOR),
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            double_sided=double_sided,
            extras={"hakowan": extras} if extras else None,
        )

    pbr = {
        "baseColorFactor": _reflectance_to_base_color(reflectance),
    }
    _apply_image_or_checker(pbr, builder, reflectance, extras)

    if isinstance(reflectance, Isocontour):
        _apply_isocontour(pbr, extras, custom_attrs, view, reflectance)

    # Roughness / metallic factors.
    if isinstance(mat, Principled):
        if isinstance(mat.roughness, (float, int)):
            pbr["roughnessFactor"] = float(mat.roughness)
        elif not isinstance(mat.roughness, ScalarField):
            pbr["roughnessFactor"] = 0.5
            logger.warning(
                "WebGL backend: textured Principled.roughness type "
                f"{type(mat.roughness).__name__} not supported; using 0.5."
            )
        if isinstance(mat.metallic, (float, int)):
            pbr["metallicFactor"] = float(mat.metallic)
        elif not isinstance(mat.metallic, ScalarField):
            pbr["metallicFactor"] = 0.0
            logger.warning(
                "WebGL backend: textured Principled.metallic type "
                f"{type(mat.metallic).__name__} not supported; using 0.0."
            )
        _apply_textured_pbr_factor(pbr, extras, custom_attrs, view, mat)
    elif isinstance(mat, RoughPlastic):
        # RoughPlastic.alpha is float-only across all backends (grammar types it
        # as float; Mitsuba/Blender don't bake a texture for it either).
        pbr["metallicFactor"] = 0.0
        if isinstance(mat.alpha, (float, int)):
            pbr["roughnessFactor"] = float(mat.alpha)
        else:
            pbr["roughnessFactor"] = 0.3
            logger.warning(
                "WebGL backend: textured RoughPlastic.alpha not supported; "
                "using fallback 0.3."
            )
    elif isinstance(mat, Plastic):
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 0.1
    elif isinstance(mat, Diffuse):
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 1.0

    _apply_normal_map(pbr, builder, view.normal_map)
    _apply_bump_map(extras, builder, view.bump_map)
    return MaterialResult(
        pbr=pbr,
        double_sided=double_sided,
        custom_attrs=custom_attrs,
        extras={"hakowan": extras} if extras else None,
    )
