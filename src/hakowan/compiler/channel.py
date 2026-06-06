from .view import View
from .attribute import compute_scaled_attribute
from .color import apply_colormap
from .texture import apply_texture
from .utils import unique_name

from ..common import logger
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
from ..grammar.channel.material import (
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
    ThinPrincipled,
)
from ..grammar.channel.curvestyle import Bend
from ..grammar.dataframe import DataFrame
from ..grammar.mark import Mark
from ..grammar.scale import Attribute, to_attribute
from ..grammar.texture import Texture, Uniform, Image

import numpy as np

### Public API


def preprocess_channels(view: View):
    """Preprocess channels in a view.

    Determine the active position, normal, size, uv and material channels. Among these, position,
    normal and uv channels can be automatically generate from data frame if not specified. Size and
    material will be set to default if not specified.

    Args:
        view (View): The view to be pre-processed. Update will be made in place.
    """
    _preprocess_channels(view)


def process_channels(view: View):
    """Process the channels in a view.

    This step applies scales and textures on the corresponding data.

    Args:
        view (View): The view to be processed. Update will be made in place.
    """
    _process_channels(view)


### Private API


def _preprocess_channels(view: View):
    assert view.data_frame is not None
    for channel in view.channels:
        match channel:
            case Position():
                if view.position_channel is None:
                    view.position_channel = channel
            case Normal():
                if view.normal_channel is None:
                    view.normal_channel = channel
            case Size():
                if view.size_channel is None:
                    view.size_channel = channel
            case VectorField():
                if view.vector_field_channel is None:
                    view.vector_field_channel = channel
            case Covariance():
                if view.covariance_channel is None:
                    view.covariance_channel = channel
            case Shape():
                if view.shape_channel is None:
                    view.shape_channel = channel
            case Material():
                if view.material_channel is None:
                    view.material_channel = channel
            case BumpMap():
                if view.bump_map is None:
                    view.bump_map = channel
            case NormalMap():
                if view.normal_map is None:
                    view.normal_map = channel
            case _:
                raise NotImplementedError(
                    f"Channel type {type(channel)} is not supported"
                )

    # Generate default material channel if not specified.
    if view.material_channel is None:
        view.material_channel = Plastic(diffuse_reflectance=Uniform(color="ivory"))


def _process_channels(view: View):
    assert view.data_frame is not None
    df = view.data_frame
    if view.position_channel is not None:
        assert isinstance(view.position_channel, Position)
        assert isinstance(view.position_channel.data, Attribute)
        attr = view.position_channel.data
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
    if view.normal_channel is not None:
        assert isinstance(view.normal_channel, Normal)
        assert isinstance(view.normal_channel.data, Attribute)
        attr = view.normal_channel.data
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
    if view.size_channel is not None:
        assert isinstance(view.size_channel, Size)
        assert isinstance(view.size_channel.data, (Attribute, float))
        if isinstance(view.size_channel.data, Attribute):
            attr = view.size_channel.data
            compute_scaled_attribute(df, attr)
            view._active_attributes.append(attr)
    if view.vector_field_channel is not None:
        assert isinstance(view.vector_field_channel, VectorField)
        assert isinstance(view.vector_field_channel.data, Attribute)
        attr = view.vector_field_channel.data
        if view.vector_field_channel.normalize:
            _normalize_vector_field(df, attr)
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
        match view.vector_field_channel.style:
            case Bend():
                style = view.vector_field_channel.style
                style.direction = to_attribute(style.direction)
                compute_scaled_attribute(df, style.direction)
                view._active_attributes.append(style.direction)
    if view.covariance_channel is not None:
        assert isinstance(view.covariance_channel, Covariance)
        assert isinstance(view.covariance_channel.data, Attribute)
        attr = view.covariance_channel.data
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
    if view.shape_channel is not None:
        assert isinstance(view.shape_channel, Shape)
        assert view.shape_channel.base_shape in ["sphere", "cube", "disk"]
        if view.shape_channel.orientation is not None:
            view.shape_channel.orientation = to_attribute(
                view.shape_channel.orientation
            )
            attr = view.shape_channel.orientation
            compute_scaled_attribute(df, attr)
            view._active_attributes.append(attr)
    if view.bump_map is not None:
        tex = view.bump_map.texture
        assert tex is not None
        if isinstance(tex, Texture):
            view._active_attributes += apply_texture(df, tex, view.uv_attribute)
            view.uv_attribute = tex._uv
    if view.normal_map is not None:
        tex = view.normal_map.texture
        assert tex is not None
        if isinstance(tex, Texture):
            if isinstance(tex, Image):
                if not tex.raw:
                    logger.warning(
                        "Normal map texture image not in raw format may lead to incorrect result"
                    )
            view._active_attributes += apply_texture(df, tex, view.uv_attribute)
            view.uv_attribute = tex._uv
    if view.material_channel is not None:
        _validate_back_side(view)
        _process_material(view, df, view.material_channel)
        if view.material_channel.back_side is not None:
            _process_material(view, df, view.material_channel.back_side)


def _validate_back_side(view: View):
    """Normalize ``material_channel.back_side`` and warn on unsupported uses.

    ``back_side`` is only meaningful for surface marks, and a nested
    ``back_side`` on the back-side material itself has no meaning (a facet has
    exactly two sides). Both are dropped here (the view's material is a deep
    copy, so this never mutates the user's spec). Done once so all backends see
    a clean ``back_side``.
    """
    mat = view.material_channel
    assert mat is not None
    back = mat.back_side
    if back is None:
        return
    if view.mark is not Mark.Surface:
        logger.warning(
            "Material.back_side is only supported for surface marks; "
            f"ignoring it for the {view.mark} mark."
        )
        mat.back_side = None
        return
    if back.back_side is not None:
        logger.warning(
            "A nested Material.back_side on a back-side material is not "
            "meaningful; ignoring it."
        )
        back.back_side = None


def _process_material(view: View, df: DataFrame, mat: Material):
    """Resolve the texture-typed fields (color/alpha/roughness/metallic) of a
    single material against the data frame, accumulating active attributes and
    threading the shared UV attribute. Called for both the front material and
    its ``back_side`` (whose textures must share the front's UVs).
    """
    match mat:
        case Diffuse():
            if isinstance(mat.reflectance, Texture):
                tex = mat.reflectance
                view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                view.uv_attribute = tex._uv
                apply_colormap(df, tex)
        case RoughConductor() | RoughDielectric():
            if isinstance(mat.alpha, Texture):
                tex = mat.alpha
                view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                view.uv_attribute = tex._uv
                apply_colormap(df, tex)  # TODO: is this needed?
        case Conductor() | Dielectric() | ThinDielectric() | Hair():
            # Nothing to do.
            pass
        case RoughPlastic() | Plastic():
            if isinstance(mat.diffuse_reflectance, Texture):
                tex = mat.diffuse_reflectance
                view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                view.uv_attribute = tex._uv
                apply_colormap(df, tex)
            if isinstance(mat.specular_reflectance, Texture):
                tex = mat.specular_reflectance
                view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                view.uv_attribute = tex._uv
        case Principled() | ThinPrincipled():
            if isinstance(mat.color, Texture):
                tex = mat.color
                view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                view.uv_attribute = tex._uv
                apply_colormap(df, tex)
            if isinstance(mat.metallic, Texture):
                tex = mat.metallic
                view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                view.uv_attribute = tex._uv
            if isinstance(mat.roughness, Texture):
                tex = mat.roughness
                view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                view.uv_attribute = tex._uv
        case _:
            raise NotImplementedError(f"Channel type {type(mat)} is not supported")


def _normalize_vector_field(df: DataFrame, attr: Attribute):
    """Rescale a vector-field attribute to unit length in place.

    A new attribute holding the unit-length vectors is created, and ``attr`` is
    repointed to it so that any scale attached to ``attr`` is subsequently
    applied on top of the normalized field. This is idempotent across recompiles
    (guarded by ``attr._internal_name`` being already set).
    """
    if attr._internal_name is not None:
        # Already processed in a previous compile pass.
        return
    mesh = df.mesh
    assert mesh is not None
    if mesh.is_attribute_indexed(attr.name):
        logger.warning(
            "Vector field normalization is not supported for indexed attributes; "
            "skipping normalization."
        )
        return

    src = mesh.attribute(attr.name)
    values = np.asarray(src.data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        logger.warning(
            "Vector field normalization expects a vector attribute; skipping."
        )
        return

    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    unit = values / lengths

    unit_name = unique_name(mesh, f"_unit_{attr.name}")
    mesh.create_attribute(
        unit_name,
        element=src.element_type,
        usage=src.usage,
        initial_values=unit,
    )
    attr.name = unit_name
