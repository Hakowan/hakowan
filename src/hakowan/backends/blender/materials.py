"""Material, shader and texture-node construction for the Blender backend."""

from ...common import logger
from ...compiler import View
from ...grammar.channel.material import (
    Conductor,
    Dielectric,
    Diffuse,
    Hair,
    Plastic,
    Principled,
    RoughConductor,
    RoughDielectric,
    RoughPlastic,
    ThinDielectric,
    ThinPrincipled,
)
from ...grammar.scale import Attribute
from ...grammar.texture import (
    ScalarField,
    Checkerboard,
    Image,
    Texture,
    Uniform,
)

import numpy as np
import lagrange

import bpy
from ._common import _ensure_nodes


class _MaterialMixin:
    _conductor_colors: dict[str, tuple[float, float, float]] = {
        "Au": (1.0, 0.78, 0.34),
        "Ag": (0.97, 0.96, 0.91),
        "Cu": (0.95, 0.64, 0.54),
        "Al": (0.91, 0.92, 0.92),
        "Fe": (0.56, 0.57, 0.58),
        "Cr": (0.55, 0.55, 0.55),
        "Pt": (0.67, 0.64, 0.59),
        "W": (0.50, 0.50, 0.50),
        "Ti": (0.62, 0.58, 0.54),
        "Ni": (0.66, 0.63, 0.58),
        "V": (0.55, 0.55, 0.55),
        "none": (0.8, 0.8, 0.8),
    }

    # Common IOR values for named Mitsuba dielectric presets.

    _ior_presets: dict[str, float] = {
        "vacuum": 1.0,
        "air": 1.000277,
        "water": 1.333,
        "ice": 1.31,
        "glass": 1.5,
        "bk7": 1.5046,
        "diamond": 2.419,
        "fused_quartz": 1.458,
        "polycarbonate": 1.584,
        "acrylic": 1.49,
        "sodium_chloride": 1.544,
        "amber": 1.55,
        "pet": 1.575,
    }

    def _resolve_ior(self, ior: str | float) -> float:
        """Resolve an IOR value from a string preset name or float."""
        if isinstance(ior, (int, float)):
            return float(ior)
        return self._ior_presets.get(ior, 1.5)

    def _create_material(
        self,
        view: View,
        index: int,
        *,
        color_layer_name: str | None = None,
        override_color: tuple[float, float, float, float] | None = None,
        material_suffix: str | None = None,
        uv_layer_name: str | None = None,
    ):
        """Create Blender material from view's material channel.

        Args:
            view: View with material channel.
            index: Material index.
            color_layer_name: If set, use mesh/curve color attribute (ScalarField).
            override_color: If set, use this RGBA as base color (e.g. per-point).
            material_suffix: Optional suffix for material name (e.g. point index).
            uv_layer_name: If set, use UV layer for texture mapping (Checkerboard).

        Returns:
            Blender material or None.
        """
        if view.material_channel is None:
            return None

        mat_data = view.material_channel
        mat_name = f"material_{index:03d}"
        if material_suffix is not None:
            mat_name = f"{mat_name}_{material_suffix}"
        mat = bpy.data.materials.new(name=mat_name)
        _ensure_nodes(mat)
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Hair and ThinDielectric use dedicated shader node graphs that the
        # front/back Principled-mix path below does not cover, so a back_side on
        # such a front material is ignored.
        if (
            isinstance(mat_data, (Hair, ThinDielectric))
            and mat_data.back_side is not None
        ):
            logger.warning(
                f"Blender backend: back_side is ignored for "
                f"{type(mat_data).__name__} front materials."
            )
        if isinstance(mat_data, Hair):
            return self._create_hair_material(mat, mat_data, nodes, links)
        if isinstance(mat_data, ThinDielectric):
            return self._create_thin_dielectric_material(mat, mat_data, nodes, links)

        # Create Principled BSDF
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)

        # Material output. The BSDF is linked to it below — directly when there
        # is no back_side, or through a Backfacing mix shader when there is.
        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (600, 0)

        # Base color: checkerboard, scalar field, override, or material default.
        checkerboard_tex = self._get_checkerboard_texture(view)
        if checkerboard_tex is not None and uv_layer_name is not None:
            # Create checkerboard shader node network
            checker_node = self._create_checkerboard_shader(
                nodes, links, checkerboard_tex, uv_layer_name
            )
            if checker_node is not None:
                links.new(checker_node.outputs["Color"], bsdf.inputs["Base Color"])
            else:
                # Fallback if checkerboard creation failed
                bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        elif (
            image_tex := self._get_image_texture(view)
        ) is not None and uv_layer_name is not None:
            tex_node = self._build_image_texture_node(
                image_tex, nodes, links, uv_layer_name, is_data=False
            )
            if tex_node is not None:
                links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
            else:
                bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        elif color_layer_name is not None:
            attr_node = nodes.new(type="ShaderNodeAttribute")
            attr_node.location = (-200, 0)
            attr_node.attribute_name = color_layer_name
            links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
        elif override_color is not None:
            bsdf.inputs["Base Color"].default_value = override_color
        else:
            color = self._extract_material_color(mat_data)
            if color is not None:
                bsdf.inputs["Base Color"].default_value = color
            else:
                bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

        # Configure based on material type (non-color inputs).
        self._apply_bsdf_inputs(
            mat,
            bsdf,
            mat_data,
            warn_unsupported=(color_layer_name is None and override_color is None),
        )

        # Normal / bump maps drive the Principled BSDF's "Normal" input. When
        # both are present the normal map feeds the bump node so they compose
        # (mirrors Mitsuba's bumpmap-over-normalmap nesting). Both require UVs.
        normal_socket = None
        if uv_layer_name is not None and view.normal_map is not None:
            normal_socket = self._build_normal_map_node(
                view.normal_map, nodes, links, uv_layer_name
            )
        if uv_layer_name is not None and view.bump_map is not None:
            bump_socket = self._build_bump_map_node(
                view.bump_map, nodes, links, uv_layer_name, normal_socket
            )
            if bump_socket is not None:
                normal_socket = bump_socket
        if normal_socket is not None and "Normal" in bsdf.inputs:
            links.new(normal_socket, bsdf.inputs["Normal"])

        # Front/back routing: link the front BSDF straight to the output, or —
        # when a back_side material is set — mix it with a separate back BSDF
        # selected by the Geometry node's "Backfacing" output (1.0 on back
        # faces, which routes the MixShader to its second shader input).
        back = mat_data.back_side
        if back is None:
            links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        else:
            back_bsdf = self._build_back_bsdf(mat, back, nodes, links, normal_socket)
            geo = nodes.new(type="ShaderNodeNewGeometry")
            geo.location = (0, -900)
            mix = nodes.new(type="ShaderNodeMixShader")
            mix.location = (300, 0)
            links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
            links.new(bsdf.outputs["BSDF"], mix.inputs[1])  # Fac=0 → front
            links.new(back_bsdf.outputs["BSDF"], mix.inputs[2])  # Fac=1 → back
            links.new(mix.outputs["Shader"], output.inputs["Surface"])
            mat.use_backface_culling = False

        # Two-sided rendering: disable backface culling
        if mat_data.two_sided:
            mat.use_backface_culling = False

        return mat

    def _apply_bsdf_inputs(self, mat, bsdf, mat_data, *, warn_unsupported: bool):
        """Set the non-color Principled BSDF inputs for a material type.

        Operates on the given ``bsdf`` node so it can configure both the front
        BSDF and a back_side BSDF. ``mat`` is the owning Blender material (used
        only to disable backface culling for inherently two-sided types).
        """
        match mat_data:
            case Diffuse():
                bsdf.inputs["Roughness"].default_value = 1.0
                bsdf.inputs["Metallic"].default_value = 0.0

            case Plastic() | RoughPlastic():
                bsdf.inputs["Roughness"].default_value = 1
                bsdf.inputs["Metallic"].default_value = 0.0
                bsdf.inputs["Coat Weight"].default_value = 1
                bsdf.inputs["Coat IOR"].default_value = 1.49
                if isinstance(mat_data, RoughPlastic):
                    # ``alpha`` is the microfacet roughness of the glossy coat.
                    alpha = (
                        mat_data.alpha
                        if isinstance(mat_data.alpha, (int, float))
                        else 0.1
                    )
                    bsdf.inputs["Coat Roughness"].default_value = float(alpha)
                else:
                    bsdf.inputs["Coat Roughness"].default_value = 0

            case RoughConductor():
                bsdf.inputs["Metallic"].default_value = 1.0
                alpha = (
                    mat_data.alpha if isinstance(mat_data.alpha, (int, float)) else 0.1
                )
                bsdf.inputs["Roughness"].default_value = float(alpha)

            case Conductor():
                bsdf.inputs["Metallic"].default_value = 1.0
                bsdf.inputs["Roughness"].default_value = 0.0

            case RoughDielectric():
                bsdf.inputs[
                    "Transmission Weight"
                ].default_value = mat_data.specular_transmittance
                # ``specular_reflectance`` is a [0,1] reflectance multiplier where
                # 1.0 means "unchanged" (Mitsuba convention). Blender's "Specular
                # IOR Level" is neutral at 0.5 (1.0 ≈ double specular), so map
                # multiplier → 0.5 * multiplier.
                bsdf.inputs["Specular IOR Level"].default_value = (
                    0.5 * mat_data.specular_reflectance
                )
                bsdf.inputs["IOR"].default_value = self._resolve_ior(mat_data.int_ior)
                alpha = (
                    mat_data.alpha if isinstance(mat_data.alpha, (int, float)) else 0.1
                )
                bsdf.inputs["Roughness"].default_value = float(alpha)
                bsdf.inputs["Metallic"].default_value = 0.0

            case Dielectric():
                bsdf.inputs[
                    "Transmission Weight"
                ].default_value = mat_data.specular_transmittance
                # See RoughDielectric: remap [0,1] reflectance multiplier (1.0 =
                # unchanged) onto Blender's 0.5-neutral "Specular IOR Level".
                bsdf.inputs["Specular IOR Level"].default_value = (
                    0.5 * mat_data.specular_reflectance
                )
                bsdf.inputs["IOR"].default_value = self._resolve_ior(mat_data.int_ior)
                bsdf.inputs["Roughness"].default_value = 0.0
                bsdf.inputs["Metallic"].default_value = 0.0

            case ThinPrincipled():
                bsdf.inputs["Roughness"].default_value = float(
                    mat_data.roughness
                    if isinstance(mat_data.roughness, (int, float))
                    else 0.5
                )
                bsdf.inputs["Metallic"].default_value = float(
                    mat_data.metallic
                    if isinstance(mat_data.metallic, (int, float))
                    else 0.0
                )
                bsdf.inputs["Transmission Weight"].default_value = mat_data.spec_trans
                bsdf.inputs["IOR"].default_value = mat_data.eta
                mat.use_backface_culling = False

            case Principled():
                bsdf.inputs["Roughness"].default_value = float(
                    mat_data.roughness
                    if isinstance(mat_data.roughness, (int, float))
                    else 0.5
                )
                bsdf.inputs["Metallic"].default_value = float(
                    mat_data.metallic
                    if isinstance(mat_data.metallic, (int, float))
                    else 0.0
                )
                bsdf.inputs["Anisotropic"].default_value = mat_data.anisotropic
                bsdf.inputs["Transmission Weight"].default_value = mat_data.spec_trans
                bsdf.inputs["IOR"].default_value = mat_data.eta
                bsdf.inputs["Sheen Weight"].default_value = mat_data.sheen

            case _:
                if warn_unsupported:
                    logger.warning(
                        f"Material type {type(mat_data)} not fully supported, using default"
                    )

    def _build_back_bsdf(self, mat, back_mat, nodes, links, normal_socket):
        """Build a Principled BSDF for the back face from a uniform-color material.

        Back faces support a flat (uniform) color only; textured back colors
        (ScalarField/Image/Checkerboard) fall back to gray with a warning.
        """
        back_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        back_bsdf.location = (0, -500)
        color = self._extract_material_color(back_mat)
        if color is None:
            logger.warning(
                "Blender backend: back_side color is not a uniform color "
                "(textured back faces are not supported); using gray."
            )
            color = (0.8, 0.8, 0.8, 1.0)
        back_bsdf.inputs["Base Color"].default_value = color
        self._apply_bsdf_inputs(mat, back_bsdf, back_mat, warn_unsupported=True)
        if normal_socket is not None and "Normal" in back_bsdf.inputs:
            links.new(normal_socket, back_bsdf.inputs["Normal"])
        return back_bsdf

    def _create_hair_material(self, mat, mat_data: Hair, nodes, links):
        """Create a Principled Hair BSDF material.

        Args:
            mat: Blender material.
            mat_data: Hair material data.
            nodes: Shader node tree nodes.
            links: Shader node tree links.

        Returns:
            Blender material.
        """
        hair_bsdf = nodes.new(type="ShaderNodeBsdfHairPrincipled")
        hair_bsdf.location = (0, 0)
        # Use melanin concentration parametrization
        hair_bsdf.parametrization = "MELANIN"
        hair_bsdf.inputs["Melanin"].default_value = mat_data.eumelanin / 8.0
        hair_bsdf.inputs["Melanin Redness"].default_value = (
            mat_data.pheomelanin / (mat_data.eumelanin + mat_data.pheomelanin)
            if (mat_data.eumelanin + mat_data.pheomelanin) > 0
            else 0.5
        )
        hair_bsdf.inputs["Roughness"].default_value = 0.3
        hair_bsdf.inputs["Coat"].default_value = 0.0

        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)
        links.new(hair_bsdf.outputs["BSDF"], output.inputs["Surface"])

        return mat

    def _create_thin_dielectric_material(
        self, mat, mat_data: ThinDielectric, nodes, links
    ):
        """Create a thin dielectric (thin glass) material.

        Models a thin sheet of glass where the two refractions cancel out:
        light passes straight through while Fresnel reflections are preserved.
        Built as a Fresnel-driven mix of Transparent BSDF and Glossy BSDF.

        Args:
            mat: Blender material.
            mat_data: ThinDielectric material data.
            nodes: Shader node tree nodes.
            links: Shader node tree links.

        Returns:
            Blender material.
        """
        ior = self._resolve_ior(mat_data.int_ior)

        # Transparent BSDF: light passes straight through
        transparent = nodes.new(type="ShaderNodeBsdfTransparent")
        transparent.location = (-200, 100)

        # Glass BSDF: reflection + refraction for a thicker glass appearance
        glass = nodes.new(type="ShaderNodeBsdfGlass")
        glass.location = (-200, -100)
        glass.inputs["Roughness"].default_value = 0.0
        glass.inputs["IOR"].default_value = ior

        # Fresnel node drives the mix based on viewing angle and IOR
        fresnel = nodes.new(type="ShaderNodeFresnel")
        fresnel.location = (-400, 0)
        fresnel.inputs["IOR"].default_value = ior

        # Scale Fresnel by specular_reflectance to control reflection intensity
        scale = nodes.new(type="ShaderNodeMath")
        scale.location = (-200, 0)
        scale.operation = "MULTIPLY"
        links.new(fresnel.outputs["Fac"], scale.inputs[0])
        scale.inputs[1].default_value = mat_data.specular_reflectance

        # Mix Shader: factor=0 → transparent, factor=1 → glass
        mix = nodes.new(type="ShaderNodeMixShader")
        mix.location = (0, 0)
        links.new(scale.outputs["Value"], mix.inputs["Fac"])
        links.new(transparent.outputs["BSDF"], mix.inputs[1])
        links.new(glass.outputs["BSDF"], mix.inputs[2])

        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (200, 0)
        links.new(mix.outputs["Shader"], output.inputs["Surface"])

        # Thin dielectric is inherently two-sided
        mat.use_backface_culling = False

        return mat

    def _create_checkerboard_shader(
        self, nodes, links, checkerboard_tex: Checkerboard, uv_layer_name: str
    ):
        """Create a checkerboard shader node network.

        Args:
            nodes: Shader node tree nodes.
            links: Shader node tree links.
            checkerboard_tex: Checkerboard texture configuration.
            uv_layer_name: Name of the UV layer to use.

        Returns:
            Output node that provides the checkerboard color, or None if failed.
        """
        # UV Map node
        uv_node = nodes.new(type="ShaderNodeUVMap")
        uv_node.location = (-800, 0)
        uv_node.uv_map = uv_layer_name

        # Mapping node for scaling
        mapping_node = nodes.new(type="ShaderNodeMapping")
        mapping_node.location = (-600, 0)
        mapping_node.inputs["Scale"].default_value = (
            float(checkerboard_tex.size),
            float(checkerboard_tex.size),
            1.0,
        )
        links.new(uv_node.outputs["UV"], mapping_node.inputs["Vector"])

        # Checker Texture node
        checker_node = nodes.new(type="ShaderNodeTexChecker")
        checker_node.location = (-400, 0)
        checker_node.inputs["Scale"].default_value = 1.0  # Scale is handled by mapping
        links.new(mapping_node.outputs["Vector"], checker_node.inputs["Vector"])

        # Extract colors from texture1 and texture2
        color1 = self._texture_to_color(checkerboard_tex.texture1)
        color2 = self._texture_to_color(checkerboard_tex.texture2)

        # Set checker colors
        if color1 is not None:
            checker_node.inputs["Color1"].default_value = color1
        if color2 is not None:
            checker_node.inputs["Color2"].default_value = color2

        return checker_node

    def _texture_to_color(self, texture) -> tuple[float, float, float, float] | None:
        """Convert a TextureLike to an RGBA color.

        Args:
            texture: Texture or color value.

        Returns:
            RGBA tuple or None if cannot be converted.
        """
        if isinstance(texture, Uniform):
            return self._extract_color(texture.color)
        elif isinstance(texture, (str, int, float, list, tuple)):
            return self._extract_color(texture)
        else:
            # Cannot convert complex textures like ScalarField or nested Checkerboard
            logger.warning(
                f"Cannot convert texture type {type(texture)} to color for checkerboard"
            )
            return None

    def _extract_material_color(
        self, mat_data
    ) -> tuple[float, float, float, float] | None:
        """Extract base color from any supported material type.

        Args:
            mat_data: Material channel data.

        Returns:
            RGBA tuple or None.
        """
        match mat_data:
            case Diffuse():
                return self._extract_color(mat_data.reflectance)
            case Plastic() | RoughPlastic():
                return self._extract_color(mat_data.diffuse_reflectance)
            case Principled() | ThinPrincipled():
                return self._extract_color(mat_data.color)
            case RoughConductor() | Conductor():
                name = mat_data.material
                rgb = self._conductor_colors.get(name, (0.8, 0.8, 0.8))
                return (rgb[0], rgb[1], rgb[2], 1.0)
            case Dielectric() | ThinDielectric() | RoughDielectric():
                # Glass-like materials: white/clear base
                return (1.0, 1.0, 1.0, 1.0)
            case _:
                return None

    def _extract_color(self, color_data) -> tuple[float, float, float, float] | None:
        """Extract RGBA color from various color representations.

        Args:
            color_data: Color data (str, tuple, or texture).

        Returns:
            RGBA tuple or None.
        """
        from ...common.to_color import to_color

        if isinstance(color_data, str):
            rgb = to_color(color_data)
            return (rgb[0], rgb[1], rgb[2], 1.0)
        elif isinstance(color_data, (list, tuple)):
            if len(color_data) == 3:
                return (color_data[0], color_data[1], color_data[2], 1.0)
            elif len(color_data) == 4:
                return tuple(color_data)
        # Texture types (e.g. ScalarField) are handled via mesh color attributes
        return None

    def _get_scalar_field_color_attr(self, view: View):
        """If the view's material uses a ScalarField color, return (attr_name, element_type).

        Returns:
            (attr_name, element_type) from the mesh, or None if not a scalar field color.
        """
        if view.material_channel is None or view.data_frame is None:
            return None
        mat_data = view.material_channel
        mesh = view.data_frame.mesh

        def check(tex):
            if isinstance(tex, ScalarField):
                if not isinstance(tex.data, Attribute) or not getattr(
                    tex.data, "_internal_color_field", None
                ):
                    return None
                name = tex.data._internal_color_field
                if not mesh.has_attribute(name):
                    return None
                attr = mesh.attribute(name)
                return (name, attr.element_type)
            return None

        match mat_data:
            case Diffuse():
                out = check(mat_data.reflectance)
            case Plastic() | RoughPlastic():
                out = check(mat_data.diffuse_reflectance)
            case Principled() | ThinPrincipled():
                out = check(mat_data.color)
            case _:
                out = None
        return out

    def _get_checkerboard_texture(self, view: View):
        """If the view's material uses a Checkerboard texture, return the texture.

        Returns:
            Checkerboard texture or None.
        """
        if view.material_channel is None or view.data_frame is None:
            return None
        mat_data = view.material_channel

        def check(tex):
            if isinstance(tex, Checkerboard):
                return tex
            return None

        match mat_data:
            case Diffuse():
                return check(mat_data.reflectance)
            case Plastic() | RoughPlastic():
                return check(mat_data.diffuse_reflectance)
            case Principled() | ThinPrincipled():
                return check(mat_data.color)
            case _:
                return None

    def _get_image_texture(self, view: View) -> Image | None:
        """Return the material's reflectance/color ``Image`` texture, or None."""
        if view.material_channel is None:
            return None
        mat_data = view.material_channel

        def check(tex):
            return tex if isinstance(tex, Image) else None

        match mat_data:
            case Diffuse():
                return check(mat_data.reflectance)
            case Plastic() | RoughPlastic():
                return check(mat_data.diffuse_reflectance)
            case Principled() | ThinPrincipled():
                return check(mat_data.color)
            case _:
                return None

    def _resolve_uv_attr_name(self, view: View) -> str | None:
        """Internal name of the UV attribute used by any UV-bearing texture.

        Checks the base-color Checkerboard/Image plus the normal/bump map
        textures (the compiler stores the resolved UV on ``texture._uv``).
        Returns the first available, or None.
        """
        candidates: list = []
        cb = self._get_checkerboard_texture(view)
        if cb is not None:
            candidates.append(cb)
        img = self._get_image_texture(view)
        if img is not None:
            candidates.append(img)
        for channel in (view.normal_map, view.bump_map):
            if channel is not None and isinstance(channel.texture, Texture):
                candidates.append(channel.texture)
        for tex in candidates:
            uv = getattr(tex, "_uv", None)
            if isinstance(uv, Attribute) and uv._internal_name is not None:
                return uv._internal_name
        return None

    def _load_blender_image(self, image: Image, *, is_data: bool):
        """Load an ``Image`` texture into a Blender image datablock.

        Applies the texture's ``saturation`` / ``whiteness`` adjustments via PIL
        when non-default, and sets the colour space ("Non-Color" for raw/data
        images such as normal and bump maps; "sRGB" otherwise).
        """
        from pathlib import Path as _Path

        path = str(_Path(image.filename).resolve())
        if image.saturation != 1.0 or image.whiteness != 0.0:
            import tempfile
            from PIL import Image as PILImage, ImageEnhance

            img = PILImage.open(path).convert("RGBA")
            if image.saturation != 1.0:
                img = ImageEnhance.Color(img).enhance(image.saturation)
            if image.whiteness != 0.0:
                white = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
                img = PILImage.blend(img, white, alpha=image.whiteness)
            tmp = tempfile.mktemp(suffix=".png")
            img.save(tmp)
            path = tmp
        bimg = bpy.data.images.load(path)
        bimg.colorspace_settings.name = (
            "Non-Color" if (is_data or image.raw) else "sRGB"
        )
        return bimg

    def _build_image_texture_node(
        self,
        image: Image,
        nodes,
        links,
        uv_layer_name: str,
        *,
        is_data: bool,
        y: int = 0,
    ):
        """Create a UV-mapped ``ShaderNodeTexImage`` node, or None on failure."""
        try:
            bimg = self._load_blender_image(image, is_data=is_data)
        except Exception as e:  # pragma: no cover - bad path / unreadable image
            logger.warning(
                f"Blender backend: failed to load image '{image.filename}': {e}"
            )
            return None
        uv_node = nodes.new(type="ShaderNodeUVMap")
        uv_node.location = (-800, y)
        uv_node.uv_map = uv_layer_name
        tex_node = nodes.new(type="ShaderNodeTexImage")
        tex_node.location = (-600, y)
        tex_node.image = bimg
        links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
        return tex_node

    def _build_normal_map_node(self, normal_map, nodes, links, uv_layer_name: str):
        """Build a normal-map node chain; return its ``Normal`` output socket."""
        tex = normal_map.texture
        if not isinstance(tex, Image):
            logger.warning(
                f"Blender backend: NormalMap.texture type {type(tex).__name__} "
                "not supported (only Image is wired)."
            )
            return None
        tex_node = self._build_image_texture_node(
            tex, nodes, links, uv_layer_name, is_data=True, y=-400
        )
        if tex_node is None:
            return None
        nm = nodes.new(type="ShaderNodeNormalMap")
        nm.location = (-400, -400)
        nm.uv_map = uv_layer_name
        links.new(tex_node.outputs["Color"], nm.inputs["Color"])
        return nm.outputs["Normal"]

    def _build_bump_map_node(
        self, bump_map, nodes, links, uv_layer_name: str, base_normal_socket
    ):
        """Build a bump node chain; return its ``Normal`` output socket.

        ``base_normal_socket`` (e.g. from a normal map) is fed into the bump
        node's ``Normal`` input so the two compose.
        """
        tex = bump_map.texture
        if not isinstance(tex, Image):
            logger.warning(
                f"Blender backend: BumpMap.texture type {type(tex).__name__} "
                "not supported (only Image is wired)."
            )
            return None
        tex_node = self._build_image_texture_node(
            tex, nodes, links, uv_layer_name, is_data=True, y=-700
        )
        if tex_node is None:
            return None
        bump = nodes.new(type="ShaderNodeBump")
        bump.location = (-400, -700)
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = float(bump_map.scale)
        links.new(tex_node.outputs["Color"], bump.inputs["Height"])
        if base_normal_socket is not None:
            links.new(base_normal_socket, bump.inputs["Normal"])
        return bump.outputs["Normal"]

    def _copy_mesh_color_to_blender(
        self,
        lagrange_mesh,
        bpy_mesh,
        attr_name: str,
        element_type,
        color_layer_name: str = "Color",
    ):
        """Copy Lagrange mesh color attribute to a Blender mesh color attribute."""
        attr = lagrange_mesh.attribute(attr_name)
        data = np.asarray(attr.data)
        if data.ndim == 2 and data.shape[1] >= 3:
            colors = (
                data[:, :4]
                if data.shape[1] >= 4
                else np.column_stack([data[:, :3], np.ones(len(data))])
            )
        else:
            return

        if element_type == lagrange.AttributeElement.Vertex:
            bpy_mesh.color_attributes.new(
                name=color_layer_name,
                type="FLOAT_COLOR",
                domain="POINT",
            )
            layer = bpy_mesh.color_attributes[color_layer_name]
            if len(colors) != len(layer.data):
                logger.warning(
                    f"Color attribute '{attr_name}' has {len(colors)} entries but "
                    f"Blender mesh has {len(layer.data)} vertices; truncating to the shorter length."
                )
            for i, c in enumerate(colors[: len(layer.data)]):
                layer.data[i].color = (
                    float(c[0]),
                    float(c[1]),
                    float(c[2]),
                    float(c[3]),
                )
        elif element_type == lagrange.AttributeElement.Facet:
            bpy_mesh.color_attributes.new(
                name=color_layer_name,
                type="FLOAT_COLOR",
                domain="CORNER",
            )
            layer = bpy_mesh.color_attributes[color_layer_name]
            loop_idx = 0
            for face in bpy_mesh.polygons:
                c = colors[face.index]
                for _ in range(face.loop_total):
                    if loop_idx < len(layer.data):
                        layer.data[loop_idx].color = (
                            float(c[0]),
                            float(c[1]),
                            float(c[2]),
                            float(c[3]),
                        )
                    loop_idx += 1
        else:
            logger.warning(
                f"Blender backend: unsupported color element type {element_type}"
            )

    def _copy_mesh_uv_to_blender(
        self,
        lagrange_mesh,
        bpy_mesh,
        attr_name: str,
        uv_layer_name: str = "UVMap",
    ):
        """Copy Lagrange mesh UV attribute to a Blender mesh UV map.

        Args:
            lagrange_mesh: Lagrange mesh with UV attribute.
            bpy_mesh: Blender mesh to copy UV data into.
            attr_name: Name of the UV attribute in the Lagrange mesh.
            uv_layer_name: Name of the UV layer to create in Blender.
        """
        if not lagrange_mesh.has_attribute(attr_name):
            logger.warning(f"UV attribute '{attr_name}' not found in mesh")
            return

        # Create UV layer
        if uv_layer_name not in bpy_mesh.uv_layers:
            bpy_mesh.uv_layers.new(name=uv_layer_name)
        uv_layer = bpy_mesh.uv_layers[uv_layer_name]

        # Handle indexed attributes (common after compilation/finalization)
        if lagrange_mesh.is_attribute_indexed(attr_name):
            indexed_attr = lagrange_mesh.indexed_attribute(attr_name)
            uv_values = np.asarray(indexed_attr.values.data)
            uv_indices = np.asarray(indexed_attr.indices.data)

            # UV values should be 2D (N x 2)
            if uv_values.ndim != 2 or uv_values.shape[1] < 2:
                logger.warning(
                    f"UV attribute '{attr_name}' has invalid shape: {uv_values.shape}"
                )
                return

            # Indexed UVs: expand using indices to corner UVs
            for i, idx in enumerate(uv_indices.flat):
                if i < len(uv_layer.data) and idx < len(uv_values):
                    uv_layer.data[i].uv = (
                        float(uv_values[idx, 0]),
                        float(uv_values[idx, 1]),
                    )
        else:
            # Handle non-indexed attributes
            attr = lagrange_mesh.attribute(attr_name)
            uv_data = np.asarray(attr.data)

            # UV data should be 2D (N x 2)
            if uv_data.ndim != 2 or uv_data.shape[1] < 2:
                logger.warning(
                    f"UV attribute '{attr_name}' has invalid shape: {uv_data.shape}"
                )
                return

            element_type = attr.element_type

            if element_type == lagrange.AttributeElement.Vertex:
                # Vertex UVs: expand to corner UVs
                for poly in bpy_mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        loop = bpy_mesh.loops[loop_idx]
                        vertex_idx = loop.vertex_index
                        if vertex_idx < len(uv_data):
                            uv_layer.data[loop_idx].uv = (
                                float(uv_data[vertex_idx, 0]),
                                float(uv_data[vertex_idx, 1]),
                            )
            elif element_type == lagrange.AttributeElement.Corner:
                # Corner/loop UVs: direct mapping
                for i, uv in enumerate(uv_data):
                    if i < len(uv_layer.data):
                        uv_layer.data[i].uv = (float(uv[0]), float(uv[1]))
            else:
                logger.warning(
                    f"Blender backend: unsupported UV element type {element_type}"
                )
