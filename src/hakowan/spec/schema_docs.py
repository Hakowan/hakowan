"""Human-facing descriptions injected into the generated JSON Schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MODEL_DESCRIPTIONS = {
    "AffineScaleSpec": "Applies a linear or homogeneous affine matrix to an attribute.",
    "AnnotationSpec": "A deterministic screen-space text annotation.",
    "AffineTransformSpec": "Applies a global affine matrix to a layer's geometry.",
    "AttributeSpec": "References a named mesh attribute and an ordered scale pipeline.",
    "BendStyleSpec": "Controls curve bending from a direction attribute.",
    "BoundaryTransformSpec": "Extracts geometric boundaries and optional attribute discontinuities.",
    "BumpMapChannelSpec": "Perturbs surface shading normals from a height texture.",
    "ChannelsSpec": "One optional slot per visual role; each compiled view uses the first channel of each kind encountered from root to leaf.",
    "CheckerboardTextureSpec": "Alternates two texture or color values in UV space.",
    "ClipScaleSpec": "Clamps attribute values to a closed numeric domain.",
    "ClipTransformSpec": "Cuts geometry against a plane and keeps its positive half-space.",
    "ComputeTransformSpec": "Computes named coordinate, normal, or component attributes.",
    "ConductorMaterialSpec": "A smooth metallic conductor selected by optical preset name.",
    "CovarianceChannelSpec": "Maps per-point 3x3 covariance matrices to anisotropic glyphs.",
    "CustomScaleSpec": "Applies a safe expression or externally resolved function to an attribute.",
    "DielectricMaterialSpec": "A smooth transmissive dielectric with optional participating medium.",
    "DiffuseMaterialSpec": "A matte material whose reflectance may be constant or data-driven.",
    "DirectionalLightSpec": "A distant light with parallel rays along a world-space direction.",
    "EnvironmentSpec": "Environment-map lighting, orientation, scale, and visibility settings.",
    "ExplodeTransformSpec": "Separates facet groups outward from the mesh center.",
    "ExpressionSpec": "A restricted, validated expression evaluated with one attribute value.",
    "ExternalDataSpec": "References in-memory data by an application-defined resolver identifier.",
    "FilterTransformSpec": "Keeps mesh elements for which a condition evaluates true.",
    "FunctionRefSpec": "References a trusted application function by resolver identifier.",
    "FurTransformSpec": "Generates hair strands along a surface vector field.",
    "HairMaterialSpec": "A hair material controlled by pigments, colors, gradients, and variation.",
    "ImageTextureSpec": "Samples an image through a mesh UV attribute.",
    "InheritNodeSpec": "Applies partial layer properties to one child node.",
    "IsocontourTextureSpec": "Alternates two textures in bands derived from a scalar field.",
    "LayerNodeSpec": "A leaf visualization layer with no child composition nodes.",
    "LayerPropertiesSpec": "Partial geometry source, mark, visual channels, transforms, and labels contributed by one node.",
    "LayoutNodeSpec": "Places child visualizations side by side along one axis.",
    "LegendSpec": "Presentation settings for a continuous colorbar or categorical legend.",
    "LogScaleSpec": "Applies a logarithm with the selected base to scalar data.",
    "MediumSpec": "Defines the homogeneous medium enclosed by a dielectric material.",
    "MeshFileDataSpec": "References mesh data stored in an external file.",
    "NormScaleSpec": "Reduces each vector value to its selected vector norm.",
    "NormTransformSpec": "Writes vector magnitudes into a new mesh attribute.",
    "NormalChannelSpec": "Maps a vector attribute to surface shading normals.",
    "NormalMapChannelSpec": "Replaces surface shading normals from a normal-map texture.",
    "NormalizeScaleSpec": "Linearly maps an attribute domain into a target range.",
    "NormalizeTransformSpec": "Recenters and uniformly scales geometry to the canonical unit sphere.",
    "OrthographicCameraSpec": "An orthographic camera with explicit look-at vectors and clipping planes.",
    "OutputSettingsSpec": "Semantic output resolution, background, passes, and deterministic seed.",
    "OffsetScaleSpec": "Adds another mesh attribute to the current attribute.",
    "OverlayNodeSpec": "Renders two or more child nodes in the same coordinate space.",
    "PlasticMaterialSpec": "A smooth plastic with diffuse and specular components.",
    "PositionChannelSpec": "Maps a vector attribute to mark positions.",
    "PerspectiveCameraSpec": "A perspective camera with explicit look-at vectors and field of view.",
    "PointLightSpec": "An omnidirectional point light with color and intensity.",
    "PrincipalAxesTransformSpec": "Aligns mesh PCA directions with a target frame.",
    "PrincipledMaterialSpec": "A general-purpose physically based surface material.",
    "RoughConductorMaterialSpec": "A metallic conductor with microfacet roughness.",
    "RoughDielectricMaterialSpec": "A transmissive dielectric with microfacet roughness.",
    "RoughPlasticMaterialSpec": "A plastic material with microfacet roughness.",
    "ScalarFieldTextureSpec": "Maps a scalar attribute through a colormap.",
    "SceneSettingsSpec": "Optional figure-level camera, lights, environment, and output intent.",
    "ShapeChannelSpec": "Selects and optionally orients the glyph used for point marks.",
    "SizeChannelSpec": "Maps a scalar field or constant to point or curve radius.",
    "StreamlineTransformSpec": "Traces surface streamlines from a vector or cross field.",
    "ThinDielectricMaterialSpec": "A transmissive dielectric for negligible-thickness surfaces.",
    "ThinPrincipledMaterialSpec": "A thin-surface principled material with diffuse transmission.",
    "ThinLensCameraSpec": "A perspective camera with aperture and focus-distance controls.",
    "UVMeshTransformSpec": "Replaces geometry positions with UV coordinates.",
    "UniformScaleSpec": "Multiplies every attribute value by one factor.",
    "UniformTextureSpec": "Represents one constant color or scalar texture value.",
    "VectorFieldChannelSpec": "Renders a vector field as curve or arrow geometry.",
}


FIELD_DESCRIPTIONS = {
    "$schema": "Canonical schema identifier for this document version.",
    "version": "Hakowan specification format version.",
    "root": "Root layer or composition node.",
    "scene": "Optional reproducible camera, lighting, environment, and output settings.",
    "kind": "Discriminator selecting the concrete specification variant.",
    "spec": "Layer properties contributed by this composition node.",
    "child": "Single child that inherits this node's layer properties.",
    "children": "Ordered child visualization nodes.",
    "data": "Context-specific input. LayerPropertiesSpec.data is a geometry source; channel, texture, and transform data fields are attribute references.",
    "mark": "Optional mark type; unresolved marks default to surface during compilation.",
    "channels": "Unique visual channels and material contributed by this node.",
    "transforms": "Ordered runtime transform chain stored by this node.",
    "name": "Human-readable layer label used by interactive viewers.",
    "annotations": "Ordered screen-space text annotations contributed by this node.",
    "unit": "Optional physical unit associated with the referenced attribute.",
    "legend": "Automatic legend toggle or detailed legend configuration.",
    "title": "Optional display title; null derives it from the attribute name.",
    "units": "Optional display units appended to the legend title.",
    "ticks": "Number of labeled ticks for a continuous colorbar.",
    "format": "Python numeric format specifier used for legend labels.",
    "category_labels": "Optional mapping from stringified category values to labels.",
    "width": "Raster legend panel width in pixels.",
    "text": "Annotation text.",
    "font_size": "Annotation font size in pixels.",
    "anchor": "Horizontal annotation anchor relative to its position.",
    "background": "Optional annotation background color.",
    "padding": "Annotation background padding in pixels.",
    "path": "Resource path; relative paths resolve beside the loaded specification file.",
    "camera": "Optional figure camera specification.",
    "lights": "Optional ordered light list; null preserves defaults and an empty list disables explicit lights.",
    "environment": "Optional environment lighting and background settings.",
    "output": "Optional semantic output settings.",
    "eye": "World-space camera position after scene normalization.",
    "target": "World-space camera look-at target after scene normalization.",
    "up": "World-space camera or environment up vector.",
    "fov": "Perspective field of view in degrees.",
    "fov_axis": "Image axis to which the field of view applies.",
    "near": "Positive near clipping-plane distance.",
    "far": "Far clipping-plane distance, greater than near.",
    "aperture_radius": "Thin-lens aperture radius.",
    "focus_distance": "Thin-lens focus distance; zero selects the target distance.",
    "intensity": "Non-negative light intensity in backend-neutral relative units.",
    "enabled": "Whether environment lighting is enabled.",
    "visible": "Whether the environment map is visible to the camera.",
    "rotation": "Environment rotation in degrees around its up axis.",
    "height": "Output image height in pixels.",
    "passes": "Ordered semantic render-pass requests.",
    "sampler_seed": "Deterministic renderer sampler seed.",
    "id": "Non-empty application-defined identifier resolved outside the JSON document.",
    "roi_box": "Optional axis-aligned region of interest as minimum and maximum corners.",
    "position": "Optional position channel.",
    "normal": "Optional normal channel or plane normal, depending on the containing model.",
    "size": "Optional size channel or checkerboard dimension, depending on the containing model.",
    "shape": "Optional point-glyph shape channel.",
    "vector_field": "Optional vector-field channel.",
    "covariance": "Optional covariance channel.",
    "material": "Optional material encoding or conductor preset name.",
    "bump_map": "Optional bump-map channel.",
    "normal_map": "Optional normal-map channel.",
    "scales": "Ordered attribute scale pipeline, evaluated in array order.",
    "factor": "Uniform multiplicative factor.",
    "base": "Logarithm base; must be greater than one at runtime.",
    "domain": "Inclusive input or clipping domain as [minimum, maximum].",
    "domain_min": "Optional explicit input-domain minimum; inferred from data when omitted.",
    "domain_max": "Optional explicit input-domain maximum; inferred from data when omitted.",
    "range": "Output colormap interval as [minimum, maximum].",
    "range_min": "Target output-range minimum.",
    "range_max": "Target output-range maximum.",
    "matrix": "Numeric affine or linear transformation matrix.",
    "order": "Vector norm order.",
    "offset": "Attribute whose values are added to the current attribute.",
    "function": "Safe expression or trusted external function reference.",
    "source": "Restricted expression source, limited to 1024 characters and 128 AST nodes.",
    "color": "Constant or texture-driven base color.",
    "reflectance": "Diffuse reflectance as a constant color or texture.",
    "diffuse_reflectance": "Diffuse plastic reflectance as a color or texture.",
    "specular_reflectance": "Specular reflection strength or texture.",
    "specular_transmittance": "Specular transmission strength.",
    "texture": "Texture supplying values for this channel.",
    "texture1": "Texture or constant used for the first alternating region.",
    "texture2": "Texture or constant used for the second alternating region.",
    "uv": "Optional two-channel UV attribute; an existing UV attribute is inferred when omitted.",
    "raw": "Whether image values bypass color-transfer decoding.",
    "saturation": "Image saturation multiplier; zero produces grayscale.",
    "whiteness": "Blend amount toward white, from zero to one.",
    "ratio": "Relative width of the first isocontour region.",
    "num_contours": "Number of contour repetitions per unit scalar interval.",
    "colormap": "Built-in palette name, colorcet palette name, identity, or explicit color list.",
    "categories": "Whether values are discrete categories rather than continuous quantities.",
    "reverse": "Whether to reverse colormap direction.",
    "two_sided": "Whether the material renders both surface orientations.",
    "back_side": "Optional independent material for back-facing surface facets.",
    "distribution": "Microfacet distribution used by a rough material.",
    "alpha": "Microfacet roughness or scalar texture.",
    "roughness": "Surface roughness as a constant scalar or texture.",
    "metallic": "Metallic response as a constant scalar or texture.",
    "anisotropic": "Principled-material anisotropy amount.",
    "spec_trans": "Principled-material specular transmission amount.",
    "eta": "Principled-material interior index of refraction.",
    "spec_tint": "Principled-material specular tint amount.",
    "sheen": "Principled-material sheen amount.",
    "sheen_tint": "Principled-material sheen tint amount.",
    "flatness": "Thin-surface or subsurface flatness control.",
    "diff_trans": "Diffuse transmission amount for thin principled surfaces.",
    "int_ior": "Interior index of refraction or named preset.",
    "ext_ior": "Exterior index of refraction or named preset.",
    "medium": "Optional homogeneous interior medium.",
    "albedo": "Single-scattering albedo of a participating medium.",
    "scale": "Numeric strength or density multiplier.",
    "eumelanin": "Dark brown/black pigment concentration for hair.",
    "pheomelanin": "Red/yellow pigment concentration for hair.",
    "root_color": "Optional hair color at strand roots.",
    "tip_color": "Optional hair color at strand tips.",
    "color_variation": "Per-strand brightness variation from zero to one.",
    "direction": "Attribute defining curve-bending directions.",
    "bend_type": "Curve bend algorithm: normal, ribbon, or smooth.",
    "base_shape": "Point glyph primitive: sphere, disk, or cube.",
    "orientation": "Optional vector attribute orienting each point glyph.",
    "refinement_level": "Vector-field sampling refinement level.",
    "style": "Optional curve style for vector-field marks.",
    "end_type": "Vector endpoint style: tapered point, arrowhead, or flat cap.",
    "normalize": "Whether vectors or layout cells are normalized before use.",
    "full": "Whether covariance values store full matrices instead of square-root factors.",
    "condition": "Predicate controlling whether each selected mesh element is retained.",
    "point": "Point lying on the clipping plane.",
    "frame": "Target orthonormal frame whose columns receive PCA directions.",
    "orthonormalize_frame": "Whether to QR-orthonormalize the supplied frame.",
    "normalize_normals": "Whether normal attributes are normalized after geometry normalization.",
    "normalize_tangents_bitangents": "Whether tangent and bitangent attributes are normalized.",
    "x": "Output attribute name for vertex x coordinates.",
    "y": "Output attribute name for vertex y coordinates.",
    "z": "Output attribute name for vertex z coordinates.",
    "vertex_normal": "Output attribute name for computed vertex normals.",
    "facet_normal": "Output attribute name for computed facet normals.",
    "component": "Output attribute name for connected-component identifiers.",
    "pieces": "Facet attribute grouping geometry into exploded pieces.",
    "magnitude": "Distance multiplier used when separating pieces.",
    "norm_attr_name": "Output attribute name for computed vector magnitudes.",
    "attributes": "Attribute names whose discontinuities count as boundaries.",
    "vec_field": "Vector or cross-field attribute used to generate curves.",
    "n": "Requested number of seeds, streamlines, or fur strands.",
    "cross_field": "Whether the input is a four-fold rotationally symmetric cross field.",
    "length": "Maximum object-space trace or strand length; null selects an automatic limit.",
    "seed": "Deterministic random seed.",
    "min_length": "Minimum number of retained streamline sample points.",
    "max_steps": "Maximum streamline edge-crossing steps; null selects a mesh-dependent limit.",
    "id_attr_name": "Output attribute name containing streamline identifiers.",
    "lift": "Fur root lift angle in degrees.",
    "curl": "Fur curvature strength toward the surface-flow direction.",
    "segments": "Number of line segments generated per fur strand.",
    "root_radius": "Fur radius at strand roots; null selects an automatic value.",
    "tip_radius": "Fur radius at strand tips.",
    "randomness": "Natural strand variation from zero to one.",
    "follow_surface": "Whether fur follows the surface instead of leaning analytically away.",
    "clump": "Strength with which child-hair tips converge on guides.",
    "spread": "Child-hair root scatter radius; null selects an automatic value.",
    "axis": "World axis along which layout children are placed.",
    "gap": "Spacing between layout cells as a fraction of mean cell diameter.",
}


FIELD_OVERRIDES = {
    (
        "AnnotationSpec",
        "position",
    ): "Normalized image position with origin at the top-left.",
    ("AnnotationSpec", "color"): "Annotation text color.",
    ("LegendSpec", "position"): "Side on which the legend panel is placed.",
    ("LegendSpec", "format"): "Python numeric format specifier for tick labels.",
    ("AttributeSpec", "unit"): "Optional physical unit shown in legends and manifests.",
    (
        "LayerPropertiesSpec",
        "annotations",
    ): "Ordered screen-space annotations for this node.",
    (
        "ScalarFieldTextureSpec",
        "legend",
    ): "Automatic legend toggle or detailed configuration.",
    ("PointLightSpec", "position"): "World-space point-light position.",
    ("PointLightSpec", "color"): "Point-light RGB or named color.",
    ("DirectionalLightSpec", "color"): "Directional-light RGB or named color.",
    (
        "DirectionalLightSpec",
        "direction",
    ): "World-space direction traveled by light rays.",
    ("BendStyleSpec", "direction"): "Attribute defining curve-bending directions.",
    (
        "EnvironmentSpec",
        "path",
    ): "Optional environment-map path resolved beside the specification.",
    ("EnvironmentSpec", "scale"): "Environment-light intensity multiplier.",
    (
        "OrthographicCameraSpec",
        "scale",
    ): "Full vertical world-space extent of the orthographic view.",
    ("OutputSettingsSpec", "width"): "Output image width in pixels.",
    ("OutputSettingsSpec", "background"): "Initial light or dark background mode.",
    (
        "ClipTransformSpec",
        "normal",
    ): "Three-component plane normal; the positive half-space is retained.",
    (
        "ComputeTransformSpec",
        "normal",
    ): "Output attribute name for automatically selected normals.",
    ("ChannelsSpec", "normal"): "Optional mapping of an attribute to shading normals.",
    ("CheckerboardTextureSpec", "size"): "Number of checker cells along each UV axis.",
    ("BumpMapChannelSpec", "scale"): "Strength of the bump perturbation.",
    ("MediumSpec", "scale"): "Density multiplier for the participating medium.",
    (
        "MaterialBaseSpec",
        "back_side",
    ): "Optional independent material for back-facing facets.",
    (
        "LayerPropertiesSpec",
        "data",
    ): "Optional file-backed or externally resolved mesh data.",
    (
        "FilterTransformSpec",
        "data",
    ): "Attribute tested by the predicate; null uses vertex positions.",
    ("PositionChannelSpec", "data"): "Vector attribute supplying mark positions.",
    ("NormalChannelSpec", "data"): "Vector attribute supplying shading normals.",
    ("SizeChannelSpec", "data"): "Constant radius or scalar attribute supplying radii.",
    (
        "VectorFieldChannelSpec",
        "data",
    ): "Vector attribute supplying arrow or curve directions.",
    ("CovarianceChannelSpec", "data"): "Nine-channel per-vertex covariance attribute.",
    (
        "ScalarFieldTextureSpec",
        "data",
    ): "Scalar attribute mapped through the selected colormap.",
    (
        "IsocontourTextureSpec",
        "data",
    ): "Scalar attribute from which contour bands are generated.",
}

FIELD_ROLES = {
    ("LayerPropertiesSpec", "data"): "geometry_source",
    ("FilterTransformSpec", "data"): "attribute_reference",
    ("IsocontourTextureSpec", "data"): "attribute_reference",
    ("NormTransformSpec", "data"): "attribute_reference",
    ("NormalChannelSpec", "data"): "attribute_reference",
    ("PositionChannelSpec", "data"): "attribute_reference",
    ("ScalarFieldTextureSpec", "data"): "attribute_reference",
    ("SizeChannelSpec", "data"): "attribute_or_constant",
    ("VectorFieldChannelSpec", "data"): "attribute_reference",
    ("CovarianceChannelSpec", "data"): "attribute_reference",
    ("BendStyleSpec", "direction"): "attribute_reference",
    ("CheckerboardTextureSpec", "uv"): "attribute_reference",
    ("ImageTextureSpec", "uv"): "attribute_reference",
    ("OffsetScaleSpec", "offset"): "attribute_reference",
    ("ShapeChannelSpec", "orientation"): "attribute_reference",
    ("ExplodeTransformSpec", "pieces"): "attribute_reference",
    ("StreamlineTransformSpec", "vec_field"): "attribute_reference",
}


ROOT_DESCRIPTIONS = {
    "$schema": FIELD_DESCRIPTIONS["$schema"],
    "version": FIELD_DESCRIPTIONS["version"],
    "root": FIELD_DESCRIPTIONS["root"],
    "scene": FIELD_DESCRIPTIONS["scene"],
}


ROOT_EXAMPLE = {
    "$schema": "https://hakowan.github.io/hakowan/schema/v1.json",
    "version": "1.1",
    "root": {
        "kind": "layer",
        "spec": {
            "data": {"kind": "mesh_file", "path": "mesh.ply", "roi_box": None},
            "mark": "surface",
            "channels": {
                "material": {
                    "kind": "diffuse",
                    "reflectance": {
                        "kind": "scalar_field",
                        "data": {"name": "temperature", "scales": []},
                        "colormap": "viridis",
                    },
                }
            },
            "transforms": [],
            "name": "temperature",
        },
    },
    "scene": {
        "camera": {
            "kind": "perspective",
            "eye": [2, 3, 4],
            "target": [0, 0, 0],
            "up": [0, 1, 0],
            "fov": 35,
        },
        "lights": [
            {
                "kind": "directional",
                "direction": [0, 0, -1],
                "color": "white",
                "intensity": 2,
            }
        ],
        "environment": {"enabled": False},
        "output": {
            "width": 1024,
            "height": 800,
            "background": "dark",
            "passes": ["beauty"],
            "sampler_seed": 0,
        },
    },
}


def _canonical_model_name(name: str) -> str:
    prefix = "hakowan__spec__model__"
    if name.startswith(prefix):
        return name[len(prefix) :].rsplit("__", 1)[0]
    return name


def enrich_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Add complete human-facing descriptions and examples to generated schema."""
    result = deepcopy(schema)
    result["description"] = (
        "Canonical, versioned Hakowan 3D visualization specification. Unknown "
        "fields and unknown discriminated variants are rejected."
    )
    result["x-hakowan-conventions"] = {
        "geometry_source": ("Only LayerPropertiesSpec.data selects mesh geometry."),
        "attribute_reference": (
            "Channel, texture, and transform attribute inputs reference named fields "
            "inside the selected geometry source."
        ),
        "channel_composition": (
            "Each channel key is a semantic slot. Along a root-to-leaf path, the "
            "first value for a slot wins; unrelated slots compose."
        ),
        "figure": "Portable visualization intent serialized as FigureSpec.",
        "config": (
            "Invocation-time renderer policy; an explicit Config overrides all "
            "Figure-derived rendering configuration."
        ),
    }
    result["examples"] = [ROOT_EXAMPLE]
    for field_name, property_schema in result.get("properties", {}).items():
        property_schema.setdefault(
            "description",
            ROOT_DESCRIPTIONS.get(field_name, FIELD_DESCRIPTIONS[field_name]),
        )
    for model_name, definition in result.get("$defs", {}).items():
        canonical_name = _canonical_model_name(model_name)
        definition.setdefault(
            "description",
            MODEL_DESCRIPTIONS.get(
                canonical_name,
                f"Supporting schema definition for {canonical_name}.",
            ),
        )
        for field_name, property_schema in definition.get("properties", {}).items():
            description = FIELD_OVERRIDES.get(
                (canonical_name, field_name), FIELD_DESCRIPTIONS.get(field_name)
            )
            if description is None:
                raise KeyError(
                    f"Missing schema description for {model_name}.{field_name}"
                )
            property_schema.setdefault("description", description)
            role = FIELD_ROLES.get((canonical_name, field_name))
            if role is not None:
                property_schema["x-hakowan-role"] = role
    return result


__all__ = ["enrich_schema"]
