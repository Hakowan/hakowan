# Material

In 3D data visualization, material is the new color! By assigning a material to the 3D geometry, we
can not only achieve more photorealistic visual quality but also encode data in more channels than
just RGB. Hakowan leverages [Mitsuba's material
support](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html) to provide a
number of material-based visual channels.

All materials support the following parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `two_sided` | `bool` | Whether the material is two-sided. (default: false) |
| `back_side` | `Material \| None` | An independent material for the back face of each facet. (default: none) |

A material can be specified via the `Layer.channel()` method.
```py
# Assume `m` is a `Material` object
l = hkw.layer().channel(material=m)
```

Hakowan also provides the shortcut method `Layer.material()` to specify material directly.
```py
l = hkw.layer().material("Diffuse", "orange", two_sided=True)
l = hkw.layer().material("Principled", "ivory", roughness=0.2, metallic=0.8)
```


## Diffuse material

![Diffuse material](../images/spot_diffuse.png){: style="width:170px"}
![Diffuse material](../images/spot_diffuse_scalar.png){: style="width:170px"}
![Diffuse material](../images/spot_diffuse_uv.png){: style="width:170px"}
![Diffuse material](../images/spot_diffuse_tex.png){: style="width:170px"}

`Diffuse` material provides a matte shading to the shape. It is good for visualizing scalar field
data without any interference from specular highlight.

| Channel | Type | Description |
|---------|------|-------------|
| `reflectance` | [TextureLike][hakowan.grammar.texture.texture.TextureLike] | Base color of the material (default: 0.5) |

Here is a simple example of creating a `Diffuse` material.

```py
m = hkw.material.Diffuse("ivory")
```

Here, we assign a uniform color, "blue", to the `reflectance` channel. To encode actual data, we
need to assign a [texture](texture.md) to the `reflectance` channel.

```py
m = hkw.material.Diffuse(hkw.texture.ScalarField(data = "attr_name"))
```

Check the [Mitsuba
doc](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-diffuse-material-diffuse)
for more details.

## Conductor material

![Conductor material](../images/spot_conductor_Au.png){: style="width:170px"}
![Conductor material](../images/spot_conductor_Ag.png){: style="width:170px"}
![Conductor material](../images/spot_conductor_Cu.png){: style="width:170px"}
![Conductor material](../images/spot_conductor_CuO.png){: style="width:170px"}

`Conductor` material gives the geometry a metallic look and feel. It takes a `material` parameter as
input, but it cannot be used to encode any data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `material` | `str` | Material type (see [supported material types](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#conductor-ior-list)) |

Here is a snippet to create a `Conductor` material of type `Al`.

```py
m = hkw.material.Conductor(material="Al")
```

Check the [Mitsuba
doc](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-conductor-conductor)
for more details.

## Rough conductor material

![RoughConductor material](../images/spot_rough_conductor_Au.png){: style="width:170px"}
![RoughConductor material](../images/spot_rough_conductor_Ag.png){: style="width:170px"}
![RoughConductor material](../images/spot_rough_conductor_Cu.png){: style="width:170px"}
![RoughConductor material](../images/spot_rough_conductor_CuO.png){: style="width:170px"}

`RoughConductor` material gives the geometry a matte metallic look and feel. It takes a `material`
parameter just like `Conductor`. In addition, it takes a `distribution` parameter and an `alpha`
channel.

| Channel | Type | Description |
|---------|------|-------------|
| `alpha` | [Texture] [hakowan.grammar.texture.texture.Texture] or `float` | Roughness value from 0 (smooth) to 1 (rough) (default: 0.1) |

| Parameter | Type | Description |
|-----------|------|-------------|
| `material` | `str` | Material type (see [supported material types](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#conductor-ior-list)) |
| `distribution` | `str` | Microfacet normal distribution: `beckmann`, `ggx`, or `phong` (default: `beckmann`) |

Here is a simple example of using `RoughConductor`.

```py
m = hkw.material.RoughConductor(material = "Cu")
```

Here is a more complex example where we map an attribute to the alpha channel:

```py
m = hkw.material.RoughConductor(
    material="Cu", alpha=hkw.texture.ScalarField(data="attr_name")
)
```

Note that `alpha`=0.1 is relatively rough, and `alpha` from 0.3 to 0.7 is very rough. Beyond 0.7,
the result does not look very realistic.

Check out the [Mitsuba
doc](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#rough-conductor-material-roughconductor)
for more details.

## Plastic material

![Plastic material](../images/spot_plastic.png){: style="width:200px"}
![Plastic material](../images/spot_plastic_uv.png){: style="width:200px"}
![Plastic material](../images/spot_plastic_tex.png){: style="width:200px"}

`Plastic` material provides a smooth plastic look and feel.
Hakowan exposes two visual channels in this material:

| Channel | Type | Description |
|---------|------|-------------|
| `diffuse_reflectance` | [TextureLike][hakowan.grammar.texture.texture.TextureLike] | Base color of the material (default: 0.5) |
| `specular_reflectance` | [Texture][hakowan.grammar.texture.texture.Texture] or `float` | Specular reflectance component (default: 1.0) |

Here is a snippet to create a `Plastic` material:

```py
m = hkw.material.Plastic(
    diffuse_reflectance=hkw.texture.ScalarField(data="attr_name"),
    specular_reflectance=hkw.texture.ScalarField(data="attr2_name")
)
```

Check out the [Mitsuba
doc](
https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-plastic-material-plastic
)
for more details.

## Rough plastic material

![RoughPlastic material](../images/spot_rough_plastic.png){: style="width:200px"}
![RoughPlastic material](../images/spot_rough_plastic_uv.png){: style="width:200px"}
![RoughPlastic material](../images/spot_rough_plastic_tex.png){: style="width:200px"}

Similar to `RoughConductor`, the `distribution` parameter controls the microfacet normal distribution;
valid values are `beckmann` (default), `ggx`, and `phong`. The `alpha` parameter controls the
roughness of the material.

| Channel | Type | Description |
|---------|------|-------------|
| `diffuse_reflectance` | [TextureLike][hakowan.grammar.texture.texture.TextureLike] | Base color of the material (default: 0.5) |
| `specular_reflectance` | [Texture][hakowan.grammar.texture.texture.Texture] or `float` | Specular reflectance component (default: 1.0) |

Addition to the above visual channels, this material also expose the following parameters.

| Parameter | Type | Description |
|---------|------|-------------|
| `distribution` | `str` | Microfacet normal distribution: `beckmann`, `ggx`, or `phong` (default: `beckmann`) |
| `alpha` | `float` | Roughness value from 0 (smooth) to 1 (rough) (default: 0.1) |

Here is a snippet to create a `RoughPlastic` material:

```py
m = hkw.material.RoughPlastic(
    diffuse_reflectance=hkw.texture.ScalarField(data="attr_name"),
    specular_reflectance=hkw.texture.ScalarField(data="attr2_name"),
    alpha=0.1
)
```

Note that `alpha`=0.1 is relatively rough, and `alpha` from 0.3 to 0.7 is very rough. Beyond 0.7,
the result does not look very realistic.

Check out the [Mitsuba
doc](
https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#rough-plastic-material-roughplastic
)
for more details.

## Principled material

![Principled material](../images/spot_principled.png){: style="width:170px"}
![Principled material](../images/spot_principled_scalar.png){: style="width:170px"}
![Principled material](../images/spot_principled_uv.png){: style="width:170px"}
![Principled material](../images/spot_principled_tex.png){: style="width:170px"}

`Principled` material is the most versatile material. It is based on paper "[Physically Based
Shading](https://www.disneyanimation.com/publications/physically-based-shading-at-disney/)" and its
extension. It can be used to approximate almost all other materials.

| Channel | Type | Description |
|---------|------|-------------|
| `color` | [TextureLike][hakowan.grammar.texture.texture.TextureLike] | Base color of the material (default: 0.5) |
| `roughness` | [Texture][hakowan.grammar.texture.texture.Texture] or `float` | Roughness value from 0 (smooth) to 1 (rough) (default: 0.5) |
| `metallic` | [Texture][hakowan.grammar.texture.texture.Texture] or `float` | Metallic value from 0 (not metallic) to 1 (very metallic) (default: 0.0) |

Other `Principled` properties are uniform parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `anisotropic` | `float` | Anisotropy amount from 0 to 1 (default: `0.0`) |
| `spec_trans` | `float` | Specular transmission from 0 to 1 (default: `0.0`) |
| `eta` | `float` | Interior index of refraction, greater than 0 (default: `1.5`) |
| `spec_tint` | `float` | Specular tint toward the base color from 0 to 1 (default: `0.0`) |
| `sheen` | `float` | Sheen amount from 0 to 1 (default: `0.0`) |
| `sheen_tint` | `float` | Sheen tint toward the base color from 0 to 1 (default: `0.0`) |
| `flatness` | `float` | Blend between thin and volumetric subsurface scattering from 0 to 1 (default: `0.0`) |

As a rule of thumb, `roughness` and `metallic` are effective at encoding binary categorical data,
but not very effective for quantitative data.

Here is a snippet to create a `Principled` material:

```py
m = hkw.material.Principled(
    color=hkw.texture.ScalarField(data="attr_name"),
    roughness=hkw.texture.ScalarField(data="attr_name2"),
    metallic=hkw.texture.ScalarField(data="attr_name3"),
)
```

Here is another example where the color, roughness and metallic channels are uniform across the
entire shape.

```py
m = hkw.material.Principled(color="#252525", roughness=0.2, metallic=0.8)
```

Check out the [Mitsuba
doc](
https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#the-principled-bsdf-principled
)
for more details.

## Thin Principled material

`ThinPrincipled` material is a thin variant of the Principled material designed for thin surfaces
such as paper, leaves, or cloth. In addition to all the channels and parameters from the `Principled`
material, it also supports diffuse transmission.

| Channel | Type | Description |
|---------|------|-------------|
| `color` | [TextureLike][hakowan.grammar.texture.texture.TextureLike] | Base color of the material (default: 0.5) |
| `roughness` | [Texture][hakowan.grammar.texture.texture.Texture] or `float` | Roughness value from 0 (smooth) to 1 (rough) (default: 0.5) |
| `metallic` | [Texture][hakowan.grammar.texture.texture.Texture] or `float` | Metallic value from 0 (not metallic) to 1 (very metallic) (default: 0.0) |
| `diff_trans` | `float` | Diffuse transmission amount from 0 to 1 (default: 0.0) |

Here is a snippet to create a `ThinPrincipled` material:

```py
m = hkw.material.ThinPrincipled(
    color="ivory",
    roughness=0.3,
    diff_trans=0.5  # 50% light transmission
)
```

Check out the [Mitsuba
doc](
https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#thin-principled-bsdf-thinprincipled
)
for more details.

## Dielectric material

![Dielectric material](../images/spot_dielectric.png){: style="width:200px"}
![Dielectric material](../images/spot_dielectric_medium.png){: style="width:200px"}
![Dielectric material](../images/spot_dielectric_medium_2.png){: style="width:200px"}

`Dielectric` material provides a smooth glossy look and feel. This material exposes no visual
channels and have the following parameters.


| Parameter | Type | Description |
|---------|------|-------------|
| `int_ior` | `str` or `float` | Interior index of refraction (default: `bk7`, see [supported ior list](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#ior-table-list)) |
| `ext_ior` | `str` or `float` | Exterior index of refraction (default: `air`, see [supported ior list](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#ior-table-list)) |
| `medium` | `Medium` or `None` | Medium of the enclosed material (default: `None`) |
| `specular_reflectance` | `float` | Specular reflectance (default: `1.0`) |
| `specular_transmittance` | `float` | Specular transmittance (default: `1.0`) |

Note that by default, `Dielectric` material does not expose a color channel. Color is specified
indirectly through the `medium` parameter. Medium describes the material/medium enclosed by the
shape. It can provide fancy effects such as sub-surface scattering. However, `medium` setting
requires the `volpath` integrator.

Here is a snippet for creating a `Dielectric` material.

```py
m = hkw.material.Dielectric(int_ior="water")
```

Check out the [Mitsuba
doc](
https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#smooth-dielectric-material-dielectric
)
for more details.

## Rough dielectric material

![RoughDielectric material](../images/spot_rough_dielectric.png){: style="width:200px"}
![RoughDielectric material](../images/spot_rough_dielectric_medium.png){: style="width:200px"}
![RoughDielectric material](../images/spot_rough_dielectric_medium_2.png){: style="width:200px"}

`RoughDielectric` material provides a matte glass look and feel. Similar to the `RoughConductor`
material, This material exposes an `alpha` visual channel.

| Channel | Type | Description |
|---------|------|-------------|
| `alpha` | [Texture] [hakowan.grammar.texture.texture.Texture] or `float` | Roughness value from 0 (smooth) to 1 (rough) (default: 0.1) |

Although `alpha` channel is available to encode data, based on our experiments, its effect is
somewhat limited.
Here are the parameters of this material.

| Parameter | Type | Description |
|-----------|------|-------------|
| `int_ior` | `str` or `float` | Interior index of refraction (default: `bk7`, see [supported ior list](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#ior-table-list)) |
| `ext_ior` | `str` or `float` | Exterior index of refraction (default: `air`, see [supported ior list](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#ior-table-list)) |
| `medium` | `Medium` or `None` | Medium of the enclosed material (default: `None`) |
| `specular_reflectance` | `float` | Specular reflectance (default: `1.0`) |
| `specular_transmittance` | `float` | Specular transmittance (default: `1.0`) |
| `distribution` | `str` | Microfacet normal distribution: `beckmann`, `ggx`, or `phong` (default: `beckmann`) |

Here is a snippet for creating a `RoughDielectric` material.

```py
m = hkw.material.RoughDielectric(int_ior="water")
```

## Thin dielectric material

![ThinDielectric material](../images/spot_thin_dielectric.png){: style="width:200px"}

`ThinDielectric` material provides a thin glossy shell look and feel for a given shape. It is designed
for thin surfaces like glass, plastic film, or soap bubbles. Unlike the regular `Dielectric` material
which assumes a volumetric object, this material models surfaces with negligible thickness. It does not
expose any visual channels.

| Parameter | Type | Description |
|-----------|------|-------------|
| `int_ior` | `str` or `float` | Interior index of refraction (default: `bk7`, see [supported ior list](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#ior-table-list)) |
| `ext_ior` | `str` or `float` | Exterior index of refraction (default: `air`, see [supported ior list](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#ior-table-list)) |
| `medium` | `Medium` or `None` | Medium of the enclosed material (default: `None`) |
| `specular_reflectance` | `float` | Specular reflectance (default: `1.0`) |
| `specular_transmittance` | `float` | Specular transmittance (default: `1.0`) |

Here is a snippet for creating a `ThinDielectric` material for a glass-like surface:

```py
m = hkw.material.ThinDielectric(int_ior="glass")
```

Check out the [Mitsuba
doc](
https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#thin-dielectric-material-thindielectric
)
for more details.

Here is a snippet for creating a `ThinDielectric` material.

```py
m = hkw.material.ThinDielectric(int_ior="water")
```

## Hair material

![Hair material](../images/spot_hair.png){: style="width:200px"}

`Hair` has uniform coloring controls rather than attribute-driven visual channels. It accepts:

| Parameter | Type | Description |
|-----------|------|-------------|
| `eumelanin` | `float` | Dark/brown pigment concentration (default: `1.3`); ignored when a color or root/tip gradient is set |
| `pheomelanin` | `float` | Reddish-yellow pigment concentration (default: `0.2`); ignored when a color or gradient is set |
| `color` | `TextureLike` or `None` | Uniform RGB/named color; defaults to `None`, which uses melanin. Data-driven colors are unsupported |
| `root_color` | `ColorLike` or `None` | Root color for a root-to-tip gradient; overrides `color` and melanin when set |
| `tip_color` | `ColorLike` or `None` | Tip color for a root-to-tip gradient |
| `color_variation` | `float` | Per-strand brightness jitter from 0 to 1 (default: `0.0`) |

Backend behavior differs: Blender evaluates gradients and variation per strand. Mitsuba averages a
root/tip gradient into one color and ignores variation. WebGL approximates hair with diffuse shading,
averages gradients, and falls back to brown for melanin. Data-driven hair colors are unsupported.

Here is a snippet for creating a `Hair` material.

```py
m = hkw.material.Hair(eumelanin=0.3, pheomelanin=0.1)
```

Check out the [Mitsuba
doc](
https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#hair-material-hair
)
for more details.

## Two-sided material

Sometimes, the mesh we have may not be correctly oriented. Inverted facets will appear to be black.
To avoid such artifact, all material classes support an optional `two_sided` parameter to enabled
two-sided normals.

```py
m = hkw.material.Diffuse("salmon", two_sided=True)
```

## Two-color material

To shade the front and back faces of each facet differently, set the `back_side` parameter to an
independent material. The owning material is used for the front face and `back_side` for the back
face. This implies two-sided rendering regardless of `two_sided`, and only applies to surface marks.
A nested `back_side` on the back material itself is ignored.

```py
m = hkw.material.Diffuse(
    "salmon",
    back_side=hkw.material.Diffuse("skyblue"),
)
```

## Bump mapped material

A bump map can be applied to any material by setting the `bump_map` texture and `bump_scale`
parameter. The former parameters defines the raw bump map texture where the latter defines the
multiplicative factor to scale the bump magnitude.

```py
m = hkw.material.Diffuse(
    "salmon",
    bump_map=hkw.texture.Image("bump_map.png"),
    bump_scale=0.5,
)
```
See the [Moon example](../examples/moon.md) for an application of bump map.
