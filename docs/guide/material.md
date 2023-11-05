# Material

In 3D data visualization, material is the new color! By assigning a material to the 3D geometry, we
can not only achieve more photorealistic visual quality but also encode data in more channels than
just RGB. Hakowan leverages [Mitsuba's material
support](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html) to provide a
number of material-based visual channels.

## Diffuse material

`Diffuse` material evenly scatter incoming light in all directions. This material model provide a
single channel, `reflectance`, for encoding data. The `reflectance` channel can be thought of as a
color field. Here is a simple example of creating a diffuse material.

```py
m = hkw.material.Diffuse(reflectance = "blue")
```

Here, we assign a uniform color, "blue", to the `reflectance` channel. To encode actual data, we
need to assign a [texture](texture.md) to the `reflectance` channel.

```py
m = hkw.material.Diffuse(reflectance = hkw.texture.ScalarField(data = "attr_name"))
```

Check the [Mitsuba
doc](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-diffuse-material-diffuse)
for more details.

## Conductor material

`Conductor` material gives the geometry a metallic look and feel. It takes a `material` parameter as
input, but it cannot be used to encode any data. The set of supported conductor materials can be
found in the [Mitsuba doc](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-conductor-conductor).

```py
m = hkw.material.Conductor(material="Al")
```

Check the [Mitsuba
doc](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-conductor-conductor)
for more details.

## Rough conductor material

`RoughConductor` material gives the geometry a matte metallic look and feel. It takes a `material`
parameter just like `Conductor`. In addition, it also takes a `distribution` parameter and a `alpha`
channel. The distribution parameter is used to determine the microfacet normal distribution. The
supported distributions are `ggx` and `beckmann` (default). The alpha channel represents the
roughness of the surface, and it can be used to encode user-defined data. Here is a simple example
of using `RoughConductor`.

```py
m = hkw.material.RoughConductor(material = "Cu")
```

Here is a more complex example where we map an attribute to the alpha channel:

```py
m = hkw.material.RoughConductor(
    material="Cu", alpha=hkw.texture.ScalarField(data="attr_name")
)
```

Check out the [Mitsuba
doc](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#rough-conductor-material-roughconductor)
for more details.

## Plastic material

`Plastic` material provides a plastic look and feel. This material expose two potential channels for
encoding data: `diffuse_reflectance` and `specular_reflectance`.

```py
m = hkw.material.Plastic(
    diffuse_reflecctance=hkw.texture.ScalarField(data="attr_name"),
    specular_reflectance=hkw.texture.ScalarField(data="attr2_name")
)
```

Check out the [Mitsuba
doc](
https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-plastic-material-plastic
)
for more details.

## Rough plastic material

`RoughPlastic` material is the matte version of the `Plastic` material. It inherits the same visual
channels, i.e. `diffuse_reflectance` and `specular_reflectance`, for data encoding. In addition, it
also support the `distribution` and `alpha` parameter. Similar to `RoughConductor`, the
`distribution` parameter controls the microfacet normal distribution, and its valid values are `ggx`
and `beckmann` (default). The `alpha` parameter control the roughness of the material.

```py
m = hkw.material.RoughPlastic(
    diffuse_reflecctance=hkw.texture.ScalarField(data="attr_name"),
    specular_reflectance=hkw.texture.ScalarField(data="attr2_name"),
    alpha=0.1
)
```

Check out the [Mitsuba
doc](
https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#rough-plastic-material-roughplastic
)
for more details.

## Principled material

`Principled` material is the most versatile material. It is based on paper "[Physically Based
Shading](https://www.disneyanimation.com/publications/physically-based-shading-at-disney/)" and its
extension. This material expose three channels capable of encoding data: `color`, `roughness` and
`metallic`.

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
