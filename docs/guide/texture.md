# Texture

In 3D data visualization, texture defines the mapping from a 3D element to colors or values.

## Uniform texture

`Uniform` texture maps all 3D elements to the same value.

```py
t = hkw.texture.Uniform(color="ivory")
```

## Image texture

`Image` texture maps a 3D element to color based on the UV coordinates and an image texture.

```py
t = hkw.texture.Image(uv="uv_attr_name", filename="texture.png")
```

| Field | Type | Meaning |
|-------|------|---------|
| `filename` | PathLike | Path to the image file |
| `uv` | AttributeLike \| None | UV attribute (auto-detected if None) |
| `raw` | bool | Treat texture as linear (no sRGB decode). Set `True` for normal maps and other non-color data. Default `False`. |
| `saturation` | float | Saturation multiplier. `1.0` = full color, `0.0` = grayscale. Default `1.0`. |
| `whiteness` | float | Blend toward pure white. `0.0` = original color, `1.0` = pure white. Default `0.0`. |

```py
# Desaturate and brighten an image texture.
t = hkw.texture.Image(
    filename="texture.png",
    uv="uv_attr",
    saturation=0.5,
    whiteness=0.2,
)
```

## Scalar field texture

One of the most common use case of texture is to map a scalar field to a color field.

| Field | Type | Meaning |
|-------|------|---------|
| data  | AttributeLike | The attribute defining the scalar field |
| colormap | str \| list | A built-in or [colorcet](https://colorcet.holoviz.org) colormap name, `"identity"`, or an explicit list of colors |
| domain | tuple | The domain of the attribute |
| range | tuple | The range of colormap |
| categories | bool | Whether the data represents categories (i.e. discrete values) |
| reverse | bool | Whether to reverse the colormap direction (maps the largest value to the colormap's first color) |

```py
t = hkw.texture.ScalarField(data="attr_name")
```

See the [Heat Method](../examples/heat.md) and the [Components](../examples/components.md) examples
for application of the scalar field texture.

## Checkerboard texture

`CheckerBoard` texture maps 3D elements to one of two possible sub-textures based on a checkerboard
pattern.

```py
t = hkw.texture.CheckBoard(
    uv="uv_attr_name",
    texture1=hkw.texture.Uniform(color=0.2),
    texture2=hkw.texture.Uniform(color=0.8),
)
```

## Isocontour texture

`Isocontour` texture maps 3D elements to one of two possible sub-textures based on the iso-contour of
a given scalar field.

```py
t = hkw.texture.Isocontour(
    data="attr_name",
    ratio=0.2,
    texture1=hkw.texture.Uniform(color=0.2),
    texture2=hkw.texture.Uniform(color=0.8),
)
```

See the [Heat Method](../examples/heat.md) example for an application of the isocontour texture.
