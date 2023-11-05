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

## Scalar field texture

One of the most common use case of texture is to map a scalar field to a color field.

| Field | Type | Meaning |
|-------|------|---------|
| data  | AttributeLike | The attribute defining the scalar field |
| colormap | str | The colormap to use |
| domain | tuple | The domain of the attribute |
| range | tuple | The range of colormap |

```py
t = hkw.texture.ScalarField(data="attr_name")
```

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

`Isocontour` texture maps 3D elements to one of two possible sub-textures based on the isocontour of
a given scalar field.

```py
t = hkw.texture.Isocontour(
    data="attr_name",
    ratio=0.2,
    texture1=hkw.texture.Uniform(color=0.2),
    texture2=hkw.texture.Uniform(color=0.8),
)
```
