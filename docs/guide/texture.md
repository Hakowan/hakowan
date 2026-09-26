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

For the common task of coloring geometry by one scalar attribute, prefer the
high-level layer helper:

```py
colored = hkw.layer("mesh.ply").color_by("temperature")
```

It infers the domain and creates a diffuse `ScalarField` with `viridis` and an
automatic legend. Construct `ScalarField` directly when nesting it in another
material field or texture:

```py
t = hkw.texture.ScalarField(data="temperature")
```

| Field | Type | Meaning |
|-------|------|---------|
| data | AttributeLike | Scalar attribute to visualize |
| colormap | str \| list \| None | Built-in or [colorcet](https://colorcet.holoviz.org) name, `"identity"`, or explicit colors; defaults to `"viridis"` for continuous fields and `"set1"` when `categories=True` |
| domain | tuple \| None | Input domain; inferred from finite data when omitted |
| range | tuple \| None | Subrange of the colormap; full range when omitted |
| categories | bool | Treat values as discrete categories and use `"set1"` by default |
| reverse | bool | Reverse colormap direction |
| legend | bool \| Legend | Automatic by default; `False` suppresses it |

See the [Heat Method](../examples/heat.md), [Components](../examples/components.md),
and [Legends and annotations](overlay.md) guides for advanced examples.

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
