# Attribute

In Hakowan, an attribute specifies a specific "column" of the [data](data.md) (i.e. mesh attribute)
that will be used to encode various visual [channels](channel.md) and [textures](texture.md). Each
attribute consists of a name and a [scale](scale.md). The name is the name of the mesh attribute
used as data, and the scale defines a "column"-specific transformation applied to the attribute
before mapping to visual channels.

## Creating Attributes

The `hkw.attribute()` function is a convenient alias for creating `Attribute` objects:

```py
# To specify an attribute from name alone.
# By default, scale is identity.
attr = hkw.attribute(name="normal")

# To specify an attribute from both name and scale.
attr = hkw.attribute(name="normal", scale=hkw.scale.Uniform(factor=2))

# hkw.attribute() is equivalent to hkw.scale.Attribute()
attr = hkw.scale.Attribute(name="normal", scale=hkw.scale.Uniform(factor=2))
```

## Attribute-Like Objects

Many Hakowan functions accept "attribute-like" objects, which can be either:

- A **string**: Automatically converted to an `Attribute` with the given name
- An **`Attribute` object**: Used directly

This allows for convenient shorthand notation:

```py
# These are equivalent:
ch1 = hkw.channel.Position(data=hkw.attribute(name="position"))
ch2 = hkw.channel.Position(data="position")

# These are also equivalent:
tex1 = hkw.texture.ScalarField(data=hkw.attribute(name="curvature"))
tex2 = hkw.texture.ScalarField(data="curvature")
```
