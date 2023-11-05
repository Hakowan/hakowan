# Attribute

In Hakowan, an attribute specifies a speicific "column" of the [data](data.md) (i.e. mesh attribute)
that will be used to encode various visual [channels](channel.md) and [textures](texture.md). Each
attribute consists of a name and a [scale](scale.md). The name is the name of the mesh attribute
used as data, and the scale defines a "column"-specific transformation applied to the attribute
before mapping to visual channels.

```py
# To specify an attribute from name alone.
By default, scale is identity.
attr = hkw.attribute(name="normal")

# To specify an attribute from both name and scale.
attr = hkw.attribute(name="normal", scale=hkw.scale.Uniform(factor=2))
```
