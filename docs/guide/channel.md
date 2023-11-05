# Channels

A channel represents a specific visual quantity that can be used to encode data.
For example, 2D data visualization often encode data in position, color, shape and size channels.
In 3D data visualization, we have the freedom of inventing new visual channels thanks to the diverse
material models modern rendering engine supports. We have a separate guide for [material-based
channels](material.md). In this guide, we will mainly focus on non-material channels.

## Position channel

`Position` channel represents the position of 3D marks. By default, the position channel will
use the vertex positions stored in the data frame. It is especially useful when multiple sets of
positions are available (e.g. animation or decimation).

```py
# To specify an attribute as the position channel data:
ch = hkw.channel.Position(data = hkw.attribute(name = "attr_name"))

# A shorthand that is exactly the same as above.
ch = hkw.channel.Position(data = "attr_name")
```

## Normal channel

`Normal` channel represents the normal vector field of a 3D surface. It has significant influence on
how the surface reflect lights. By default, normals will be computed from the 3D data frame. The
`Normal` channel is only relevant for `Surface` mark.

```py
# To specify an attribute as the normal channel data:
ch = hkw.channel.Normal(data = hkw.attribute(name = "attr_name"))

# Shorthand. Same as above
ch = hkw.channel.Normal(data = "attr_name")
```

## Size channel

`Size` channel represents the size of 3D marks. It is only relevant to `Point` and `Curve` marks.
For `Point` mark, size represents the radius of the point mark. For `Curve` mark, size represents
the radius of the curves.

```py
# To sepcify an attribute as the size channel data:
ch = hkw.channel.Size(data = hkw.attribute(name = "attr_name"))

# Shorthand. Same as above
ch = hkw.channel.Size(data = "attr_name")

# To assign constant size field
ch = hkw.channel.Size(data = 0.1)
```

Note that `Size` channel uses the same unit as the `Position` channel.

## Vector field channel

`VectorField` channel defines the data used for a vector field visualization. This channel is only
relevant when the mark is `Curve` as each vector is rendered using the curve geometry.

```py
# To specify an attribute as the vector field channel data.
ch = hkw.channel.VectorField(data = hkw.attribute(name = "attr_name"))

# Shorthand, same as above.
ch = hkw.channel.VectorField(data = "attr_name")
```
