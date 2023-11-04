# Layer Overview

Layer is a concept that holds the specification of the 4 key components: data, mark, channels and
transform. A layer may be complete if all its associated components are not `None`, or partial if
one or more component is `None`. A layer is created with the `hkw.layer` method.

``` py
l0 = hkw.layer()
```

Here we have created an empty layer, where all of data, mark, channels and transform are None.

An empty layer cannot be rendered because data component is required for rendering. With data is
specified, Hakowan will assign a default mark, channel or transform if any of them are `None` at the
rendering time. Fortunately, it is very easy to build on top of an existing layer with a set of
overwrite functions. For example,

``` py
l1 = l0.data("shape.obj")
```

where `l1` is a new layer created from `l0` layer with the data component set to `shape.obj`. The
method `.data` is an example of such overwrite functions. The other overwrite functions are `.mark`,
`.channel` and `.transform`. These overwrite functions can be chained together based on the [fluent
interface design pattern](https://en.wikipedia.org/wiki/Fluent_interface).

``` py
l2 = l0.data("shape.obj").mark(hkw.mark.Point)
```

Note that the overwrite functions does not change the caller object (i.e. `l0` in the above
example). This design allows the base layer `l0` to be reused over and over again. Here is a more
complex example.

``` py
mesh = lagrange.io.load_mesh("shape.obj")
position_attr_name = mesh.attr_name_vertex_to_position
base = (
    hkw.layer()
    .data(mesh)
    .mark(hkw.mark.Point)
    .channel(size=0.1)
    .transform(
        hkw.transform.Filter(
            data=hkw.attribute(position_attr_name), condition=lambda p: p[0] > 0
        )
    )
)
```

This visualization shows all vertices of the input mesh with positive x coordinate as spheres with
radius 0.1.

## Layer composition

In the following example, we will demonstrate the idea of _layer composition_.

``` py
base = hkw.layer().data("shape.obj")

surface_view = base.mark(hkw.mark.Surface)
point_view = base.mark(hkw.mark.Point)
edge_view = base.mark(hkw.mark.Curve)

composite_view = surface_view + point_view + edge_view
```

Here, `surface_view` is a visualization of the surface geometry, while `point_view` and `edge_view`
are the visualizations of vertices and edges of the geometry. The addition operations combines all
three views together to form a composite view that visualizes all three elements. TODO: Show an
example.

