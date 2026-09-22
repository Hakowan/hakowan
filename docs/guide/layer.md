# Layer Overview

Layer is a concept that holds the specification of the 4 key components: [data](data.md),
[mark](mark.md), [channels](channel.md) and [transform](transform.md). A layer may
be complete if all its associated components are not `None`, or partial if one or more component is
`None`. A layer is created with the `hkw.layer` method.

``` py
l0 = hkw.layer()
```

Here we have created an empty layer, where all of data, mark, channels and transform are None.

An empty layer cannot be rendered because the data component is required for rendering.
Fortunately, it is very easy to build on top of an existing layer with a set of
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

Note that the overwrite functions do not change the caller object (i.e. `l0` in the above
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
            data=position_attr_name, condition=lambda p: p[0] > 0
        )
    )
)
```

This visualization shows all vertices of the input mesh with positive X coordinate as spheres with
radius 0.1. Note that, in addition to filename, `.data` method can also take an actual Lagrange
`SurfaceMesh` object.

Lastly, it is also possible to directly specify the components as arguments to `hkw.layer` method.
```py
base = hkw.layer("shape.obj", mark=hkw.mark.Point)
```

## Task-oriented helpers

Common visualization intents have concise methods. Each method expands into the
same marks, channels, textures, transforms, and composition nodes described by
the core grammar, so serialization and backend behavior remain unchanged.

### Color by a scalar attribute

The default workflow needs only the data and attribute name. Hakowan infers the
domain and supplies `viridis` plus an automatic legend:

```py
colored = hkw.layer("mesh.ply").color_by("temperature")
```

Specify only the options that differ from those defaults:

```py
colored = base.color_by(
    "temperature",
    domain=(0, 100),
    legend=hkw.Legend(title="Temperature", units="°C"),
)
```

`color_by()` creates a diffuse material whose reflectance is a `ScalarField`.
It also supports categorical fields, reversed or custom colormaps, an explicit
output range, custom legends, and two-sided rendering. Use `hkw.inspect()` when
the scalar attribute name or element domain is not known.

### Overlay mesh edges

```py
with_edges = colored.show_edges(color="black", width=0.01)
```

`show_edges()` returns the original visualization overlaid with a curve-mark
view of the same data. The input layer is not modified.

### Add vector glyphs

```py
with_velocity = colored.glyph_vectors(
    "velocity",
    scale=0.2,
    size=0.01,
    color="white",
    normalize=False,
    end_type="arrow",
)
```

Vector `scale` controls glyph length and `size` controls thickness. By default,
the glyph layer is overlaid on the input; pass `overlay=False` to return only
the vector visualization.

### Slice with a plane

```py
upper = base.slice(normal=(0, 0, 1), offset=0.25)
```

`offset` is signed distance along the normalized plane normal. Use `point=`
instead when the plane must pass through a specific point.

### Isolate a connected component

```py
component = base.isolate_component(3)
```

By default, Hakowan computes connected-component IDs in a temporary
`component` attribute and keeps the requested facet group. To use existing
labels instead:

```py
region = base.isolate_component(8, attribute="region", compute=False)
```

The generated filter uses the restricted expression system and remains fully
serializable.

### Compare two layers

```py
comparison = before.compare(
    after,
    axis="x",
    gap=0.1,
    normalize=True,
    labels=("Before", "After"),
)
```

`compare()` is a convenience over `juxtapose()`. Labels name layers in the
interactive WebGL controls; static Blender and Mitsuba images do not draw them.

## Layer composition

In the following example, we will demonstrate the idea of _layer composition_.

``` py
base = hkw.layer("shape.obj")

surface_view = base.mark(hkw.mark.Surface)
point_view = base.mark(hkw.mark.Point)
edge_view = base.mark(hkw.mark.Curve)

composite_view = surface_view + point_view + edge_view
```

Here, `surface_view` is a visualization of the surface geometry, while `point_view` and `edge_view`
are the visualizations of vertices and edges of the geometry. The addition operations combines all
three views together to form a composite view that visualizes all three elements.

## Layer comparison

While `+` overlays layers in the same coordinate space, the `|` operator places layers _side by
side_ so they can be compared.

``` py
comparison = hkw.layer("before.obj") | hkw.layer("after.obj")
```

This lays out the two layers in a horizontal row, automatically translating them apart so they do
not overlap. The original relative scale of each layer is preserved.

The `&` operator is the vertical analogue of `|`: it stacks layers in a column instead of a row.

``` py
comparison = hkw.layer("before.obj") & hkw.layer("after.obj")
```

!!! note
    Python binds `&` tighter than `|`, so `a | b & c` parses as `a | (b & c)`. Parenthesise when
    mixing the two operators.

For more control, use the [`juxtapose`][hakowan.grammar.layer.layer.Layer.juxtapose] method, which
`|` and `&` call with default settings (`axis="x"` and `axis="y"` respectively):

``` py
comparison = base.juxtapose(
    other,
    axis="y",        # lay out along Y instead of the default X
    gap=0.2,         # spacing between cells, as a fraction of the mean cell size
    normalize=True,  # scale each cell to equal size before placing
)
```

Each operand of `|` becomes one _cell_. Cells may themselves be composite layers, so `+`, `|`, and
`&` combine freely:

``` py
# Compare a bare surface against the same surface with its wireframe overlaid.
surface = hkw.layer("shape.obj").mark(hkw.mark.Surface)
edges = hkw.layer("shape.obj").mark(hkw.mark.Curve)
comparison = surface | (surface + edges)
```

Nesting `|` and `&` builds a 2-D **grid** — each operator lays out its own operands along its axis,
so a row of cells can be stacked over another, or several rows tiled into a matrix:

``` py
top    = a | b           # a row
grid   = (a | b) & c     # the a–b row stacked above c
matrix = (a | b) & (c | d)  # a 2x2 grid
```

Cells are spaced by their bounding spheres, so they never overlap — even as you rotate each cell
in the interactive viewer.

## Annotations

Attach deterministic screen-space labels with `annotate()`:

```py
layer = hkw.layer("shape.obj").annotate(
    "Simulation A",
    position=(0.5, 0.05),
    anchor="center",
    background="black",
)
```

Annotations follow layer inheritance and are deduplicated when composite views
are compiled. See [Legends and annotations](overlay.md).

