# Grammar Overview

The grammar of a language provides a set of structural rules to combine words
together to form semantically meaningful statements. Similarly, from the data
visualization community, the seminal work of [The Grammar of
Graphics](https://link.springer.com/book/10.1007/0-387-28695-0) from Leland
Wilkinson proposed a structured way to systematically decompose a graphic (i.e.
chart) into a set of visual and mathematical components. Much like the recipe
of a dish, each component is independent, and together they provide a complete
specification of a chart. The grammar of graphics is a highly flexible
way of describing the desired chart while abstracting away much of the tedious
details related to chart drawing.

Hakowan is a 3D data visualization grammar based on the concept of the grammar
of graphics. It identifies 4 key components in 3D data visualization:
[_data_](data.md), [_mark_](mark.md), [_channels_](channel.md) and
[_transform_](transform.md). A complete specification of these 4 components
forms a [_layer_](layer.md) in the visualization. Each component of a layer can
be overwrite, and multiple layers can be combined together to generate
composite visualization.

## Creating a layer

For the rest of this document, we will assume the `hakowan` package has been imported as the alias `hkw`.

```py
import hakowan as hkw
```

Layer is the most fundamental object within Hakowan as it contains the complete
specification of all 4 key components. The following snippet shows how to
create a layer with default settings.

```py
l = hkw.layer()
```
