# Hakowan
Hakowan is a 3D data visualization grammar. It is inspired by the grammar of graphics, and it is
designed for easily creating beautiful 3D visualizations.

## Install

Hakowan relies on Lagrange and Mitsuba to provide geometry processing and rendering capabilities.
Both Hakowan and its dependencies can be simply installed via pip:

```sh
pip install git+ssh://git@github.com/qnzhou/hakowan.git
```

## Key Concept

Based on the principles from grammar of graphics, Hakowan decomposes a 3D visualization into the
four basic components: data, mark, channel and transform.  A complete specification of all four components
forms a __layer__, and multiple layers can be combined together to create the final visualization.
All components have reasonable default settings.  Here is a simple example:

```python
import hakowan

layer = hakowan.layer()             # Create an empty layer.
layer = layer.data(mesh)            # Set data to a mesh (lagrange.SurfaceMesh object).
layer = layer.mark(hakowan.SURFACE) # Set mark to surface.
layer = layer.channel(
    color = "facet_id",             # Set color channel to a mesh attribute.
    color_map = "turbo",            # Use turbo color map.
    roughness = 0.5,                # Set roughness channel to a const value.
    metallic = 0.75,                # Set metallic channel to a const value.
)

config = hakowan.RenderConfig()   # Global render settings.
config.filename = "output.exr"   # The output image filename.

hakowan.render(layer, config)
```


### Data

Data component specifies the data frame of the visualization. In 3D visualization, a data frame
contains the geometry and any attributes associated with various elements of the geometry.
We use `lagrange.SurfaceMesh` to represent a 3D data frame. It is capable of representing arbitrary
polygonal mesh, point cloud.  It support scalar, vector and tensor attributes associated with
vertices, facets, edges and  corners. It also supports indexed attributes and value attributes.

To specify a data:

```python
mesh = lagrange.io.load_mesh("input_mesh.obj")
layer = layer.data(mesh)
```

### Mark

Mark specifies the type of geometry to render.  Available options are the following:
* `hakowan.SURFACE`: Use the surface mesh as geometry.
* `hakowan.POINT`: Use points as geometry.
* `hakowan.CURVE`: Use curves as geometry.

### Channel

Channel specifies the mapping between data and visual channels.
The following channels are supported:
* `color`: The base color channel. It can be a hex color value (e.g. `#0f0f0f`), or a valid CSS
  color name (e.g. `gray`) or an attribute name from the current layer's data frame.
* `roughness`: The roughness of the material. It can be a value in [0, 1] range or an attribute name
  from the current layer's data frame.
* `metallic`: The metallic-ness of the material. It can be a value in [0, 1] range or an attribute
  name from the current layer's data frame.
* `size`: The size channel.  Only used by point and curve marks.
* `color_map`: The color map.  Must be one of the predefined color maps: `viridis`, `turbo`,
  `coolwarm`, `magma`, `plasma` and `inferno`.
* `alpha`: It specifies the transparency of the material.  (Not yet supported.)
* `position`: Specifies the vertex positions.  (Not yet supported.)
* `normal`: Specifies the normal attribute to use. (Not yet supported.)
* `uv`: Specifies the uv attribute. (Not yet supported.)

### Transform

Transform specifies the transformation of the data frame before rendering. It supports the following
types:

* Rotation and translation of the geometry as a whole. (TODO)
* Filtering of geometric elements. (TODO)
* Rotation and translation of geometric elements. (TODO)
* Generation of new geometries such as isoline. (TODO)

