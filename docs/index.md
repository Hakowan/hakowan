# Hakowan: A 3D Data Visualization Grammar

Hakowan is a 3D data visualization grammar based on the concept of [The Grammar
of Graphics](https://link.springer.com/book/10.1007/0-387-28695-0). It is
designed for creating compelling SIGGRAPH-quality 3D data visualizations with
minimal setup. It provides a concise, high-level declarative API in python and
is powered by project [Lagrange](https://opensource.adobe.com/lagrange-docs/)
and [Mitsuba](https://mitsuba.readthedocs.io/en/stable/index.html) for data
processing and rendering.

## Installation

Hakowan can be installed via `pip` from PyPI:

``` sh
pip install hakowan
```

Note that hakowan requires python 3.11 and above.

## Quick start

Let `shape.obj` be a mesh that you would like to visualize:

``` py
import hakowan as hkw

base = hkw.layer("shape.obj")
hkw.render(base, filename="output.exr")
```

The above code creates a single visualization [_layer_](guide/layer.md) using
`shape.obj` as the [_data_](guide/data.md). This layer is then rendered into an image
named `output.exr`.

Hakowan's grammar decompose a 3D visualization into layers, where each layer
provides a specification of one or more of the following items:

* [__Data__](guide/data.md): consists of the geometry as well as
[_attributes_](guide/data.md#Attribute) associated with the geometry.

* [__Mark__](guide/mark.md): determines the geometry type (e.g. point, curve or surface).

* [__Channel__](guide/channel.md): define the mapping from data attributes to the available visual channels.

* [__Transform__](guide/transform.md): is the data transformation that should be carried out before visualization.

## Citation

``` bibtex
@software{zhou2023hakowan,
    title = {Hakowan: A 3D Data Visualization Grammar},
    author = {Qingnan Zhou and Zhicheng Liu}
    version = {0.1.0},
    year = 2023,
}
```
