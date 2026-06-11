# Hakowan: A 3D Data Visualization Grammar

![Hakowan teaser](images/teaser.webp)

Hakowan is a 3D data visualization grammar based on the concept of [The Grammar
of Graphics](https://link.springer.com/book/10.1007/0-387-28695-0). It is
designed for creating compelling SIGGRAPH-quality 3D data visualizations with
minimal setup. It provides a concise, high-level declarative API in python and
is powered by project [Lagrange](https://opensource.adobe.com/lagrange-docs/)
for geometry processing and supports three rendering backends:
[WebGL](https://threejs.org/) (default, interactive browser viewer),
[Mitsuba](https://mitsuba.readthedocs.io/en/stable/index.html) (photorealistic), and
[Blender](https://www.blender.org/) (Cycles/EEVEE).

## Installation

Hakowan can be installed via `pip` from PyPI. The WebGL backend ships with the
base install; the heavier Mitsuba and Blender backends are optional extras:

``` sh
pip install hakowan                  # WebGL only (default)
pip install hakowan[mitsuba]         # add the Mitsuba backend
pip install hakowan[blender]         # add the Blender backend (Python 3.13)
pip install hakowan[mitsuba,blender] # all backends
```

Note that Hakowan requires Python 3.11 and above (Python 3.13 for the Blender backend).

## Quick start

Let `shape.obj` be a mesh that you would like to visualize:

``` py
import hakowan as hkw

base = hkw.layer("shape.obj")
hkw.render(base, filename="output.html")
```

The above code creates a single visualization [_layer_](guide/layer.md) using
`shape.obj` as the [_data_](guide/data.md). This layer is then rendered with the
default WebGL backend into an interactive viewer named `output.html`, which you
can open in any modern browser.

Hakowan's grammar decomposes a 3D visualization into layers, where each layer
provides a specification of one or more of the following items:

* [__Data__](guide/data.md): consists of the geometry as well as
[_attributes_](guide/attribute.md) associated with the geometry.

* [__Mark__](guide/mark.md): determines the geometry type (e.g. point, curve or surface).

* [__Channel__](guide/channel.md): defines the mapping from data attributes to the available visual channels.

* [__Transform__](guide/transform.md): is the data transformation that should be carried out before visualization.

## Citation

``` bibtex
@software{hakowan,
    title = {Hakowan: A 3D Data Visualization Grammar},
    version = {0.5.2},
    year = 2026,
}
```
