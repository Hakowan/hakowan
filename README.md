# Hakowan

![Hakowan teaser](docs/images/teaser.webp)

Hakowan is a 3D data visualization grammar. It is inspired by the grammar of graphics, and it is
designed for easily creating beautiful 3D visualizations.

## Install

Hakowan relies on [Lagrange](https://opensource.adobe.com/lagrange-docs/) for geometry processing
and supports three rendering backends:

| Backend | Description | Extra |
|---------|-------------|-------|
| **WebGL** (default) | Interactive browser viewer — ships with base install | *(none)* |
| **Mitsuba** | Photorealistic offline renderer | `hakowan[mitsuba]` |
| **Blender** | Cycles/EEVEE renderer (requires Python 3.13) | `hakowan[blender]` |

```sh
# WebGL only (default)
pip install hakowan

# Mitsuba backend
pip install hakowan[mitsuba]

# Blender backend (Python 3.13 required)
pip install hakowan[blender]

# All backends
pip install hakowan[mitsuba,blender]
```

Note that Hakowan requires Python 3.11 and above (Python 3.13 for the Blender backend).

To check which backends are available in your environment:

```py
import hakowan as hkw
print(hkw.list_backends())
```

## Quick start

The following example renders an interactive HTML viewer using the WebGL backend:

```py
import hakowan as hkw

layer = (
    hkw.layer("mesh.obj")
    .mark(hkw.mark.Surface)
    .channel(material=hkw.material.Diffuse(reflectance="orange"))
)

result = hkw.render(layer, filename="viewer.html", backend="webgl")
# Open viewer.html in any modern browser
```

## Documentation

[HTML](https://hakowan.github.io/hakowan/)

```bibtex
@software{hakowan,
    title = {Hakowan: A 3D Data Visualization Grammar},
    version = {0.5.2},
    year = 2026,
}
```
