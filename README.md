# Hakowan
Hakowan is a 3D data visualization grammar. It is inspired by the grammar of graphics, and it is
designed for easily creating beautiful 3D visualizations.

## Install

Hakowan relies on [Lagrange](https://opensource.adobe.com/lagrange-docs/) for geometry processing
and supports three rendering backends: [Mitsuba](https://www.mitsuba-renderer.org/) (default,
photorealistic), [Blender](https://www.blender.org/) (Cycles/EEVEE), and WebGL (interactive
browser viewer). Hakowan and its core dependencies can be installed via pip:

```sh
pip install hakowan
```

Note that Hakowan requires Python 3.11 and above.

## Quick start

```py
import hakowan as hkw

base = hkw.layer("mesh.obj")            # Create a base layer
hkw.render(base, filename="image.exr")  # Render!
```

## Documentation

[HTML](https://hakowan.github.io/hakowan/)

```bibtex
@software{hakowan,
    title = {Hakowan: A 3D Data Visualization Grammar},
    version = {0.4.4},
    year = 2026,
}
```

