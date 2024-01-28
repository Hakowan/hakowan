# Hakowan
Hakowan is a 3D data visualization grammar. It is inspired by the grammar of graphics, and it is
designed for easily creating beautiful 3D visualizations.

## Install

Hakowan relies on [Lagrange](https://opensource.adobe.com/lagrange-docs/) and
[Mitsuba](https://www.mitsuba-renderer.org/) to provide geometry processing and rendering
capabilities. Both Hakowan and its dependencies can be simply installed via pip:

```sh
pip install hakowan
```

Note that hakowan requires python 3.11 and above.

## Quick start

```py
import hakowan as hkw

base = hkw.layer("mesh.obj")            # Create a base layer
hkw.render(base, filename="image.exr")  # Render!
```

## Documentation

[HTML](https://qnzhou.github.io/hakowan/)

```bibtex
@software{hakowan,
    title = {Hakowan: A 3D Data Visualization Grammar},
    version = {0.3.2},
    year = 2024,
}
```

