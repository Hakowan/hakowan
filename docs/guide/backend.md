# Rendering Backends

Hakowan offers three backends: **WebGL** for interactive browser viewing,
**Mitsuba** for photorealistic rendering, and **Blender** for Cycles or EEVEE
rendering. WebGL is included in the base install and is the default; Mitsuba
and Blender are optional extras (`pip install hakowan[mitsuba]` /
`pip install hakowan[blender]`).

## Available Backends

### Mitsuba Backend

Use Mitsuba for photorealistic renders with advanced lighting and materials.
Install the optional backend with `pip install hakowan[mitsuba]`.

**Advantages:**
- High-quality photorealistic rendering
- Fast rendering with GPU acceleration
- Support for advanced materials and lighting

**Requirements:**
- Install the `mitsuba` extra (`pip install hakowan[mitsuba]`)

**Usage:**

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
hkw.render(layer, filename="output.exr", backend="mitsuba")
```

For edge-derived `Curve` marks, Mitsuba supports vertex `ScalarField` colors on `Diffuse` and
`Principled` materials when each edge has the same resolved BSDF at both endpoints. Hakowan groups
edges with identical BSDFs into curve shapes. If material values vary across an edge, rendering
raises `NotImplementedError`; use a tube mesh for smoothly varying color. Attribute-driven materials
on vector-field curves are not supported.

### Blender Backend

Use Blender to render with Cycles or EEVEE, save editable `.blend` scenes, or
produce render passes such as albedo, depth, normal, and facet ID.

**Advantages:**
- Integration with Blender ecosystem
- Access to Blender's Cycles and EEVEE rendering engines
- Can save Blender scene files (.blend) for further editing
- Supports render passes: albedo, depth, normal, facet ID

**Requirements:**
- Install the `blender` extra (`pip install hakowan[blender]`); requires Python 3.13

**Usage:**

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
# Explicitly specify Blender backend
hkw.render(layer, filename="output.png", backend="blender")

# Optionally save the Blender scene file
hkw.render(
    layer,
    filename="output.png",
    backend="blender",
    blend_file="scene.blend"
)
```

### WebGL Backend

WebGL creates an interactive HTML viewer for exploring a scene. It is included
in the base install and is the default backend. Set `offline=True` to save the
viewer with the assets needed to use it without a network connection.

**Advantages:**
- Default backend — ships with the base install
- Immediate interactive browser output
- The complete scene can be shared as an HTML viewer
- Optional offline viewer assets
- Live albedo, depth, and normal passes
- Deterministic headless snapshots with `hakowan[observe]`

**Requirements:**
- Interactive viewer: none beyond the base install and a modern browser
- `snapshot()` / `observe()`: `pip install "hakowan[observe]"` followed by
  `playwright install chromium`

**Usage:**

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
hkw.render(layer, filename="output.html", backend="webgl")
```

The interactive viewer shows a layer list when the scene contains multiple
rendered layers. Each layer has a visibility checkbox; unnamed layers receive
generated labels. Use the **−** button to minimize the list and **layers +** to
restore it. Minimizing the panel does not change layer visibility.

**Embed in a webpage:**

Use `embed_html()` to add the viewer to a webpage:

```py
result = hkw.render(layer, filename="output.html", backend="webgl")
iframe = result.embed_html(src="visualizations/output.html", height=420)
```

Set `src` to the published viewer path. If the viewer and embedding page are in
the same directory, omit `src`. Adjust `title`, `width`, and `height` as needed.

For offline sharing, render with `offline=True` and publish `output.html` and
its sibling `output_assets/` directory together, preserving their relative paths.

**Background:**

The WebGL beauty view defaults to a soft dark studio gradient; `"light"` is
another gradient, and `"black"` matches an opaque black raster background:

```py
hkw.render(layer, filename="output.html", backend="webgl", background="dark")  # default
hkw.render(layer, filename="output.html", backend="webgl", background="light")
hkw.render(layer, filename="output.html", backend="webgl", background="black")
```

The `background` option sets the initial look. The interactive viewer's **☀ / ☾
button** switches between light and dark gradients, including when it starts
black. Transparent materials, such as `ThinDielectric` glass, show the active
background. Use `figure.output(background="black")` to request a black
background in WebGL and opaque raster renders.

## Render result

Regardless of backend, `hkw.render()` returns a `RenderResult`:

```py
result = hkw.render(layer, filename="output.png")

result.path       # main output path, or None if no filename was given
result.image      # in-memory image (Mitsuba only); None for Blender/WebGL
result.outputs    # manifest: {"main": ..., "<pass>": <path or "interactive">}
result.backend    # name of the backend that produced the result
```

`result.outputs` lists every artifact the render produced, including per-pass
sidecars (see [render passes](config.md#render-passes)). A `RenderResult` is
also path-like, so it can be passed straight to `open()` or `pathlib.Path`
when a main output file was written. For notebook display, use `result.image`
(Mitsuba).

## Backend Management

### List Available Backends

You can query which backends are currently available:

```py
import hakowan as hkw

backends = hkw.list_backends()
print(f"Available backends: {backends}")
# Output: Available backends: ['blender', 'mitsuba', 'webgl']
```

WebGL is always listed (it is part of the base install); Mitsuba and Blender appear only when their
extras (`hakowan[mitsuba]` / `hakowan[blender]`) are installed.

### Inspect Backend Capabilities

Capability declarations can be queried without importing heavyweight backend
modules:

```py
caps = hkw.backend_capabilities("webgl")
print(caps.marks)
print(caps.render_passes)
print(caps.features)
print(caps.limitations)

# JSON-safe form; includes declarations for unavailable optional backends too.
payload = caps.to_dict()
all_capabilities = hkw.list_backend_capabilities()
```

`hkw.validate(layer, backend=..., strict=True)` uses these declarations to
reject requested behavior that the selected backend would approximate or
ignore.

### Set Default Backend

When `backend=` is not given, Hakowan defaults to **WebGL** — its dependency ships with the base
install, so it is always available. The heavier Mitsuba and Blender backends are never auto-selected;
request them explicitly per render with `backend=`, or change the process-wide default with
`set_default_backend()`.

You can change the default backend for all subsequent render calls:

```py
import hakowan as hkw

# Make Mitsuba the default backend
hkw.set_default_backend("mitsuba")

# Now this will use Mitsuba
layer = hkw.layer("mesh.obj")
hkw.render(layer, filename="output.png")

# You can still override per render call
hkw.render(layer, filename="output.html", backend="webgl")
```

## Backend-Specific Options

Different backends may support different options passed as keyword arguments to `hkw.render()`.

### Mitsuba Backend Options

```py
# Save Mitsuba scene configuration to YAML
hkw.render(
    layer,
    filename="output.exr",
    backend="mitsuba",
    yaml_file="scene.yaml"
)
```

### Blender Backend Options

```py
# Use EEVEE instead of Cycles (faster)
hkw.render(
    layer,
    filename="output.png",
    backend="blender",
    engine="BLENDER_EEVEE"
)

# Save Blender scene file for further editing
hkw.render(
    layer,
    filename="output.png",
    backend="blender",
    blend_file="scene.blend"
)
```

### WebGL Backend Options

```py
# Standard interactive viewer output.
hkw.render(layer, filename="output.html", backend="webgl")

# Include assets alongside the HTML for offline use.
hkw.render(
    layer,
    filename="output.html",
    backend="webgl",
    offline=True,
)
```

Publish `output.html` and `output_assets/` together when sharing an offline
viewer. See [Snapshot and observation](observation.md) for PNG capture.

## Choosing a Backend

**Use Mitsuba when:**
- You want high-quality photorealistic rendering
- You need fast rendering with GPU acceleration
- You're creating publication-quality visualizations

**Use Blender when:**
- You want Cycles or EEVEE rendering with Blender materials
- You want to further edit the scene in Blender
- You need render passes (albedo, depth, normal, facet ID)

**Use WebGL when:**
- You want an interactive, shareable viewer in a browser
- You need fast turnaround with no render time
- You're embedding visualizations in web pages or notebooks

## Troubleshooting

If a backend is not available, `hkw.list_backends()` will simply omit it. You can check
which optional dependencies are missing:

```
pip install hakowan[mitsuba]  # enables the Mitsuba backend
pip install hakowan[blender]  # enables the Blender backend (Python 3.13)
```

The WebGL backend is part of the base install and is always available.
