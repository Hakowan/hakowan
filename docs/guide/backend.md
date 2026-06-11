# Rendering Backends

Hakowan supports multiple rendering backends, allowing you to choose the renderer that best suits
your needs. Three backends are available: **WebGL** (interactive browser viewer), **Mitsuba**
(photorealistic), and **Blender** (Cycles/EEVEE). WebGL ships with the base install and is used by
default; Mitsuba and Blender are optional extras (`pip install hakowan[mitsuba]` /
`pip install hakowan[blender]`).

## Available Backends

### Mitsuba Backend

The Mitsuba backend is based on the
[Mitsuba 3](https://www.mitsuba-renderer.org/) physically-based rendering system, which provides
high-quality, photorealistic rendering with advanced lighting and material models. It is an
optional extra — install it with `pip install hakowan[mitsuba]`.

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

### Blender Backend

The Blender backend uses [Blender](https://www.blender.org/)'s rendering engine through the `bpy`
Python API. This backend is useful if you want to leverage Blender's rendering capabilities or
integrate Hakowan into a Blender-based workflow.

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

The WebGL backend generates a self-contained, interactive HTML file using
[three.js](https://threejs.org/) and the glTF 2.0 format. No server is required — open
the HTML file in any browser to explore the scene interactively. It ships with the base install
and is the default backend.

**Advantages:**
- Default backend — ships with the base install, no extra to add
- Instant interactive 3D viewer in any browser
- No rendering time — output is generated immediately
- Self-contained single HTML file (or optional sidecar GLB)
- Supports render passes: albedo, depth, normal

**Requirements:**
- None — `pygltflib` is a core dependency, so the WebGL backend is always available

**Usage:**

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
hkw.render(layer, filename="output.html", backend="webgl")
```

**Background:**

The beauty view uses a soft "studio" radial gradient — a bright spot in the centre falling off
towards the edges. Two presets are available via the `background` option:

```py
hkw.render(layer, filename="output.html", backend="webgl", background="dark")   # default
hkw.render(layer, filename="output.html", backend="webgl", background="light")
```

The `background` option only sets the *initial* look. The interactive viewer includes a small
**☀ / ☾ button** (top-right) to toggle light/dark at any time. Transparent materials (e.g.
`ThinDielectric` glass) refract whichever background is active.

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

### Set Default Backend

When `backend=` is not given, Hakowan uses the backend set via `set_default_backend()`, falling
back to the first available backend in the order Mitsuba → Blender → WebGL. On a base install only
WebGL is available, so it is the default; installing the `mitsuba` or `blender` extra makes that
backend the auto-selected default unless you override it.

You can change the default backend for all subsequent render calls:

```py
import hakowan as hkw

# Set WebGL as the default backend
hkw.set_default_backend("webgl")

# Now this will use WebGL
layer = hkw.layer("mesh.obj")
hkw.render(layer, filename="output.html")

# You can still override per render call
hkw.render(layer, filename="output.exr", backend="mitsuba")
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
# Write sidecar GLB file instead of embedding in HTML
hkw.render(
    layer,
    filename="output.html",
    backend="webgl",
    embed=False       # writes output.glb alongside output.html
)
```

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
