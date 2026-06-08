# Rendering Backends

Hakowan supports multiple rendering backends, allowing you to choose the renderer that best suits
your needs. Three backends are available: **Mitsuba** (default), **Blender**, and **WebGL**.

## Available Backends

### Mitsuba Backend

The Mitsuba backend is the default renderer in Hakowan. It is based on the
[Mitsuba 3](https://www.mitsuba-renderer.org/) physically-based rendering system, which provides
high-quality, photorealistic rendering with advanced lighting and material models.

**Advantages:**
- High-quality photorealistic rendering
- Fast rendering with GPU acceleration
- Support for advanced materials and lighting
- Default backend, no additional configuration needed

**Usage:**

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
# Mitsuba is the default backend
hkw.render(layer, filename="output.exr")

# Or explicitly specify Mitsuba
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
- `bpy` must be installed (`pip install bpy`)

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
the HTML file in any browser to explore the scene interactively.

**Advantages:**
- Instant interactive 3D viewer in any browser
- No rendering time — output is generated immediately
- Self-contained single HTML file (or optional sidecar GLB)
- Supports render passes: albedo, depth, normal

**Requirements:**
- `pygltflib` must be installed (`pip install pygltflib`)

**Usage:**

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
hkw.render(layer, filename="output.html", backend="webgl")
```

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

Note that the Blender backend requires `bpy` and the WebGL backend requires `pygltflib`.

### Set Default Backend

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

**Use Mitsuba (default) when:**
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
pip install bpy        # enables the Blender backend
pip install pygltflib  # enables the WebGL backend
```
