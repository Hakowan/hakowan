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

The WebGL backend generates an interactive HTML viewer using
[three.js](https://threejs.org/) and glTF 2.0. The scene data is embedded in the
HTML. By default, the viewer imports version-pinned Three.js modules from unpkg;
`offline=True` writes those modules into a sibling asset directory instead.
WebGL ships with the base install and is the default backend.

**Advantages:**
- Default backend — ships with the base install
- Immediate interactive browser output
- Embedded glTF scene data
- Optional offline module bundle
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

For scenes with multiple named layers, the viewer shows a layer list with
visibility checkboxes. Use the **−** button to minimize the list; use **layers +**
to restore it. Minimizing the panel does not change layer visibility.

**Embed in a webpage:**

Render the viewer to an HTML file, publish it on an HTTP(S) static host, and paste
an iframe into the page where it should appear:

```py
result = hkw.render(layer, filename="output.html", backend="webgl")
print(result.embed_html(src="https://example.org/visualizations/output.html", height=420))
```

```html
<iframe src="https://example.org/visualizations/output.html" title="Interactive 3D visualization" width="100%" height="420" loading="lazy" style="border:0;"></iframe>
```

If the embedding page and viewer are published in the same directory, call
`result.embed_html()` without `src`: it uses the generated viewer's filename,
even when the render output path on your computer is absolute. Set `src` to the
published path or URL for any other layout. You can also set `title` and `width`.
The viewer's styles, scripts, and controls stay inside the iframe; no Three.js
code is needed in the embedding page.

The default output embeds scene data but loads Three.js modules from unpkg. For
an offline/self-hosted deployment, render with `offline=True` and publish both
`output.html` **and** the sibling `output_assets/` directory at the same relative
paths. The viewer's host must allow framing (for example, through its
`frame-ancestors` policy), and the embedding page must allow the viewer's URL in
its iframe/CSP policy. Hosts that strip iframes or disallow scripts cannot
display the interactive viewer; link to the HTML page instead.

**Background:**

The WebGL beauty view defaults to a soft dark studio gradient; `"light"` is
another gradient, and `"black"` matches an opaque black raster background:

```py
hkw.render(layer, filename="output.html", backend="webgl", background="dark")  # default
hkw.render(layer, filename="output.html", backend="webgl", background="light")
hkw.render(layer, filename="output.html", backend="webgl", background="black")
```

The `background` option sets the *initial* look. The interactive viewer's
**☀ / ☾ button** switches to light/dark gradients, including when it starts
black. Transparent materials (e.g. `ThinDielectric` glass) refract the active
background. Use `figure.output(background="black")` to request a black background
in WebGL **and** opaque raster renders. Orthographic `scale` is now the full
vertical extent in WebGL and Mitsuba, independent of aspect ratio. WebGL Diffuse
shading no longer adds the glTF dielectric specular lobe, but Three.js PMREM
lighting and Mitsuba path tracing can still give slightly different colors.

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
# Default: compact HTML that imports Three.js from the pinned CDN URL.
hkw.render(layer, filename="output.html", backend="webgl")

# Portable offline folder: output.html + output_assets/.
hkw.render(
    layer,
    filename="output.html",
    backend="webgl",
    offline=True,
)
```

The first offline build downloads the pinned core, controls, geometry utilities,
and environment-map loader dependencies into `~/.cache/hakowan/three/` and
verifies their SHA-256 digests. Subsequent bundles are copied from that cache.
See [Snapshot and observation](observation.md) for deterministic PNG capture.

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
