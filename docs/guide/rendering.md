# Rendering and viewers

## Choose an output or inspect a scene

Choose by the result you need, not by an assumed speed difference:

| Task | API | Result |
|---|---|---|
| Let a person explore or share a scene | `render(..., backend="webgl")` | Interactive HTML viewer |
| Produce a final image with advanced lighting and materials | `render(..., backend="mitsuba")` or `backend="blender"` | Raster image and backend artifacts |
| Capture one WebGL view for a preview or image check | `snapshot(...)` | One image with camera metadata and optional raw pass data |
| Inspect several views or query visible geometry and attributes | `observe(...)` | Images, contact sheet, manifest, diagnostics, and inspection queries |

**`render()` produces a deliverable; `observe()` captures evidence for inspection.**
Both can produce images. `snapshot()` is the single-view, single-pass capture API.
See [Capture and visual inspection](observation.md) for the capture workflow.

All three accept a `Layer` or `Figure` independently. You do not need to call
`render()` before `snapshot()` or `observe()`. An observation already contains its
snapshots; do not capture the same views separately to retrieve them.

For AI agents and programmatic checks, use `observe()` when you need pixel
picking, visibility, visible attribute statistics, or several views/passes. Use
`snapshot()` when one known view and pass are sufficient. Use `render()` when
the requested deliverable is a viewer, a Mitsuba/Blender image, or a scene file.

Speed and quality depend on the backend and settings, not the function name.
WebGL capture is suited to quick inspection, but starting Chromium and capturing
many views/passes have costs. Mitsuba and Blender support advanced rendering;
high-quality settings can take longer. `render(backend="webgl")` creates a viewer,
not a slow photorealistic raster. Capture currently uses WebGL through headless
Chromium; `observe()` is not a multi-view wrapper around Mitsuba or Blender.

## Render a scene

WebGL is the default backend:

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
result = hkw.render(layer, filename="viewer.html")
print(result.path)
```

### Work with the result

Every backend returns a `RenderResult`:

```py
result.path       # main output path, or None if no filename was given
result.image      # in-memory image (Mitsuba only); None for Blender/WebGL
result.outputs    # manifest: {"main": ..., "<pass>": <path or "interactive">}
result.backend    # backend that produced the result
```

`result.outputs` includes per-pass sidecars; see [render passes](config.md#render-passes).
When a main file was written, the result is path-like and can be passed to
`open()` or `pathlib.Path`. Use `result.image` for Mitsuba notebook display.
Full signatures are in the [render API reference](../api/render.md).

## Choose and configure a backend

WebGL ships with the base install. Mitsuba and Blender are optional; their
installation requirements and options are grouped below by backend.

```py
print(hkw.list_backends())  # installed, available backends
hkw.set_default_backend("mitsuba")
hkw.render(layer, filename="output.exr")
# An explicit backend overrides the process-wide default.
hkw.render(layer, filename="viewer.html", backend="webgl")
```

Without a changed default or explicit `backend=`, Hakowan uses WebGL. It does
not automatically select Mitsuba or Blender just because they are installed.

### Inspect capabilities

Capability declarations can be queried without importing heavyweight backend
modules, including declarations for unavailable optional backends:

```py
caps = hkw.backend_capabilities("webgl")
print(caps.marks, caps.render_passes, caps.features, caps.limitations)
payload = caps.to_dict()  # JSON-safe
all_capabilities = hkw.list_backend_capabilities()
hkw.validate(layer, backend="webgl", strict=True)
```

Strict validation rejects requested behavior that the selected backend would
approximate or ignore. Use it before selecting a backend for an automated task.

## WebGL

Use WebGL for an interactive, shareable browser viewer. It requires only the
base install and a modern browser, and supports live albedo, depth, and normal
passes. Headless raster capture has separate requirements documented in the
[capture guide](observation.md#installation).

```py
hkw.render(layer, filename="viewer.html", backend="webgl")
```

### Viewer controls and backgrounds

For multiple rendered layers, the viewer shows visibility checkboxes. Unnamed
layers receive generated labels. The **−** button minimizes the list;
**layers +** restores it without changing visibility.

The beauty view defaults to a soft dark studio gradient. `"light"` selects a
light gradient; `"black"` selects an opaque black background:

```py
hkw.render(layer, filename="viewer.html", background="light")
```

The **☀ / ☾** button switches light/dark gradients, even when the viewer starts
black. Transparent materials such as `ThinDielectric` show the active background.
Use `figure.output(background="black")` for black backgrounds in WebGL and
opaque raster renders.

### Embed a viewer

```py
result = hkw.render(layer, filename="viewer.html", backend="webgl")
iframe = result.embed_html(src="visualizations/viewer.html", height=420)
```

Set `src` to the published viewer path, or omit it when the embedding page and
viewer share a directory. Adjust `title`, `width`, and `height` as needed.

### Share an offline viewer

```py
hkw.render(layer, filename="viewer.html", backend="webgl", offline=True)
```

Publish `viewer.html` and its sibling `viewer_assets/` directory together,
preserving relative paths. Browsers block local ES-module imports from `file://`;
serve the folder over HTTP:

```sh
python -m http.server --directory /path/to/output 8000
```

Open `http://127.0.0.1:8000/viewer.html`. Offline viewer packaging is a `render()`
workflow, not an observation output.

### Curve material support

Mesh-edge `Curve` marks support vertex- and facet-valued `ScalarField` colors on
`Diffuse`, `Principled`, and `Plastic` materials. Vertex values color endpoints;
facet values color each edge from its first incident facet, which is the owning
line facet for a line mesh.

## Mitsuba

Use Mitsuba for photorealistic images with advanced materials and lighting,
including GPU-accelerated rendering where supported. Install the optional extra:

```sh
pip install "hakowan[mitsuba]"
```

```py
hkw.render(layer, filename="output.exr", backend="mitsuba")
# Also save the Mitsuba scene configuration.
hkw.render(layer, filename="output.exr", backend="mitsuba", yaml_file="scene.yaml")
```

### Curve material limitations

For edge-derived `Curve` marks, Mitsuba supports vertex- and facet-valued
`ScalarField` inputs on `Diffuse` and `Principled` materials. Each facet value
is expanded to the segment's two endpoints; on a shared mesh edge, the first
incident facet supplies the value. Vertex-valued inputs must resolve to the
same BSDF at both endpoints of each edge; varying values require a tube mesh.
Attribute-driven materials on vector-field curves remain unsupported.

## Blender

Use Blender for Cycles or EEVEE images, editable `.blend` scenes, or albedo,
depth, normal, and facet-ID passes. The extra requires Python 3.13:

```sh
pip install "hakowan[blender]"
```

```py
hkw.render(layer, filename="output.png", backend="blender", blend_file="scene.blend")
# Select EEVEE instead of Cycles.
hkw.render(layer, filename="output.png", backend="blender", engine="BLENDER_EEVEE")
```

## Troubleshooting

`hkw.list_backends()` omits unavailable optional backends. If Mitsuba or Blender
is missing, install its extra in the Python environment running Hakowan and
check the requirements above. WebGL is part of the base install and is always
available. For Chromium installation or capture limitations, use the
[capture guide](observation.md).
