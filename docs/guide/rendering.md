# Rendering and capture

Use `hkw.render()` to produce a backend output, `hkw.snapshot()` to capture one
browser-rendered image, and `hkw.observe()` to collect images with structured
inspection data. They accept the same `Layer` or `Figure`; you do not need to
call one before another.

| Goal | API | Output |
|---|---|---|
| Open or share an interactive scene | `hkw.render(..., backend="webgl")` | HTML viewer |
| Produce a backend-rendered image or scene | `hkw.render(..., backend="mitsuba")` or `backend="blender"` | Raster image and/or backend artifacts |
| Capture one known view and pass | `hkw.snapshot(...)` | One `Snapshot`; optionally a PNG and raw-data sidecar |
| Inspect a scene across views or passes | `hkw.observe(...)` | One `Observation` with snapshots, contact sheet, manifest, and query methods |

## Render a scene

In Python, WebGL is the default backend and writes an interactive HTML viewer:

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
result = hkw.render(layer, filename="viewer.html")
print(result.path)
```

Choose Mitsuba or Blender for their raster-rendering and scene-file workflows;
install the corresponding optional extra first. `render()` returns a
`RenderResult` with the selected backend, output paths, and— for Mitsuba—an
in-memory image. Backend-specific options and capabilities are covered in the
[rendering backends guide](backend.md); the full signature is in the
[render API reference](../api/render.md).

## Capture one snapshot

Use a snapshot when the camera and pass are known and you need one deterministic
raster plus its capture metadata:

```py
snapshot = hkw.snapshot(
    layer,
    view="isometric",
    pass_name="beauty",
    filename="preview.png",
)
```

A `Snapshot` includes the image, resolved camera and projection state, bounds,
and diagnostics. Analytical passes such as `depth` or `normal` can also include
raw NumPy data. Snapshot capture currently uses WebGL through headless Chromium;
install `hakowan[observe]` and run `playwright install chromium` first.

## Collect an observation

Use an observation when you need several views or passes, or want to query what
is visible in the captured scene:

```py
observation = hkw.observe(
    layer,
    views=["front", "right", "top", "isometric"],
    passes=["beauty", "depth", "normal", "element_id", "layer_id"],
    output_dir="inspection",
)

front = observation.snapshot("front", "beauty")
hit = observation.pick("front", (320, 240))
```

An `Observation` bundles the requested snapshots with a contact sheet, manifest,
scene summaries, and inspection methods. Request both `element_id` and
`layer_id` for picking and structured visibility queries. See the
[Snapshot and observation guide](observation.md) for pass details, camera
presets, pixel picking, visibility queries, and output files.

## Do I need all three?

No. Pick the API for the result you need:

- `render()` for an interactive viewer or a render from Mitsuba/Blender.
- `snapshot()` for one static browser capture.
- `observe()` for multi-view, multi-pass evidence and programmatic inspection.

`observe()` already contains multiple `Snapshot` objects, so do not call
`snapshot()` separately just to obtain those same captures. `render()` is a
separate backend-output workflow; use it alongside capture only when you also
need its viewer or backend-specific render artifact.
