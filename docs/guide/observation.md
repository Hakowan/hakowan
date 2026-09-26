# Snapshot and observation

Hakowan can produce deterministic raster evidence for automated inspection,
vision models, regression checks, and publication workflows. The default path
uses the existing WebGL scene through headless Chromium.

## Mental model

Hakowan separates visualization intent from captured evidence:

```text
Figure ──capture one view/pass──▶ Snapshot
   │
   └──capture several views/passes──▶ Observation
```

- A [`Figure`](figure.md) is the visualization definition: its layer tree plus
  camera, lighting, environment, and output intent. It is neither an image nor
  evidence that the result is visible.
- A `Snapshot` is one raster for one resolved camera and one render pass. It
  carries the image together with camera matrices, bounds, optional raw pass
  data, diagnostics, paths, and hashes.
- An `Observation` is a collection of snapshots plus scene-level evidence. It
  adds a contact sheet, scene and layer summaries, a manifest, and queries for
  picking, visibility, visible attribute ranges, and occlusion.

| Goal | API | Result |
|---|---|---|
| Save one user-facing PNG | `snapshot(..., pass_name="beauty")` | One `Snapshot` and PNG |
| Capture one analytical pass | `snapshot(..., pass_name="depth")` | One `Snapshot`, PNG, and raw NumPy array |
| Inspect several views or passes | `observe(...)` | One `Observation` containing many snapshots |
| Build visual evidence for an agent or regression check | `observe(...)` | Contact sheet, manifest, diagnostics, and queryable pass data |
| Create an interactive browser viewer | `render(..., backend="webgl")` | HTML, not an observation |

Use `snapshot()` when the required camera and pass are already known. Use
`observe()` when choosing a view, checking visibility, or comparing multiple
passes is part of the task. Both compile the same `Layer` or `Figure`; neither
changes the visualization definition.


## Installation

```sh
pip install "hakowan[observe]"
playwright install chromium
```

With Pixi:

```sh
pixi shell -e observe
playwright install chromium
```

Three.js modules are downloaded once into `~/.cache/hakowan/three/` (or
`HAKOWAN_CACHE_DIR`) and reused, so later captures do not require the CDN.

## One snapshot

```py
snapshot = hkw.snapshot(
    layer,
    view="isometric",
    pass_name="beauty",
    resolution=(768, 768),
    filename="figure.png",
)

snapshot.image             # PIL image
snapshot.data              # raw pass array, when applicable
snapshot.camera            # resolved CameraState
snapshot.world_to_camera   # 4x4 matrix
snapshot.projection        # 4x4 projection matrix
snapshot.bounds            # rendered scene bounds
snapshot.diagnostics       # structured validation diagnostics
```

Supported deterministic view presets:

- `front`, `back`
- `left`, `right`
- `top`, `bottom`
- `isometric`

Set `up_axis="z"` for Z-up datasets. Presets frame the compiled scene's bounding
sphere with a fixed margin. Supply `camera=CameraState(...)` to override a
preset explicitly.

## Render passes


Without ID passes, depth and normal arrays are decoded from the GPU pass image;
depth therefore has 8-bit display quantization. When `element_id` or `layer_id`
is requested for the same view, Hakowan reuses the pixel-center raycast data for
full `float32` depth and world normals.

| Pass | Image | Raw `Snapshot.data` |
|---|---|---|
| `beauty` | shaded RGBA | `None` |
| `albedo` | unlit material color | `uint8[H,W,3]` |
| `depth` | normalized display depth | camera-space `float32[H,W]`; background is NaN |
| `normal` | view normal display | world-space `float32[H,W,3]`; background is NaN |
| `element_id` | deterministic false color | `uint32[H,W]` |
| `layer_id` | deterministic false color | `uint32[H,W]` |

ID pass background pixels use `hkw.workflow.observation.BACKGROUND_ID` (`2**32 - 1`).
For surface marks, `element_id` identifies the rendered triangle. Polygonal
surfaces are triangulated before capture, and the retained observation scene
uses the same triangle IDs for picking, region queries, and attribute extrema.
For instanced point and curve geometry, `element_id` identifies the rendered
instance.

## Multi-view observation

```py
observation = hkw.observe(
    layer,
    views=["front", "right", "top", "isometric"],
    passes=["beauty", "depth", "normal", "element_id", "layer_id"],
    resolution=(512, 512),
    output_dir="observation",
)
```

The output directory contains:

```text
observation/
├── front_beauty.png
├── front_depth.png
├── front_depth.npy
├── front_normal.png
├── front_normal.npy
├── ...
├── contact_sheet.png
└── manifest.json
```

Every pass with raw data writes a NumPy `.npy` sidecar. The manifest records
both `path` and `data_path`; load arrays with `np.load(..., allow_pickle=False)`.

Individual results are keyed by `(view, pass)`:

```py
front_depth = observation.snapshot("front", "depth")
all_snapshots = observation.snapshots
contact_sheet = observation.contact_sheet
manifest = observation.manifest
scene = observation.scene_summary
```

The manifest records scene bounds, layer names and mark types, camera state,
world-to-camera and projection matrices, raw data dtypes, output paths, and
validation diagnostics.

## Pixel-to-data lookup

Request the required geometry passes, then probe a pixel:

```py
observation = hkw.observe(
    layer,
    views=["front"],
    passes=["depth", "normal", "element_id", "layer_id"],
)

hit = observation.pick("front", (320, 240))
if hit is not None:
    print(hit.layer_id, hit.element_id)
    print(hit.depth, hit.world_position, hit.normal)
    print(hit.attributes)
```

`pick()` returns `None` for background pixels. Facet attributes are returned for
surface hits. Vertex attributes are returned directly for point hits and
averaged across a triangle for surface hits. Indexed attributes are currently
omitted from the attribute dictionary.

## Structured visibility queries

Request `element_id` and `layer_id` together to enable region, visibility,
attribute, and occlusion queries. Add `depth` for per-layer depth ranges.

```py
observation = hkw.observe(
    layer,
    views=["front", "right"],
    passes=["depth", "element_id", "layer_id"],
)
```

Summarize a half-open pixel rectangle `(x0, y0, x1, y1)`:

```py
region = observation.region(100, 80, 300, 260, view="front")
print(region.background_fraction)
for layer in region.layers:
    print(
        layer.name,
        layer.visible_pixel_count,
        layer.visible_element_ids,
        layer.visible_element_fraction,
        layer.visible_bounds,    # bounds of actually visible pixels
        layer.projected_bounds,  # projected bounds of all layer geometry
        layer.depth_range,
    )
```

Query complete captured views, optionally selecting a layer by ID or name:

```py
visible = observation.visible_elements("surface", view="front")
```

Compute statistics over only the visible source elements carrying an attribute:

```py
stats = observation.attribute_extrema(
    "stress", layer="surface", view="front"
)

# Restrict the same statistics to a pixel rectangle.
regional = observation.attribute_extrema(
    "stress", layer="surface", view="front", bounds=(100, 80, 300, 260)
)
for item in stats:
    print(item.minimum, item.maximum, item.mean)
    print(item.minimum_element_id, item.maximum_element_id)
```

Scalar attributes use their value as the extremum criterion. Vector attributes
use magnitude while retaining component-wise minimum, maximum, mean, and the
original sample at each extremum. For surfaces, facet fields sample visible
facets and vertex fields sample the union of vertices belonging to visible
facets. Indexed attributes are rejected rather than ambiguously flattened.

Projected depth ordering can identify likely occlusion:

```py
for item in observation.occlusion_report(view="front"):
    print(
        item.occluder_layer_name,
        "covers",
        item.occluded_layer_name,
        item.projected_coverage,
        item.fully_hidden,
    )
```

Occlusion records are geometric evidence, not an opacity proof: they report a
closer layer whose visible pixel bounds overlap a farther layer's projected
geometry. Visibility and occlusion summaries are added automatically to the
observation manifest when both ID passes are captured.

## Explicit cameras

```py
camera = hkw.CameraState(
    eye=(2.0, 3.0, 4.0),
    target=(0.0, 0.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fov=35.0,
    near=0.01,
    far=100.0,
)

snapshot = hkw.snapshot(layer, camera=camera)
```

`view` remains the result label when an explicit camera is supplied. In an
`observe()` call, pass `cameras={"front": camera}` to override selected presets.

Custom camera keys also become artifact name prefixes. They may contain spaces
and Unicode text, but empty labels, `.`/`..`, path separators, and NUL bytes are
rejected so `output_dir` remains the artifact boundary. The `timeout=` value
applies to navigation and every Playwright capture operation.

## Offline WebGL viewers

Interactive HTML normally imports Three.js from a CDN. Produce a portable
folder bundle instead:

```py
hkw.render(
    layer,
    filename="viewer.html",
    backend="webgl",
    offline=True,
)
```

This writes `viewer.html` and `viewer_assets/`. The first offline build downloads
the pinned Three.js core, controls, geometry utilities, and environment-map
loader dependencies into Hakowan's cache and verifies their SHA-256 digests.
Subsequent bundles are copied from that cache and work without network access.
Keep the HTML and asset directory together. Because browsers block JavaScript
modules loaded from `file://`, serve the directory over HTTP:

```sh
python -m http.server --directory /path/to/output 8000
```

Then open `http://127.0.0.1:8000/viewer.html`.

## Determinism and limitations

Hakowan fixes camera framing, viewport size, device pixel ratio, pass behavior,
random seeds already present in the layer, layer ordering, and output naming.
WebGL output can still differ across browser/GPU implementations at antialiased
edges. Raw ID and raycast data are sampled at pixel centers and are preferable
to false-color PNGs for programmatic comparison.

The headless API currently supports `backend="webgl"`. Mitsuba and Blender keep
their existing render APIs but do not yet implement the `Snapshot` contract.
