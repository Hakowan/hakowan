# Capture and visual inspection

Use `observe()` to capture one or more WebGL views with structured evidence
for AI agents, programmatic checks, or visual inspection. For interactive viewers
and Mitsuba/Blender presentation images, use the [rendering guide](rendering.md).

## Mental model

Hakowan separates visualization intent from captured evidence:

A [`Figure`](figure.md) defines the visualization, including its layer tree,
camera, lighting, and output intent. A `Snapshot` captures one view and pass,
with an image, camera matrices, bounds, optional raw data, diagnostics, paths,
and hashes. An `Observation` collects snapshots and adds a contact sheet,
manifest, scene summaries, and picking and visibility queries.

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

## One capture

```py
import hakowan as hkw
import numpy as np

layer = hkw.layer(
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
).mark("point")
observation = hkw.observe(
    layer,
    views=["isometric"],
    passes=["beauty"],
    resolution=(768, 768),
)
capture = observation.snapshot("isometric", "beauty")
capture.image.save("figure.png")

capture.image             # PIL image
capture.data              # raw pass array, when applicable
capture.camera            # resolved CameraState
capture.world_to_camera   # 4x4 matrix
capture.projection        # 4x4 projection matrix
capture.bounds            # rendered scene bounds
capture.diagnostics       # structured validation diagnostics
```

Beauty snapshots default to a transparent background, preserving alpha around the
rendered scene. Pass `background="light"`, `"dark"`, `"black"`, or `"white"` to
choose an opaque background; `"auto"` and `"transparent"` capture with transparency.

Supported deterministic view presets:

- `front`, `back`
- `left`, `right`
- `top`, `bottom`
- `isometric`

Set `up_axis="z"` for Z-up datasets. Presets frame the compiled scene's bounding
sphere with a fixed margin. Supply `cameras={"front": camera}` with a
`CameraState` to override a preset explicitly.

## Render passes

Request `element_id` or `layer_id` when you need full-precision depth and
normal data. Without those passes, depth and normals have display-image
precision.

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

observation = hkw.observe(
    layer,
    views=["custom"],
    passes=["beauty"],
    cameras={"custom": camera},
)
snapshot = observation.snapshot("custom", "beauty")
```

The camera key is used as the view label and in output filenames. Use
`views=["front"], cameras={"front": camera}` to override a named preset.
`observe()` does not inherit a camera from `Config`; pass that camera explicitly
in the `cameras` mapping when you need to preserve it.

## Determinism and limitations

Hakowan uses consistent camera framing and output settings for repeatable
captures. Exact pixels may vary across browser and GPU implementations,
especially at antialiased edges. Use raw ID, depth, or normal data for
programmatic comparisons rather than false-color images.

The headless API currently supports `backend="webgl"`. Mitsuba and Blender keep
their existing render APIs but do not yet implement the `Snapshot` contract.
