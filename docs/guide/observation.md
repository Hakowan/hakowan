# Snapshot and observation

Hakowan can produce deterministic raster evidence for automated inspection,
vision models, regression checks, and publication workflows. The default path
uses the existing WebGL scene through headless Chromium.

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

ID pass background pixels use `hkw.observation.BACKGROUND_ID` (`2**32 - 1`).
For surface marks, `element_id` identifies the rendered triangle. For instanced
point and curve geometry it identifies the rendered instance. Polygonal source
facets may produce multiple rendered triangle IDs after triangulation.

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
four version-pinned Three.js modules into Hakowan's cache; the generated bundle
works without network access afterward. Keep the HTML and asset directory
together.

## Determinism and limitations

Hakowan fixes camera framing, viewport size, device pixel ratio, pass behavior,
random seeds already present in the layer, layer ordering, and output naming.
WebGL output can still differ across browser/GPU implementations at antialiased
edges. Raw ID and raycast data are sampled at pixel centers and are preferable
to false-color PNGs for programmatic comparison.

The headless API currently supports `backend="webgl"`. Mitsuba and Blender keep
their existing render APIs but do not yet implement the `Snapshot` contract.
