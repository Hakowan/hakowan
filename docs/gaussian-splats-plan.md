# Gaussian Splats Support Plan

## Decision

Implement Gaussian splats as a first-class `Splat` mark, initially on the WebGL backend using [Spark 2.2](https://github.com/sparkjsdev/spark). The first supported source format is GraphDECO/INRIA PLY.

Do not extend the existing `Point + Covariance` renderer. It emits instanced or baked triangle ellipsoids and lacks projected Gaussian kernels, opacity composition, view-dependent spherical harmonics, and camera-dependent sorting.

## Renderer choice

| Option | Decision | Reason |
|---|---|---|
| Existing point renderer | Reject | Ellipsoid geometry is not Gaussian splatting. |
| Custom WebGL shaders and sorter | Reject | Duplicates specialized projection, sorting, SH, and packing code. |
| Three.js r186 native splats | Follow-up | Standards-aligned, but requires `WebGPURenderer`; classic `WebGLRenderer` is unsupported. |
| Spark 2.2 | Use for V1 | Retains `WebGLRenderer`, supports mixed meshes and splats, global sorting, common formats, raycasting, and WebGL2. |

Three r186 native splats work through `WebGPURenderer`'s WebGL fallback, but a runtime probe confirmed that `material.onBeforeCompile` is not invoked there. Switching now would break Hakowan's isocontour, data-driven roughness/metallic, back-face color, and custom render-pass shader paths.

## Proposed API

```python
# Standard GraphDECO PLY: attributes detected and normalized automatically.
layer = hkw.layer("capture.ply").mark(hkw.mark.Splat)
hkw.render(layer, filename="capture.html", backend="webgl")
```

Programmatic data:

```python
layer = (
    hkw.layer(positions)
    .mark("splat")
    .channel(
        gaussian=hkw.channel.Gaussian(
            scale="scale",                    # N x 3, positive linear scale
            rotation="rotation",              # N x 4, normalized xyzw quaternion
            opacity="opacity",                # N, linear [0, 1]
            spherical_harmonics="sh",         # N x 3/12/27/48
            degree=3,
            color_space="srgb_rec709_display",
        )
    )
)
```

A dedicated `Gaussian` channel matches GraphDECO and Khronos scale/rotation/opacity/SH semantics directly and avoids decomposing covariance matrices during ordinary rendering.

## Delivery slices

### Grammar, ingestion, and canonical schema

Affected areas:

- `src/hakowan/grammar/mark/mark.py`
- `src/hakowan/grammar/channel/channel.py`
- `src/hakowan/grammar/layer/layer.py`
- `src/hakowan/grammar/dataframe/adapters.py`
- `src/hakowan/compiler/view.py`
- `src/hakowan/compiler/channel.py`
- `src/hakowan/spec/model.py`
- `src/hakowan/spec/codec.py`
- `src/hakowan/spec/schema_docs.py`
- `src/hakowan/workflow/validation.py`
- `src/hakowan/backends/__init__.py`

Changes:

- Add `Mark.Splat` and the canonical `"splat"` literal.
- Add `Gaussian` channel and `.channel(gaussian=...)`.
- Bump the canonical schema minor version.
- Detect GraphDECO PLY properties:
  - `scale_0..2`
  - `rot_0..3`
  - `opacity`
  - `f_dc_0..2`
  - exactly 0, 9, 24, or 45 `f_rest_*` values
- Normalize during ingestion:
  - `scale = exp(raw_scale)`
  - quaternion `wxyz -> xyzw`, then normalize
  - `opacity = sigmoid(raw_opacity)`
  - group SH coefficients into degree 0-3 arrays
- Validate counts, vertex-domain placement, finite values, positive scales, nonzero rotations, opacity range, and complete SH bands.
- Skip ordinary PBR material defaulting for splat views.

### Bounds and transform semantics

Current bounds use point centers only. Required behavior:

- Compute conservative 3-sigma splat bounds from scale and rotation.
- Use splat-aware bounds for global normalization, juxtaposition, camera fitting, clipping diagnostics, and observation summaries.
- Treat `Filter` on splats as vertex filtering, like `Point`.
- Make `Normalize` scale Gaussian axes along with centers.
- Preserve rigid and uniform affine transforms as object transforms.
- For shear or nonuniform affine transforms, transform covariance and decompose only when exporting; never silently distort splats.

Relevant files include:

- `src/hakowan/compiler/view.py`
- `src/hakowan/compiler/scene.py`
- `src/hakowan/compiler/transform.py`
- `src/hakowan/workflow/framing.py`
- `src/hakowan/workflow/observation.py`
- `src/hakowan/workflow/validation.py`

### WebGL rendering and asset delivery

Affected areas:

- `src/hakowan/backends/webgl/render.py`
- new focused `src/hakowan/backends/webgl/splat.py`
- `src/hakowan/backends/webgl/template.py`
- `src/hakowan/backends/webgl/templates/viewer.html`
- `src/hakowan/backends/webgl/assets.py`
- `src/hakowan/render/__init__.py`
- `src/hakowan/workflow/observation.py` (temporary snapshot viewer and camera changes)

Changes:

- Upgrade Three.js from `0.170.0` to at least `0.180.0`, Spark's peer floor.
- Upgrade the temporary viewer used by `workflow/observation.py` too; its capture path currently builds HTML with a separate Three.js version.
- Pin and integrity-check Spark 2.2.0:
  - module size: 2,881,309 bytes
  - SHA-256: `d5c3b3722e4e121836b7d26f1260c2a2750973130adfce0aed97bc312b54ead7`
- Dynamically import Spark only when the scene contains splats.
- Create one `SparkRenderer` and one `SplatMesh` per splat layer.
- Emit compiled splats as binary little-endian PLY; Spark rejects ASCII PLY.
  Re-encode canonical linear scales to GraphDECO log scales, linear opacity to
  logit opacity, xyzw rotations to wxyz, and restore GraphDECO's channel-major
  `f_rest_*` property order. Round-trip emitted PLY through Spark's decoder.
- Extend viewer cell grouping and bounds to include `SplatMesh`. It renders as
  an `Object3D`, not a Three.js `Mesh`, so node tags alone do not make current
  geometry discovery see it. Compute its pivot from the splat-aware 3-sigma
  bounds defined above.
- Wait for the initial Spark sort before `__hakowan_loaded`, and await a fresh
  sort after each camera/object transform before deterministic snapshot readback.
- Support mixed opaque meshes and splats in one scene.

Large captures require an explicit asset policy:

```python
hkw.render(layer, filename="capture.html", assets="sidecar")  # practical default
hkw.render(layer, filename="capture.html", assets="embed")    # portable, memory-heavy
```

Every generated splat sidecar must appear in `RenderResult.outputs`.

### Observation contract

V1 supports:

- Interactive beauty rendering.
- Deterministic beauty snapshots.
- Camera state and splat-aware scene bounds.
- Layer visibility and object rotation.

Initially unsupported:

- Normal pass: Gaussian fields have no canonical surface normal.
- Existing material-swap depth and albedo passes.
- Element and layer IDs until alpha-composition semantics are defined.
- Interactive clipping of splats.

Unsupported requests must fail validation rather than return misleading images. Backend capabilities need mark-specific pass declarations.
The interactive viewer must also hide or disable unsupported pass and clipping
controls for splat scenes; backend validation cannot reject later UI clicks.

## Deferred work

- SPZ, SPLAT, KSPLAT, and SOG ingestion. Spark can render them, but Hakowan's Python compiler cannot yet inspect, filter, transform, or bound them.
- Khronos `KHR_gaussian_splatting` glTF import/export. Keep the internal representation compatible so later export is mechanical.
- Mitsuba support via `ellipsoids` and `volprim_rf_basic`; this requires separate integrator and mixed-scene semantics.
- Blender support after the pinned Blender runtime exposes a stable native splat API.
- Migration to Three's native renderer after Hakowan's `onBeforeCompile` and custom GLSL paths have moved to TSL.

## Acceptance checks

- Real binary GraphDECO PLY at SH degrees 0 and 3 loads and schema-round-trips.
- Filter, normalize, affine transform, layout, and camera fitting preserve splat extent.
- One splat layer and mixed mesh/splat scenes render in Chromium.
- Sorting updates after camera orbit and object rotation.
- Layer toggles hide the complete splat object.
- Online, offline, embedded, and sidecar outputs load.
- Existing WebGL material patches and observation behavior remain unchanged for non-splat scenes.
- No per-splat triangle expansion; payload and GPU work remain linear in splat count.

## Scoping evidence

- Lagrange loads GraphDECO PLY properties, but exposes them as separate scalar attributes; grouping and activation transforms are required.
- Three r186 native splats rendered successfully through `WebGPURenderer`'s WebGL fallback.
- That renderer did not invoke Hakowan's required `onBeforeCompile` path.
- Spark 2.2 rendered anisotropic Gaussian splats successfully through classic `WebGLRenderer` from binary PLY.

## References

- [KHR_gaussian_splatting](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting)
- [Three.js r186 GaussianSplat](https://github.com/mrdoob/three.js/blob/r186/examples/jsm/objects/GaussianSplat.js)
- [Three.js r186 release](https://github.com/mrdoob/three.js/releases/tag/r186)
- [Spark](https://github.com/sparkjsdev/spark)
- [Mitsuba ellipsoids](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_shapes.html#ellipsoids-ellipsoids)
