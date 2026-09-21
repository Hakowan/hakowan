# Figure and scene configuration

A `Figure` combines a layer tree with scene-level camera, lighting, environment,
and semantic output settings. Use it when one serialized specification must
reproduce the intended image rather than only its geometry and encodings.

```py
figure = (
    hkw.figure(layer)
    .camera(
        "perspective",
        eye=(2, 3, 4),
        target=(0, 0, 0),
        up=(0, 1, 0),
        fov=35,
    )
    .light(
        "directional",
        direction=(0, 0, -1),
        color="white",
        intensity=2,
    )
    .environment("studio.exr", scale=1, visible=False)
    .output(
        width=1024,
        height=800,
        background="dark",
        passes=("beauty", "depth", "normal"),
        sampler_seed=0,
    )
)

hkw.render(figure, filename="result.html")
```

All fluent methods return a new immutable `Figure`.

## Camera

Supported camera kinds:

```py
figure.camera("perspective", eye=(2, 3, 4), fov=35)
figure.camera("orthographic", eye=(2, 3, 4))
figure.camera(
    "thin_lens",
    eye=(2, 3, 4),
    fov=35,
    aperture_radius=0.1,
    focus_distance=4,
)
```

### Automatic framing

High-level camera modes compile the layer tree, inspect its final normalized
geometry, and resolve immediately to a concrete serializable camera:

```py
# Fit the complete scene using a canonical direction.
figure = hkw.figure(layer).camera(
    "fit", direction="isometric", margin=0.08
)

# Fit one named layer or one component encoded by a scalar attribute.
figure = hkw.figure(layers).camera("fit", layer="surface")
figure = hkw.figure(layer).camera(
    "fit", component=("component", 3), direction="front"
)

# Fit an explicit box in normalized scene coordinates.
figure = hkw.figure(layer).camera(
    "fit", bounds=((-1, -1, -1), (1, 1, 1))
)
```

`direction` accepts `front`, `back`, `left`, `right`, `top`, `bottom`,
`isometric`, or an explicit three-vector. Set `up_axis="z"` for Z-up data.
Framing uses the Figure output dimensions when computing aspect ratio.

Principal-axis framing derives the camera direction from the selected points:

```py
figure = hkw.figure(layer).camera(
    "principal_axis", axis=0, sign="+", margin=0.1
)
```

Attribute-extremum framing targets the vertex or facet whose scalar value (or
vector magnitude) is smallest or largest:

```py
figure = hkw.figure(layer).camera(
    "attribute_extremum",
    attribute="stress",
    extremum="max",
    direction="isometric",
)
```

A section view aligns the camera with a plane normal and targets that plane.
It does not clip geometry; combine it with `hkw.transform.Clip` when an actual
cutaway is required:

```py
figure = hkw.figure(layer).camera(
    "section",
    normal=(0, 0, 1),
    offset=0.25,
    projection="orthographic",
)
```

Use `turntable()` to create immutable Figures with evenly spaced fitted cameras:

```py
frames = hkw.figure(layer).turntable(count=12, elevation=20)
for index, frame in enumerate(frames):
    hkw.render(frame, filename=f"turntable_{index:02d}.html")
```

Automatic framing produces ordinary `PerspectiveCamera`,
`OrthographicCamera`, or `ThinLensCamera` values. Canonical JSON therefore
contains resolved coordinates and remains independent of framing code at load
time.

Camera coordinates are evaluated after Hakowan normalizes the compiled scene.
All cameras support `eye`, `target`, `up`, `near`, and `far`. Perspective and
thin-lens cameras additionally support `fov` and `fov_axis`; orthographic
cameras use `scale` as the full vertical world-space extent.

The WebGL backend approximates thin-lens cameras as perspective and reports the
degradation through strict validation.

## Lights

Add any number of direct lights:

```py
figure = figure.light(
    "point",
    position=(3, 4, 5),
    color="warmwhite",
    intensity=20,
)

figure = figure.light(
    "directional",
    direction=(0, 0, -1),
    color="white",
    intensity=2,
)
```

`DirectionalLight.direction` is the world-space direction traveled by light
rays. Intensity is a backend-neutral relative scalar; physical renderer output
can still differ due to different light-unit conventions.

`figure.clear_lights()` removes direct point and directional lights. Environment
lighting is controlled independently.

## Environment

```py
figure = figure.environment(
    "studio.exr",
    scale=1.0,
    up=(0, 1, 0),
    rotation=180,
    visible=False,
)
```

Set `enabled=False` to remove environment lighting. `visible` controls whether
the environment is visible to the camera while still allowing it to illuminate
the scene. Relative paths in schema files resolve beside the schema document.

## Output intent

```py
figure = figure.output(
    width=1920,
    height=1080,
    background="light",
    passes=("beauty", "albedo", "depth", "normal"),
    sampler_seed=7,
)
```

Output settings describe semantic intent. Filename, backend, browser executable,
render device, and performance-specific sample counts remain invocation options.

## Precedence

The render precedence is:

```text
explicit Config/backend keyword > Figure scene setting > Config default
```

Passing `config=` deliberately overrides the complete declarative scene config:

```py
preview_config = hkw.config()
preview_config.sensor.location = (0, 0, 8)
hkw.render(figure, config=preview_config, filename="debug.html")
```

Backend keyword arguments override individual backend-facing settings. For
example, `background="dark"` overrides a figure's light WebGL background.

## Snapshots and observations

`snapshot()` uses the figure camera and output resolution when no explicit
camera, view, or resolution is supplied:

```py
snapshot = hkw.snapshot(figure)
```

Explicit observation arguments win:

```py
snapshot = hkw.snapshot(
    figure,
    view="right",
    resolution=(512, 512),
    background="dark",
)
```

`observe()` continues to use canonical multi-view cameras, but inherits figure
lighting, environment, resolution, background, and requested passes unless the
caller supplies replacements.

## Canonical schema

Serializing a `Figure` emits schema version 1.1 and a top-level `scene` block:

```py
spec = figure.to_spec(data_ids={id(mesh): "mesh"})
assert spec.version == "1.1"
assert spec.scene.camera.kind == "perspective"
```

Version 1.0 layer-only documents remain readable and reconstruct as `Layer`.
Documents containing scene settings reconstruct as `Figure`.
