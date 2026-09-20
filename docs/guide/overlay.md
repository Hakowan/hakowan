# Legends and annotations

Hakowan treats legends and annotations as semantic figure overlays. They are
part of the layer specification, survive canonical JSON round-trips, appear in
WebGL viewers and beauty snapshots, and are composited into PNG/JPEG/WebP/TIFF
outputs from Mitsuba and Blender.

## Automatic scalar-field legends

A visible color `ScalarField` produces a continuous colorbar by default:

```py
layer = hkw.layer(mesh).material(
    "Diffuse",
    hkw.texture.ScalarField(
        hkw.attribute("temperature", unit="°C"),
        colormap="inferno",
    ),
)
```

The title defaults to the mesh attribute name. Units come from
`hkw.attribute(..., unit=...)`. The compiler records the effective numeric
domain after user-supplied attribute scales and before color normalization.

Customize the colorbar with `hkw.Legend`:

```py
texture = hkw.texture.ScalarField(
    hkw.attribute("temperature", unit="°C"),
    colormap="inferno",
    legend=hkw.Legend(
        title="Temperature",
        ticks=6,
        format=".1f",
        position="right",
        width=180,
    ),
)
```

Suppress it explicitly:

```py
texture = hkw.texture.ScalarField("temperature", legend=False)
```

`colormap="identity"` does not create a legend because the source is interpreted
as color data rather than a scalar quantity.

## Categorical legends

Set `categories=True` to emit one swatch per unique value:

```py
texture = hkw.texture.ScalarField(
    "component",
    colormap="set1",
    categories=True,
    legend=hkw.Legend(
        title="Component",
        category_labels={
            "0.0": "Body",
            "1.0": "Handle",
            "2.0": "Fastener",
        },
    ),
)
```

Unmapped categories use the configured numeric format.

## Scale metadata

A legend records the attribute scale pipeline:

```py
field = hkw.attribute(
    "pressure",
    unit="Pa",
    scale=hkw.scale.Log(base=10) * hkw.scale.Clip(domain=(0, 6)),
)
```

The WebGL and raster legend displays `log → clip`. Numeric legend values are in
the transformed domain used by the colormap. This is explicit in observation
manifests under each legend's `scale` field.

## Screen-space annotations

Use `Layer.annotate()` for deterministic text overlays:

```py
layer = (
    hkw.layer(mesh)
    .annotate(
        "Peak stress region",
        position=(0.5, 0.06),
        anchor="center",
        color="white",
        background="black",
        font_size=18,
        padding=5,
    )
)
```

`position` is normalized image space with `(0, 0)` at the top-left and `(1, 1)`
at the bottom-right. Supported anchors are `left`, `center`, and `right`.

You can also construct the reusable object directly:

```py
label = hkw.Annotation("Simulation A", position=(0.02, 0.02))
layer = hkw.layer(mesh, annotations=[label])
```

Annotations inherited by multiple child views are deduplicated when the scene is
compiled.

## Backend behavior

| Backend/output | Behavior |
|---|---|
| WebGL viewer | HTML/CSS legend panels and annotations over the interactive canvas. |
| `snapshot()` / `observe()` beauty pass | Pillow-composited overlays; legend panels extend the image left or right. |
| Mitsuba LDR file | Pillow-composited after rendering. The in-memory Mitsuba tensor remains the uncomposited render. |
| Blender LDR file | Pillow-composited after rendering and format conversion. |
| EXR/HDR files | Overlay is skipped with a warning because raster text cannot be represented faithfully. |
| Diagnostic passes | No overlays; raw depth, normal, and ID dimensions stay aligned. |

For a right-side legend of width 180, a `512×512` beauty image becomes
`692×512`. Raw diagnostic pass arrays retain the requested `512×512` dimensions.

## Observation manifests

`hkw.observe()` includes semantic overlay metadata even when inspecting passes
that do not draw the overlays:

```py
observation.manifest["legends"]
observation.manifest["annotations"]
```

This lets an AI associate colors with fields, ranges, units, scale pipelines,
and category labels without inferring them from pixels.
