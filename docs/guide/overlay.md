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
    categories=True,
    legend=hkw.Legend(
        title="Component",
        category_labels={
            "0": "Body",
            "1": "Handle",
            "2": "Fastener",
        },
    ),
)
```

Integer-valued categories accept compact keys such as `"0"`; exact float keys
such as `"0.0"` remain supported and take precedence when both are present.
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

## World-space callouts

Use `WorldAnnotation` to attach a label to a point in the **local coordinates of
the layer's geometry**, rather than a fixed pixel location:

```py
layer = hkw.layer(mesh).annotate(
    hkw.WorldAnnotation(
        "Peak stress",
        position=(0.4, 0.2, 0.1),
        color="white",
        background="black",
    )
)
```

The compiler applies the layer transform and scene layout to the anchor, so
the callout follows its view when the scene is juxtaposed. In the WebGL viewer,
the text remains readable in screen space, offset from the point and connected
by an arrow. Its placement follows camera orbit, zoom, projection changes, and
per-object rotation smoothly. Callouts behind the camera, outside the viewport,
or hidden by geometry are not displayed; crowded scenes may hide labels that
cannot be placed without overlap. Existing `Annotation` remains a fixed
screen-space overlay.

WebGL and raster callouts use the same placement policy: try the same ordered
directions, expand the distance until the text clears the rendered geometry,
keep a six-pixel viewport inset and a twelve-pixel exclusion around the anchor,
and avoid fixed overlays and previously placed labels. If no legal position
exists, hide the callout rather than cover the mesh. WebGL additionally avoids
its controls and preserves the previous placement during interactive movement.
For identical cameras and viewport sizes, initial placements are comparable;
browser and Pillow font measurements can produce small pixel differences.

Beauty snapshots and LDR images from Mitsuba and Blender project the point
through their fixed render camera. Mitsuba casts a camera ray per annotation and
hides anchors behind other geometry; WebGL beauty snapshots currently only
project anchors, and Blender raster overlays do not perform this occlusion test.
Diagnostic passes never include labels. As with other raster overlays, EXR/HDR
output skips compositing.

## Backend behavior

| Backend/output | Behavior |
|---|---|
| WebGL viewer | HTML/CSS legend panels and annotations over the interactive canvas. |
| `snapshot()` / `observe()` beauty pass | Pillow-composited overlays within the original image bounds. |
| Mitsuba LDR file | Pillow-composited within the rendered image after rendering. The in-memory Mitsuba tensor remains uncomposited. |
| Blender LDR file | Pillow-composited within the rendered image after format conversion. |
| EXR/HDR files | Overlay is skipped with a warning because raster text cannot be represented faithfully. |
| Diagnostic passes | No overlays; raw depth, normal, and ID dimensions stay aligned. |

Raster legends do not change the requested output dimensions. A left or right
legend is drawn over that edge of the beauty image with an eight-pixel inset.
The panel uses the same rounded corners and semi-transparent dark background as
the WebGL legend. Pixels outside that rounded rectangle retain the original
render, including its alpha values. An explicit light or dark raster background
override still flattens transparent render pixels before overlays are drawn.

## Observation manifests

`hkw.observe()` includes semantic overlay metadata even when inspecting passes
that do not draw the overlays:

```py
observation.manifest["legends"]
observation.manifest["annotations"]
```

This lets an AI associate colors with fields, ranges, units, scale pipelines,
and category labels without inferring them from pixels.
