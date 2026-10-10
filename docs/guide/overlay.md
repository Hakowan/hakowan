# Legends and annotations

Add legends and labels to layers to explain the data shown in a visualization.
They appear in WebGL viewers, snapshots, and supported raster image outputs.

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

The title defaults to the mesh attribute name, and units come from
`hkw.attribute(..., unit=...)`. Legend values reflect the attribute's configured
scale before the colormap is applied.

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

Raster outputs overlay the legend when its requested width fits beside the rendered view. If it
does not fit, the image canvas expands to make room for the full legend panel.

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

`position` uses normalized image coordinates with `(0, 0)` at the top-left and
`(1, 1)` at the bottom-right. Both values must be in `[0, 1]`; out-of-range
positions raise `ValueError`. Supported anchors are `left`, `center`, and `right`.

You can also construct the reusable object directly:

```py
label = hkw.Annotation("Simulation A", position=(0.02, 0.02))
layer = hkw.layer(mesh, annotations=[label])
```

Identical inherited screen annotations are deduplicated across the scene and drawn
once on the shared canvas; they are not repeated in each juxtaposed view.

## World-space callouts

Use `WorldAnnotation` to attach a label to a point in a layer's 3D geometry,
rather than at a fixed position on the image:

```py
layer = hkw.layer(mesh).annotate(
    hkw.WorldAnnotation(
        "Peak stress",
        position=(0.4, 0.2, 0.1),
        color="white",
        background="black",
        offset=(0, -10),  # label sits 10 px above the point
        arrow=False,      # omit the leader; the default is True
    )
)
```

Use the 3D coordinates of the point in the layer's geometry, after any
geometry-changing transforms. The callout follows the point when layers are
transformed or arranged into multiple views, while the label stays screen-facing.
When `arrow` is true (the default), a leader connects the label to the point;
`arrow=False` omits it.

`offset` is an optional screen-space pixel vector from the point to the nearest
edge of the label. `+x` is right and `+y` is down; for example,
`offset=(0, -10)` places the label 10 pixels above the point. With
`offset=None`, Hakowan places the label automatically. An explicit offset may
overlap geometry or place the label partly outside the viewport.

Automatic placement looks for available space around the point. If there is no
room for a readable label, Hakowan may hide it. In WebGL, placement adjusts as
you move the camera.

Callouts outside the view are not displayed. Mitsuba and WebGL hide callouts whose
points are obscured by geometry; Blender may show callouts even when their points
are hidden. In WebGL, a callout centered on a point glyph ignores that glyph's own
surface, but other glyphs—including those in the same layer—can still occlude it.
This applies to both instanced and baked point geometry. A free-position callout
that is not centered on a source point does not exclude any glyph.
Mitsuba also ignores self-occlusion for anchors inside an edge-derived curve tube,
but only when the ray hits the segment containing the anchor. Other segments in
the same layer can occlude it, including segments split into material groups.

## Backend behavior

| Output | Behavior |
|---|---|
| WebGL viewer | Legends and annotations appear with the interactive scene. |
| Beauty images and snapshots | Include legends and annotations; expand the canvas when a legend cannot fit beside the view. |
| Mitsuba `result.image` | Does not include raster overlays. |
| Blender `result.image` | Includes raster overlays. |
| EXR/HDR and diagnostic passes | Do not include legends or annotations. |

Raster legends and annotations are drawn over the beauty image when the panels fit.
If a legend panel exceeds the available width, the output canvas expands to make
room. Use the `background` option to flatten transparent renders onto a light,
dark, or black background.

## Observation manifests

`hkw.observe()` includes semantic overlay metadata even when inspecting passes
that do not draw the overlays:

```py
observation.manifest["legends"]
observation.manifest["annotations"]
```

Use the observation manifest to retrieve legend and annotation details
programmatically, without inferring them from rendered pixels.
