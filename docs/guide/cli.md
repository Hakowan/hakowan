# Command line

Hakowan provides a convenient `hkw` command to streamline common visualization tasks.
Use it to render an image, create an interactive viewer, or inspect a mesh:

```sh
hkw render mesh.obj
hkw view mesh.obj
hkw inspect mesh.obj
```

Run `hkw --help` for the available actions. Each action has its own help:

```sh
hkw render --help
hkw view --help
hkw inspect --help
```

## Render an image

`hkw render` writes a still image using Mitsuba, then Blender. One of those
optional backends must be installed; the command reports an error if neither is
available. Use `hkw view` for an interactive WebGL viewer. The default output is
the input filename with a `.png` extension; use `-o` to choose another path.

```sh
hkw render mesh.ply -o mesh.png --resolution 1600 1200
hkw render mesh.ply --material temperature
hkw render mesh.ply --wireframe --wire-thickness 1
```


`--facet-id` requests a per-facet color image and selects Blender. It reports an
error if Blender is unavailable.

## Create an interactive viewer

`hkw view` uses WebGL to create an interactive HTML viewer. By default, it
writes the viewer to a temporary `.html` file and opens it. Pass `-o` to keep a
viewer at a chosen path, or `--no-open` to save it as the input filename with a
`.html` extension. `--turn-table` instead saves an animated GIF; WebGL
turn-table capture requires `hakowan[observe]` and Playwright Chromium.

```sh
hkw view mesh.obj -o viewer.html
hkw view mesh.obj --z-up --wireframe
```

Use `--no-open` to save the viewer without opening it automatically.

## Compare meshes

Pass up to six input meshes to either `render` or `view`; Hakowan arranges them
in a grid. With multiple inputs, each mesh is normalized to a comparable size.

```sh
hkw view front.obj side.obj back.obj -o comparison.html
```

## Inspect mesh attributes

`hkw inspect` lists a mesh's attributes, their domains, channel counts, and
minimum/maximum values. It does not render the mesh.

```sh
hkw inspect model.ply
```

Use this before selecting a field with `--material` or `--normal`.

## Common options

Most visualization options are shared by `render` and `view`. `--facet-id` is
available only with `render`. Run the action's `--help` for all options and
defaults.

| Option | Purpose |
|---|---|
| `-o`, `--output PATH` | Choose the output file. |
| `--resolution WIDTH HEIGHT` | Set output dimensions in pixels. |
| `--z-up` | Use Z as the vertical axis. |
| `--material NAME` | Choose a material, or use an attribute name to color by that field. |
| `--color COLOR` | Set the material color. |
| `-w`, `--wireframe` | Show mesh edges. |
| `--wire-thickness PIXELS` | Set wireframe, seam, or field-line thickness. |
| `--point-cloud` | Display the mesh as a point cloud. |
| `--clip AXIS[:VALUE]` | Slice the mesh with a plane, keeping the selected side of `x`, `y`, `z`, `-x`, `-y`, or `-z`. `VALUE` is a fraction of the bounding-box size (default `0.5`); the cross-section is left open, not capped. |
| `--albedo`, `--depth`, `--shading-normal` | Request diagnostic render passes. |
| `--seams`, `--uv` | Show UV seams or the UV layout. |
| `--vector-field ATTR`, `--cross-field ATTR` | Visualize a vector or cross field. |

With Mitsuba or Blender, diagnostic passes are saved alongside the beauty
image. WebGL image capture uses the requested pass as its output and supports
one diagnostic pass at a time.
