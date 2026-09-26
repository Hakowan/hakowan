# Transform

Transform is applied to the entire data frame as a pre-processing step. It is useful to modify the
geometry and/or computing new attributes from existing attributes.

## Filter transform

Filter transform extracts a subset of the 3D elements based on a user provided condition.

```py
tr = hkw.transform.Filter(data="attr_name", condition=lambda value: value > 0)
```

Note that if `data` parameter is `None`, the mesh vertex position attribute will be used by default.
See the [Smoothed Particle Hydrodynamics example](../examples/sph.md) for an actual usage of the filter
transform.

## Clip transform

Clip transform cuts the mesh with a plane, keeping only the half that the plane normal points into.
Unlike the [Filter transform](#filter-transform), which keeps or drops whole facets, the clip
transform slices through triangles: facets straddling the plane are cut so that only the part on the
positive side is kept (partial triangles are produced). The exposed cross-section is left open (it is
not capped).

```py
# Keep the half-space where dot(normal, x - point) >= 0.
tr = hkw.transform.Clip(point=[0, 0, 0], normal=[1, 0, 0])
```

The plane is defined in the data/object coordinate space, consistent with the
[Filter transform](#filter-transform).

The [WebGL backend](backend.md) additionally offers an *interactive* clipping plane in the viewer
(the `clip` button), independent of this transform, so you can slide a plane through the scene live
without re-rendering.

## UVMesh transform

UVMesh transform extract the corresponding UV mesh from a given 3D mesh.

```py
tr = hkw.transform.UVMesh(uv="attr_name")
```

## Affine transform

Affine transform applies an affine transformation to the given 3D mesh.

```py
tr = hkw.transform.Affine(matrix=np.eye(4))
```

Because affine transform is a very common operation, Hakowan provide the following shortcuts:

```py
# Assume `v` is a offset vector.
l = hkw.layer().translate(v)

# Assume `s` is a scaling factor.
l = hkw.layer().scale(s)

# Assume `axis` is a vector representing rotational axis,
# and `theta` is the rotation angle in radian.
l = hkw.layer().rotate(v, theta)
```

## Compute transform

Compute transform is designed to compute a set of commonly used attributes from the data frame.
Currently, the following attributes can be computed:

* X|Y|Z coordinates
* Normal
* Component

```py
# Add per-facet component id attribute named "comp"
tr = hkw.transform.Compute(component="comp")
```

See the [Penny example](../examples/penny.md) for an actual usage of the compute
transform.

## Explode transform

Explode transform breaks a mesh into pieces based on the specified `pieces` attribute.
A piece is defined as a set of facets that has the same value of the `pieces` attribute.
Each piece will be moved away from the input bounding box center by an amount scaled by
the `magnitude` parameter.

```py
tr = hkw.transform.Explode(pieces="comp_id", magnitude=2)
```

Please see the [Powell-Sabin example](../examples/powell-sabin.md) for an actual use case of the
explode transform.

## Norm transform

Norm transform computes the row-wise norm of a given vector attribute.

```py
# Compute speed from a velocity vector field.
tr = hkw.transform.Norm(data="velocity", norm_attr_name="speed")
```

See the [Smoothed Particle Hydrodynamics example](../examples/sph.md) for an example usage of the norm
transform.

## PrincipalAxes transform

PrincipalAxes transform aligns the principal directions of the mesh vertex positions (computed via
PCA) with a target orthonormal frame. Principal axes are ordered by descending eigenvalue, so the
largest-variance direction maps to column 0 of `frame`, second-largest to column 1, and so on.

```py
# Align the mesh so its longest axis points along world-x, second-longest along world-y.
tr = hkw.transform.PrincipalAxes(frame=np.eye(3))
```

By default, `frame` is QR-orthonormalized so mildly skewed inputs still produce a proper rotation.
Set `orthonormalize_frame=False` if you guarantee an orthonormal input frame.

```py
# Custom target frame (e.g. flipped axes for camera-friendly orientation).
frame = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]], dtype=float).T
tr = hkw.transform.PrincipalAxes(frame=frame)
```

## Normalize transform

Normalize transform recenters and uniformly scales the mesh so it fits a unit box centered at the
origin (the bounding-box center is moved to the origin and the bounding-box diagonal is scaled to 2,
i.e. the geometry fits inside the unit sphere). It is handy for bringing meshes from unrelated
coordinate systems to a comparable on-screen size — for example when laying several meshes side by
side with [`Layer.juxtapose()`](layer.md).

```py
tr = hkw.transform.Normalize()
```

By default, normal and tangent/bitangent attributes are re-normalized to unit length; set
`normalize_normals=False` or `normalize_tangents_bitangents=False` to leave them untouched.

Unlike a layer-level [Affine transform](#affine-transform) (and the `scale` / `translate` shortcuts),
Normalize mutates the data-frame vertices in place. It therefore normalizes the geometry as it
currently stands — after any earlier mesh-mutating transforms — and ignores prior global affine
transforms, matching the behavior of the [PrincipalAxes transform](#principalaxes-transform).

## Streamline transform

`Streamline` replaces a triangulated surface with curves traced through a
tangent vector field. It supports ordinary vector fields and four-fold
rotationally symmetric (4-RoSy) cross fields. The resulting mesh contains
polyline vertices and two-vertex facets suitable for the `Curve` mark.

```py
streamlines = (
    hkw.layer(mesh)
    .transform(
        hkw.transform.Streamline(
            vec_field="velocity",
            cross_field=False,
            n=100,
        )
    )
    .mark("Curve")
    .channel(size=hkw.channel.Size(data=0.5, space="screen"))
)
```

### Input field domains

The input must be a three-channel vector attribute on a triangular surface.
Hakowan resolves every supported domain to one tangent direction per facet:

| Attribute domain | Conversion before tracing |
|---|---|
| `facet` | Used directly, then projected into the facet tangent plane. |
| `vertex` | Levi-Civita transported into each facet frame and symmetry-aware averaged; ordinary fields use 1-RoSy averaging and cross fields use 4-RoSy averaging. |
| `corner` | The three corner vectors are arithmetically averaged per triangle. |
| `indexed` | Indexed corner values are expanded and arithmetically averaged per triangle. |

Corner and indexed cross fields do not currently receive the symmetry-aware
transport used for vertex fields. Prefer facet or vertex storage when field
representatives can differ by sign or quarter-turn.

### Tracing behavior

Seeds are selected by area-aware blue-noise sampling. Each seed is traced in
both directions through exact triangle-edge crossings. Directions are parallel
transported between adjacent facet tangent frames. For a cross field, the
transported direction snaps to the closest of four equivalent arms, and Hakowan
traces both orthogonal bidirectional axes. A request for `n` seeds can therefore
produce up to `n` ordinary streamlines or `2 * n` cross-field streamlines.

Zero directions, missing forward crossings, mesh boundaries, the length limit,
and the step limit can terminate a trace. The surface must already be
triangulated.

### Length and limits

| Parameter | Meaning |
|---|---|
| `n=50` | Number of sampled seed facets. |
| `cross_field=True` | Interpret the input as a 4-RoSy field; use `False` for an ordinary vector field. |
| `length=None` | Maximum object-space length **per half-trace**. `None` traces until a boundary or another termination condition. A complete forward-plus-backward streamline can approach `2 * length`. |
| `seed=0` | Deterministic seed for sampling fallback. |
| `min_length=3` | Minimum retained sample-point count. |
| `max_steps=None` | Edge-crossing cap per half-trace; `None` uses half the facet count. |
| `id_attr_name="_hakowan_streamline_id"` | Per-vertex integer attribute identifying each output streamline. |

`length` is measured on the data-frame mesh before layer-level affine
transforms. It is not a fraction of the bounding box. To use a relative limit,
compute the object-space value explicitly from the input bounds.

The output streamline ID can drive categorical color:

```py
streamlines = streamlines.material(
    "Diffuse",
    hkw.texture.ScalarField(
        "_hakowan_streamline_id",
        categories=True,
    ),
)
```

## Fur transform

Fur transform replaces the mesh with fur/hair strands that flow along a per-facet vector field.
Each strand is a short, tapered curve that grows from the surface, leans in the direction of the
field, and curls back toward the surface flow — so a dense collection reads as realistic fur
combed along the field. The output is a vertex-only mesh whose 2-vertex polylines encode strand
segments, suitable for the `Curve` mark.

```py
# Grow 12000 fur strands along a per-facet vector field attribute "flow".
tr = hkw.transform.Fur(vec_field="flow", n=12000, length=0.18, lift=35)

# Visualize as curves. Pair with a Hair material for realistic fur in the
# Blender backend (Principled Hair BSDF).
l = hkw.layer(mesh).transform(tr).mark("Curve").material("Hair")
```

Vertex- or corner-domain vector attributes are automatically averaged to per-facet before growing.
Strands are seeded area-uniformly across the surface and grow along the mesh face normals, so a
correctly oriented (outward-wound) mesh produces fur standing off the surface.

Key parameters:

* `n` — number of strands (default 2000). Increase for denser, more realistic fur.
* `length` — strand length in object space; `None` (default) picks 5% of the bounding-box diagonal.
* `lift` — root lift-off angle in degrees, `0` flat along the field, `90` straight up (default 30).
* `curl` — how strongly the strand curls back toward the surface flow as it grows (default 0.35).
* `segments` — segments per strand; higher gives smoother curls (default 6).
* `root_radius` / `tip_radius` — root and tip radii; `root_radius=None` (default) picks 6% of
  `length`, `tip_radius=0` gives a pointed hair tip.
* `randomness` — amount of natural per-strand variation in `[0, 1]` (default 0.3).
* `follow_surface` — when `True`, trace each strand as a short streamline *on* the surface (so it
  follows the field and the surface curvature) with only a gentle lift/curl, so the fur hugs the
  surface instead of standing off it. Default `False` (analytic strands that lean off the surface).
* `children` — number of child hairs grown around each guide strand for dense, clumped fur
  (default 0 = guides only). Only the Blender backend expands children, via a Geometry Nodes
  modifier that duplicates each guide, scatters the child roots, and pulls the child tips back
  toward the guide. This keeps the guide count (and the geometry the transform produces) low while
  render-time fur stays dense.
* `clump` — how strongly child-hair tips converge onto their guide, in `[0, 1]` (default 0.6).
* `spread` — child-root scatter radius; `None` (default) picks 15% of `length`.
* `seed` — RNG seed for seeding and per-strand variation.

For dense fur, prefer a modest guide count with children (e.g. `n=5000, children=15`) over a huge
`n`: the guides stay cheap and Geometry Nodes multiplies them into clumped fur only at render time.

Fur color comes from the `Hair` material: the `eumelanin` / `pheomelanin` pigments give a natural
palette (black → brown → red → blonde), or set a constant `color` (RGB / named) for any hue, e.g.
`.material("Hair", color=[0.15, 0.35, 0.95])` for blue fur.

For richer coats, mix colors procedurally: `root_color` / `tip_color` give a root-to-tip gradient
(dark undercoat → lighter tips) and `color_variation` adds per-strand brightness jitter, e.g.
`.material("Hair", root_color=[0.05, 0.02, 0.01], tip_color=[0.9, 0.65, 0.32], color_variation=0.4)`.
These are evaluated in the Blender hair shader (child hairs inherit them); the Mitsuba and WebGL
backends collapse a gradient to its average color.

The strand radius taper is baked onto the output mesh, so it drives the curve thickness
automatically. Paired with a `Hair` material, the Blender backend renders the strands as native
Cycles hair *primitives* (thin 3D hair with proper self-shadowing and translucency under the
Principled Hair BSDF) — the realistic path — while the Mitsuba and WebGL backends render the same
strands as tapered tubes. An explicit `size` channel overrides the baked taper. For dense,
realistic fur, use many thin strands (e.g. `n=40000` with a small `root_radius`); native hair
primitives keep high strand counts cheap.

## Boundary transform

Boundary transform extracts the boundary of a mesh. The boundary consists of edges that are only
adjacent to one facet. This is useful for visualizing mesh boundaries or seams in UV coordinates.

```py
# Extract boundary edges
tr = hkw.transform.Boundary()

# Extract boundary considering discontinuities in specific attributes
# (e.g., UV seams where UV coordinates are discontinuous)
tr = hkw.transform.Boundary(attributes=["uv"])
```

The `attributes` parameter allows you to specify which attributes should be considered when
determining boundaries. Edges where the specified attributes are discontinuous will also be
considered as boundaries, even if they are not geometric boundaries.

Here is an example of extracting and visualizing mesh boundaries:

```py
mesh = lagrange.io.load_mesh("shape.obj")
base = hkw.layer(mesh)

# Extract boundary as curves
boundary = base.transform(hkw.transform.Boundary()).mark("Curve")
boundary = boundary.material("Diffuse", "black").channel(size=0.01)

# Combine with surface visualization
layer = base.mark("Surface") + boundary
```

## Combining multiple transforms

Multiple transforms can be chained together using `*` operator.

```py
# Add per-facet component id attribute named "comp"
compute_tr = hkw.transform.Compute(component="comp")

# Filter transform to select the first component.
filter_tr = hkw.transform.Filter(data="comp", condition=lambda id: id==0)

# Transforms are carried out from left to right.
tr = component * filter_tr
```

When specifying transformation via `Layer.transform()` methods, the transforms are applied in the
order of the specification.

```py
# Transform `tr1` is applied before `tr2`.
hkw.layer().transform(tr1).transform(tr2)
```
