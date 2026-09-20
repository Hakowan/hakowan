# Canonical specification

Hakowan's canonical specification is the stable, JSON-safe description of a
visualization. It is intended for saved figures, cache keys, web tools, and
constrained AI output. Runtime objects such as `SurfaceMesh`, Python callables,
compiler attributes, and backend objects do not appear directly in the document.

The normative machine contract is the published [v1 JSON Schema](../schema/v1.json).
`hkw.schema()` returns the same document. Every object rejects unknown fields,
and every polymorphic object uses a required `kind` discriminator.

## API

```py
import hakowan as hkw

spec = hkw.to_spec(layer)
payload = spec.to_dict()
text = spec.to_json()
canonical = spec.to_json(canonical=True)
spec.save("figure.json")

parsed = hkw.load_spec("figure.json")       # FigureSpec only
layer = hkw.from_spec(parsed)               # FigureSpec/dict -> Layer
layer = hkw.from_json(text)                 # JSON -> Layer
layer = hkw.load_layer("figure.json")       # file -> Layer, resolves relative paths
schema = hkw.schema()                       # JSON Schema dict
```

`Layer.to_spec()` and `Layer.to_json()` are equivalent convenience methods.
The complete Python reference is under [Canonical specification API](../api/spec.md).

## Document root

| Field | Required | Value |
|---|---:|---|
| `$schema` | yes | `https://hakowan.github.io/hakowan/schema/v1.json` |
| `version` | yes | `1.0` |
| `root` | yes | A composition node |

A different version, unknown field, or unknown `kind` is rejected during parsing.

## Composition nodes

Every node contributes a `spec` containing partial layer properties. Parent
properties have precedence over descendant properties, matching Hakowan's
runtime layer tree.

| `kind` | Fields | Semantics |
|---|---|---|
| `layer` | `spec` | Leaf visualization layer. |
| `inherit` | `spec`, `child` | Applies this node's properties to one child. |
| `overlay` | `spec`, `children` | Renders at least two children in one coordinate space; runtime `+`. |
| `layout` | `spec`, `axis`, `gap`, `normalize`, `children` | Places at least two children along `x`, `y`, or `z`; runtime `|`, `&`, or `juxtapose()`. |

`gap` defaults to `0.05` times the mean cell diameter. `normalize=false`
preserves relative object scale. Child order is significant.

## Layer properties

A node's `spec` accepts:

| Field | Type | Meaning |
|---|---|---|
| `data` | data reference or `null` | Mesh and optional region of interest. |
| `mark` | `point`, `curve`, `surface`, or `null` | Mark override. The compiler defaults unresolved marks to `surface`. |
| `channels` | object | At most one value for each visual channel. |
| `transforms` | array | Transform operations in application order. |
| `name` | string or `null` | Viewer-facing layer label. |

Defaults and explicit `null` values are retained in canonical output so two
canonical documents can be compared byte-for-byte.

## Data references

### Mesh file

```json
{
  "kind": "mesh_file",
  "path": "data/mesh.ply",
  "roi_box": [[-1, -1, -1], [1, 1, 1]]
}
```

`path` may be relative or absolute. `roi_box` is optional and contains the
minimum and maximum 3D corners.

### External in-memory data

```json
{
  "kind": "external",
  "id": "simulation-frame-42",
  "roi_box": null
}
```

Mesh buffers are deliberately not embedded. Serialize an in-memory mesh by
providing an identifier:

```py
spec = hkw.to_spec(layer, data_ids={id(mesh): "simulation-frame-42"})
```

Restore it through a mapping or callable:

```py
layer = hkw.from_spec(
    spec,
    data_resolver={"simulation-frame-42": mesh},
)

# Callable form: Callable[[str], DataFrameLike]
layer = hkw.from_spec(spec, data_resolver=load_mesh_by_id)
```

Missing identifiers raise `SpecConversionError`; they never produce placeholder
data.

## Attributes and scales

An attribute reference contains its mesh attribute name and an ordered scale
pipeline:

```json
{
  "name": "velocity",
  "scales": [
    {"kind": "norm", "order": 2},
    {
      "kind": "normalize",
      "range_min": 0.005,
      "range_max": 0.02,
      "domain_min": null,
      "domain_max": null
    }
  ]
}
```

Scale arrays are evaluated from first to last.

| `kind` | Fields | Meaning |
|---|---|---|
| `uniform` | `factor` | Multiply values by a constant. |
| `log` | `base=10` | Apply a logarithm. Non-positive values should be rejected by strict validation. |
| `clip` | `domain` | Clamp to `[minimum, maximum]`. |
| `normalize` | `range_min`, `range_max`, `domain_min=null`, `domain_max=null` | Map an explicit or inferred domain into a target range. |
| `affine` | `matrix` | Apply a linear or homogeneous matrix. |
| `norm` | `order=2` | Reduce vectors to scalar magnitudes. |
| `offset` | `offset` | Add another `AttributeSpec`. |
| `custom` | `function` | Apply a restricted expression or trusted external function. |

## Channels

`channels` is keyed rather than represented as a list. Duplicate channels are
rejected at the canonical boundary instead of relying on runtime shadowing.

| Key | `kind` | Fields | Applicable marks |
|---|---|---|---|
| `position` | `position` | `data` | all |
| `normal` | `normal` | `data` | surface |
| `size` | `size` | `data` constant or attribute | point, curve |
| `shape` | `shape` | `base_shape=sphere`, `orientation=null` | point |
| `vector_field` | `vector_field` | `data`, `refinement_level=0`, `style=null`, `end_type=point`, `normalize=false` | curve |
| `covariance` | `covariance` | `data`, `full=false` | point |
| `material` | material variant | material-specific fields | all |
| `bump_map` | `bump_map` | `texture`, `scale=1` | surface |
| `normal_map` | `normal_map` | `texture` | surface |

A vector-field `style`, when present, currently supports:

```json
{
  "kind": "bend",
  "direction": {"name": "bend_direction", "scales": []},
  "bend_type": "n"
}
```

`bend_type` is `n` (normal), `r` (ribbon), or `s` (smooth).

## Textures

A texture-valued field accepts one of the following objects. Color-valued
positions additionally accept a number, named/hex color string, or numeric list.

| `kind` | Fields | Meaning |
|---|---|---|
| `uniform` | `color` | Constant color or scalar. |
| `image` | `path`, `uv=null`, `raw=false`, `saturation=1`, `whiteness=0` | Image sampled through UV coordinates. |
| `checkerboard` | `uv=null`, `texture1=0.8`, `texture2=0.2`, `size=8` | Alternating UV-space textures. |
| `isocontour` | `data`, `ratio=0.1`, `texture1=0.4`, `texture2=0.2`, `num_contours=8` | Scalar contour bands. |
| `scalar_field` | `data`, `colormap=viridis`, `domain=null`, `range=null`, `categories=false`, `reverse=false` | Scalar-to-color mapping. |

## Materials

Every material also accepts `two_sided=false` and `back_side=null`.

| `kind` | Fields beyond common material fields |
|---|---|
| `diffuse` | `reflectance=0.5` |
| `conductor` | `material` preset name |
| `rough_conductor` | `material`, `distribution=beckmann`, `alpha=0.1` |
| `plastic` | `diffuse_reflectance=0.5`, `specular_reflectance=1` |
| `rough_plastic` | plastic fields, `distribution=beckmann`, `alpha=0.1` |
| `principled` | `color=0.5`, `roughness=0.5`, `metallic=0`, `anisotropic=0`, `spec_trans=0`, `eta=1.5`, `spec_tint=0`, `sheen=0`, `sheen_tint=0`, `flatness=0` |
| `thin_principled` | principled fields, `diff_trans=0` |
| `dielectric` | `int_ior=bk7`, `ext_ior=air`, `medium=null`, `specular_reflectance=1`, `specular_transmittance=1` |
| `thin_dielectric` | dielectric fields |
| `rough_dielectric` | dielectric fields, `distribution=beckmann`, `alpha=0.1` |
| `hair` | `eumelanin=1.3`, `pheomelanin=0.2`, `color=null`, `root_color=null`, `tip_color=null`, `color_variation=0` |

A dielectric `medium` has `albedo=0.75` and `scale=1`.
Backend-specific material degradation is reported by
[`hkw.validate(..., strict=True)`](compile.md#validation).

## Transforms

Transform arrays are applied from first to last.

| `kind` | Fields and defaults |
|---|---|
| `filter` | `data=null`, `condition=null`; null data means vertex positions and null condition keeps all elements. |
| `clip` | `point=[0,0,0]`, `normal=[1,0,0]` |
| `uv_mesh` | `uv=null`; null selects the mesh UV attribute. |
| `affine` | required `matrix` |
| `principal_axes` | identity `frame`, `orthonormalize_frame=true` |
| `normalize` | `normalize_normals=true`, `normalize_tangents_bitangents=true` |
| `compute` | optional output names: `x`, `y`, `z`, `normal`, `vertex_normal`, `facet_normal`, `component` |
| `explode` | `pieces`, `magnitude=1` |
| `norm` | `data`, `norm_attr_name`, `order=2` |
| `boundary` | `attributes=[]` |
| `streamline` | `vec_field`, `n=50`, `cross_field=true`, `length=null`, `seed=0`, `min_length=3`, `max_steps=null`, `id_attr_name=_hakowan_streamline_id` |
| `fur` | `vec_field`, `n=2000`, `length=null`, `lift=30`, `curl=0.35`, `segments=6`, `root_radius=null`, `tip_radius=0`, `randomness=0.3`, `follow_surface=false`, `children=0`, `clump=0.6`, `spread=null`, `seed=0` |

## Expressions and functions

### Restricted expressions

```json
{
  "kind": "expression",
  "source": "value >= 0 and value <= 1 and isfinite(value)"
}
```

Available values:

- `value`: complete scalar or vector value;
- `x`, `y`, `z`: components 0, 1, and 2, or `null` when absent;
- `true`, `false`, `null`.

Allowed operations:

- numeric literals, lists, and tuples;
- `+`, `-`, `*`, `/`, `//`, `%`, and bounded literal powers;
- comparisons, `in`, and `not in`;
- `and`, `or`, and `not`;
- integer indexing;
- `abs`, `min`, `max`, `isfinite`, and `norm`.

Attribute access, imports, comprehensions, keyword arguments, arbitrary calls,
and non-literal or large exponents are rejected. Expressions are limited to
1024 characters and 128 AST nodes. Hakowan interprets the validated AST; it does
not call `eval()`.

### External functions

```json
{"kind": "function", "id": "my-scale-v2"}
```

Resolvers may be mappings or callables:

```py
# Serialization
function_ids: Mapping[Callable, str] | Callable[[Callable], str]

# Reconstruction
function_resolver: Mapping[str, Callable] | Callable[[str], Callable]
```

An arbitrary callable without an identifier raises `SpecConversionError`.
Python source, bytecode, closures, and pickle payloads are never serialized.

## Path resolution

- `hkw.from_spec(spec)` and `hkw.from_json(text)` interpret relative paths using
  the process working directory unless `base_dir=` is supplied.
- `hkw.load_layer("dir/figure.json")` resolves relative mesh and image paths
  against `dir/`.
- The unresolved logical path is retained when converting the restored layer
  back to a specification.
- Paths are serialized with POSIX separators.
- Absolute paths remain absolute.

## Canonicalization guarantees

`spec.to_json(canonical=True)` guarantees for the same `FigureSpec`:

- sorted object keys;
- compact separators and no insignificant whitespace;
- explicit defaults and nulls;
- UTF-8-compatible JSON text;
- rejection of NaN and Infinity;
- deterministic output suitable for hashing and cache keys.

Canonicalization does **not** normalize file contents, hash external resources,
resolve symbolic links, or claim identical pixels across renderer versions.
Child order, transform order, numeric types, and resource identifiers remain
semantically significant.

## Version compatibility

Version `1.0` is the only accepted version. Documents with another `$schema`,
version, unknown field, or unknown variant are rejected. There is currently no
implicit migration path: callers must explicitly convert a future incompatible
document before loading it. Within the `1.x` line, additions must remain
backward-readable; semantic reinterpretation requires a new major version.

## Unsupported serialization cases

The canonical boundary deliberately rejects:

- in-memory meshes without a `data_ids` mapping or callable;
- arbitrary callables without a `function_ids` mapping or callable;
- duplicate channel kinds within one runtime layer node;
- unknown runtime subclasses of marks, channels, materials, textures, scales,
  transforms, or curve styles;
- NaN and Infinity;
- binary mesh or image payloads embedded directly in JSON.

Render `Config`—camera, lights, film, sampler, integrator, and passes—is not yet
part of version 1.0. Serialize it separately until those concepts move into the
figure grammar. Consequently, version 1.0 reproduces the layer tree but does not
by itself guarantee the same camera and lighting.

## Complete handwritten example

```json
{
  "$schema": "https://hakowan.github.io/hakowan/schema/v1.json",
  "version": "1.0",
  "root": {
    "kind": "overlay",
    "spec": {
      "data": {"kind": "mesh_file", "path": "mesh.ply", "roi_box": null},
      "mark": null,
      "channels": {},
      "transforms": [],
      "name": null
    },
    "children": [
      {
        "kind": "layer",
        "spec": {
          "data": null,
          "mark": "surface",
          "channels": {
            "material": {
              "kind": "diffuse",
              "two_sided": false,
              "back_side": null,
              "reflectance": {
                "kind": "scalar_field",
                "data": {
                  "name": "temperature",
                  "scales": [
                    {
                      "kind": "normalize",
                      "range_min": 0,
                      "range_max": 1,
                      "domain_min": 0,
                      "domain_max": 100
                    }
                  ]
                },
                "colormap": "viridis",
                "domain": null,
                "range": null,
                "categories": false,
                "reverse": false
              }
            }
          },
          "transforms": [],
          "name": "temperature"
        }
      },
      {
        "kind": "layer",
        "spec": {
          "data": null,
          "mark": "curve",
          "channels": {
            "size": {"kind": "size", "data": 0.005},
            "material": {
              "kind": "diffuse",
              "two_sided": false,
              "back_side": null,
              "reflectance": "black"
            }
          },
          "transforms": [
            {"kind": "boundary", "attributes": []}
          ],
          "name": "boundary"
        }
      }
    ]
  }
}
```

Omitted optional fields receive the defaults recorded in the JSON Schema.
`spec.to_json()` emits them explicitly in canonical runtime output.
