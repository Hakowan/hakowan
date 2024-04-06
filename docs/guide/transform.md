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
