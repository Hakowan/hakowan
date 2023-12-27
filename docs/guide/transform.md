# Transform

Transform is applied to the entire data frame as a pre-processing step. It is useful to modify the
geometry and/or computing new attributes from existing attributes.

## Filter transform

Filter transform extracts a subset of the 3D elements based on a user provided condition.

```py
tr = hkw.transform.Filter(data="attr_name", condition=lambda value: value > 0)
```

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
