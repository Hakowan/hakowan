# Mark

Mark defines the type of geometry a visualization should use. Currently mark can be either
`hkw.mark.Surface`, `hkw.mark.Curve` or `hkw.mark.Point`.

* __Surface__: The surface mark indicates the geometry represents a 3D surface.
* __Curve__: The curve mark indicates the geometry represents a set of 3D curves.
* __Point__: The point mark indices the geometry represents a set of 3D points.

Here is an example of specifying the mark.

```py
# As arguement to hkw.layer method.
l = hkw.layer(mark = hkw.mark.Surface)

# Or use .mark overwrite method.
l = hkw.layer().mark(hkw.mark.Surface)
```
