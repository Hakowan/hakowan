# Scale

A scale specifies an attribute-specific transformation that should be carried out before mapping it
to various visual [channels](channel.md).
Note that a scale can be associated with one or more [Attributes](attribute.md).

## Uniform scale

`Uniform` scale rescale the attribute uniformly based on the specified factor.

```py
s = hkw.scale.Uniform(factor = 2)
```

## Log scale

It is quite common that we want to visualize the log of the data. The `Log` scale provides this
capability.

```py
s = hkw.scale.Log(base = 2)
```

In this example, we have created a log scale with base 2.

## Clip scale

`Clip` scale truncates the data with the specified domain. Data values below the
domain minimum will be set to the domain minimum, and vice versa for the maximum.

```py
s = hkw.scale.Clip(domain=[0, 1])
```

## Normalize scale

`Normalize` scale rescales the attribute such that the specified domain maps to the
specified range. For example,

```py
s = hkw.scale.Normalize(range_min=0, range_max=1, domain_min=-10, domain_max=10)
```

The scale `s` will rescale a attribute such that the data within [-10, 10] is now within [0, 1].
If the domain is not specified, the maximum and minimum value of the attribute will be used as the
domain.

```py
s = hkw.scale.Normalize(range_min=np.zeros(3), range_max=np.ones(3))
```

In this example, we are rescaling a vector attribute so that it fits within the box from [0, 0, 0]
to [1, 1, 1].

## Affine scale

`Affine` scale applies a affine transformation to a vector-valued attribute.

```py
s = hkw.scale.Affine(matrix = np.eye(3) * 2)
```

## Offset scale

`Offset` scale offsets the current attribute by another attribute. It is useful for showing deformed
data where the rest position and displacement field are stored in separate attributes.

```py
s = hkw.scale.Offset(offset=hkw.attribute("displacement"))
```

## Custom scale

`Custom` scale allows one to use arbitrary scaling function. The function should take a single data
value (either a scalar or a vector) as input and output the scaled data value. For example, one can
reproduce the effect of uniform scaling by 2 using the following custom scale.

```py
s = hkw.scale.Custom(function = lambda x : x * 2)
```

## Combining multiple scales

It is often necessary to apply multiple scales on an attribute. Hakowan provides an easy way of
combining scales together.

```py
s1 = hkw.scale.Log(base = 2)
s2 = hkw.scale.Uniform(factor = 2)

# Chain s1 and s2 together. s1 will be applied before s2.
s = s1 * s2
```
