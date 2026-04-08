# Render

::: hakowan.render.render

## Related Functions

### set_default_backend

Set the default rendering backend for subsequent render calls.

```py
import hakowan as hkw

# Set Blender as default
hkw.set_default_backend("blender")

# All subsequent renders will use Blender unless overridden
hkw.render(layer, filename="output.png")
```

::: hakowan.backends.set_default_backend

### list_backends

List all available rendering backends.

```py
import hakowan as hkw

backends = hkw.list_backends()
print(backends)
# Output: ['mitsuba', 'blender']
```

::: hakowan.backends.list_backends
