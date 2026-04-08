# Rendering Backends

Hakowan supports multiple rendering backends, allowing you to choose the renderer that best suits
your needs. Currently, two backends are available: **Mitsuba** (default) and **Blender**.

## Available Backends

### Mitsuba Backend

The Mitsuba backend is the default renderer in Hakowan. It is based on the
[Mitsuba 3](https://www.mitsuba-renderer.org/) physically-based rendering system, which provides
high-quality, photorealistic rendering with advanced lighting and material models.

**Advantages:**
- High-quality photorealistic rendering
- Fast rendering with GPU acceleration
- Support for advanced materials and lighting
- Default backend, no additional configuration needed

**Usage:**

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
# Mitsuba is the default backend
hkw.render(layer, filename="output.exr")

# Or explicitly specify Mitsuba
hkw.render(layer, filename="output.exr", backend="mitsuba")
```

### Blender Backend

The Blender backend uses [Blender](https://www.blender.org/)'s rendering engine through the `bpy`
Python API. This backend is useful if you want to leverage Blender's rendering capabilities or
integrate Hakowan into a Blender-based workflow.

**Advantages:**
- Integration with Blender ecosystem
- Access to Blender's rendering features
- Can save Blender scene files (.blend) for further editing

**Requirements:**
- Blender's `bpy` module must be installed
- Typically only available when running Python from within Blender or with a special bpy build

**Usage:**

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
# Explicitly specify Blender backend
hkw.render(layer, filename="output.png", backend="blender")

# Optionally save the Blender scene file
hkw.render(
    layer,
    filename="output.png",
    backend="blender",
    blend_file="scene.blend"
)
```

## Backend Management

### List Available Backends

You can query which backends are currently available:

```py
import hakowan as hkw

backends = hkw.list_backends()
print(f"Available backends: {backends}")
# Output: Available backends: ['mitsuba', 'blender']
```

Note that the Blender backend will only appear if `bpy` is installed.

### Set Default Backend

You can change the default backend for all subsequent render calls:

```py
import hakowan as hkw

# Set Blender as the default backend
hkw.set_default_backend("blender")

# Now this will use Blender
layer = hkw.layer("mesh.obj")
hkw.render(layer, filename="output.png")

# You can still override per render call
hkw.render(layer, filename="output.exr", backend="mitsuba")
```

## Backend-Specific Options

Different backends may support different options passed as keyword arguments to `hkw.render()`.

### Mitsuba Backend Options

```py
# Save Mitsuba scene configuration to YAML
hkw.render(
    layer,
    filename="output.exr",
    backend="mitsuba",
    yaml_file="scene.yaml"
)
```

### Blender Backend Options

```py
# Save Blender scene file
hkw.render(
    layer,
    filename="output.png",
    backend="blender",
    blend_file="scene.blend"
)
```

## Choosing a Backend

**Use Mitsuba (default) when:**
- You want high-quality photorealistic rendering
- You need fast rendering with GPU acceleration
- You're creating publication-quality visualizations

**Use Blender when:**
- You're already working in a Blender-based workflow
- You want to further edit the scene in Blender
- You need specific Blender features

## Troubleshooting

If a backend is not available, you'll see an informational message when importing Hakowan:

```
Blender backend not available (bpy not installed)
```

To use the Blender backend, you need to install Blender and ensure the `bpy` module is available
in your Python environment.
