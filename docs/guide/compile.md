# Compile

The `compile` function is responsible for converting a layer specification into a renderable scene.
This function is automatically called by `hkw.render()`, but can also be used directly for
advanced workflows or debugging.

## Overview

The compilation process transforms the high-level declarative layer specification into a low-level
scene representation that can be rendered by a backend. During compilation, Hakowan:

1. Flattens the layer tree into individual views
2. Applies all transforms to the data
3. Processes channels and applies scales
4. Generates geometry and attributes for rendering
5. Computes the global scene transformation

## Basic Usage

```py
import hakowan as hkw

# Create a layer
layer = hkw.layer("mesh.obj").mark(hkw.mark.Surface)

# Compile the layer into a scene
scene = hkw.compile(layer)

# The scene can then be rendered
# (normally you would just call hkw.render() which does both)
```

## When to Use Compile Directly

In most cases, you don't need to call `compile()` directly because `hkw.render()` does it
automatically. However, there are scenarios where direct compilation is useful:

### Debugging and Inspection

You can compile a layer to inspect the resulting scene without rendering:

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
scene = hkw.compile(layer)

# Inspect the compiled scene
print(f"Number of views: {len(scene.views)}")
for i, view in enumerate(scene):
    print(f"View {i}: {view.mark}, {view.data_frame.mesh.num_vertices} vertices")
```

### Custom Rendering Workflows

If you're building a custom rendering pipeline or integrating Hakowan with other tools:

```py
import hakowan as hkw

layer = hkw.layer("mesh.obj")
scene = hkw.compile(layer)

# Access the compiled mesh data
for view in scene:
    mesh = view.data_frame.mesh
    # Process mesh data with custom tools
    # ...
```

### Testing and Validation

When writing tests or validating layer specifications:

```py
import hakowan as hkw

def test_layer_compilation():
    layer = hkw.layer("mesh.obj").mark(hkw.mark.Surface)
    scene = hkw.compile(layer)
    
    assert len(scene.views) == 1
    assert scene.views[0].mark == hkw.mark.Surface
```

## Compilation Process

The compilation process consists of several stages:

1. **Layer Tree Condensation**: The layer tree is flattened into individual views, where each
   path from root to leaf becomes a view.

2. **Transform Application**: All transforms specified in each view are applied to the data frame.

3. **Channel Preprocessing**: Channels are preprocessed and default values are assigned.

4. **Channel Processing**: Channels are processed, scales are applied, and attributes are computed.

5. **Data Frame Finalization**: The data frame is finalized and prepared for rendering.

6. **Global Transform Computation**: The global scene transformation is computed.

## Scene Object

The result of compilation is a `Scene` object that contains:

- **views**: A list of `View` objects, each representing a compiled layer
- **global_transform**: The global transformation matrix applied to the entire scene

Each `View` object contains:

- **data_frame**: The processed mesh data and attributes
- **mark**: The mark type (Point, Curve, or Surface)
- **channels**: The processed channels
- **material_channel**: The material specification
- **transform**: The transform chain applied to this view
- **global_transform**: The global transformation for this view

## Example: Multi-Layer Compilation

```py
import hakowan as hkw

mesh = lagrange.io.load_mesh("shape.obj")
base = hkw.layer(mesh)

# Create multiple layers
surface = base.mark(hkw.mark.Surface)
wireframe = base.mark(hkw.mark.Curve).channel(size=0.01)
vertices = base.mark(hkw.mark.Point).channel(size=0.05)

# Combine layers
combined = surface + wireframe + vertices

# Compile - this creates 3 views
scene = hkw.compile(combined)

print(f"Total views: {len(scene.views)}")
# Output: Total views: 3
```

## See Also

- [Render](../api/render.md) - The render function that calls compile
- [Layer](layer.md) - Layer specification
- [Transform](transform.md) - Data transformations
