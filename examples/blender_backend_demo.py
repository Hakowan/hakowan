#!/usr/bin/env python
"""
Demo script showing how to use the Blender backend in hakowan.

Usage:
    python examples/blender_backend_demo.py
"""

import hakowan as hkw
import lagrange
import numpy as np

# List available backends
print(f"Available rendering backends: {hkw.list_backends()}")

# Create a simple mesh (icosphere)
mesh = lagrange.SurfaceMesh()
vertices = np.array([
    [0, 0, 1],
    [0, 0.894, -0.447],
    [0.851, -0.276, -0.447],
    [-0.851, -0.276, -0.447],
    [0, -0.894, -0.447],
], dtype=np.float64)
facets = np.array([
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    [0, 4, 1],
    [1, 4, 2],
    [2, 4, 3],
    [3, 4, 1],
    [1, 3, 2],
], dtype=np.uint32)
mesh.add_vertices(vertices)
mesh.add_triangles(facets)

# Create a layer with plastic material
layer_plastic = hkw.layer(mesh).material("Plastic", "ivory")

# Configure rendering
config = hkw.config()
config.film.width = 800
config.film.height = 600

# Example 1: Render with default (Mitsuba) backend
print("\nRendering with Mitsuba backend...")
try:
    hkw.render(layer_plastic, config, filename="output_mitsuba.png")
    print("✓ Mitsuba render complete: output_mitsuba.png")
except Exception as e:
    print(f"✗ Mitsuba render failed: {e}")

# Example 2: Render with Blender backend (Plastic material)
print("\nRendering with Blender backend (Plastic material)...")
try:
    hkw.render(layer_plastic, config, filename="output_blender_plastic.png", backend="blender", samples=64, engine="CYCLES")
    print("✓ Blender render complete: output_blender_plastic.png")
except Exception as e:
    print(f"✗ Blender render failed: {e}")

# Example 2b: Different material types
print("\nRendering different material types with Blender...")
materials_to_test = [
    ("Diffuse", "red", "diffuse"),
    ("RoughPlastic", "blue", "roughplastic"),
    ("Principled", "green", "principled"),
]

for mat_type, color, suffix in materials_to_test:
    try:
        layer = hkw.layer(mesh).material(mat_type, color)
        hkw.render(
            layer,
            config,
            filename=f"output_{suffix}.png",
            backend="blender",
            samples=32,
            engine="CYCLES"
        )
        print(f"✓ {mat_type} material render complete: output_{suffix}.png")
    except Exception as e:
        print(f"✗ {mat_type} material render failed: {e}")

# Example 3: Set default backend to Blender
print("\nChanging default backend to Blender...")
hkw.set_default_backend("blender")
print(f"Default backend is now: blender")

# Render without specifying backend (will use Blender now)
try:
    hkw.render(layer_plastic, config, filename="output_default.png", samples=32, engine="EEVEE")
    print("✓ Default backend render complete: output_default.png")
except Exception as e:
    print(f"✗ Default backend render failed: {e}")

# Example 4: Save .blend file for debugging
print("\nRendering with .blend file export for debugging...")
try:
    hkw.render(
        layer_plastic, 
        config, 
        filename="output_debug.png", 
        backend="blender",
        samples=64,
        blend_file="debug_scene.blend"  # Save scene for inspection
    )
    print("✓ Render complete with debug file: output_debug.png")
    print("  Debug scene saved to: debug_scene.blend")
    print("  Open in Blender with: blender debug_scene.blend")
except Exception as e:
    print(f"✗ Debug render failed: {e}")

# Example 5: Custom environment lighting
print("\nRendering with custom environment lighting...")
try:
    from hakowan.setup.emitter import Envmap
    from pathlib import Path
    
    # Configure environment map (using default museum.exr)
    config_env = hkw.config()
    config_env.film.width = 800
    config_env.film.height = 600
    config_env.emitters = [
        Envmap(
            scale=2.0,      # Brighter lighting
            rotation=90.0,  # Rotate HDRI 90 degrees
            up=[0, 1, 0]   # Y-up
        )
    ]
    
    hkw.render(
        layer_plastic, 
        config_env, 
        filename="output_envmap.png", 
        backend="blender",
        samples=64,
        engine="CYCLES"
    )
    print("✓ Environment map render complete: output_envmap.png")
except Exception as e:
    print(f"✗ Environment map render failed: {e}")

# Example 6: Environment lighting with transparent background
print("\nRendering with transparent background...")
try:
    config_transparent = hkw.config()
    config_transparent.film.width = 800
    config_transparent.film.height = 600
    config_transparent.emitters = [
        Envmap(scale=1.5, rotation=180.0)
    ]
    
    hkw.render(
        layer_plastic, 
        config_transparent, 
        filename="output_transparent.png", 
        backend="blender",
        samples=64,
        engine="CYCLES",
        transparent_background=True      # Transparent background (HDRI still provides lighting)
    )
    print("✓ Transparent background render complete: output_transparent.png")
    print("  (HDRI provides lighting, background is transparent)")
except Exception as e:
    print(f"✗ Transparent background render failed: {e}")

print("\nDemo complete!")
