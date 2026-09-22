#!/usr/bin/env python
"""Demo script showing checkerboard texture support in Blender backend."""

import hakowan as hkw
import lagrange
import numpy as np

# Create a simple plane mesh with UV coordinates
mesh = lagrange.SurfaceMesh()

# Create a plane (2 triangles forming a square)
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
], dtype=np.float64)

facets = np.array([
    [0, 1, 2],
    [0, 2, 3],
], dtype=np.uint32)

# UV coordinates (one per vertex)
uv_coords = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
], dtype=np.float64)

mesh.add_vertices(vertices)
mesh.add_triangles(facets)

# Add UV attribute
uv_attr_id = mesh.create_attribute(
    "uv",
    element=lagrange.AttributeElement.Vertex,
    usage=lagrange.AttributeUsage.UV,
    num_channels=2,
    initial_values=uv_coords,
)

# Create a checkerboard texture
checkerboard_tex = hkw.texture.Checkerboard(
    uv="uv",
    texture1=hkw.texture.Uniform(color="#ffffff"),  # White
    texture2=hkw.texture.Uniform(color="#000000"),  # Black
    size=8,  # 8x8 checkerboard
)

# Create layer with plastic material using checkerboard texture
layer = hkw.layer(mesh).material("Plastic", diffuse_reflectance=checkerboard_tex)

# Configure rendering
config = hkw.config()
config.film.width = 800
config.film.height = 600
config.sampler.sample_count = 64

# Set camera to look at the plane from an angle
config.sensor.location = [0.5, -1.5, 1.0]
config.sensor.target = [0.5, 0.5, 0.0]
config.sensor.fov = 45

# Render using Blender backend
hkw.render(
    layer,
    config=config,
    filename="blender_checkerboard_output.png",
    backend="blender",
)

print("Checkerboard texture rendered successfully!")
print("Output saved to: blender_checkerboard_output.png")
