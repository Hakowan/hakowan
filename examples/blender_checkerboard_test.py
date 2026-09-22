#!/usr/bin/env python
"""Test script for checkerboard texture with different materials and colors."""

import hakowan as hkw
import lagrange
import numpy as np


def create_plane_with_uv():
    """Create a plane mesh with UV coordinates."""
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
    mesh.create_attribute(
        "uv",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.UV,
        num_channels=2,
        initial_values=uv_coords,
    )
    
    return mesh


def test_plastic_material():
    """Test checkerboard with Plastic material."""
    print("Testing Plastic material with checkerboard...")
    mesh = create_plane_with_uv()
    
    checkerboard_tex = hkw.texture.Checkerboard(
        uv="uv",
        texture1=hkw.texture.Uniform(color="red"),
        texture2=hkw.texture.Uniform(color="blue"),
        size=8,
    )
    
    layer = hkw.layer(mesh).material("Plastic", diffuse_reflectance=checkerboard_tex)
    
    config = hkw.config()
    config.film.width = 400
    config.film.height = 400
    config.sampler.sample_count = 32
    config.sensor.location = [0.5, -1.5, 1.0]
    config.sensor.target = [0.5, 0.5, 0.0]
    
    hkw.render(layer, config=config, filename="test_plastic_checker.png", backend="blender")
    print("✓ Plastic material test passed")


def test_principled_material():
    """Test checkerboard with Principled material."""
    print("Testing Principled material with checkerboard...")
    mesh = create_plane_with_uv()
    
    checkerboard_tex = hkw.texture.Checkerboard(
        uv="uv",
        texture1=hkw.texture.Uniform(color=(0.9, 0.9, 0.9)),  # Light gray
        texture2=hkw.texture.Uniform(color=(0.1, 0.1, 0.1)),  # Dark gray
        size=4,  # 4x4 checkerboard
    )
    
    layer = hkw.layer(mesh).material("Principled", color=checkerboard_tex, roughness=0.3, metallic=0.5)
    
    config = hkw.config()
    config.film.width = 400
    config.film.height = 400
    config.sampler.sample_count = 32
    config.sensor.location = [0.5, -1.5, 1.0]
    config.sensor.target = [0.5, 0.5, 0.0]
    
    hkw.render(layer, config=config, filename="test_principled_checker.png", backend="blender")
    print("✓ Principled material test passed")


def test_diffuse_material():
    """Test checkerboard with Diffuse material."""
    print("Testing Diffuse material with checkerboard...")
    mesh = create_plane_with_uv()
    
    checkerboard_tex = hkw.texture.Checkerboard(
        uv="uv",
        texture1=0.9,  # Using scalar values
        texture2=0.1,
        size=16,  # Fine 16x16 checkerboard
    )
    
    layer = hkw.layer(mesh).material("Diffuse", reflectance=checkerboard_tex)
    
    config = hkw.config()
    config.film.width = 400
    config.film.height = 400
    config.sampler.sample_count = 32
    config.sensor.location = [0.5, -1.5, 1.0]
    config.sensor.target = [0.5, 0.5, 0.0]
    
    hkw.render(layer, config=config, filename="test_diffuse_checker.png", backend="blender")
    print("✓ Diffuse material test passed")


if __name__ == "__main__":
    test_plastic_material()
    test_principled_material()
    test_diffuse_material()
    print("\n✅ All checkerboard tests passed!")
