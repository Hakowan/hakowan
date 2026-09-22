#!/usr/bin/env python
"""Test script for UV attribute scaling with checkerboard texture in Blender backend."""

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
    
    # UV coordinates (one per vertex) - single tile
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


def test_no_scale():
    """Test checkerboard without UV scale (8x8 pattern in 1 tile)."""
    print("Testing checkerboard without UV scale...")
    mesh = create_plane_with_uv()
    
    checkerboard_tex = hkw.texture.Checkerboard(
        uv="uv",  # No scale
        texture1=hkw.texture.Uniform(color="white"),
        texture2=hkw.texture.Uniform(color="black"),
        size=8,
    )
    
    layer = hkw.layer(mesh).material("Diffuse", reflectance=checkerboard_tex)
    
    config = hkw.config()
    config.film.width = 400
    config.film.height = 400
    config.sampler.sample_count = 32
    config.sensor.location = [0.5, -1.5, 1.0]
    config.sensor.target = [0.5, 0.5, 0.0]
    
    hkw.render(layer, config=config, filename="test_no_scale.png", backend="blender")
    print("✓ No scale test passed")


def test_with_uniform_scale():
    """Test checkerboard with UV scale (16x16 pattern in 1 tile due to 2x scale)."""
    print("Testing checkerboard with Uniform scale (factor=2.0)...")
    mesh = create_plane_with_uv()
    
    # Scale UV by 2.0 - this should double the number of tiles
    checkerboard_tex = hkw.texture.Checkerboard(
        uv=hkw.attribute(name="uv", scale=hkw.scale.Uniform(factor=2.0)),
        texture1=hkw.texture.Uniform(color="red"),
        texture2=hkw.texture.Uniform(color="blue"),
        size=8,  # 8x8 pattern, but with 2x UV scale = 16x16 effective
    )
    
    layer = hkw.layer(mesh).material("Principled", color=checkerboard_tex)
    
    config = hkw.config()
    config.film.width = 400
    config.film.height = 400
    config.sampler.sample_count = 32
    config.sensor.location = [0.5, -1.5, 1.0]
    config.sensor.target = [0.5, 0.5, 0.0]
    
    hkw.render(layer, config=config, filename="test_with_scale.png", backend="blender")
    print("✓ Uniform scale test passed")


def test_with_affine_scale():
    """Test checkerboard with Affine scale (non-uniform scaling)."""
    print("Testing checkerboard with Affine scale...")
    mesh = create_plane_with_uv()
    
    # Create affine transformation: scale U by 3, V by 1.5
    M = np.array([
        [3.0, 0.0],
        [0.0, 1.5],
    ])
    
    checkerboard_tex = hkw.texture.Checkerboard(
        uv=hkw.attribute(name="uv", scale=hkw.scale.Affine(matrix=M)),
        texture1=hkw.texture.Uniform(color="green"),
        texture2=hkw.texture.Uniform(color="yellow"),
        size=4,
    )
    
    layer = hkw.layer(mesh).material("Plastic", diffuse_reflectance=checkerboard_tex)
    
    config = hkw.config()
    config.film.width = 400
    config.film.height = 400
    config.sampler.sample_count = 32
    config.sensor.location = [0.5, -1.5, 1.0]
    config.sensor.target = [0.5, 0.5, 0.0]
    
    hkw.render(layer, config=config, filename="test_affine_scale.png", backend="blender")
    print("✓ Affine scale test passed")


if __name__ == "__main__":
    test_no_scale()
    test_with_uniform_scale()
    test_with_affine_scale()
    print("\n✅ All UV scale tests passed!")
