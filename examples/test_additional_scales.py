#!/usr/bin/env python
"""Test additional scale types for UV attributes in Blender backend."""

import hakowan as hkw
import lagrange
import numpy as np


def create_plane_with_uv():
    """Create a plane mesh with UV coordinates ranging from 0 to 10."""
    mesh = lagrange.SurfaceMesh()
    
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
    
    # UV coordinates with larger range for testing normalization
    uv_coords = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
    ], dtype=np.float64)
    
    mesh.add_vertices(vertices)
    mesh.add_triangles(facets)
    
    mesh.create_attribute(
        "uv",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.UV,
        num_channels=2,
        initial_values=uv_coords,
    )
    
    return mesh


def test_normalize_scale():
    """Test checkerboard with Normalize scale."""
    print("Testing checkerboard with Normalize scale...")
    mesh = create_plane_with_uv()

    # Normalize UV to [0, 1] range (from [0, 10] domain)
    # For 2D UV data, need to provide 2D range
    checkerboard_tex = hkw.texture.Checkerboard(
        uv=hkw.attribute(
            name="uv",
            scale=hkw.scale.Normalize(
                range_min=np.array([0.0, 0.0]),
                range_max=np.array([1.0, 1.0])
            )
        ),
        texture1=hkw.texture.Uniform(color="cyan"),
        texture2=hkw.texture.Uniform(color="magenta"),
        size=8,
    )
    
    layer = hkw.layer(mesh).material("Diffuse", reflectance=checkerboard_tex)
    
    config = hkw.config()
    config.film.width = 400
    config.film.height = 400
    config.sampler.sample_count = 32
    config.sensor.location = [0.5, -1.5, 1.0]
    config.sensor.target = [0.5, 0.5, 0.0]
    
    hkw.render(layer, config=config, filename="test_normalize_scale.png", backend="blender")
    print("✓ Normalize scale test passed")


def test_clip_scale():
    """Test checkerboard with Clip scale."""
    print("Testing checkerboard with Clip scale...")
    mesh = create_plane_with_uv()

    # Clip UV to [2, 5] range, then normalize to [0, 1]
    checkerboard_tex = hkw.texture.Checkerboard(
        uv=hkw.attribute(
            name="uv",
            scale=hkw.scale.Clip(domain=(2.0, 5.0)) * hkw.scale.Normalize(
                range_min=np.array([0.0, 0.0]),
                range_max=np.array([1.0, 1.0])
            )
        ),
        texture1=hkw.texture.Uniform(color="orange"),
        texture2=hkw.texture.Uniform(color="purple"),
        size=4,
    )
    
    layer = hkw.layer(mesh).material("Principled", color=checkerboard_tex)
    
    config = hkw.config()
    config.film.width = 400
    config.film.height = 400
    config.sampler.sample_count = 32
    config.sensor.location = [0.5, -1.5, 1.0]
    config.sensor.target = [0.5, 0.5, 0.0]
    
    hkw.render(layer, config=config, filename="test_clip_scale.png", backend="blender")
    print("✓ Clip scale test passed")


def test_custom_scale():
    """Test checkerboard with Custom scale."""
    print("Testing checkerboard with Custom scale...")
    mesh = create_plane_with_uv()
    
    # Custom function: take square root of UV values
    def sqrt_func(val):
        return np.sqrt(np.clip(val, 0, None))
    
    checkerboard_tex = hkw.texture.Checkerboard(
        uv=hkw.attribute(name="uv", scale=hkw.scale.Custom(function=sqrt_func)),
        texture1=hkw.texture.Uniform(color="pink"),
        texture2=hkw.texture.Uniform(color="brown"),
        size=8,
    )
    
    layer = hkw.layer(mesh).material("Plastic", diffuse_reflectance=checkerboard_tex)
    
    config = hkw.config()
    config.film.width = 400
    config.film.height = 400
    config.sampler.sample_count = 32
    config.sensor.location = [0.5, -1.5, 1.0]
    config.sensor.target = [0.5, 0.5, 0.0]
    
    hkw.render(layer, config=config, filename="test_custom_scale.png", backend="blender")
    print("✓ Custom scale test passed")


if __name__ == "__main__":
    test_normalize_scale()
    test_clip_scale()
    test_custom_scale()
    print("\n✅ All additional scale tests passed!")
