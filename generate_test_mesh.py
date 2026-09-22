#!/usr/bin/env python3
"""
Generate a simple test mesh with UV coordinates for testing LIC visualization.

This creates a cylinder or sphere with UV mapping.
"""

import argparse
import numpy as np
import lagrange


def create_cylinder_with_uv(radius=1.0, height=2.0, radial_segments=32, height_segments=16):
    """
    Create a cylinder mesh with UV coordinates.
    
    UV mapping:
        U: wraps around the cylinder (0 to 1)
        V: goes from bottom to top (0 to 1)
    """
    vertices = []
    facets = []
    uvs = []
    
    # Generate vertices and UVs
    for i in range(height_segments + 1):
        v = i / height_segments
        y = height * (v - 0.5)
        
        for j in range(radial_segments):
            u = j / radial_segments
            theta = 2 * np.pi * u
            
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            
            vertices.append([x, y, z])
            uvs.append([u, v])
    
    # Generate faces
    for i in range(height_segments):
        for j in range(radial_segments):
            # Current ring
            current = i * radial_segments + j
            next_j = i * radial_segments + (j + 1) % radial_segments
            
            # Next ring
            next_ring = (i + 1) * radial_segments + j
            next_ring_j = (i + 1) * radial_segments + (j + 1) % radial_segments
            
            # Two triangles per quad
            facets.append([current, next_ring, next_j])
            facets.append([next_j, next_ring, next_ring_j])
    
    # Create mesh
    vertices = np.array(vertices, dtype=np.float64)
    facets = np.array(facets, dtype=np.uint32)
    
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(vertices)
    mesh.add_triangles(facets)
    
    # Add UV coordinates as indexed attribute
    uvs = np.array(uvs, dtype=np.float64)
    uv_indices = facets  # Same topology as mesh
    
    mesh.create_attribute(
        "uv",
        element=lagrange.AttributeElement.Indexed,
        usage=lagrange.AttributeUsage.UV,
        initial_values=uvs,
        initial_indices=uv_indices,
    )
    
    return mesh


def create_sphere_with_uv(radius=1.0, lat_segments=32, lon_segments=64):
    """
    Create a sphere mesh with UV coordinates.
    
    UV mapping:
        U: longitude (0 to 1)
        V: latitude (0 to 1, from south pole to north pole)
    """
    vertices = []
    facets = []
    uvs = []
    
    # Generate vertices and UVs
    for i in range(lat_segments + 1):
        v = i / lat_segments
        phi = np.pi * v  # Latitude angle from 0 (south) to pi (north)
        
        for j in range(lon_segments):
            u = j / lon_segments
            theta = 2 * np.pi * u  # Longitude angle
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)
            
            vertices.append([x, y, z])
            uvs.append([u, v])
    
    # Generate faces
    for i in range(lat_segments):
        for j in range(lon_segments):
            current = i * lon_segments + j
            next_j = i * lon_segments + (j + 1) % lon_segments
            
            next_ring = (i + 1) * lon_segments + j
            next_ring_j = (i + 1) * lon_segments + (j + 1) % lon_segments
            
            # Skip degenerate triangles at poles
            if i > 0:
                facets.append([current, next_ring, next_j])
            if i < lat_segments - 1:
                facets.append([next_j, next_ring, next_ring_j])
    
    # Create mesh
    vertices = np.array(vertices, dtype=np.float64)
    facets = np.array(facets, dtype=np.uint32)
    
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(vertices)
    mesh.add_triangles(facets)
    
    # Add UV coordinates
    uvs = np.array(uvs, dtype=np.float64)
    uv_indices = facets
    
    mesh.create_attribute(
        "uv",
        element=lagrange.AttributeElement.Indexed,
        usage=lagrange.AttributeUsage.UV,
        initial_values=uvs,
        initial_indices=uv_indices,
    )
    
    return mesh


def main():
    parser = argparse.ArgumentParser(description="Generate test mesh with UV coordinates")
    parser.add_argument('output', type=str, help='Output mesh file (.obj)')
    parser.add_argument('--type', type=str, default='cylinder', choices=['cylinder', 'sphere'],
                       help='Type of mesh to generate (default: cylinder)')
    parser.add_argument('--segments', type=int, default=32,
                       help='Number of segments (default: 32)')
    
    args = parser.parse_args()
    
    if args.type == 'cylinder':
        print(f"Generating cylinder with {args.segments} radial segments...")
        mesh = create_cylinder_with_uv(radial_segments=args.segments, height_segments=args.segments//2)
    else:
        print(f"Generating sphere with {args.segments} longitude segments...")
        mesh = create_sphere_with_uv(lat_segments=args.segments//2, lon_segments=args.segments)
    
    print(f"Mesh: {mesh.num_vertices} vertices, {mesh.num_facets} facets")
    
    lagrange.io.save_mesh(args.output, mesh)
    print(f"✓ Saved to {args.output}")


if __name__ == '__main__':
    main()
