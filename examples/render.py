#!/usr/bin/env python

""" A command line rendering example using hakowan. """

import argparse
import hakowan
import logging
import lagrange
import numpy as np
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input_mesh", help="Input mesh file")
    parser.add_argument("output_image", help="Output image file")
    parser.add_argument(
        "--euler",
        default=[0, 0, 0],
        nargs=3,
        help="Transformation defined in Euler angles",
    )
    parser.add_argument("-H", "--height", help="Output image height", default=800)
    parser.add_argument("-W", "--width", help="Output image width", default=1024)
    parser.add_argument("-s", "--num-samples", help="Number of samples", default=64)
    parser.add_argument("-c", "--color", help="Base color", default="ivory")
    parser.add_argument("-r", "--roughness", help="Material roughness", default=0.5)
    parser.add_argument("-m", "--metallic", help="Material metallic", default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    mesh = lagrange.io.load_mesh(args.input_mesh)
    vertices = mesh.vertices
    max_side = np.amax(np.amax(vertices, axis=0) - np.amin(vertices, axis=0))
    mesh.create_attribute(
        name="x",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=mesh.vertices[:, 0].copy(),
    )

    # id = lagrange.compute_facet_normal(mesh)
    # name = mesh.get_attribute_name(id)
    # abs_map = lambda x: np.absolute(x)

    id = mesh.create_attribute(
        "index",
        element=lagrange.AttributeElement.Facet,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.arange(mesh.num_facets),
    )

    try:
        roughness = float(args.roughness)
    except ValueError:
        roughness = args.roughness

    try:
        metallic = float(args.metallic)
    except ValueError:
        metallic = args.metallic

    base = hakowan.layer().data(mesh)
    surface_view = base.mark(hakowan.SURFACE).channel(
        color=args.color,
        roughness=roughness,
        metallic=metallic,
    )
    point_view = base.mark(hakowan.POINT).channel(
        size=0.01 * max_side,
    )

    transform = np.identity(4)
    transform[:3, :3] = Rotation.from_euler("xyz", args.euler, degrees=True).as_matrix()
    config = hakowan.RenderConfig()
    config.filename = args.output_image
    config.transform = transform
    config.width = args.width
    config.height = args.height
    config.num_samples = args.num_samples
    config.sampler_type = "multijitter"
    config.envmap_scale = 1

    hakowan.render(surface_view, config)


if __name__ == "__main__":
    main()
