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
    parser.add_argument("-c", "--color", help="Shape color", default="ivory")
    parser.add_argument(
        "-m",
        "--material",
        help="Material",
        default="diffuse",
        choices=[
            "conductor",
            "diffuse",
            "plastic",
            "roughconductor",
            "roughplastic",
        ],
    )
    parser.add_argument(
        "-p",
        "--material-preset",
        help="Material preset",
        default="Au",
        choices=[
            "Ag",
            "Al",
            "Au",
            "Cr",
            "CrI",
            "Cu",
            "Cu2O",
            "CuO",
            "Hg",
            "Ir",
            "Li",
            "MgO",
            "TiC",
            "TiN",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base = hakowan.layer().data(args.input_mesh)
    surface_view = base.mark(hakowan.SURFACE).channel(
        color=args.color, material=args.material, material_preset=args.material_preset
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

    hakowan.render(surface_view, config)


if __name__ == "__main__":
    main()
