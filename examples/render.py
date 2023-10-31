#!/usr/bin/env python

""" A command line rendering example using hakowan. """

import argparse
import hakowan as hkw
import logging
import lagrange
import numpy as np
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input_mesh", help="Input mesh file")
    parser.add_argument("output_image", help="Output image file")
    parser.add_argument(
        "-Z", "--z-up", help="Make Z axis the up direction", action="store_true"
    )
    parser.add_argument("-H", "--height", help="Output image height", default=800)
    parser.add_argument("-W", "--width", help="Output image width", default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    hkw.logger.setLevel(logging.INFO)

    # Create a base layer.
    base = (
        hkw.layer.Layer()
        .data(args.input_mesh)
        .mark(hkw.mark.Surface)
        .channel(material=hkw.material.RoughConductor(material="Al"))
    )

    # Setup configuration.
    config = hkw.config.Config()
    config.film.width = args.width
    config.film.height = args.height
    if args.z_up:
        config.z_up()

    # Render!
    hkw.render(base, config, filename=args.output_image)


if __name__ == "__main__":
    main()
