#!/usr/bin/env python

""" A command line rendering example using hakowan. """

import argparse
import hakowan.beta as hkw
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
    parser.add_argument(
        "-s", "--num-samples", help="Number of samples", default=64, type=int
    )
    parser.add_argument("-c", "--color", help="Base color", default="ivory")
    parser.add_argument("-r", "--roughness", help="Material roughness", default=0.5)
    parser.add_argument("-m", "--metallic", help="Material metallic", default=0.0)
    parser.add_argument("--uv-scale", help="UV scale", default=1.0, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    hkw.logger.setLevel(logging.INFO)

    mesh = lagrange.io.load_mesh(args.input_mesh)
    uv_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV)

    base = hkw.layer.Layer().data(mesh).mark(hkw.mark.Surface)
    if len(uv_ids) > 0:
        uv_name = mesh.get_attribute_name(uv_ids[0])
        checkerboard = hkw.channel.Diffuse(
            reflectance=hkw.texture.CheckerBoard(
                uv=hkw.Attribute(name=uv_name, scale=hkw.scale.Uniform(factor=10)),
                texture1=hkw.texture.Uniform(color=0.2),
                texture2=hkw.texture.Uniform(color=0.8),
            )
        )
        conductor = hkw.channel.Conductor(material="Hg")

        base = base.channel(material=conductor)
    else:
        mesh.create_attribute(
            "x",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=mesh.vertices[:, 0].copy(),
        )
        base = base.channel(
            material=hkw.channel.Diffuse(
                reflectance=hkw.texture.Isocontour(
                    data=hkw.Attribute(name="x", scale=hkw.scale.Uniform(factor=0.1)),
                    ratio=0.1,
                    texture1=hkw.texture.Uniform(color=0.9),
                    texture2=hkw.texture.ScalarField(
                        data=hkw.Attribute(name="x"), colormap="viridis"
                    ),
                )
            ),
        )

    config = hkw.config.Config()
    config.z_up()
    config.sampler.sample_count = args.num_samples
    hkw.render(base, config)


if __name__ == "__main__":
    main()
