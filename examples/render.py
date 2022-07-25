#!/usr/bin/env python

""" A command line rendering example using hakowan. """

import argparse
import hakowan
import logging
import mshio
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
    parser.add_argument("-s", "--num-samples", help="Number of samples",
            default=64)
    parser.add_argument("-c", "--color", help="Shape color", default="ivory")
    return parser.parse_args()


def extract_vertices(msh):
    all_vertices = []
    for vertex_block in msh.nodes.entity_blocks:
        n = vertex_block.num_nodes_in_block
        vertices = np.array(vertex_block.data).reshape((n, 3))
        all_vertices.append(vertices)
    vertices = np.vstack(all_vertices)
    return vertices


def extract_faces(msh):
    all_faces = []
    for face_block in msh.elements.entity_blocks:
        if face_block.element_type != 2:
            logging.warning(
                "Skipping non-triangle element block with type {}".format(
                    face_block.element_type
                )
            )
            continue
        n = face_block.num_elements_in_block
        faces = np.array(face_block.data).reshape((n, 4))[:, 1:4] - 1
        all_faces.append(faces)

    faces = np.vstack(all_faces)
    return faces


def msh_to_dataframe(msh):
    data = hakowan.grammar.layer_data.DataFrame()
    vertices = extract_vertices(msh)
    faces = extract_faces(msh)
    data.geometry = hakowan.grammar.layer_data.Attribute(vertices, faces)
    return data


def main():
    args = parse_args()

    mesh = mshio.load_msh(args.input_mesh)
    data = msh_to_dataframe(mesh)

    base = hakowan.layer(data=data)
    surface_view = base.mark(hakowan.SURFACE).channel(color=args.color)

    transform = np.identity(4)
    transform[:3, :3] = Rotation.from_euler("xyz", args.euler, degrees=True).as_matrix()
    config = hakowan.RenderConfig()
    config.filename = args.output_image
    config.transform = transform
    config.width = args.width
    config.height = args.height
    config.num_samples = args.num_samples

    hakowan.render(surface_view, config)


if __name__ == "__main__":
    main()
