import struct
import zlib
import numpy as np

import lagrange


def serialize_mesh(vertices, faces, normals=None, colors=None, uvs=None):
    """Serialize mesh data.  Note that vertices, normals, colors and uvs share
    the same index buffer (faces).
    """
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3

    num_vertices = len(vertices)
    num_faces = len(faces)

    name = b"Generated with Hakowan"
    header1 = int("041c", 16)
    header2 = int("0004", 16)
    header = struct.pack("<HH", header1, header2)
    data_flags = 1 << 13
    # Double precision.

    vertex_data = b"".join([struct.pack("<ddd", v[0], v[1], v[2]) for v in vertices])
    face_data = b"".join([struct.pack("<III", f[0], f[1], f[2]) for f in faces])

    if normals is not None and len(normals) == num_vertices:
        data_flags |= 1
        normal_data = b"".join([struct.pack("<ddd", n[0], n[1], n[2]) for n in normals])
    else:
        normal_data = b""

    if colors is not None and len(colors) == num_vertices:
        data_flags |= 8
        color_data = b"".join([struct.pack("<ddd", c[0], c[1], c[2]) for c in colors])
    else:
        color_data = b""

    if uvs is not None and len(uvs) == num_vertices:
        data_flags |= 2
        uv_data = b"".join([struct.pack("<dd", uv[0], uv[1]) for uv in uvs])
    else:
        uv_data = b""

    mesh_header = struct.pack(
        "<I{}sQQ".format(len(name) + 1), data_flags, name, num_vertices, num_faces
    )

    mesh_data = (
        mesh_header + vertex_data + normal_data + uv_data + color_data + face_data
    )
    mesh_data = zlib.compress(mesh_data)
    footer = struct.pack("<QI", 0, 1)

    data = header + mesh_data + footer
    return data


def serialize_mesh_ply(vertices, faces, normals=None, colors=None, uvs=None):
    mesh = lagrange.SurfaceMesh()
    mesh.vertices = vertices
    mesh.facets = faces

    if normals is not None:
        mesh.create_attribute(
            "vertex_normal",
            lagrange.AttributeElement.Vertex,
            lagrange.AttributeUsage.Normal,
            normals,
            np.array([], dtype=np.intc),
        )

    if colors is not None:
        if len(colors) == len(vertices):
            mesh.create_attribute(
                "vertex_color",
                lagrange.AttributeElement.Vertex,
                lagrange.AttributeUsage.Color,
                colors[:, :3].copy(),
            )
        elif len(colors) == len(faces):
            mesh.create_attribute(
                "facet_color",
                lagrange.AttributeElement.Facet,
                lagrange.AttributeUsage.Color,
                colors[:, :3].copy(),
            )

    if uvs is not None:
        mesh.create_attribute(
            "vertex_uv",
            lagrange.AttributeElement.Vertex,
            lagrange.AttributeUsage.UV,
            colors,
            np.array([], dtype=np.intc),
        )

    return lagrange.io.serialize_mesh(mesh, "ply")
