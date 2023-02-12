""" MSH format utility functions"""

import mshio
import numpy as np


def extract_vertices(spec):
    """Extract vertices from msh spec."""
    all_vertices = []
    for vertex_block in spec.nodes.entity_blocks:
        n = vertex_block.num_nodes_in_block
        vertices = np.array(vertex_block.data).reshape((n, 3))
        all_vertices.append(vertices)
    if len(all_vertices) == 0:
        return np.array([])
    else:
        return np.vstack(all_vertices)


def extract_faces(spec):
    """Extract faces from msh spec. (Triangles only at the moment.)"""
    all_faces = []
    for face_block in spec.elements.entity_blocks:
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

    if len(all_faces) == 0:
        return np.array([])
    else:
        return np.vstack(all_faces)


def extract_node_attributes(spec):
    """Extract node attributes."""
    attrs = []
    for data in spec.node_data:
        header = data.header
        name = header.string_tags[0]
        num_fields = header.int_tags[1]
        num_entries = header.int_tags[2]

        attr_values = np.ndarray((num_entries, num_fields))
        for entry in data.entries:
            tag = entry.tag - 1
            attr_values[tag, :] = entry.data
        attrs.append(attr_values)

    return attrs
