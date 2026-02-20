import lagrange


def unique_name(mesh: lagrange.SurfaceMesh, name: str):
    count = 0
    attr_name = name
    while mesh.has_attribute(attr_name):
        count += 1
        attr_name = f"{name}_{count}"
    return attr_name


def get_default_uv(mesh: lagrange.SurfaceMesh):
    uv_attr_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV)
    assert len(uv_attr_ids) != 0, "No UV attribute found in mesh"
    return mesh.get_attribute_name(uv_attr_ids[0])
