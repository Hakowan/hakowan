import lagrange

def unique_name(mesh: lagrange.SurfaceMesh, name: str):
    count = 0
    attr_name = name
    while mesh.has_attribute(attr_name):
        count += 1
        attr_name = f"{name}_{count}"
    return attr_name
