"""Correctness test for the Blender backend's facet-ID render pass.

The facet-ID pass (``config.facet_id = True``) writes a second image in which
every pixel encodes the zero-based index of the mesh face visible there, packed
into the RGB channels (``fid = (R << 16) | (G << 8) | B``).

This test verifies that encoding is *geometrically* correct: for every
foreground pixel we reconstruct the camera ray Blender used and cast it against
the rendered geometry with ``lagrange.raycasting``.  The face the ray hits must
equal the face decoded from the rendered pixel.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

if sys.platform == "win32" and os.environ.get("CI") == "true":
    pytest.skip("bpy crashes on Windows CI runners", allow_module_level=True)

bpy = pytest.importorskip("bpy")
import lagrange
import lagrange.core as lacore
import lagrange.raycasting as raycasting
from PIL import Image as PILImage

import hakowan as hkw


def _grid_mesh(n: int = 3, lo: float = -1.0, hi: float = 1.0) -> lagrange.SurfaceMesh:
    """Build an ``n x n`` quad grid on the z=0 plane, triangulated.

    Produces ``2 * n * n`` triangles that all face the default camera, so every
    facet is visible (no occlusion) and projects to a distinct image region.
    """
    coords = np.linspace(lo, hi, n + 1)
    idx = {}
    verts = []
    for j in range(n + 1):
        for i in range(n + 1):
            idx[(i, j)] = len(verts)
            verts.append([coords[i], coords[j], 0.0])
    tris = []
    for j in range(n):
        for i in range(n):
            a, b = idx[(i, j)], idx[(i + 1, j)]
            c, d = idx[(i + 1, j + 1)], idx[(i, j + 1)]
            tris.append([a, b, c])
            tris.append([a, c, d])
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array(verts, dtype=float))
    mesh.add_triangles(np.array(tris, dtype=np.uint32))
    return mesh


def _decode_facet_ids(png_path) -> tuple[np.ndarray, np.ndarray]:
    """Decode an RGB-packed facet-ID image. Returns (ids, foreground_mask)."""
    img = np.array(PILImage.open(png_path).convert("RGBA"))
    r = img[..., 0].astype(np.int64)
    g = img[..., 1].astype(np.int64)
    b = img[..., 2].astype(np.int64)
    ids = (r << 16) | (g << 8) | b
    foreground = img[..., 3] > 127
    return ids, foreground


def _rendered_geometry(scene) -> lagrange.SurfaceMesh:
    """Rebuild the rendered geometry in world space as a triangle soup.

    Each output triangle carries a per-vertex scalar ``fid`` equal to the
    Blender polygon index it came from -- the same index the facet-ID pass
    encodes (``poly.index``).  Vertices are *not* shared between facets so that
    ray-cast attribute interpolation recovers the integer id exactly.
    """
    mesh_objs = [o for o in scene.objects if o.type == "MESH"]
    # The facet-ID pass colors each mesh object with its own polygon index, so
    # ids would collide across objects.  The single-layer scene under test must
    # therefore contain exactly one mesh.
    assert len(mesh_objs) == 1, f"expected 1 mesh object, got {len(mesh_objs)}"

    verts: list[np.ndarray] = []
    tris: list[list[int]] = []
    fids: list[int] = []
    for obj in mesh_objs:
        m2w = np.array(obj.matrix_world)
        data = obj.data
        for poly in data.polygons:
            loop_verts = [
                data.loops[li].vertex_index
                for li in range(poly.loop_start, poly.loop_start + poly.loop_total)
            ]
            world = []
            for vi in loop_verts:
                co = data.vertices[vi].co
                world.append((m2w @ np.array([co[0], co[1], co[2], 1.0]))[:3])
            # Fan-triangulate the polygon; every sub-triangle keeps poly.index.
            for k in range(1, len(world) - 1):
                base = len(verts)
                verts.extend([world[0], world[k], world[k + 1]])
                tris.append([base, base + 1, base + 2])
                fids.extend([poly.index] * 3)

    src = lagrange.SurfaceMesh()
    src.add_vertices(np.array(verts, dtype=float))
    src.add_triangles(np.array(tris, dtype=np.uint32))
    src.create_attribute(
        "fid",
        element=lacore.AttributeElement.Vertex,
        usage=lacore.AttributeUsage.Scalar,
        initial_values=np.array(fids, dtype=np.float64).reshape(-1, 1),
    )
    return src


def _camera_rays(scene, cam, width: int, height: int):
    """Reconstruct the world-space (origin, directions) Blender used per pixel.

    ``view_frame`` gives the four corners of the camera frustum in camera-local
    space; bilinear interpolation across them yields the exact per-pixel ray,
    automatically accounting for fov, aspect ratio and sensor fit.
    """
    m2w = cam.matrix_world
    origin = np.array(m2w.translation)
    local = [np.array(v) for v in cam.data.view_frame(scene=scene)]
    world = [np.array(m2w @ v) for v in cam.data.view_frame(scene=scene)]

    def corner(right: bool, top: bool) -> np.ndarray:
        for lv, wv in zip(local, world):
            if (lv[0] > 0) == right and (lv[1] > 0) == top:
                return wv
        raise RuntimeError("camera frame corner not found")

    tr, tl = corner(True, True), corner(False, True)
    br, bl = corner(True, False), corner(False, False)

    # Pixel centers; PNG row 0 is the top of the image.
    s = ((np.arange(width) + 0.5) / width)[None, :, None]  # left -> right
    t = ((np.arange(height) + 0.5) / height)[:, None, None]  # top  -> bottom
    top_edge = tl * (1 - s) + tr * s
    bot_edge = bl * (1 - s) + br * s
    points = top_edge * (1 - t) + bot_edge * t  # (H, W, 3)
    dirs = points - origin
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return origin, dirs


def _raycast_facet_ids(src, origins, directions) -> np.ndarray:
    """Cast rays against ``src`` and return the hit facet id (or -1 on miss)."""
    target = lagrange.SurfaceMesh()
    target.add_vertices(np.ascontiguousarray(origins, dtype=np.float64))
    dir_id = target.create_attribute(
        "dir",
        element=lacore.AttributeElement.Vertex,
        usage=lacore.AttributeUsage.Normal,  # raycasting requires Normal usage
        initial_values=np.ascontiguousarray(directions, dtype=np.float64),
    )
    fid_id = src.get_attribute_id("fid")
    raycasting.project_directional(
        src,
        target,
        attribute_ids=[fid_id],
        project_vertices=False,
        direction=dir_id,
        cast_mode=raycasting.CastMode.OneWay,
        fallback_mode=raycasting.FallbackMode.Constant,
        default_value=-1.0,
    )
    return np.asarray(target.attribute("fid").data).ravel().round().astype(np.int64)


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="headless Blender render not supported in CI",
)
def test_facet_id_matches_raycast(tmp_path):
    width = height = 80
    mesh = _grid_mesh(3)  # 18 triangles
    num_facets = mesh.num_facets

    config = hkw.config()
    config.film.width = width
    config.film.height = height
    config.sampler.sample_count = 1
    config.facet_id = True

    layer = hkw.layer().data(mesh).mark(hkw.mark.Surface)
    out = tmp_path / "grid.png"
    hkw.render(layer, config, filename=out, backend="blender", engine="BLENDER_EEVEE")

    fid_png = tmp_path / "grid_facet_id.png"
    assert fid_png.exists() and fid_png.stat().st_size > 0

    decoded, foreground = _decode_facet_ids(fid_png)
    assert foreground.any(), "facet-ID image is entirely background"

    scene = bpy.context.scene
    src = _rendered_geometry(scene)
    origin, dirs = _camera_rays(scene, scene.camera, width, height)

    rows, cols = np.where(foreground)
    origins = np.tile(origin, (rows.size, 1))
    cast_ids = _raycast_facet_ids(src, origins, dirs[rows, cols])
    decoded_ids = decoded[rows, cols]

    hit = cast_ids >= 0
    # Rays through foreground pixels should overwhelmingly hit the geometry.
    assert hit.mean() > 0.99, f"only {hit.mean():.3f} of foreground rays hit"

    # Decoded id must equal the ray-cast hit id everywhere the ray hit.
    match = cast_ids == decoded_ids
    accuracy = (match & hit).sum() / hit.sum()
    assert accuracy > 0.99, f"facet-id accuracy too low: {accuracy:.4f}"

    # Pixels well inside a facet (4-neighbors agree) must match *exactly* --
    # any disagreement there is a real encoding/orientation bug, not boundary
    # anti-aliasing.
    interior = np.zeros_like(foreground)
    interior[1:-1, 1:-1] = (
        (decoded[1:-1, 1:-1] == decoded[:-2, 1:-1])
        & (decoded[1:-1, 1:-1] == decoded[2:, 1:-1])
        & (decoded[1:-1, 1:-1] == decoded[1:-1, :-2])
        & (decoded[1:-1, 1:-1] == decoded[1:-1, 2:])
        & foreground[1:-1, 1:-1]
        & foreground[:-2, 1:-1]
        & foreground[2:, 1:-1]
        & foreground[1:-1, :-2]
        & foreground[1:-1, 2:]
    )
    interior_mask = interior[rows, cols]
    interior_mismatch = (~match & hit & interior_mask).sum()
    assert interior_mismatch == 0, f"{interior_mismatch} interior pixels mismatched"

    # Every facet must be rendered somewhere (the grid fully faces the camera).
    rendered = set(decoded_ids[interior_mask].tolist())
    assert rendered == set(range(num_facets)), (
        f"missing facet ids: {set(range(num_facets)) - rendered}"
    )


def test_lossless_render_state_applies_and_restores():
    """The discrete-pass lossless context must zero AA/filter/dither and
    restore every touched setting on exit (no render needed)."""
    from hakowan.backends.blender.render import BlenderBackend

    backend = BlenderBackend()
    backend._clear_scene()
    scene = bpy.context.scene

    # Seed deliberately "lossy" settings to prove they are restored.
    scene.render.filter_size = 1.5
    scene.render.dither_intensity = 0.7
    scene.view_settings.view_transform = "Standard"
    before = (
        scene.render.engine,
        scene.render.filter_size,
        scene.render.dither_intensity,
        scene.view_settings.view_transform,
        scene.eevee.taa_render_samples,
    )

    with backend._lossless_render_state():
        assert scene.render.engine == "BLENDER_EEVEE"
        assert scene.render.filter_size == 0.0
        assert scene.render.dither_intensity == 0.0
        assert scene.view_settings.view_transform == "Raw"
        assert scene.eevee.taa_render_samples == 1

    after = (
        scene.render.engine,
        scene.render.filter_size,
        scene.render.dither_intensity,
        scene.view_settings.view_transform,
        scene.eevee.taa_render_samples,
    )
    assert before == after
