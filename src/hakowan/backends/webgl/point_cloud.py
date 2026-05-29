"""Point mark renderer.

For each vertex of the input mesh we bake the chosen base shape (sphere, disk,
or cube) into a single combined triangle mesh. This avoids glTF mesh
instancing complexity while staying correct for moderate point counts
(~10k–50k spheres at refinement 0). Phase 4 may swap in
``EXT_mesh_gpu_instancing`` for larger point clouds.
"""

from __future__ import annotations

from functools import lru_cache

import lagrange
import numpy as np

from ...common import logger
from ...compiler import View
from ...grammar.scale import Attribute

from .builder import GLTFBuilder
from .mesh_extract import _find_color_field_name, _read_color_attribute
from .material_translate import translate_material


# ---------------------------------------------------------------------- #
# Base shapes                                                              #
# ---------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def _icosphere(refinement: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(positions, normals, triangles)`` for a unit icosphere."""
    phi = (1.0 + 5 ** 0.5) / 2.0
    verts = np.array(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    tris = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [2, 11, 10], [4, 5, 11], [9, 1, 5], [8, 7, 1], [6, 10, 7],
            [4, 9, 5], [9, 8, 1], [8, 6, 7], [6, 2, 10], [2, 4, 11],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        ],
        dtype=np.uint32,
    )
    for _ in range(refinement):
        verts, tris = _subdivide_on_sphere(verts, tris)
    positions = verts.astype(np.float32)
    normals = positions.copy()  # unit-sphere normals coincide with positions.
    return positions, normals, tris


def _subdivide_on_sphere(
    verts: np.ndarray, tris: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    edge_midpoints: dict[tuple[int, int], int] = {}
    new_verts: list[np.ndarray] = list(verts)
    new_tris: list[list[int]] = []

    def get_mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in edge_midpoints:
            return edge_midpoints[key]
        m = (verts[a] + verts[b]) / 2.0
        m = m / np.linalg.norm(m)
        new_verts.append(m)
        idx = len(new_verts) - 1
        edge_midpoints[key] = idx
        return idx

    for v1, v2, v3 in tris.tolist():
        m12 = get_mid(v1, v2)
        m23 = get_mid(v2, v3)
        m31 = get_mid(v3, v1)
        new_tris.extend(
            [
                [v1, m12, m31],
                [v2, m23, m12],
                [v3, m31, m23],
                [m12, m23, m31],
            ]
        )
    return np.array(new_verts, dtype=np.float64), np.array(new_tris, dtype=np.uint32)


@lru_cache(maxsize=2)
def _disk(segments: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit disk in the XY plane facing +Z. One central vertex + ring."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    positions = np.vstack([np.zeros((1, 3)), ring]).astype(np.float32)
    normals = np.zeros_like(positions)
    normals[:, 2] = 1.0
    tris = np.zeros((segments, 3), dtype=np.uint32)
    tris[:, 1] = np.arange(1, segments + 1)
    tris[:, 2] = np.roll(np.arange(1, segments + 1), -1)
    return positions, normals, tris


@lru_cache(maxsize=1)
def _cube() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit cube centred at the origin (extent ±1) with flat per-face normals."""
    # Six faces × four corners → 24 vertices so each face can carry its own
    # normal without sharing across edges.
    face_normals = np.array(
        [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ],
        dtype=np.float32,
    )
    face_quads = [
        # (vertex order CCW when viewed from outside)
        [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
        [(-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1)],
        [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)],
        [(-1, -1, 1), (-1, -1, -1), (1, -1, -1), (1, -1, 1)],
        [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
        [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)],
    ]
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    tris: list[list[int]] = []
    for face_idx, quad in enumerate(face_quads):
        base = len(positions)
        for v in quad:
            positions.append(list(v))
            normals.append(face_normals[face_idx].tolist())
        tris.append([base + 0, base + 1, base + 2])
        tris.append([base + 0, base + 2, base + 3])
    return (
        np.array(positions, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(tris, dtype=np.uint32),
    )


_BASE_SHAPE_BUILDERS = {
    "sphere": lambda: _icosphere(refinement=0),
    "disk": lambda: _disk(segments=16),
    "cube": lambda: _cube(),
}


# ---------------------------------------------------------------------- #
# Orientation                                                              #
# ---------------------------------------------------------------------- #


def _rotation_matrix_z_to(target: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation that maps +Z onto the unit vector ``target``."""
    z = np.array([0.0, 0.0, 1.0])
    t = np.asarray(target, dtype=np.float64)
    n = np.linalg.norm(t)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    t = t / n
    cos_a = float(np.dot(z, t))
    if cos_a > 0.999999:
        return np.eye(3, dtype=np.float32)
    if cos_a < -0.999999:
        # 180° flip around X axis.
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    axis = np.cross(z, t)
    sin_a = np.linalg.norm(axis)
    axis = axis / sin_a
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
    return R.astype(np.float32)


def _covariance_matrices(view: View, n: int) -> np.ndarray | None:
    """Per-point 3x3 stretch/rotation matrices from the covariance channel.

    Mirrors the Mitsuba backend (``mitsuba/shape.py:extract_transform_from_covariances``):
    when ``full`` the attribute stores the covariance matrix Σ and we return
    ``M = U·√S`` (so that ``Σ = M·Mᵀ``) via SVD; otherwise the attribute already
    stores ``M`` directly. Returns ``None`` when no covariance channel is set or
    the attribute can't be resolved.
    """
    cov = view.covariance_channel
    if cov is None:
        return None
    assert isinstance(cov.data, Attribute)
    name = cov.data._internal_name
    assert name is not None
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    if name is None or not mesh.has_attribute(name):
        logger.warning(
            f"WebGL backend: covariance attribute '{name}' missing; ignoring."
        )
        return None
    data = np.asarray(mesh.attribute(name).data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 9 or data.shape[0] != n:
        logger.warning(
            f"WebGL backend: covariance attribute shape {data.shape} is not "
            f"({n}, 9); ignoring covariance."
        )
        return None
    sigma = data.reshape(-1, 3, 3)
    if cov.full:
        U, S, _Vh = np.linalg.svd(sigma)
        # M = U · diag(√S); Σ = M·Mᵀ.
        M = U @ (np.sqrt(S)[:, :, None] * np.eye(3)[None, :, :])
        return M.astype(np.float32)
    return sigma.astype(np.float32)


def _orientation_matrices(view: View, n: int) -> np.ndarray | None:
    """Per-point 3x3 orientation matrices, or ``None`` if no rotation is set."""
    sc = view.shape_channel
    if sc is None or sc.orientation is None:
        return None
    assert isinstance(sc.orientation, Attribute)
    name = sc.orientation._internal_name
    assert name is not None
    assert view.data_frame is not None
    normals = np.asarray(
        view.data_frame.mesh.attribute(name).data, dtype=np.float32
    )
    if normals.shape[0] != n:
        logger.warning(
            f"WebGL backend: orientation attribute length {normals.shape[0]} "
            f"!= vertex count {n}; ignoring."
        )
        return None
    out = np.empty((n, 3, 3), dtype=np.float32)
    for i in range(n):
        out[i] = _rotation_matrix_z_to(normals[i])
    return out


# ---------------------------------------------------------------------- #
# Size resolution                                                          #
# ---------------------------------------------------------------------- #


def _extract_sizes(view: View, n: int, default_size: float = 0.01) -> np.ndarray:
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    if view.size_channel is None:
        return np.full(n, default_size, dtype=np.float32)
    data = view.size_channel.data
    if isinstance(data, float):
        return np.full(n, float(data), dtype=np.float32)
    if isinstance(data, Attribute):
        name = data._internal_name
        assert name is not None
        attr = mesh.attribute(name).data
        arr = np.asarray(attr, dtype=np.float32).reshape(-1)
        if arr.shape[0] != n:
            logger.warning(
                f"WebGL backend: size attribute length {arr.shape[0]} != "
                f"vertex count {n}; padding/truncating."
            )
            if arr.shape[0] < n:
                arr = np.pad(arr, (0, n - arr.shape[0]), constant_values=default_size)
            else:
                arr = arr[:n]
        return arr
    logger.warning(
        f"WebGL backend: unsupported size channel type {type(data).__name__}; "
        f"using default {default_size}."
    )
    return np.full(n, default_size, dtype=np.float32)


# ---------------------------------------------------------------------- #
# Build node                                                               #
# ---------------------------------------------------------------------- #


def add_point_view(builder: GLTFBuilder, view: View) -> int:
    """Bake the configured base shape per vertex and emit a single mesh node."""
    assert view.data_frame is not None
    mesh = view.data_frame.mesh

    base_shape = (
        view.shape_channel.base_shape
        if view.shape_channel is not None
        else "sphere"
    )
    if base_shape not in _BASE_SHAPE_BUILDERS:
        logger.warning(
            f"WebGL backend: point shape '{base_shape}' not supported; "
            "falling back to sphere."
        )
        base_shape = "sphere"

    centers = np.asarray(mesh.vertices, dtype=np.float32)
    n_points = centers.shape[0]
    if n_points == 0:
        logger.warning("WebGL backend: point view has 0 vertices; skipping.")
        return -1

    # Covariance encodes a full per-point stretch+rotation, so when present it
    # supersedes the orientation channel (matching the Mitsuba backend) and the
    # size acts as an extra uniform scale (default 1.0 instead of 0.01).
    covariances = _covariance_matrices(view, n_points)
    sizes = _extract_sizes(
        view, n_points, default_size=1.0 if covariances is not None else 0.01
    )
    if covariances is not None:
        linear = covariances * sizes[:, None, None]  # (N, 3, 3)
    else:
        rotations = _orientation_matrices(view, n_points)
        linear = (
            rotations * sizes[:, None, None] if rotations is not None else None
        )

    base_positions, base_normals, base_tris = _BASE_SHAPE_BUILDERS[base_shape]()
    n_base_verts = base_positions.shape[0]
    n_base_tris = base_tris.shape[0]

    # Expand: per-point linear transform + translate.
    if linear is None:
        # Uniform scale only — the fast path.
        pos = (
            base_positions[None, :, :] * sizes[:, None, None]
            + centers[:, None, :]
        ).reshape(-1, 3).astype(np.float32)
        nor = np.broadcast_to(
            base_normals[None, :, :], (n_points, n_base_verts, 3)
        ).reshape(-1, 3).astype(np.float32)
    else:
        # out = (L @ base) + center, with L (N,3,3) the per-point linear map.
        transformed = np.einsum("nij,kj->nki", linear, base_positions)
        pos = (transformed + centers[:, None, :]).reshape(-1, 3).astype(np.float32)
        # Normals transform by the inverse-transpose of L (then renormalize),
        # which is required for non-uniform/anisotropic covariance stretches and
        # reduces to the rotation itself for pure orientation matrices.
        normal_mats = np.transpose(np.linalg.pinv(linear), (0, 2, 1))
        transformed_normals = np.einsum("nij,kj->nki", normal_mats, base_normals)
        lengths = np.linalg.norm(transformed_normals, axis=2, keepdims=True)
        nor = (
            (transformed_normals / np.maximum(lengths, 1e-12))
            .reshape(-1, 3)
            .astype(np.float32)
        )

    offsets = (np.arange(n_points, dtype=np.uint32) * n_base_verts)[:, None, None]
    idx = (base_tris[None, :, :] + offsets).reshape(-1).astype(np.uint32)

    # Per-vertex colors from ScalarField, repeated for each shape copy.
    color_name = _find_color_field_name(view)
    colors: np.ndarray | None = None
    if color_name is not None:
        per_point = _read_color_attribute(mesh, color_name)
        if per_point.shape[0] != n_points:
            logger.warning(
                f"WebGL backend: color attribute length {per_point.shape[0]} "
                f"!= vertex count {n_points}; dropping color."
            )
        else:
            colors = np.repeat(per_point, n_base_verts, axis=0).astype(np.float32)

    result = translate_material(view, builder)
    pbr = result.pbr
    if colors is not None:
        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
    if result.extras is not None:
        pbr["extras"] = result.extras
    material_idx = builder.add_material(pbr, double_sided=result.double_sided)

    # Replicate per-source-vertex custom attributes across all base-mesh
    # vertices, so e.g. _ROUGHNESS_0 per point becomes per-bake-vertex.
    custom_attrs: dict[str, np.ndarray] = {}
    for name, arr in result.custom_attrs.items():
        if arr.shape[0] != n_points:
            logger.warning(
                f"WebGL backend: custom attr '{name}' length {arr.shape[0]} "
                f"!= point count {n_points}; dropping."
            )
            continue
        if arr.ndim == 1:
            custom_attrs[name] = np.repeat(arr, n_base_verts)
        else:
            custom_attrs[name] = np.repeat(arr, n_base_verts, axis=0)

    transform = np.asarray(view.global_transform, dtype=np.float64)
    logger.debug(
        f"WebGL backend: point view → {n_points} '{base_shape}' marks, "
        f"{n_points * n_base_verts} verts, {n_points * n_base_tris} tris."
    )
    return builder.add_mesh_node(
        positions=pos,
        indices=idx,
        normals=nor,
        colors=colors,
        custom_attributes=(custom_attrs or None),
        material_idx=material_idx,
        transform_4x4=transform,
    )
