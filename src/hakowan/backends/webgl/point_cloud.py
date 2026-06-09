"""Point mark renderer.

Each vertex of the input mesh draws the chosen base shape (sphere, disk, or
cube). When the per-point placement is a translation · rotation · axis-aligned
scale — i.e. uniform size, an orientation rotation, or a ``full`` covariance
ellipsoid — we emit a single prototype shape GPU-instanced via
``EXT_mesh_gpu_instancing`` (one mesh + per-point TRANSLATION/ROTATION/SCALE),
which keeps the GLB tiny for large clouds.

The baked fallback (every shape copy expanded into one combined triangle mesh)
handles the cases instancing can't express: raw (non-``full``) covariance, which
may encode an arbitrary 3x3 stretch/shear, and per-point custom material
attributes (e.g. a per-point roughness field), which have no instance slot
beyond ``_COLOR_0``.
"""

from __future__ import annotations

from functools import lru_cache

import lagrange
import numpy as np

from ...common import logger
from ...compiler import View
from ...grammar.scale import Attribute

from .builder import GLTFBuilder
from .mesh_extract import (
    _find_color_field_name,
    _read_color_attribute,
    primitive_arrays,
)
from .material_translate import translate_material


# ---------------------------------------------------------------------- #
# Base shapes (lagrange.primitive)                                         #
# ---------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def _icosphere(refinement: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(positions, normals, triangles)`` for a unit icosphere.

    A subdivided icosahedron projected onto the unit sphere. ``lagrange`` bakes
    flat per-face normals, but a centred unit sphere's smooth normal at a vertex
    is just its position, so we use the shared vertices (12 at level 0) with
    radial normals for smooth shading.
    """
    mesh = lagrange.primitive.generate_subdivided_sphere(
        base_shape=lagrange.primitive.generate_icosahedron(),
        radius=1.0,
        subdiv_level=refinement,
    )
    positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    tris = np.ascontiguousarray(mesh.facets, dtype=np.uint32)
    normals = positions.copy()  # unit-sphere normals coincide with positions.
    return positions, normals, tris


@lru_cache(maxsize=2)
def _disk(segments: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit disk in the XY plane facing +Z (central vertex + ring)."""
    mesh = lagrange.primitive.generate_disc(
        radius=1.0,
        normal=[0.0, 0.0, 1.0],
        radial_sections=segments,
        triangulate=True,
    )
    positions, normals, indices = primitive_arrays(mesh)
    return positions, normals, indices.reshape(-1, 3)


@lru_cache(maxsize=1)
def _cube() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cube centred at the origin (extent ±1) with flat per-face normals."""
    mesh = lagrange.primitive.generate_rounded_cube(
        width=2.0,
        height=2.0,
        depth=2.0,
        bevel_radius=0.0,
        triangulate=True,
    )
    positions, normals, indices = primitive_arrays(mesh)
    return positions, normals, indices.reshape(-1, 3)


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
    normals = np.asarray(view.data_frame.mesh.attribute(name).data, dtype=np.float32)
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


def _decompose_linear(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split per-point ``M = R·diag(s)`` into rotation ``R`` and axis scale ``s``.

    Used for ``full`` covariance, where ``M = U·diag(√S)`` has orthogonal
    columns: the per-axis scale is each column's length and ``R`` is ``M`` with
    its columns normalised. A reflection (``det < 0``) is folded into a flipped
    column — harmless since the base shape is symmetric, so the ellipsoid is
    unchanged while ``R`` stays a proper rotation a quaternion can encode.
    """
    scales = np.linalg.norm(matrices, axis=1)  # (N, 3) per-column norms
    safe = np.maximum(scales, 1e-12)
    rotations = matrices / safe[:, None, :]
    flip = np.linalg.det(rotations) < 0.0
    rotations[flip, :, 2] *= -1.0
    return rotations.astype(np.float32), scales.astype(np.float32)


def _identity_quaternions(n: int) -> np.ndarray:
    """``n`` identity quaternions in xyzw order."""
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 3] = 1.0
    return quats


def _matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """Vectorised (N,3,3) proper-rotation → (N,4) quaternion (xyzw).

    Shepperd's method: pick, per matrix, the branch with the largest
    denominator (trace or a diagonal term) for numerical stability, then
    normalise.
    """
    m = matrices
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    trace = m00 + m11 + m22

    case0 = trace > 0.0
    case1 = (~case0) & (m00 >= m11) & (m00 >= m22)
    case2 = (~case0) & (~case1) & (m11 >= m22)
    case3 = ~(case0 | case1 | case2)

    n = m.shape[0]
    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)
    w = np.empty(n)

    s = np.sqrt(np.maximum(trace + 1.0, 1e-12)) * 2.0  # s = 4w
    w[case0] = (0.25 * s)[case0]
    x[case0] = ((m21 - m12) / s)[case0]
    y[case0] = ((m02 - m20) / s)[case0]
    z[case0] = ((m10 - m01) / s)[case0]

    s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2.0  # s = 4x
    w[case1] = ((m21 - m12) / s)[case1]
    x[case1] = (0.25 * s)[case1]
    y[case1] = ((m01 + m10) / s)[case1]
    z[case1] = ((m02 + m20) / s)[case1]

    s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2.0  # s = 4y
    w[case2] = ((m02 - m20) / s)[case2]
    x[case2] = ((m01 + m10) / s)[case2]
    y[case2] = (0.25 * s)[case2]
    z[case2] = ((m12 + m21) / s)[case2]

    s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2.0  # s = 4z
    w[case3] = ((m10 - m01) / s)[case3]
    x[case3] = ((m02 + m20) / s)[case3]
    y[case3] = ((m12 + m21) / s)[case3]
    z[case3] = (0.25 * s)[case3]

    quats = np.stack([x, y, z, w], axis=1)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    return quats.astype(np.float32)


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


def _resolve_base_shape(view: View) -> str:
    base_shape = (
        view.shape_channel.base_shape if view.shape_channel is not None else "sphere"
    )
    if base_shape not in _BASE_SHAPE_BUILDERS:
        logger.warning(
            f"WebGL backend: point shape '{base_shape}' not supported; "
            "falling back to sphere."
        )
        base_shape = "sphere"
    return base_shape


def add_point_view(builder: GLTFBuilder, view: View) -> int:
    """Render the point mark, GPU-instanced when the placement allows it."""
    assert view.data_frame is not None
    mesh = view.data_frame.mesh

    base_shape = _resolve_base_shape(view)
    centers = np.asarray(mesh.vertices, dtype=np.float32)
    n_points = centers.shape[0]
    if n_points == 0:
        logger.warning("WebGL backend: point view has 0 vertices; skipping.")
        return -1

    # Covariance encodes a full per-point stretch+rotation, so when present it
    # supersedes the orientation channel (matching the Mitsuba backend) and the
    # size acts as an extra uniform scale (default 1.0 instead of 0.01).
    covariances = _covariance_matrices(view, n_points)
    cov_full = (
        view.covariance_channel.full if view.covariance_channel is not None else False
    )
    sizes = _extract_sizes(
        view, n_points, default_size=1.0 if covariances is not None else 0.01
    )
    result = translate_material(view, builder)

    # Instancing carries only TRANSLATION/ROTATION/SCALE + _COLOR_0. Raw
    # (non-``full``) covariance may encode an arbitrary 3x3 (shear), and per-point
    # custom material attributes have no instance slot — both keep the baked path.
    raw_covariance = covariances is not None and not cov_full
    if result.custom_attrs or raw_covariance:
        return _add_baked_points(
            builder, view, mesh, result, base_shape, centers, sizes, covariances
        )
    return _add_instanced_points(
        builder, view, mesh, result, base_shape, centers, sizes, covariances
    )


def _point_instance_colors(view: View, mesh, n_points: int) -> np.ndarray | None:
    """Per-point linear RGB from a ScalarField reflectance, or ``None``."""
    color_name = _find_color_field_name(view)
    if color_name is None:
        return None
    per_point = _read_color_attribute(mesh, color_name)
    if per_point.shape[0] != n_points:
        logger.warning(
            f"WebGL backend: color attribute length {per_point.shape[0]} "
            f"!= vertex count {n_points}; dropping color."
        )
        return None
    return np.ascontiguousarray(per_point[:, :3], dtype=np.float32)


def _add_instanced_points(
    builder: GLTFBuilder,
    view: View,
    mesh,
    result,
    base_shape: str,
    centers: np.ndarray,
    sizes: np.ndarray,
    covariances: np.ndarray | None,
) -> int:
    n_points = centers.shape[0]
    if covariances is not None:
        # ``full`` covariance: M = U·diag(√S) → rotation U + per-axis scale √S,
        # with ``size`` an extra uniform scale on top.
        rotations, axis_scales = _decompose_linear(covariances)
        quaternions = _matrices_to_quaternions(rotations)
        scales = (axis_scales * sizes[:, None]).astype(np.float32)
    else:
        rotations = _orientation_matrices(view, n_points)
        quaternions = (
            _matrices_to_quaternions(rotations)
            if rotations is not None
            else _identity_quaternions(n_points)
        )
        scales = np.repeat(sizes[:, None], 3, axis=1).astype(np.float32)

    base_positions, base_normals, base_tris = _BASE_SHAPE_BUILDERS[base_shape]()
    proto_idx = base_tris.reshape(-1).astype(np.uint32)

    instance_colors = _point_instance_colors(view, mesh, n_points)
    pbr = result.pbr
    if instance_colors is not None:
        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
    if result.extras is not None:
        pbr["extras"] = result.extras
    material_idx = builder.add_material(pbr, double_sided=result.double_sided)

    transform = np.asarray(view.global_transform, dtype=np.float64)
    logger.debug(
        f"WebGL backend: point view → {n_points} instanced '{base_shape}' marks "
        f"({base_positions.shape[0]}-vert prototype)."
    )
    return builder.add_instanced_mesh_node(
        positions=base_positions,
        indices=proto_idx,
        normals=base_normals,
        translations=np.ascontiguousarray(centers, dtype=np.float32),
        rotations=quaternions,
        scales=scales,
        instance_colors=instance_colors,
        material_idx=material_idx,
        transform_4x4=transform,
    )


def _add_baked_points(
    builder: GLTFBuilder,
    view: View,
    mesh,
    result,
    base_shape: str,
    centers: np.ndarray,
    sizes: np.ndarray,
    covariances: np.ndarray | None,
) -> int:
    n_points = centers.shape[0]
    if covariances is not None:
        linear = covariances * sizes[:, None, None]  # (N, 3, 3)
    else:
        rotations = _orientation_matrices(view, n_points)
        linear = rotations * sizes[:, None, None] if rotations is not None else None

    base_positions, base_normals, base_tris = _BASE_SHAPE_BUILDERS[base_shape]()
    n_base_verts = base_positions.shape[0]
    n_base_tris = base_tris.shape[0]

    # Expand: per-point linear transform + translate.
    if linear is None:
        # Uniform scale only — the fast path.
        pos = (
            (base_positions[None, :, :] * sizes[:, None, None] + centers[:, None, :])
            .reshape(-1, 3)
            .astype(np.float32)
        )
        nor = (
            np.broadcast_to(base_normals[None, :, :], (n_points, n_base_verts, 3))
            .reshape(-1, 3)
            .astype(np.float32)
        )
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
        f"WebGL backend: point view → {n_points} baked '{base_shape}' marks, "
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
