"""Resolve high-level camera framing requests into concrete scene cameras."""

from __future__ import annotations

from typing import Any, Literal

import lagrange
import numpy as np
import numpy.typing as npt

from .compiler import compile
from .grammar.figure import (
    Camera,
    OrthographicCamera,
    FovAxis,
    PerspectiveCamera,
    ThinLensCamera,
)
from .grammar.layer import Layer

DirectionPreset = Literal[
    "front", "back", "left", "right", "top", "bottom", "isometric"
]
Projection = Literal["perspective", "orthographic", "thin_lens"]
Extremum = Literal["min", "max"]
LayerSelector = str | int | None
DirectionLike = DirectionPreset | npt.ArrayLike
ComponentSelector = tuple[str, int | float] | None


def _world_points(view: Any) -> npt.NDArray[np.float64]:
    assert view.data_frame is not None
    vertices = np.asarray(view.data_frame.mesh.vertices, dtype=np.float64)
    if not vertices.size:
        return np.empty((0, 3), dtype=np.float64)
    transform = np.asarray(view.global_transform, dtype=np.float64)
    return (transform[:3, :3] @ vertices.T).T + transform[:3, 3]


def _selected_views(scene: Any, selector: LayerSelector) -> list[Any]:
    if selector is None:
        return list(scene)
    if isinstance(selector, int):
        if selector < 0 or selector >= len(scene):
            raise IndexError(f"Layer index {selector} is outside [0, {len(scene)})")
        return [scene[selector]]
    matches = [view for view in scene if view.name == selector]
    if not matches:
        names = [view.name for view in scene if view.name is not None]
        raise ValueError(f"No compiled layer named {selector!r}; available names: {names}")
    return matches


def _attribute_values(mesh: lagrange.SurfaceMesh, name: str) -> tuple[Any, np.ndarray]:
    if not mesh.has_attribute(name):
        raise ValueError(f"Mesh has no attribute {name!r}")
    if mesh.is_attribute_indexed(name):
        raise ValueError(
            f"Framing by indexed attribute {name!r} is ambiguous; convert it to a vertex or facet attribute"
        )
    attribute = mesh.attribute(name)
    values = np.asarray(attribute.data)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return attribute, values


def _component_points(
    view: Any, selector: ComponentSelector
) -> npt.NDArray[np.float64]:
    points = _world_points(view)
    if selector is None:
        return points
    name, requested = selector
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    attribute, values = _attribute_values(mesh, name)
    if values.shape[1] != 1:
        raise ValueError(f"Component attribute {name!r} must be scalar")
    selected = np.isclose(values[:, 0], requested)
    if attribute.element_type == lagrange.AttributeElement.Vertex:
        return points[selected]
    if attribute.element_type == lagrange.AttributeElement.Facet:
        vertex_ids: set[int] = set()
        for facet in np.flatnonzero(selected):
            vertex_ids.update(int(value) for value in mesh.get_facet_vertices(int(facet)))
        return points[sorted(vertex_ids)] if vertex_ids else points[:0]
    raise ValueError(
        f"Component attribute {name!r} must be defined on vertices or facets"
    )


def _selection_points(
    scene: Any,
    layer: LayerSelector,
    component: ComponentSelector,
) -> tuple[list[Any], npt.NDArray[np.float64]]:
    views = _selected_views(scene, layer)
    chunks = [_component_points(view, component) for view in views]
    chunks = [chunk for chunk in chunks if chunk.size]
    if not chunks:
        detail = f" for component {component!r}" if component is not None else ""
        raise ValueError(f"Camera framing selection contains no points{detail}")
    return views, np.concatenate(chunks, axis=0)


def _direction(
    value: DirectionLike, up_axis: Literal["y", "z"]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if up_axis == "z":
        presets = {
            "front": (0.0, -1.0, 0.0),
            "back": (0.0, 1.0, 0.0),
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
            "top": (0.0, 0.0, 1.0),
            "bottom": (0.0, 0.0, -1.0),
            "isometric": (1.0, -1.0, 1.0),
        }
        default_up = np.array([0.0, 0.0, 1.0])
        pole_up = np.array([0.0, 1.0, 0.0])
    else:
        presets = {
            "front": (0.0, 0.0, 1.0),
            "back": (0.0, 0.0, -1.0),
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
            "top": (0.0, 1.0, 0.0),
            "bottom": (0.0, -1.0, 0.0),
            "isometric": (1.0, 1.0, 1.0),
        }
        default_up = np.array([0.0, 1.0, 0.0])
        pole_up = np.array([0.0, 0.0, -1.0])
    if isinstance(value, str):
        if value not in presets:
            raise ValueError(f"Unknown camera direction {value!r}")
        result = np.asarray(presets[value], dtype=np.float64)
        up = pole_up if value == "top" else -pole_up if value == "bottom" else default_up
    else:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (3,):
            raise ValueError("Camera direction must contain three values")
        up = default_up
    length = np.linalg.norm(result)
    if length <= 1e-12:
        raise ValueError("Camera direction must be non-zero")
    result /= length
    if abs(float(np.dot(result, up))) > 0.999:
        up = pole_up
    return result, up


def _limiting_half_angle(
    fov: float,
    fov_axis: FovAxis,
    resolution: tuple[int, int],
) -> float:
    half = np.radians(fov) * 0.5
    aspect = resolution[0] / resolution[1]
    if fov_axis == "x":
        half_x, half_y = half, np.arctan(np.tan(half) / aspect)
    elif fov_axis == "y":
        half_x, half_y = np.arctan(np.tan(half) * aspect), half
    elif fov_axis == "diagonal":
        half_y = np.arctan(np.tan(half) / np.sqrt(aspect * aspect + 1.0))
        half_x = np.arctan(np.tan(half_y) * aspect)
    elif fov_axis == "larger":
        if aspect >= 1.0:
            half_x, half_y = half, np.arctan(np.tan(half) / aspect)
        else:
            half_x, half_y = np.arctan(np.tan(half) * aspect), half
    else:  # smaller
        if aspect >= 1.0:
            half_x, half_y = np.arctan(np.tan(half) * aspect), half
        else:
            half_x, half_y = half, np.arctan(np.tan(half) / aspect)
    return float(min(half_x, half_y))


def _tuple3(values: npt.ArrayLike) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(3)
    return float(array[0]), float(array[1]), float(array[2])


def _camera_from_points(
    points: npt.NDArray[np.float64],
    *,
    direction: DirectionLike,
    up_axis: Literal["y", "z"],
    margin: float,
    projection: Projection,
    resolution: tuple[int, int],
    fov: float,
    fov_axis: FovAxis,
    target: npt.ArrayLike | None = None,
    up: npt.ArrayLike | None = None,
    near: float | None = None,
    far: float | None = None,
    aperture_radius: float = 0.1,
    focus_distance: float = 0.0,
) -> Camera:
    if margin < 0.0:
        raise ValueError("Camera margin must be non-negative")
    if projection not in {"perspective", "orthographic", "thin_lens"}:
        raise ValueError(f"Unknown camera projection {projection!r}")
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError("Camera framing resolution must be positive")
    eye_direction, default_up = _direction(direction, up_axis)
    target_array = (
        (points.min(axis=0) + points.max(axis=0)) * 0.5
        if target is None
        else np.asarray(target, dtype=np.float64)
    )
    if target_array.shape != (3,):
        raise ValueError("Camera target must contain three values")
    up_array = default_up if up is None else np.asarray(up, dtype=np.float64)
    if up_array.shape != (3,) or np.linalg.norm(up_array) <= 1e-12:
        raise ValueError("Camera up must be a non-zero three-vector")
    up_array /= np.linalg.norm(up_array)
    if abs(float(np.dot(eye_direction, up_array))) > 0.999:
        raise ValueError("Camera up must not be parallel to the viewing direction")

    radius = float(np.linalg.norm(points - target_array, axis=1).max(initial=0.0))
    radius = max(radius, 1e-6)
    if projection == "orthographic":
        forward = -eye_direction
        right = np.cross(forward, up_array)
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, forward)
        relative = points - target_array
        half_width = float(np.max(np.abs(relative @ right), initial=0.0))
        half_height = float(np.max(np.abs(relative @ camera_up), initial=0.0))
        aspect = resolution[0] / resolution[1]
        scale = 2.0 * max(half_height, half_width / aspect, 1e-6) * (1.0 + margin)
        distance = max(radius * 2.0, 1.0)
    else:
        limiting = _limiting_half_angle(fov, fov_axis, resolution)
        distance = radius / max(float(np.sin(limiting)), 1e-4) * (1.0 + margin)
        scale = 2.0
    resolved_near = near if near is not None else max(1e-4, distance - radius * 1.1)
    resolved_far = far if far is not None else distance + radius * 1.1
    eye = target_array + eye_direction * distance
    eye_tuple = _tuple3(eye)
    target_tuple = _tuple3(target_array)
    up_tuple = _tuple3(up_array)
    if projection == "orthographic":
        return OrthographicCamera(
            eye=eye_tuple,
            target=target_tuple,
            up=up_tuple,
            near=float(resolved_near),
            far=float(resolved_far),
            scale=scale,
        )
    if projection == "thin_lens":
        return ThinLensCamera(
            eye=eye_tuple,
            target=target_tuple,
            up=up_tuple,
            near=float(resolved_near),
            far=float(resolved_far),
            fov=fov,
            fov_axis=fov_axis,
            aperture_radius=aperture_radius,
            focus_distance=focus_distance or distance,
        )
    return PerspectiveCamera(
        eye=eye_tuple,
        target=target_tuple,
        up=up_tuple,
        near=float(resolved_near),
        far=float(resolved_far),
        fov=fov,
        fov_axis=fov_axis,
    )


def _compiled_selection(
    root: Layer,
    *,
    layer: LayerSelector,
    component: ComponentSelector,
    bounds: npt.ArrayLike | None,
) -> tuple[Any, list[Any], npt.NDArray[np.float64]]:
    scene = compile(root, preserve_attributes=True)
    if bounds is not None:
        if layer is not None or component is not None:
            raise ValueError("bounds cannot be combined with layer or component selection")
        array = np.asarray(bounds, dtype=np.float64)
        if array.shape != (2, 3) or np.any(array[0] > array[1]):
            raise ValueError("bounds must have shape (2, 3) with minimum <= maximum")
        points = np.array(
            [
                [x, y, z]
                for x in array[:, 0]
                for y in array[:, 1]
                for z in array[:, 2]
            ],
            dtype=np.float64,
        )
        return scene, list(scene), points
    views, points = _selection_points(scene, layer, component)
    return scene, views, points


def _canonical_axis(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result = values.copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result = -result
    return result


def resolve_camera(
    root: Layer,
    mode: Literal["fit", "principal_axis", "attribute_extremum", "section"],
    *,
    direction: DirectionLike = "isometric",
    layer: LayerSelector = None,
    component: ComponentSelector = None,
    bounds: npt.ArrayLike | None = None,
    projection: Projection = "perspective",
    margin: float = 0.08,
    resolution: tuple[int, int] = (1024, 800),
    up_axis: Literal["y", "z"] = "y",
    fov: float = 35.0,
    fov_axis: Literal["x", "y", "diagonal", "smaller", "larger"] = "smaller",
    up: npt.ArrayLike | None = None,
    near: float | None = None,
    far: float | None = None,
    axis: int = 0,
    sign: Literal["+", "-"] = "+",
    attribute: str | None = None,
    extremum: Extremum = "max",
    normal: npt.ArrayLike | None = None,
    offset: float = 0.0,
    aperture_radius: float = 0.1,
    focus_distance: float = 0.0,
) -> Camera:
    """Resolve a high-level framing request to a concrete serializable camera."""
    camera_direction: DirectionLike = direction
    if sign not in {"+", "-"}:
        raise ValueError("Principal-axis sign must be '+' or '-'")
    if extremum not in {"min", "max"}:
        raise ValueError("Attribute extremum must be 'min' or 'max'")
    _, views, points = _compiled_selection(
        root, layer=layer, component=component, bounds=bounds
    )
    target: npt.ArrayLike | None = None
    if mode == "principal_axis":
        if axis not in (0, 1, 2):
            raise ValueError("Principal axis must be 0, 1, or 2")
        centered = points - points.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        if axis >= vectors.shape[0]:
            raise ValueError(
                f"Principal axis {axis} is unavailable for this {len(points)}-point selection"
            )
        principal = _canonical_axis(vectors[axis])
        if sign == "-":
            principal = -principal
        camera_direction = principal
        candidate_up = _canonical_axis(vectors[(axis + 1) % len(vectors)])
        up = candidate_up
    elif mode == "attribute_extremum":
        if attribute is None:
            raise ValueError("attribute_extremum framing requires attribute=")
        candidates: list[tuple[float, np.ndarray]] = []
        for view in views:
            assert view.data_frame is not None
            mesh = view.data_frame.mesh
            if not mesh.has_attribute(attribute):
                continue
            source, values = _attribute_values(mesh, attribute)
            magnitudes = values[:, 0] if values.shape[1] == 1 else np.linalg.norm(values, axis=1)
            index = int(np.argmin(magnitudes) if extremum == "min" else np.argmax(magnitudes))
            world = _world_points(view)
            if source.element_type == lagrange.AttributeElement.Vertex:
                position = world[index]
            elif source.element_type == lagrange.AttributeElement.Facet:
                ids = np.asarray(mesh.get_facet_vertices(index), dtype=np.int64)
                position = world[ids].mean(axis=0)
            else:
                raise ValueError(
                    f"Attribute {attribute!r} must be defined on vertices or facets"
                )
            candidates.append((float(magnitudes[index]), position))
        if not candidates:
            raise ValueError(f"Selected layers have no attribute {attribute!r}")
        chosen = min(candidates, key=lambda item: item[0]) if extremum == "min" else max(candidates, key=lambda item: item[0])
        target = chosen[1]
    elif mode == "section":
        if normal is None:
            raise ValueError("section framing requires normal=")
        section_normal = np.asarray(normal, dtype=np.float64)
        if section_normal.shape != (3,) or np.linalg.norm(section_normal) <= 1e-12:
            raise ValueError("Section normal must be a non-zero three-vector")
        section_normal /= np.linalg.norm(section_normal)
        center = (points.min(axis=0) + points.max(axis=0)) * 0.5
        target = center + section_normal * (offset - float(np.dot(section_normal, center)))
        camera_direction = section_normal
    elif mode != "fit":
        raise ValueError(f"Unknown camera framing mode: {mode!r}")
    return _camera_from_points(
        points,
        direction=camera_direction,
        up_axis=up_axis,
        margin=margin,
        projection=projection,
        resolution=resolution,
        fov=fov,
        fov_axis=fov_axis,
        target=target,
        up=up,
        near=near,
        far=far,
        aperture_radius=aperture_radius,
        focus_distance=focus_distance,
    )


def turntable_cameras(
    root: Layer,
    *,
    count: int = 12,
    elevation: float = 20.0,
    start: float = 0.0,
    layer: LayerSelector = None,
    component: ComponentSelector = None,
    projection: Projection = "perspective",
    margin: float = 0.08,
    resolution: tuple[int, int] = (1024, 800),
    up_axis: Literal["y", "z"] = "y",
    fov: float = 35.0,
) -> tuple[Camera, ...]:
    """Return evenly spaced concrete cameras around a selected scene region."""
    if count <= 0:
        raise ValueError("Turntable count must be positive")
    _, _, points = _compiled_selection(
        root, layer=layer, component=component, bounds=None
    )
    elevation_rad = np.radians(elevation)
    cameras: list[Camera] = []
    for azimuth in np.linspace(start, start + 360.0, count, endpoint=False):
        angle = np.radians(azimuth)
        if up_axis == "z":
            direction = (
                np.sin(angle) * np.cos(elevation_rad),
                -np.cos(angle) * np.cos(elevation_rad),
                np.sin(elevation_rad),
            )
        else:
            direction = (
                np.sin(angle) * np.cos(elevation_rad),
                np.sin(elevation_rad),
                np.cos(angle) * np.cos(elevation_rad),
            )
        cameras.append(
            _camera_from_points(
                points,
                direction=direction,
                up_axis=up_axis,
                margin=margin,
                projection=projection,
                resolution=resolution,
                fov=fov,
                fov_axis="smaller",
            )
        )
    return tuple(cameras)


__all__ = ["resolve_camera", "turntable_cameras"]
