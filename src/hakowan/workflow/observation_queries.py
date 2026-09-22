"""Structured queries over deterministic observation buffers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import lagrange
import numpy as np

from ..grammar.mark import Mark
from .validation import Diagnostic


if TYPE_CHECKING:
    from .observation import Observation, Snapshot

_BACKGROUND_ID = int(np.uint32(0xFFFFFFFF))


@dataclass(frozen=True, slots=True)
class LayerVisibility:
    """Visible pixels and elements for one layer in one captured view."""

    view: str
    layer_id: int
    name: str
    mark: str
    visible_pixel_count: int
    visible_element_ids: tuple[int, ...]
    total_element_count: int | None
    visible_element_fraction: float | None
    visible_bounds: tuple[int, int, int, int] | None
    projected_bounds: tuple[int, int, int, int] | None
    depth_range: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe visibility record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegionSummary:
    """Layer and background coverage within a half-open pixel rectangle."""

    view: str
    bounds: tuple[int, int, int, int]
    pixel_count: int
    background_pixel_count: int
    background_fraction: float
    layers: tuple[LayerVisibility, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe region summary with nested layer records."""
        return {
            **asdict(self),
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass(frozen=True, slots=True)
class AttributeVisibility:
    """Statistics for visible source elements carrying one numeric attribute."""

    view: str
    layer_id: int
    layer_name: str
    attribute: str
    element: str
    channels: int
    sample_count: int
    criterion: str
    minimum: float | tuple[float, ...]
    maximum: float | tuple[float, ...]
    mean: float | tuple[float, ...]
    minimum_element_id: int
    maximum_element_id: int
    minimum_sample: float | tuple[float, ...]
    maximum_sample: float | tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe visible-attribute statistics."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OcclusionRecord:
    """Projected overlap where one layer is entirely closer than another."""

    view: str
    occluded_layer_id: int
    occluded_layer_name: str
    occluder_layer_id: int
    occluder_layer_name: str
    projected_coverage: float
    fully_hidden: bool
    occluded_projected_bounds: tuple[int, int, int, int]
    occluder_visible_bounds: tuple[int, int, int, int]
    occluded_depth_range: tuple[float, float]
    occluder_depth_range: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe occlusion record."""
        return asdict(self)


def _available_views(observation: Observation, pass_name: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            view
            for view, current_pass in observation.snapshots
            if current_pass == pass_name
        )
    )


def _query_views(
    observation: Observation, view: str | None, pass_name: str
) -> tuple[str, ...]:
    available = _available_views(observation, pass_name)
    if view is not None:
        if view not in available:
            raise KeyError(
                f"View {view!r} has no {pass_name!r} pass; available views: {list(available)}"
            )
        return (view,)
    if not available:
        raise ValueError(f"Query requires the {pass_name!r} observation pass")
    return available


def _data_snapshot(observation: Observation, view: str, pass_name: str) -> Snapshot:
    snapshot = observation.snapshots.get((view, pass_name))
    if snapshot is None or snapshot.data is None:
        raise ValueError(
            f"Query requires pass {pass_name!r} for view {view!r}; request it in hkw.observe(..., passes=...)"
        )
    return snapshot


def _layer_summary(observation: Observation, layer_id: int) -> tuple[str, str]:
    for layer in observation.scene_summary.layers:
        if layer.id == layer_id:
            return layer.name, layer.mark
    return f"Layer {layer_id + 1}", "unknown"


def _total_elements(observation: Observation, layer_id: int) -> int | None:
    if observation._scene is None or not 0 <= layer_id < len(observation._scene):
        return None
    view = observation._scene[layer_id]
    assert view.data_frame is not None
    if view.mark is Mark.Point:
        return int(view.data_frame.mesh.num_vertices)
    if view.mark is Mark.Surface:
        return int(view.data_frame.mesh.num_facets)
    return None


def _pixel_bounds(
    mask: np.ndarray, x0: int = 0, y0: int = 0
) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return (
        x0 + int(columns.min()),
        y0 + int(rows.min()),
        x0 + int(columns.max()) + 1,
        y0 + int(rows.max()) + 1,
    )


def _depth_range(
    depth: np.ndarray | None, mask: np.ndarray
) -> tuple[float, float] | None:
    if depth is None:
        return None
    values = np.asarray(depth[mask], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return float(values.min()), float(values.max())


def region(
    observation: Observation,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    view: str | None = None,
) -> RegionSummary:
    """Summarize a half-open pixel rectangle from layer and element ID passes.

    The query requires both ID passes for exactly one view. A matching depth
    pass is optional and adds per-layer depth ranges.
    """
    views = _query_views(observation, view, "layer_id")
    if len(views) != 1:
        raise ValueError(
            "region() requires view= when an observation has multiple views"
        )
    selected_view = views[0]
    layer_snapshot = _data_snapshot(observation, selected_view, "layer_id")
    element_snapshot = _data_snapshot(observation, selected_view, "element_id")
    layer_data = np.asarray(layer_snapshot.data, dtype=np.uint32)
    element_data = np.asarray(element_snapshot.data, dtype=np.uint32)
    if layer_data.shape != element_data.shape:
        raise ValueError("layer_id and element_id pass dimensions do not match")
    height, width = layer_data.shape
    if not all(isinstance(value, int) for value in (x0, y0, x1, y1)):
        raise TypeError("Region bounds must be integers")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"Region {(x0, y0, x1, y1)} is outside the {width}x{height} observation"
        )
    layer_crop = layer_data[y0:y1, x0:x1]
    element_crop = element_data[y0:y1, x0:x1]
    depth_snapshot = observation.snapshots.get((selected_view, "depth"))
    depth_crop = (
        np.asarray(depth_snapshot.data)[y0:y1, x0:x1]
        if depth_snapshot is not None and depth_snapshot.data is not None
        else None
    )
    background = layer_crop == _BACKGROUND_ID
    layers: list[LayerVisibility] = []
    for raw_layer_id in np.unique(layer_crop[~background]):
        layer_id = int(raw_layer_id)
        mask = layer_crop == raw_layer_id
        ids = element_crop[mask]
        elements = tuple(int(value) for value in np.unique(ids[ids != _BACKGROUND_ID]))
        total = _total_elements(observation, layer_id)
        fraction = (
            min(len(elements) / total, 1.0) if total is not None and total > 0 else None
        )
        name, mark = _layer_summary(observation, layer_id)
        layers.append(
            LayerVisibility(
                view=selected_view,
                layer_id=layer_id,
                name=name,
                mark=mark,
                visible_pixel_count=int(np.count_nonzero(mask)),
                visible_element_ids=elements,
                total_element_count=total,
                visible_element_fraction=fraction,
                visible_bounds=_pixel_bounds(mask, x0, y0),
                projected_bounds=(
                    projection[0]
                    if (
                        projection := _projected_geometry(
                            observation, selected_view, layer_id
                        )
                    )
                    is not None
                    else None
                ),
                depth_range=_depth_range(depth_crop, mask),
            )
        )
    pixel_count = int(layer_crop.size)
    background_count = int(np.count_nonzero(background))
    return RegionSummary(
        view=selected_view,
        bounds=(x0, y0, x1, y1),
        pixel_count=pixel_count,
        background_pixel_count=background_count,
        background_fraction=background_count / pixel_count,
        layers=tuple(layers),
    )


def visible_elements(
    observation: Observation,
    layer: str | int | None = None,
    *,
    view: str | None = None,
) -> tuple[LayerVisibility, ...]:
    """Return visible element IDs, pixel coverage, and projected bounds.

    Fully hidden layers are retained with zero visible pixels. ``layer`` may be
    a numeric layer ID or its compiled name; ``view=None`` queries every view
    containing a layer-ID pass.
    """
    results: list[LayerVisibility] = []
    matched_selector = layer is None
    for selected_view in _query_views(observation, view, "layer_id"):
        snapshot = _data_snapshot(observation, selected_view, "layer_id")
        height, width = np.asarray(snapshot.data).shape
        summary = region(observation, 0, 0, width, height, view=selected_view)
        visible_by_id = {item.layer_id: item for item in summary.layers}
        for known in observation.scene_summary.layers:
            if layer is not None and layer not in {known.id, known.name}:
                continue
            matched_selector = True
            visible = visible_by_id.get(known.id)
            if visible is not None:
                results.append(visible)
                continue
            total = _total_elements(observation, known.id)
            results.append(
                LayerVisibility(
                    view=selected_view,
                    layer_id=known.id,
                    name=known.name,
                    mark=known.mark,
                    visible_pixel_count=0,
                    visible_element_ids=(),
                    total_element_count=total,
                    visible_element_fraction=(
                        0.0 if total is not None and total > 0 else None
                    ),
                    visible_bounds=None,
                    projected_bounds=(
                        projection[0]
                        if (
                            projection := _projected_geometry(
                                observation, selected_view, known.id
                            )
                        )
                        is not None
                        else None
                    ),
                    depth_range=None,
                )
            )
    if not matched_selector:
        raise ValueError(f"No layer matches {layer!r}")
    return tuple(results)


def _attribute_rows(
    observation: Observation, item: LayerVisibility, attribute_name: str
) -> tuple[Any, np.ndarray, np.ndarray]:
    if observation._scene is None or not 0 <= item.layer_id < len(observation._scene):
        raise ValueError("Observation does not retain its compiled scene")
    view = observation._scene[item.layer_id]
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    if not mesh.has_attribute(attribute_name):
        raise KeyError(attribute_name)
    if mesh.is_attribute_indexed(attribute_name):
        raise ValueError(
            f"Visible statistics do not support indexed attribute {attribute_name!r}"
        )
    attribute = mesh.attribute(attribute_name)
    values = np.asarray(attribute.data)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    visible = np.asarray(item.visible_element_ids, dtype=np.int64)
    if (
        view.mark is Mark.Point
        and attribute.element_type == lagrange.AttributeElement.Vertex
    ):
        rows = visible
    elif (
        view.mark is Mark.Surface
        and attribute.element_type == lagrange.AttributeElement.Facet
    ):
        rows = visible[visible < mesh.num_facets]
    elif (
        view.mark is Mark.Surface
        and attribute.element_type == lagrange.AttributeElement.Vertex
    ):
        vertex_ids: set[int] = set()
        for facet_id in visible[visible < mesh.num_facets]:
            vertex_ids.update(
                int(value) for value in mesh.get_facet_vertices(int(facet_id))
            )
        rows = np.asarray(sorted(vertex_ids), dtype=np.int64)
    else:
        raise ValueError(
            f"Cannot map visible {view.mark.name} elements to {attribute.element_type.name.lower()} attribute {attribute_name!r}"
        )
    rows = rows[(rows >= 0) & (rows < len(values))]
    return attribute, values[rows], rows


def _numeric(value: np.ndarray) -> float | tuple[float, ...]:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(flat) == 1:
        return float(flat[0])
    return tuple(float(item) for item in flat)


def attribute_extrema(
    observation: Observation,
    attribute: str,
    *,
    layer: str | int | None = None,
    view: str | None = None,
    bounds: tuple[int, int, int, int] | None = None,
) -> tuple[AttributeVisibility, ...]:
    """Return visible-only numeric attribute statistics.

    Scalar extrema use the value itself; vector extrema use magnitude. Surface
    vertex attributes include vertices belonging to visible facets. Indexed
    attributes are rejected because their element mapping is ambiguous.
    """
    results: list[AttributeVisibility] = []
    missing: list[str] = []
    if bounds is None:
        items = visible_elements(observation, layer, view=view)
    else:
        items = tuple(
            item
            for item in region(observation, *bounds, view=view).layers
            if layer is None or layer in {item.layer_id, item.name}
        )
        if layer is not None and not items:
            raise ValueError(f"No visible layer matches {layer!r} in the region")
    for item in items:
        try:
            source, values, row_ids = _attribute_rows(observation, item, attribute)
        except KeyError:
            missing.append(item.name)
            continue
        if not values.size:
            continue
        finite = np.all(np.isfinite(values), axis=1)
        values = np.asarray(values[finite], dtype=np.float64)
        row_ids = row_ids[finite]
        if not len(values):
            continue
        criterion = (
            values[:, 0] if values.shape[1] == 1 else np.linalg.norm(values, axis=1)
        )
        minimum_index = int(np.argmin(criterion))
        maximum_index = int(np.argmax(criterion))
        results.append(
            AttributeVisibility(
                view=item.view,
                layer_id=item.layer_id,
                layer_name=item.name,
                attribute=attribute,
                element=source.element_type.name.lower(),
                channels=int(values.shape[1]),
                sample_count=int(len(values)),
                criterion="value" if values.shape[1] == 1 else "magnitude",
                minimum=_numeric(values.min(axis=0)),
                maximum=_numeric(values.max(axis=0)),
                mean=_numeric(values.mean(axis=0)),
                minimum_element_id=int(row_ids[minimum_index]),
                maximum_element_id=int(row_ids[maximum_index]),
                minimum_sample=_numeric(values[minimum_index]),
                maximum_sample=_numeric(values[maximum_index]),
            )
        )
    if not results:
        detail = f"; missing on visible layers {missing}" if missing else ""
        raise ValueError(f"No visible samples for attribute {attribute!r}{detail}")
    return tuple(results)


def _world_points(observation: Observation, layer_id: int) -> np.ndarray:
    assert observation._scene is not None
    view = observation._scene[layer_id]
    assert view.data_frame is not None
    points = np.asarray(view.data_frame.mesh.vertices, dtype=np.float64)
    transform = np.asarray(view.global_transform, dtype=np.float64)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _projected_geometry(
    observation: Observation, view_name: str, layer_id: int
) -> tuple[tuple[int, int, int, int], tuple[float, float]] | None:
    layer_snapshot = _data_snapshot(observation, view_name, "layer_id")
    height, width = np.asarray(layer_snapshot.data).shape
    points = _world_points(observation, layer_id)
    if not points.size:
        return None
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (layer_snapshot.world_to_camera @ homogeneous.T).T
    depth = -camera[:, 2]
    clip = (layer_snapshot.projection @ camera.T).T
    valid = (
        (depth >= layer_snapshot.camera.near)
        & (depth <= layer_snapshot.camera.far)
        & (np.abs(clip[:, 3]) > 1e-12)
    )
    if not np.any(valid):
        return None
    ndc = clip[valid, :3] / clip[valid, 3, None]
    x = (ndc[:, 0] + 1.0) * 0.5 * width
    y = (1.0 - ndc[:, 1]) * 0.5 * height
    x0 = max(0, min(width, int(np.floor(x.min()))))
    x1 = max(0, min(width, int(np.ceil(x.max())) + 1))
    y0 = max(0, min(height, int(np.floor(y.min()))))
    y1 = max(0, min(height, int(np.ceil(y.max())) + 1))
    if x0 >= x1 or y0 >= y1:
        return None
    visible_depth = depth[valid]
    return (x0, y0, x1, y1), (float(visible_depth.min()), float(visible_depth.max()))


def _rect_coverage(
    covering: tuple[int, int, int, int], covered: tuple[int, int, int, int]
) -> float:
    area = max(covered[2] - covered[0], 1) * max(covered[3] - covered[1], 1)
    overlap = max(0, min(covering[2], covered[2]) - max(covering[0], covered[0])) * max(
        0, min(covering[3], covered[3]) - max(covering[1], covered[1])
    )
    return overlap / area


def occlusion_report(
    observation: Observation,
    *,
    view: str | None = None,
    layer: str | int | None = None,
) -> tuple[OcclusionRecord, ...]:
    """Report likely occlusion from projected overlap and strict depth ordering.

    Records are geometric evidence rather than opacity proofs. A layer is
    ``fully_hidden`` when it projects into the view but contributes no visible
    pixels to the captured layer-ID pass.
    """
    if observation._scene is None:
        raise ValueError("Observation does not retain its compiled scene")
    results: list[OcclusionRecord] = []
    for view_name in _query_views(observation, view, "layer_id"):
        visible = visible_elements(observation, view=view_name)
        visible_by_id = {item.layer_id: item for item in visible}
        projections = {
            layer_id: _projected_geometry(observation, view_name, layer_id)
            for layer_id in range(len(observation._scene))
        }
        for back_id, back_projection in projections.items():
            if back_projection is None:
                continue
            back_name, _ = _layer_summary(observation, back_id)
            if layer is not None and layer not in {back_id, back_name}:
                continue
            back_bounds, back_depth = back_projection
            back_visible = visible_by_id.get(back_id)
            for front_id, front_visible in visible_by_id.items():
                if front_id == back_id or front_visible.visible_bounds is None:
                    continue
                front_projection = projections.get(front_id)
                if front_projection is None:
                    continue
                _, front_depth = front_projection
                if front_depth[1] >= back_depth[0]:
                    continue
                coverage = _rect_coverage(front_visible.visible_bounds, back_bounds)
                if coverage <= 0.0:
                    continue
                front_name, _ = _layer_summary(observation, front_id)
                results.append(
                    OcclusionRecord(
                        view=view_name,
                        occluded_layer_id=back_id,
                        occluded_layer_name=back_name,
                        occluder_layer_id=front_id,
                        occluder_layer_name=front_name,
                        projected_coverage=coverage,
                        fully_hidden=(
                            back_visible is None
                            or back_visible.visible_pixel_count == 0
                        ),
                        occluded_projected_bounds=back_bounds,
                        occluder_visible_bounds=front_visible.visible_bounds,
                        occluded_depth_range=back_depth,
                        occluder_depth_range=front_depth,
                    )
                )
    results.sort(
        key=lambda item: (item.view, item.occluded_layer_id, -item.projected_coverage)
    )
    return tuple(results)


def _projected_vertex_evidence(
    observation: Observation, view_name: str, layer_id: int
) -> dict[str, Any]:
    snapshot = _data_snapshot(observation, view_name, "layer_id")
    points = _world_points(observation, layer_id)
    if not len(points):
        return {"vertex_count": 0, "visible_vertex_count": 0, "clipped_fraction": 0.0}
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (snapshot.world_to_camera @ homogeneous.T).T
    depth = -camera[:, 2]
    clip = (snapshot.projection @ camera.T).T
    finite = np.all(np.isfinite(clip), axis=1) & (np.abs(clip[:, 3]) > 1e-12)
    ndc = np.full((len(points), 3), np.nan, dtype=np.float64)
    ndc[finite] = clip[finite, :3] / clip[finite, 3, None]
    inside = (
        finite
        & (depth >= snapshot.camera.near)
        & (depth <= snapshot.camera.far)
        & np.all(np.abs(ndc) <= 1.0, axis=1)
    )
    visible_count = int(np.count_nonzero(inside))
    return {
        "vertex_count": int(len(points)),
        "visible_vertex_count": visible_count,
        "clipped_fraction": 1.0 - visible_count / len(points),
    }


def _contrast_evidence(
    observation: Observation, view_name: str, foreground: np.ndarray
) -> dict[str, float] | None:
    snapshot = observation.snapshots.get((view_name, "beauty"))
    if snapshot is None:
        return None
    rgb = np.asarray(snapshot.image.convert("RGB"), dtype=np.float64) / 255.0
    height, width = foreground.shape
    if rgb.shape[0] < height or rgb.shape[1] < width or not np.any(foreground):
        return None
    rgb = rgb[:height, :width]
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    background = ~foreground
    foreground_mean = float(np.mean(luminance[foreground]))
    background_mean = (
        float(np.mean(luminance[background])) if np.any(background) else foreground_mean
    )
    return {
        "foreground_luminance": foreground_mean,
        "background_luminance": background_mean,
        "luminance_contrast": abs(foreground_mean - background_mean),
    }


def _visible_attributes(
    observation: Observation, view_name: str, layer_id: int
) -> list[dict[str, Any]]:
    if observation._scene is None or not 0 <= layer_id < len(observation._scene):
        return []
    view = observation._scene[layer_id]
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    result: list[dict[str, Any]] = []
    for attribute_id in mesh.get_matching_attribute_ids():
        name = mesh.get_attribute_name(attribute_id)
        try:
            summaries = attribute_extrema(
                observation, name, layer=layer_id, view=view_name
            )
        except (KeyError, TypeError, ValueError):
            continue
        result.extend(item.to_dict() for item in summaries)
    return result


def visual_evidence(observation: Observation) -> dict[str, Any]:
    """Build compact framing, visibility, depth, and contrast evidence."""
    result: dict[str, Any] = {"views": {}}
    layer_views = _available_views(observation, "layer_id")
    element_views = set(_available_views(observation, "element_id"))
    queryable_views = tuple(view for view in layer_views if view in element_views)
    occlusions = tuple(
        item
        for view in queryable_views
        for item in occlusion_report(observation, view=view)
    )
    for view_name in queryable_views:
        snapshot = _data_snapshot(observation, view_name, "layer_id")
        layer_data = np.asarray(snapshot.data, dtype=np.uint32)
        height, width = layer_data.shape
        foreground = layer_data != _BACKGROUND_ID
        summary = region(observation, 0, 0, width, height, view=view_name)
        visible = visible_elements(observation, view=view_name)
        occlusion_by_layer: dict[int, float] = {}
        for occlusion in occlusions:
            if occlusion.view != view_name:
                continue
            value = 1.0 if occlusion.fully_hidden else occlusion.projected_coverage
            occlusion_by_layer[occlusion.occluded_layer_id] = max(
                value, occlusion_by_layer.get(occlusion.occluded_layer_id, 0.0)
            )
        layers: list[dict[str, Any]] = []
        for visibility in visible:
            projected_pixels = (
                (visibility.projected_bounds[2] - visibility.projected_bounds[0])
                * (visibility.projected_bounds[3] - visibility.projected_bounds[1])
                if visibility.projected_bounds is not None
                else 0
            )
            geometry = next(
                layer
                for layer in observation.scene_summary.layers
                if layer.id == visibility.layer_id
            )
            layers.append(
                {
                    **visibility.to_dict(),
                    "screen_fraction": visibility.visible_pixel_count
                    / (width * height),
                    "projected_pixel_count": projected_pixels,
                    "projected_visibility_fraction": (
                        min(visibility.visible_pixel_count / projected_pixels, 1.0)
                        if projected_pixels
                        else 0.0
                    ),
                    "occluded_fraction_estimate": occlusion_by_layer.get(
                        visibility.layer_id, 0.0
                    ),
                    "empty": geometry.vertex_count == 0
                    or (geometry.mark == "surface" and geometry.facet_count == 0),
                    "projection": _projected_vertex_evidence(
                        observation, view_name, visibility.layer_id
                    ),
                    "attributes": _visible_attributes(
                        observation, view_name, visibility.layer_id
                    ),
                }
            )
        depth_order = [
            item["name"]
            for item in sorted(
                layers,
                key=lambda value: (
                    value["depth_range"][0]
                    if value["depth_range"] is not None
                    else float("inf")
                ),
            )
        ]
        result["views"][view_name] = {
            "size": [width, height],
            "occupancy": 1.0 - summary.background_fraction,
            "background_fraction": summary.background_fraction,
            "visible_bounds": _pixel_bounds(foreground),
            "depth_order_near_to_far": depth_order,
            "contrast": _contrast_evidence(observation, view_name, foreground),
            "layers": layers,
        }
    view_values = list(result["views"].values())
    result["summary"] = {
        "view_count": len(view_values),
        "minimum_occupancy": min(
            (item["occupancy"] for item in view_values), default=0.0
        ),
        "maximum_occupancy": max(
            (item["occupancy"] for item in view_values), default=0.0
        ),
        "maximum_clipped_fraction": max(
            (
                layer["projection"]["clipped_fraction"]
                for item in view_values
                for layer in item["layers"]
            ),
            default=0.0,
        ),
        "fully_hidden_layer_count": sum(
            layer["visible_pixel_count"] == 0 and not layer["empty"]
            for item in view_values
            for layer in item["layers"]
        ),
        "legend_count": len(observation.manifest.get("legends", [])),
        "legend_observed": bool(observation.manifest.get("legends"))
        and any((view, "beauty") in observation.snapshots for view in layer_views),
    }
    return result


def visual_diagnostics(
    observation: Observation,
    *,
    min_occupancy: float = 0.02,
    max_occupancy: float = 0.95,
    max_clipped_fraction: float = 0.05,
    min_contrast: float = 0.08,
    _evidence: dict[str, Any] | None = None,
) -> tuple[Diagnostic, ...]:
    """Return deterministic diagnostics for common visual failures."""
    evidence = _evidence if _evidence is not None else visual_evidence(observation)
    diagnostics: list[Diagnostic] = []
    for view_name, view in evidence["views"].items():
        path = f"views/{view_name}"
        occupancy = view["occupancy"]
        if occupancy < min_occupancy:
            diagnostics.append(
                Diagnostic(
                    code="visual.frame_empty"
                    if occupancy == 0.0
                    else "visual.occupancy_low",
                    severity="error" if occupancy == 0.0 else "warning",
                    path=path,
                    message=f"Foreground occupancy is {occupancy:.3f}.",
                    hint="Use fit_camera or move the camera closer to the scene.",
                )
            )
        elif occupancy > max_occupancy:
            diagnostics.append(
                Diagnostic(
                    code="visual.occupancy_high",
                    severity="warning",
                    path=path,
                    message=f"Foreground occupancy is {occupancy:.3f}.",
                    hint="Use fit_camera with a larger margin.",
                )
            )
        contrast = view["contrast"]
        if contrast is not None and contrast["luminance_contrast"] < min_contrast:
            diagnostics.append(
                Diagnostic(
                    code="visual.low_contrast",
                    severity="warning",
                    path=path,
                    message=(
                        "Foreground/background luminance contrast is "
                        f"{contrast['luminance_contrast']:.3f}."
                    ),
                    hint="Change material color, lighting, or background.",
                )
            )
        for layer in view["layers"]:
            layer_path = f"{path}/layers/{layer['layer_id']}"
            clipped = layer["projection"]["clipped_fraction"]
            if layer["empty"]:
                diagnostics.append(
                    Diagnostic(
                        code="visual.layer_empty",
                        severity="error",
                        path=layer_path,
                        message=f"Layer {layer['name']!r} has no vertices.",
                        hint="Check its data source and transforms.",
                    )
                )
            elif layer["visible_pixel_count"] == 0:
                diagnostics.append(
                    Diagnostic(
                        code="visual.layer_hidden",
                        severity="error",
                        path=layer_path,
                        message=f"Layer {layer['name']!r} contributes no visible pixels.",
                        hint="Change the camera, offset the layer, or adjust occluding material.",
                    )
                )
            if clipped > max_clipped_fraction:
                diagnostics.append(
                    Diagnostic(
                        code="visual.camera_clipping",
                        severity="warning",
                        path=layer_path,
                        message=f"Projected vertex clipping fraction is {clipped:.3f}.",
                        hint="Use fit_camera or increase the camera margin.",
                    )
                )
            occluded = layer["occluded_fraction_estimate"]
            if occluded >= 0.9:
                diagnostics.append(
                    Diagnostic(
                        code="visual.layer_occluded",
                        severity="warning",
                        path=layer_path,
                        message=(
                            f"Layer {layer['name']!r} has estimated projected "
                            f"occlusion {occluded:.3f}."
                        ),
                        hint="Offset the layer, use transparency, or change the camera.",
                    )
                )
    summary = evidence["summary"]
    if summary["legend_count"] and not summary["legend_observed"]:
        diagnostics.append(
            Diagnostic(
                code="visual.legend_unobserved",
                severity="warning",
                path="legends",
                message="The scene declares a legend but no beauty pass observed it.",
                hint="Include the beauty pass when capturing visual evidence.",
            )
        )
    return tuple(diagnostics)


def query_manifest(observation: Observation) -> dict[str, Any]:
    """Build JSON-safe visibility and occlusion sections for a manifest."""
    result: dict[str, Any] = {}
    layer_views = _available_views(observation, "layer_id")
    element_views = set(_available_views(observation, "element_id"))
    visibility: dict[str, Any] = {}
    for view in layer_views:
        if view not in element_views:
            continue
        snapshot = _data_snapshot(observation, view, "layer_id")
        height, width = np.asarray(snapshot.data).shape
        summary = region(observation, 0, 0, width, height, view=view)
        payload = summary.to_dict()
        payload["layers"] = [
            item.to_dict() for item in visible_elements(observation, view=view)
        ]
        visibility[view] = payload
    if visibility:
        result["visibility"] = visibility
        result["occlusion"] = [item.to_dict() for item in occlusion_report(observation)]
    evidence = visual_evidence(observation)
    result["evidence"] = evidence
    result["visual_diagnostics"] = [
        item.to_dict() for item in visual_diagnostics(observation, _evidence=evidence)
    ]
    return result


__all__ = [
    "AttributeVisibility",
    "LayerVisibility",
    "OcclusionRecord",
    "RegionSummary",
    "attribute_extrema",
    "occlusion_report",
    "query_manifest",
    "region",
    "visible_elements",
    "visual_diagnostics",
    "visual_evidence",
]
