"""Deterministic headless snapshots and multi-view observation bundles."""

from __future__ import annotations
import asyncio
import hashlib
import functools
from concurrent.futures import ThreadPoolExecutor

import base64
import copy
import io
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import lagrange
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from .backends import BackendName
from .common.overlay import composite_overlays
from .compiler import Scene, compile
from .grammar.figure import Figure, OrthographicCamera
from .grammar.layer import Layer
from .grammar.mark import Mark
from .setup import Config
from .setup.sensor import Orthographic, Perspective
from .validation import Diagnostic, validate
from .observation_queries import (
    AttributeVisibility,
    LayerVisibility,
    OcclusionRecord,
    RegionSummary,
    attribute_extrema as _attribute_extrema,
    occlusion_report as _occlusion_report,
    query_manifest as _query_manifest,
    visual_diagnostics as _visual_diagnostics,
    visual_evidence as _visual_evidence,
    region as _region,
    visible_elements as _visible_elements,
)


ViewPreset = Literal["front", "back", "left", "right", "top", "bottom", "isometric"]
PassName = Literal["beauty", "albedo", "depth", "normal", "element_id", "layer_id"]
ProjectionMode = Literal["perspective", "orthographic"]

VIEW_PRESETS: tuple[ViewPreset, ...] = (
    "front",
    "back",
    "left",
    "right",
    "top",
    "bottom",
    "isometric",
)
PASS_NAMES: tuple[PassName, ...] = (
    "beauty",
    "albedo",
    "depth",
    "normal",
    "element_id",
    "layer_id",
)
BACKGROUND_ID = np.uint32(0xFFFFFFFF)


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ObservationError(RuntimeError):
    """Raised when deterministic observation capture cannot complete."""


@dataclass(frozen=True, slots=True)
class CameraState:
    """Resolved deterministic camera used for one or more snapshots.

    ``scale`` is the full vertical extent for orthographic cameras and is
    ``None`` for perspective cameras.
    """

    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov: float = 35.0
    near: float = 0.01
    far: float = 100.0
    mode: ProjectionMode = "perspective"
    scale: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe camera-state mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LayerSummary:
    """Geometry counts and identity for one compiled observation layer."""

    id: int
    name: str
    mark: str
    vertex_count: int
    facet_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe layer summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SceneSummary:
    """Normalized scene bounds, coordinate convention, and layer summaries."""

    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    center: tuple[float, float, float]
    radius: float
    up_axis: Literal["y", "z"]
    layers: tuple[LayerSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe scene summary."""
        return {
            "bounds": [list(self.bounds[0]), list(self.bounds[1])],
            "center": list(self.center),
            "radius": self.radius,
            "up_axis": self.up_axis,
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One deterministic camera/pass raster and its machine-readable state."""

    image: Image.Image = field(repr=False, compare=False)
    data: npt.NDArray | None = field(default=None, repr=False, compare=False)
    view: str = "custom"
    pass_name: PassName = "beauty"
    camera: CameraState = field(
        default_factory=lambda: CameraState(
            eye=(0.0, 0.0, 5.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
    )
    world_to_camera: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4), repr=False, compare=False
    )
    projection: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4), repr=False, compare=False
    )
    bounds: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((2, 3)), repr=False, compare=False
    )
    depth_range: tuple[float, float] | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    path: Path | None = None
    data_path: Path | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Return paths, hashes, camera state, matrices, bounds, and diagnostics."""
        return {
            "view": self.view,
            "pass": self.pass_name,
            "path": str(self.path) if self.path is not None else None,
            "mime_type": "image/png",
            "sha256": _file_sha256(self.path),
            "data_path": str(self.data_path) if self.data_path is not None else None,
            "data_mime_type": "application/x-npy"
            if self.data_path is not None
            else None,
            "data_sha256": _file_sha256(self.data_path),
            "size": list(self.image.size),
            "camera": self.camera.to_dict(),
            "world_to_camera": self.world_to_camera.tolist(),
            "projection": self.projection.tolist(),
            "bounds": self.bounds.tolist(),
            "depth_range": list(self.depth_range) if self.depth_range else None,
            "data_dtype": str(self.data.dtype) if self.data is not None else None,
            "background_id": int(BACKGROUND_ID)
            if self.pass_name in {"element_id", "layer_id"}
            else None,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class PixelHit:
    """Geometry and data resolved at one observation pixel."""

    view: str
    pixel: tuple[int, int]
    layer_id: int
    element_id: int
    depth: float
    world_position: tuple[float, float, float]
    normal: tuple[float, float, float] | None
    attributes: dict[str, Any]


@dataclass(slots=True)
class Observation:
    """Multi-view, multi-pass evidence bundle for visual examination."""

    snapshots: dict[tuple[str, str], Snapshot]
    scene_summary: SceneSummary
    diagnostics: tuple[Diagnostic, ...] = ()
    contact_sheet: Image.Image | None = field(default=None, repr=False)
    manifest: dict[str, Any] = field(default_factory=dict)
    _scene: Scene | None = field(default=None, repr=False)

    def snapshot(self, view: str, pass_name: str) -> Snapshot:
        """Return the snapshot captured for ``view`` and ``pass_name``."""
        return self.snapshots[(view, pass_name)]

    def save(self, directory: str | Path) -> None:
        """Write PNGs, raw NumPy pass arrays, a contact sheet, and manifest."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        for (view, pass_name), item in self.snapshots.items():
            path = output / f"{view}_{pass_name}.png"
            item.image.save(path)
            object.__setattr__(item, "path", path)
            if item.data is not None:
                data_path = output / f"{view}_{pass_name}.npy"
                np.save(data_path, item.data, allow_pickle=False)
                object.__setattr__(item, "data_path", data_path)
        if self.contact_sheet is not None:
            self.contact_sheet.save(output / "contact_sheet.png")
        self.manifest["snapshots"] = [
            self.snapshots[key].to_manifest() for key in self.snapshots
        ]
        contact_sheet_path = output / "contact_sheet.png"
        self.manifest["contact_sheet"] = (
            str(contact_sheet_path) if self.contact_sheet is not None else None
        )
        self.manifest["contact_sheet_metadata"] = (
            {
                "path": str(contact_sheet_path),
                "mime_type": "image/png",
                "size": list(self.contact_sheet.size),
                "sha256": _file_sha256(contact_sheet_path),
            }
            if self.contact_sheet is not None
            else None
        )
        (output / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def pick(self, view: str, pixel: tuple[int, int]) -> PixelHit | None:
        """Resolve IDs, depth, world position, normal, and attributes at a pixel.

        The observation must contain depth, element-ID, and layer-ID passes for
        the requested view. Background pixels return ``None``.
        """
        x, y = pixel
        depth_snapshot = self.snapshots.get((view, "depth"))
        element_snapshot = self.snapshots.get((view, "element_id"))
        layer_snapshot = self.snapshots.get((view, "layer_id"))
        if depth_snapshot is None or element_snapshot is None or layer_snapshot is None:
            raise ObservationError(
                "pick() requires depth, element_id, and layer_id passes for the view."
            )
        width, height = depth_snapshot.image.size
        if not (0 <= x < width and 0 <= y < height):
            raise IndexError(f"Pixel {pixel} is outside {width}x{height}.")
        assert depth_snapshot.data is not None
        assert element_snapshot.data is not None
        assert layer_snapshot.data is not None
        depth = float(depth_snapshot.data[y, x])
        element_id = int(element_snapshot.data[y, x])
        layer_id = int(layer_snapshot.data[y, x])
        if (
            not np.isfinite(depth)
            or element_id == int(BACKGROUND_ID)
            or layer_id == int(BACKGROUND_ID)
        ):
            return None
        position = _unproject_pixel(
            pixel,
            (width, height),
            depth,
            depth_snapshot.world_to_camera,
            depth_snapshot.projection,
        )
        normal = _normal_at(self.snapshots.get((view, "normal")), pixel)
        attributes = _attributes_at(self._scene, layer_id, element_id)
        return PixelHit(
            view=view,
            pixel=pixel,
            layer_id=layer_id,
            element_id=element_id,
            depth=depth,
            world_position=(float(position[0]), float(position[1]), float(position[2])),
            normal=normal,
            attributes=attributes,
        )

    def region(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        view: str | None = None,
    ) -> RegionSummary:
        """Summarize IDs and coverage inside half-open pixel bounds."""
        return _region(self, x0, y0, x1, y1, view=view)

    def visible_elements(
        self, layer: str | int | None = None, *, view: str | None = None
    ) -> tuple[LayerVisibility, ...]:
        """Return visibility records for all layers or one ID/name selector."""
        return _visible_elements(self, layer, view=view)

    def attribute_extrema(
        self,
        attribute: str,
        *,
        layer: str | int | None = None,
        view: str | None = None,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[AttributeVisibility, ...]:
        """Summarize a numeric attribute over visible source elements.

        ``bounds`` optionally restricts the query to a half-open pixel region.
        Vector extrema are selected by magnitude while component-wise statistics
        and the original extremum samples are retained.
        """
        return _attribute_extrema(
            self, attribute, layer=layer, view=view, bounds=bounds
        )

    def occlusion_report(
        self, *, view: str | None = None, layer: str | int | None = None
    ) -> tuple[OcclusionRecord, ...]:
        """Report depth-ordered projected overlap for selected views or layers."""
        return _occlusion_report(self, view=view, layer=layer)

    def visual_evidence(self) -> dict[str, Any]:
        """Return compact framing, visibility, depth, and contrast evidence."""
        return _visual_evidence(self)

    def visual_diagnostics(
        self,
        *,
        min_occupancy: float = 0.02,
        max_occupancy: float = 0.95,
        max_clipped_fraction: float = 0.05,
        min_contrast: float = 0.08,
    ) -> tuple[Diagnostic, ...]:
        """Diagnose deterministic visual failures against explicit thresholds."""
        return _visual_diagnostics(
            self,
            min_occupancy=min_occupancy,
            max_occupancy=max_occupancy,
            max_clipped_fraction=max_clipped_fraction,
            min_contrast=min_contrast,
        )


@dataclass(slots=True)
class _CaptureResult:
    snapshots: dict[tuple[str, str], Snapshot]
    scene_summary: SceneSummary
    diagnostics: tuple[Diagnostic, ...]
    scene: Scene


def _scene_summary(scene: Scene, up_axis: Literal["y", "z"]) -> SceneSummary:
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    layers: list[LayerSummary] = []
    for index, view in enumerate(scene):
        assert view.data_frame is not None
        mesh = view.data_frame.mesh
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.size:
            transformed = (
                view.global_transform[:3, :3] @ vertices.T
            ).T + view.global_transform[:3, 3]
            current_min = np.min(transformed, axis=0)
            current_max = np.max(transformed, axis=0)
            minimum = (
                current_min if minimum is None else np.minimum(minimum, current_min)
            )
            maximum = (
                current_max if maximum is None else np.maximum(maximum, current_max)
            )
        layers.append(
            LayerSummary(
                id=index,
                name=view.name or f"Layer {index + 1}",
                mark=view.mark.name.lower() if view.mark is not None else "surface",
                vertex_count=int(mesh.num_vertices),
                facet_count=int(mesh.num_facets),
            )
        )
    if minimum is None or maximum is None:
        minimum = np.array([-1.0, -1.0, -1.0])
        maximum = np.array([1.0, 1.0, 1.0])
    center = (minimum + maximum) * 0.5
    radius = float(np.linalg.norm(maximum - minimum) * 0.5)
    if radius <= 1e-12:
        radius = 1.0
    return SceneSummary(
        bounds=(tuple(minimum.tolist()), tuple(maximum.tolist())),
        center=tuple(center.tolist()),
        radius=radius,
        up_axis=up_axis,
        layers=tuple(layers),
    )


def _camera_for_view(
    view: ViewPreset,
    summary: SceneSummary,
    resolution: tuple[int, int],
    fov: float = 35.0,
) -> CameraState:
    center = np.asarray(summary.center, dtype=np.float64)
    if summary.up_axis == "z":
        directions = {
            "front": np.array([0.0, -1.0, 0.0]),
            "back": np.array([0.0, 1.0, 0.0]),
            "left": np.array([-1.0, 0.0, 0.0]),
            "right": np.array([1.0, 0.0, 0.0]),
            "top": np.array([0.0, 0.0, 1.0]),
            "bottom": np.array([0.0, 0.0, -1.0]),
            "isometric": np.array([1.0, -1.0, 1.0]),
        }
        up = np.array([0.0, 0.0, 1.0])
        pole_up = np.array([0.0, 1.0, 0.0])
    else:
        directions = {
            "front": np.array([0.0, 0.0, 1.0]),
            "back": np.array([0.0, 0.0, -1.0]),
            "left": np.array([-1.0, 0.0, 0.0]),
            "right": np.array([1.0, 0.0, 0.0]),
            "top": np.array([0.0, 1.0, 0.0]),
            "bottom": np.array([0.0, -1.0, 0.0]),
            "isometric": np.array([1.0, 1.0, 1.0]),
        }
        up = np.array([0.0, 1.0, 0.0])
        pole_up = np.array([0.0, 0.0, -1.0])
    direction = directions[view]
    direction /= np.linalg.norm(direction)
    if view in {"top", "bottom"}:
        up = pole_up if view == "top" else -pole_up
    width, height = resolution
    aspect = width / height
    half_y = math.radians(fov) * 0.5
    half_x = math.atan(math.tan(half_y) * aspect)
    limiting = min(half_x, half_y)
    distance = summary.radius / max(math.sin(limiting), 1e-4) * 1.08
    eye = center + direction * distance
    near = max(1e-3, distance - summary.radius * 1.5)
    far = distance + summary.radius * 1.5
    return CameraState(
        eye=tuple(eye.tolist()),
        target=summary.center,
        up=tuple(up.tolist()),
        fov=fov,
        near=near,
        far=far,
    )


def _vec3(values) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _camera_state_from_figure(camera) -> CameraState:
    return CameraState(
        eye=_vec3(camera.eye),
        target=_vec3(camera.target),
        up=_vec3(camera.up),
        fov=float(getattr(camera, "fov", 35.0)),
        near=camera.near,
        far=camera.far,
        mode="orthographic"
        if isinstance(camera, OrthographicCamera)
        else "perspective",
        scale=camera.scale if isinstance(camera, OrthographicCamera) else None,
    )


def _camera_state_from_config(config: Config) -> CameraState:
    sensor = config.sensor
    return CameraState(
        eye=_vec3(sensor.location),
        target=_vec3(sensor.target),
        up=_vec3(sensor.up),
        fov=float(getattr(sensor, "fov", 35.0)),
        near=float(sensor.near_clip),
        far=float(sensor.far_clip),
        mode="orthographic" if isinstance(sensor, Orthographic) else "perspective",
        scale=sensor.scale if isinstance(sensor, Orthographic) else None,
    )


def _require_playwright():
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ObservationError(
            "snapshot()/observe() require Playwright. Install 'hakowan[observe]' "
            "and run 'playwright install chromium'."
        ) from exc
    return sync_playwright, PlaywrightError


def _decode_png(data_url: str) -> Image.Image:
    encoded = data_url.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    image.load()
    return image.convert("RGBA")


def _decode_samples(
    samples: dict[str, Any], key: str, dtype: Any, components: int = 1
) -> np.ndarray:
    raw = base64.b64decode(samples[key])
    shape: tuple[int, ...] = (samples["height"], samples["width"])
    if components > 1:
        shape += (components,)
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def _depth_from_image(
    image: Image.Image, depth_range: tuple[float, float]
) -> np.ndarray:
    encoded = np.asarray(image.convert("RGB"), dtype=np.float32)[..., 0] / 255.0
    normalized = _srgb_to_linear(encoded)
    result = depth_range[0] + normalized * (depth_range[1] - depth_range[0])
    result[normalized >= 1.0 - 1e-6] = np.nan
    return result.astype(np.float32)


def _normal_from_image(image: Image.Image, world_to_camera: np.ndarray) -> np.ndarray:
    encoded = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    view_normals = _srgb_to_linear(encoded) * 2.0 - 1.0
    background = np.all(encoded == 0.0, axis=2)
    world_rotation = np.linalg.inv(world_to_camera)[:3, :3]
    result = view_normals @ world_rotation.T
    lengths = np.linalg.norm(result, axis=2, keepdims=True)
    result /= np.maximum(lengths, 1e-12)
    result[background] = np.nan
    return result.astype(np.float32)


def _capture_sync(
    root: Layer,
    views: Sequence[str],
    passes: Sequence[PassName],
    resolution: tuple[int, int],
    *,
    cameras: dict[str, CameraState] | None,
    config: Config | None,
    background: Literal["light", "dark"],
    up_axis: Literal["y", "z"],
    timeout: float,
) -> _CaptureResult:
    if not views:
        raise ValueError("At least one view is required.")
    if not passes:
        raise ValueError("At least one pass is required.")
    width, height = resolution
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got {resolution}.")
    unknown_views = {
        view
        for view in views
        if view not in VIEW_PRESETS and (cameras is None or view not in cameras)
    }
    unknown_passes = set(passes) - set(PASS_NAMES)
    if unknown_views:
        raise ValueError(f"Unknown view preset(s): {sorted(unknown_views)}")
    if unknown_passes:
        raise ValueError(f"Unknown pass(es): {sorted(unknown_passes)}")

    report = validate(root, backend="webgl", strict=False)
    report.raise_for_errors()
    scene = compile(root, preserve_attributes=True)
    summary = _scene_summary(scene, up_axis)
    resolved_cameras = {
        view: cameras[view]
        if cameras is not None and view in cameras
        else _camera_for_view(cast(ViewPreset, view), summary, resolution)
        for view in views
    }

    render_config = copy.deepcopy(config) if config is not None else Config()
    render_config.film.width = width
    render_config.film.height = height
    first_camera = resolved_cameras[views[0]]
    render_config.sensor = Perspective(
        location=list(first_camera.eye),
        target=list(first_camera.target),
        up=list(first_camera.up),
        near_clip=first_camera.near,
        far_clip=first_camera.far,
        fov=first_camera.fov,
        fov_axis="y",
    )
    sync_playwright, playwright_error = _require_playwright()
    from .backends.webgl import WebGLBackend
    from .backends.webgl.assets import ensure_three_assets

    assets = ensure_three_assets("0.170.0")
    three_module_url = (assets / "build" / "three.module.js").resolve().as_uri()
    three_addons_url = (assets / "examples" / "jsm").resolve().as_uri() + "/"
    html = WebGLBackend().html_string(
        scene,
        render_config,
        background=background,
        title="hakowan-snapshot",
        three_module_url=three_module_url,
        three_addons_url=three_addons_url,
    )
    snapshots: dict[tuple[str, str], Snapshot] = {}
    with tempfile.TemporaryDirectory(prefix="hakowan-snapshot-") as directory:
        html_path = Path(directory) / "scene.html"
        html_path.write_text(html, encoding="utf-8")
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    args=["--allow-file-access-from-files"],
                )
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                )
                page = context.new_page()
                page.goto(
                    html_path.resolve().as_uri(),
                    wait_until="domcontentloaded",
                    timeout=int(timeout * 1000),
                )
                page.wait_for_function(
                    "window.__hakowan_loaded === true",
                    timeout=int(timeout * 1000),
                )
                page.evaluate(
                    "([w,h]) => window.__hakowan.prepareCapture(w,h)",
                    [width, height],
                )
                for view in views:
                    camera = resolved_cameras[view]
                    page.evaluate(
                        "state => window.__hakowan.setCameraState(state)",
                        camera.to_dict(),
                    )
                    needs_geometry = any(
                        item in {"element_id", "layer_id"} for item in passes
                    )
                    samples = (
                        page.evaluate(
                            "([w,h]) => window.__hakowan.sampleGeometry(w,h)",
                            [width, height],
                        )
                        if needs_geometry
                        else None
                    )
                    for pass_name in passes:
                        if pass_name in {"element_id", "layer_id"}:
                            data_url = page.evaluate(
                                "([p,w,h]) => window.__hakowan.captureIdPng(p,w,h)",
                                [pass_name, width, height],
                            )
                        else:
                            data_url = page.evaluate(
                                "p => window.__hakowan.capturePng(p)", pass_name
                            )
                        image = _decode_png(data_url)
                        state = page.evaluate("() => window.__hakowan.captureState()")
                        # Three.js Matrix4.toArray() is column-major.
                        world_to_camera = (
                            np.asarray(state["worldToCamera"], dtype=np.float64)
                            .reshape(4, 4)
                            .T
                        )
                        projection = (
                            np.asarray(state["projection"], dtype=np.float64)
                            .reshape(4, 4)
                            .T
                        )
                        depth_range = tuple(state["depthRange"])
                        if pass_name == "depth":
                            data = (
                                _decode_samples(samples, "depth", np.float32)
                                if samples is not None
                                else _depth_from_image(image, depth_range)
                            )
                        elif pass_name == "element_id":
                            assert samples is not None
                            data = _decode_samples(samples, "element", np.uint32)
                        elif pass_name == "layer_id":
                            assert samples is not None
                            data = _decode_samples(samples, "layer", np.uint32)
                        elif pass_name == "normal":
                            data = (
                                _decode_samples(
                                    samples, "normal", np.float32, components=3
                                )
                                if samples is not None
                                else _normal_from_image(image, world_to_camera)
                            )
                        elif pass_name == "albedo":
                            data = np.asarray(image.convert("RGB"), dtype=np.uint8)
                        else:
                            data = None
                        if pass_name == "beauty" and (
                            scene.legends or scene.annotations
                        ):
                            image = composite_overlays(
                                image, scene.legends, scene.annotations, background
                            )
                        snapshots[(view, pass_name)] = Snapshot(
                            image=image,
                            data=data,
                            view=view,
                            pass_name=pass_name,
                            camera=camera,
                            world_to_camera=world_to_camera,
                            projection=projection,
                            bounds=np.asarray(summary.bounds, dtype=np.float64),
                            depth_range=tuple(state["depthRange"])
                            if pass_name == "depth"
                            else None,
                            diagnostics=tuple(report.diagnostics),
                        )
                context.close()
                browser.close()
        except playwright_error as exc:
            raise ObservationError(
                "Chromium snapshot capture failed. Run 'playwright install chromium' "
                f"and verify the cached Three.js assets: {exc}"
            ) from exc
    return _CaptureResult(
        snapshots=snapshots,
        scene_summary=summary,
        diagnostics=tuple(report.diagnostics),
        scene=scene,
    )


def _capture(*args: Any, **kwargs: Any) -> _CaptureResult:
    """Run Playwright outside an active notebook event-loop thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _capture_sync(*args, **kwargs)
    call = functools.partial(_capture_sync, *args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(call).result()


def _capture_context(
    root: Layer | Figure,
    config: Config | None,
    resolution: tuple[int, int] | None,
    background: Literal["light", "dark"] | None,
) -> tuple[Layer, Config | None, tuple[int, int], Literal["light", "dark"], bool]:
    explicit_config = config is not None
    figure = root if isinstance(root, Figure) else None
    runtime_layer = cast(Layer, figure.layer if figure is not None else root)
    if figure is not None and config is None:
        config = figure.to_config()
    if resolution is None:
        if explicit_config and config is not None:
            resolution = (config.film.width, config.film.height)
        elif figure is not None and figure.scene.output is not None:
            resolution = (figure.scene.output.width, figure.scene.output.height)
        else:
            resolution = (512, 512)
    if background is None:
        if (
            figure is not None
            and not explicit_config
            and figure.scene.output is not None
        ):
            background = figure.scene.output.background
        else:
            background = "dark"
    return runtime_layer, config, resolution, background, explicit_config


def snapshot(
    root: Layer | Figure,
    *,
    view: ViewPreset | None = None,
    pass_name: PassName = "beauty",
    resolution: tuple[int, int] | None = None,
    camera: CameraState | None = None,
    config: Config | None = None,
    backend: BackendName = "webgl",
    background: Literal["light", "dark"] | None = None,
    up_axis: Literal["y", "z"] = "y",
    filename: str | Path | None = None,
    timeout: float = 60.0,
) -> Snapshot:
    """Capture one deterministic WebGL raster and optional raw pass array.

    A Figure camera is used when neither ``view`` nor ``camera`` is supplied;
    otherwise named presets frame the compiled scene. Explicit ``config``,
    ``resolution``, and ``background`` override Figure intent.

    Args:
        root: Layer or Figure to capture.
        view: Named camera preset and result label.
        pass_name: Beauty, albedo, depth, normal, element-ID, or layer-ID pass.
        resolution: Output width and height in pixels.
        camera: Explicit camera overriding the named preset.
        config: Explicit invocation configuration.
        backend: Capture backend; currently only ``webgl`` is supported.
        background: Light or dark studio background.
        up_axis: Coordinate convention used by named camera presets.
        filename: Optional PNG path; raw pass data is saved beside it as NPY.
        timeout: Chromium capture timeout in seconds.

    Returns:
        Snapshot containing the image, raw data, camera, matrices, and metadata.

    """
    if backend != "webgl":
        raise NotImplementedError("snapshot() currently supports backend='webgl'.")
    figure = root if isinstance(root, Figure) else None
    runtime_layer, resolved_config, resolution, background, explicit_config = (
        _capture_context(root, config, resolution, background)
    )
    label: str = view or "isometric"
    if camera is None and view is None:
        if (
            figure is not None
            and not explicit_config
            and figure.scene.camera is not None
        ):
            camera = _camera_state_from_figure(figure.scene.camera)
            label = "figure"
        elif explicit_config and resolved_config is not None:
            camera = _camera_state_from_config(resolved_config)
            label = "config"
    result = _capture(
        runtime_layer,
        [label],
        [pass_name],
        resolution,
        cameras={label: camera} if camera is not None else None,
        config=resolved_config,
        up_axis=up_axis,
        background=background,
        timeout=timeout,
    )
    item = result.snapshots[(label, pass_name)]
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        item.image.save(path)
        if item.data is not None:
            data_path = path.with_suffix(".npy")
            np.save(data_path, item.data, allow_pickle=False)
            object.__setattr__(item, "data_path", data_path)
        object.__setattr__(item, "path", path)
    return item


def observe(
    root: Layer | Figure,
    *,
    views: Sequence[str] | None = None,
    passes: Sequence[PassName] | None = None,
    resolution: tuple[int, int] | None = None,
    cameras: dict[str, CameraState] | None = None,
    config: Config | None = None,
    backend: BackendName = "webgl",
    background: Literal["light", "dark"] | None = None,
    up_axis: Literal["y", "z"] = "y",
    output_dir: str | Path | None = None,
    timeout: float = 60.0,
) -> Observation:
    """Capture a deterministic multi-view, multi-pass inspection bundle.

    Defaults to front, right, top, and isometric views with beauty, depth, and
    normal passes. Figure output passes are inherited when present. Request
    depth plus both ID passes to enable picking and structured visibility queries.

    Args:
        root: Layer or Figure to inspect.
        views: Ordered named view presets or labels with explicit camera overrides.
        passes: Ordered semantic pass names.
        resolution: Width and height shared by every capture.
        cameras: Explicit CameraState overrides keyed by view label.
        config: Explicit invocation configuration.
        backend: Capture backend; currently only ``webgl`` is supported.
        background: Light or dark studio background.
        up_axis: Coordinate convention used by named camera presets.
        output_dir: Optional directory for PNG, NPY, contact-sheet, and manifest files.
        timeout: Chromium capture timeout in seconds.

    Returns:
        Observation containing snapshots, structured scene evidence, and manifest.

    """
    if backend != "webgl":
        raise NotImplementedError("observe() currently supports backend='webgl'.")
    figure = root if isinstance(root, Figure) else None
    runtime_layer, resolved_config, resolution, background, explicit_config = (
        _capture_context(root, config, resolution, background)
    )
    if views is None:
        if (
            figure is not None
            and not explicit_config
            and figure.scene.camera is not None
        ):
            views = ("figure",)
            cameras = {
                **(cameras or {}),
                "figure": _camera_state_from_figure(figure.scene.camera),
            }
        else:
            views = ("front", "right", "top", "isometric")
    if passes is None:
        if (
            figure is not None
            and not explicit_config
            and figure.scene.output is not None
        ):
            passes = tuple(
                "element_id" if item == "facet_id" else item
                for item in figure.scene.output.passes
            )
        else:
            passes = ("beauty", "depth", "normal")
    assert views is not None
    assert passes is not None
    result = _capture(
        runtime_layer,
        views,
        passes,
        resolution,
        cameras=cameras,
        config=resolved_config,
        background=background,
        up_axis=up_axis,
        timeout=timeout,
    )
    contact_sheet = _contact_sheet(result.snapshots, views, passes)
    manifest = {
        "version": "1.0",
        "backend": "webgl",
        "scene": result.scene_summary.to_dict(),
        "legends": [legend.to_dict() for legend in result.scene.legends],
        "annotations": [
            annotation.to_dict() for annotation in result.scene.annotations
        ],
        "snapshots": [
            result.snapshots[(view, pass_name)].to_manifest()
            for view in views
            for pass_name in passes
        ],
        "diagnostics": [item.to_dict() for item in result.diagnostics],
    }
    observation = Observation(
        snapshots=result.snapshots,
        scene_summary=result.scene_summary,
        diagnostics=result.diagnostics,
        contact_sheet=contact_sheet,
        manifest=manifest,
        _scene=result.scene,
    )
    manifest.update(_query_manifest(observation))
    if output_dir is not None:
        observation.save(output_dir)
    return observation


def _contact_sheet(
    snapshots: dict[tuple[str, str], Snapshot],
    views: Sequence[str],
    passes: Sequence[str],
) -> Image.Image:
    images = [item.image for item in snapshots.values()]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    header_height = 28
    label_width = max(80, max(len(view) for view in views) * 7 + 12)
    cell_width = max(width, max(len(name) for name in passes) * 7 + 12)
    gutter = 2
    sheet = Image.new(
        "RGB",
        (
            label_width + len(passes) * (cell_width + gutter),
            header_height + len(views) * (height + gutter),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column, pass_name in enumerate(passes):
        x = label_width + column * (cell_width + gutter)
        draw.text((x + 4, 7), pass_name, fill="black")
    for row, view in enumerate(views):
        y = header_height + row * (height + gutter)
        draw.text((4, y + 4), view, fill="black")
        for column, pass_name in enumerate(passes):
            x = label_width + column * (cell_width + gutter)
            image_x = x + (cell_width - width) // 2
            sheet.paste(snapshots[(view, pass_name)].image.convert("RGB"), (image_x, y))
    return sheet


def _unproject_pixel(
    pixel: tuple[int, int],
    resolution: tuple[int, int],
    depth: float,
    world_to_camera: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    x, y = pixel
    width, height = resolution
    ndc = np.array(
        [
            2.0 * (x + 0.5) / width - 1.0,
            1.0 - 2.0 * (y + 0.5) / height,
            -1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    camera_point = np.linalg.inv(projection) @ ndc
    camera_point /= camera_point[3]
    direction = camera_point[:3]
    direction /= np.linalg.norm(direction)
    camera_position = direction * (depth / max(-direction[2], 1e-12))
    world = np.linalg.inv(world_to_camera) @ np.array([*camera_position, 1.0])
    return world[:3] / world[3]


def _normal_at(
    normal_snapshot: Snapshot | None, pixel: tuple[int, int]
) -> tuple[float, float, float] | None:
    if normal_snapshot is None or normal_snapshot.data is None:
        return None
    x, y = pixel
    value = np.asarray(normal_snapshot.data[y, x], dtype=np.float64)
    length = np.linalg.norm(value)
    if length < 1e-8:
        return None
    value /= length
    return float(value[0]), float(value[1]), float(value[2])


def _attributes_at(
    scene: Scene | None, layer_id: int, element_id: int
) -> dict[str, Any]:
    if scene is None or not (0 <= layer_id < len(scene)):
        return {}
    view = scene[layer_id]
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    result: dict[str, Any] = {}
    for attribute_id in mesh.get_matching_attribute_ids():
        name = mesh.get_attribute_name(attribute_id)
        if mesh.is_attribute_indexed(name):
            continue
        attribute = mesh.attribute(name)
        values = np.asarray(attribute.data)
        value: Any | None = None
        if (
            view.mark is Mark.Point
            and attribute.element_type == lagrange.AttributeElement.Vertex
        ):
            if element_id < len(values):
                value = values[element_id]
        elif view.mark is Mark.Surface:
            if (
                attribute.element_type == lagrange.AttributeElement.Facet
                and element_id < len(values)
            ):
                value = values[element_id]
            elif (
                attribute.element_type == lagrange.AttributeElement.Vertex
                and mesh.is_triangle_mesh
                and element_id < mesh.num_facets
            ):
                value = np.mean(values[np.asarray(mesh.facets[element_id])], axis=0)
        if value is not None:
            array = np.asarray(value)
            result[name] = array.item() if array.ndim == 0 else array.tolist()
    return result


__all__ = [
    "AttributeVisibility",
    "BACKGROUND_ID",
    "CameraState",
    "LayerSummary",
    "LayerVisibility",
    "Observation",
    "ObservationError",
    "OcclusionRecord",
    "PASS_NAMES",
    "PixelHit",
    "SceneSummary",
    "RegionSummary",
    "Snapshot",
    "VIEW_PRESETS",
    "observe",
    "snapshot",
]
