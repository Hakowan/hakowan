"""Provider-neutral, JSON-safe operations exposed by the Hakowan MCP server."""

from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import os
import re
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from threading import RLock
from pathlib import Path
from typing import Any, TypeAlias, cast

from pydantic import ValidationError as PydanticValidationError

from ..backends import BackendName, list_backend_capabilities
from ..compiler import compile as compile_layer
from ..grammar.figure import Figure, OrthographicCamera, ThinLensCamera
from ..grammar.layer import Layer
from ..inspection import inspect as inspect_data_source
from ..observation import observe
from ..render import render
from ..spec import FigureSpec, PatchError, from_spec, json_schema, patch_spec
from ..validation import validate
from .catalog import fragment_names, schema_fragment, spec_template, template_catalog


_DEFAULT_GALLERY_URL = "https://hakowan.github.io/hakowan-gallery/agent/v1/index.json"
_MAX_REMOTE_DOCUMENT_BYTES = 8 * 1024 * 1024
_DEFAULT_VISUAL_CRITERIA = {
    "min_occupancy": 0.02,
    "max_occupancy": 0.95,
    "max_clipped_fraction": 0.05,
    "min_contrast": 0.08,
}
_MAX_STORED_SPECS = 128
SpecInput: TypeAlias = dict[str, Any] | str


@dataclass(frozen=True, slots=True)
class _RemoteDocument:
    payload: bytes
    etag: str | None
    last_modified: str | None


@dataclass(frozen=True, slots=True)
class PathPolicy:
    """Restrict all agent-controlled reads and writes to one workspace root."""

    root: Path

    @classmethod
    def create(cls, root: str | Path | None = None) -> "PathPolicy":
        """Resolve a workspace policy from ``root`` or the current directory."""
        return cls(Path(root or Path.cwd()).expanduser().resolve())

    def resolve(self, value: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve a path and reject traversal or symlinks outside the root."""
        path = Path(value).expanduser()
        resolved = (
            (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        )
        if not resolved.is_relative_to(self.root):
            raise PermissionError(
                f"Path '{value}' is outside the configured workspace root '{self.root}'."
            )
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        return resolved

    def display(self, path: Path) -> str:
        """Return a stable workspace-relative path when possible."""
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path.resolve())


class HakowanMCPService:
    """Implement deterministic Hakowan tool behavior independently of MCP transport."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        gallery: str | Path | None = None,
        gallery_url: str | None = None,
    ) -> None:
        self.paths = PathPolicy.create(root)
        self.gallery = self._find_gallery(gallery)
        configured_url = (
            gallery_url
            if gallery_url is not None
            else os.environ.get("HAKOWAN_GALLERY_URL", _DEFAULT_GALLERY_URL)
        )
        self.gallery_url = configured_url.strip() or None
        self._remote_documents: dict[str, _RemoteDocument] = {}
        self._specs: OrderedDict[str, FigureSpec] = OrderedDict()
        self._spec_lock = RLock()

    def _find_gallery(self, gallery: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if gallery is not None:
            candidates.append(Path(gallery))
        if os.environ.get("HAKOWAN_GALLERY"):
            candidates.append(Path(os.environ["HAKOWAN_GALLERY"]))
        candidates.append(self.paths.root.parent / "hakowan-gallery")
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
                if (resolved / "gallery").is_dir():
                    return resolved
            except OSError:
                continue
        return None

    @staticmethod
    def _error(code: str, error: Exception, **extra: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": code,
                "type": type(error).__name__,
                "message": str(error),
                **extra,
            },
        }

    def get_backends(self) -> dict[str, Any]:
        """Return declared features and limitations for every rendering backend."""
        return {
            "ok": True,
            "backends": {
                name: capabilities.to_dict()
                for name, capabilities in list_backend_capabilities().items()
            },
        }

    @staticmethod
    def _schema_path(document: Any, location: tuple[Any, ...]) -> str:
        tokens: list[str] = []
        current = document
        for raw_token in location:
            token = str(raw_token)
            if isinstance(current, dict) and raw_token in current:
                tokens.append(token)
                current = current[raw_token]
            elif isinstance(current, list) and isinstance(raw_token, int):
                tokens.append(token)
                current = current[raw_token] if raw_token < len(current) else None
            elif isinstance(current, dict) and current.get("kind") == raw_token:
                continue
            elif "[" in token or token.endswith("Spec"):
                continue
            else:
                tokens.append(token)
                current = None
        return "".join(
            f"/{token.replace('~', '~0').replace('/', '~1')}" for token in tokens
        )

    @classmethod
    def _schema_error(
        cls, error: PydanticValidationError, document: dict[str, Any]
    ) -> dict[str, Any]:
        diagnostics = []
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        ):
            path = cls._schema_path(document, item["loc"])
            diagnostic: dict[str, Any] = {
                "code": f"schema.{item['type']}",
                "path": path,
                "message": item["msg"],
            }
            if item["type"] == "extra_forbidden":
                diagnostic["suggested_patch"] = {"op": "remove", "path": path}
            diagnostics.append(diagnostic)
        return {
            "ok": False,
            "error": {
                "code": "validate.schema",
                "type": type(error).__name__,
                "message": "FigureSpec schema validation failed.",
                "diagnostics": diagnostics,
            },
        }

    def get_schema(self, fragment: str | None = None) -> dict[str, Any]:
        """Return the full FigureSpec schema or one self-contained fragment."""
        if fragment is None:
            return {
                "ok": True,
                "schema": json_schema(),
                "available_fragments": list(fragment_names()),
            }
        try:
            return {
                "ok": True,
                "fragment": fragment,
                "schema": schema_fragment(fragment),
            }
        except KeyError as exc:
            return self._error(
                "schema.fragment_unknown",
                exc,
                fragment=fragment,
                available_fragments=list(fragment_names()),
            )

    def get_spec_template(
        self,
        name: str | None = None,
        data_id: str = "data",
        attribute: str = "value",
    ) -> dict[str, Any]:
        """List or return minimal canonical FigureSpec templates."""
        if name is None:
            return {"ok": True, "templates": template_catalog()}
        try:
            return {
                "ok": True,
                "name": name,
                "spec": spec_template(name, data_id=data_id, attribute=attribute),
            }
        except KeyError as exc:
            return self._error(
                "template.unknown",
                exc,
                name=name,
                available_templates=[item["name"] for item in template_catalog()],
            )
        except ValueError as exc:
            return self._error("template.invalid", exc, name=name)

    def inspect_data(
        self, source: str, positions: list[str] | None = None
    ) -> dict[str, Any]:
        """Inspect a workspace data file and return geometry and attribute facts."""
        try:
            path = self.paths.resolve(source, must_exist=True)
            summary = inspect_data_source(path, positions=positions)
            return {
                "ok": True,
                "source": self.paths.display(path),
                "summary": summary.to_dict(),
            }
        except Exception as exc:
            return self._error("inspect.failed", exc, source=source)

    def _store_spec(self, spec: FigureSpec) -> str:
        digest = hashlib.sha256(
            spec.to_json(canonical=True).encode("utf-8")
        ).hexdigest()
        spec_id = f"sha256:{digest}"
        with self._spec_lock:
            self._specs[spec_id] = spec
            self._specs.move_to_end(spec_id)
            while len(self._specs) > _MAX_STORED_SPECS:
                self._specs.popitem(last=False)
        return spec_id

    def _parse_spec(self, spec: SpecInput) -> FigureSpec:
        if isinstance(spec, str):
            with self._spec_lock:
                try:
                    parsed = self._specs[spec]
                except KeyError as exc:
                    raise KeyError(f"Unknown or expired spec handle: {spec}") from exc
                self._specs.move_to_end(spec)
        else:
            parsed = FigureSpec.model_validate(spec)
        self._check_resource_paths(parsed.to_dict())
        return parsed

    def get_spec(self, spec_id: str) -> dict[str, Any]:
        """Return canonical JSON for a specification stored in this server session."""
        try:
            spec = self._parse_spec(spec_id)
            return {"ok": True, "spec_id": spec_id, "spec": spec.to_dict()}
        except Exception as exc:
            return self._error("spec.not_found", exc, spec_id=spec_id)

    def _check_resource_paths(self, spec: dict[str, Any]) -> None:
        def visit(value: Any, parent: str | None = None) -> None:
            if isinstance(value, dict):
                kind = value.get("kind")
                if kind == "mesh_file" and isinstance(value.get("path"), str):
                    self.paths.resolve(value["path"], must_exist=True)
                if kind == "image" and isinstance(value.get("filename"), str):
                    self.paths.resolve(value["filename"], must_exist=True)
                if parent == "environment" and isinstance(value.get("path"), str):
                    self.paths.resolve(value["path"], must_exist=True)
                for key, child in value.items():
                    visit(child, key)
            elif isinstance(value, list):
                for child in value:
                    visit(child, parent)

        visit(spec)

    def _data_resolver(self, bindings: dict[str, str] | None):
        resolved = {
            identifier: self.paths.resolve(path, must_exist=True)
            for identifier, path in (bindings or {}).items()
        }

        def resolve(identifier: str):
            try:
                return resolved[identifier]
            except KeyError as exc:
                raise KeyError(
                    f"External data ID '{identifier}' has no data_bindings entry."
                ) from exc

        return resolve

    def _runtime(
        self, spec: SpecInput, data_bindings: dict[str, str] | None = None
    ) -> tuple[FigureSpec, Layer | Figure]:
        parsed = self._parse_spec(spec)
        runtime = from_spec(
            parsed,
            data_resolver=self._data_resolver(data_bindings),
            base_dir=self.paths.root,
        )
        return parsed, runtime

    def validate_spec(
        self,
        spec: SpecInput,
        backend: str = "webgl",
        strict: bool = True,
        data_bindings: dict[str, str] | None = None,
        compile_check: bool = True,
    ) -> dict[str, Any]:
        """Validate schema, resources, semantics, backend support, and compilation."""
        try:
            parsed, runtime = self._runtime(spec, data_bindings)
            report = validate(
                runtime,
                backend=cast(BackendName, backend),
                strict=strict,
                compile_check=compile_check,
            )
            spec_id = self._store_spec(parsed)
            return {
                "ok": True,
                "spec_id": spec_id,
                "spec": parsed.to_dict(),
                "validation": report.to_dict(),
            }
        except PydanticValidationError as exc:
            return self._schema_error(exc, spec if isinstance(spec, dict) else {})
        except Exception as exc:
            return self._error("validate.failed", exc)

    def compile_spec(
        self,
        spec: SpecInput,
        data_bindings: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compile a specification and summarize resolved views without rendering."""
        try:
            parsed, runtime = self._runtime(spec, data_bindings)
            report = validate(runtime, backend="webgl", strict=False)
            if not report.valid:
                return {"ok": False, "validation": report.to_dict()}
            scene = compile_layer(runtime, preserve_attributes=True)
            views = []
            for index, view in enumerate(scene):
                assert view.data_frame is not None
                mesh = view.data_frame.mesh
                bounds = view.bbox.tolist() if view.bbox is not None else None
                views.append(
                    {
                        "id": index,
                        "name": view.name or f"Layer {index + 1}",
                        "mark": view.mark.name.lower()
                        if view.mark is not None
                        else None,
                        "vertex_count": int(mesh.num_vertices),
                        "facet_count": int(mesh.num_facets),
                        "bounds": bounds,
                    }
                )
            return {
                "ok": True,
                "spec_id": self._store_spec(parsed),
                "validation": report.to_dict(),
                "scene": {
                    "views": views,
                    "legends": [item.to_dict() for item in scene.legends],
                    "annotations": [item.to_dict() for item in scene.annotations],
                },
            }
        except Exception as exc:
            return self._error("compile.failed", exc)

    def fit_camera(
        self,
        spec: SpecInput,
        data_bindings: dict[str, str] | None = None,
        backend: str = "webgl",
        direction: str | list[float] = "isometric",
        projection: str = "perspective",
        margin: float = 0.08,
        resolution: list[int] | None = None,
        fov: float = 35.0,
        fov_axis: str = "smaller",
        up_axis: str = "y",
    ) -> dict[str, Any]:
        """Fit a concrete camera to resolved scene geometry and return its patch."""
        try:
            parsed, runtime = self._runtime(spec, data_bindings)
            figure = runtime if isinstance(runtime, Figure) else Figure(runtime)
            size = tuple(resolution or [1024, 800])
            if len(size) != 2:
                raise ValueError("resolution must contain width and height")
            fitted = figure.camera(
                "fit",
                direction=direction,
                projection=projection,
                margin=margin,
                resolution=(int(size[0]), int(size[1])),
                fov=fov,
                fov_axis=fov_axis,
                up_axis=up_axis,
            )
            camera = fitted.scene.camera
            assert camera is not None
            camera_value = asdict(camera)
            camera_value["kind"] = (
                "orthographic"
                if isinstance(camera, OrthographicCamera)
                else "thin_lens"
                if isinstance(camera, ThinLensCamera)
                else "perspective"
            )
            payload = parsed.to_dict()
            operations: list[dict[str, Any]] = (
                [
                    {
                        "op": "replace",
                        "path": "/scene",
                        "value": {"camera": camera_value},
                    }
                ]
                if payload["scene"] is None
                else [
                    {
                        "op": "replace",
                        "path": "/scene/camera",
                        "value": camera_value,
                    }
                ]
            )
            patched = patch_spec(parsed, operations)
            canonical = patched.to_dict()
            canonical_camera = canonical["scene"]["camera"]
            operations[0]["value"] = (
                {"camera": canonical_camera}
                if operations[0]["path"] == "/scene"
                else canonical_camera
            )
            _, patched_runtime = self._runtime(canonical, data_bindings)
            report = validate(
                patched_runtime,
                backend=cast(BackendName, backend),
                strict=True,
                compile_check=True,
            )
            return {
                "ok": report.valid,
                "spec_id": self._store_spec(patched),
                "camera": canonical_camera,
                "patch": operations,
                "spec": canonical,
                "validation": report.to_dict(),
            }
        except Exception as exc:
            return self._error("camera.fit_failed", exc)

    def render_spec(
        self,
        spec: SpecInput,
        output: str,
        backend: str = "webgl",
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
        offline: bool = False,
    ) -> dict[str, Any]:
        """Validate and render a specification to a workspace output path."""
        try:
            parsed, runtime = self._runtime(spec, data_bindings)
            report = validate(
                runtime, backend=cast(BackendName, backend), strict=strict
            )
            if not report.valid:
                return {"ok": False, "validation": report.to_dict()}
            output_path = self.paths.resolve(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if offline and backend == "webgl":
                result = render(
                    runtime,
                    filename=output_path,
                    backend=cast(BackendName, backend),
                    offline=True,
                )
            else:
                result = render(
                    runtime,
                    filename=output_path,
                    backend=cast(BackendName, backend),
                )
            return {
                "ok": True,
                "spec_id": self._store_spec(parsed),
                "validation": report.to_dict(),
                "render": {
                    "backend": result.backend,
                    "path": self.paths.display(result.path) if result.path else None,
                    "outputs": {
                        name: self.paths.display(value)
                        if isinstance(value, Path)
                        else value
                        for name, value in result.outputs.items()
                    },
                },
            }
        except Exception as exc:
            return self._error("render.failed", exc, output=output)

    def _visual_criteria(self, values: dict[str, float] | None) -> dict[str, float]:
        criteria = dict(_DEFAULT_VISUAL_CRITERIA)
        if values is not None:
            unknown = set(values) - set(criteria)
            if unknown:
                raise ValueError(f"Unknown visual criteria: {sorted(unknown)}")
            criteria.update({key: float(value) for key, value in values.items()})
        if not 0.0 <= criteria["min_occupancy"] <= criteria["max_occupancy"] <= 1.0:
            raise ValueError("occupancy criteria must satisfy 0 <= min <= max <= 1")
        for key in ("max_clipped_fraction", "min_contrast"):
            if not 0.0 <= criteria[key] <= 1.0:
                raise ValueError(f"{key} must be in [0, 1]")
        return criteria

    def _client_observation_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(manifest)
        for snapshot in result.get("snapshots", []):
            for key in ("path", "data_path"):
                if snapshot.get(key):
                    snapshot[key] = self.paths.display(Path(snapshot[key]))
        if result.get("contact_sheet"):
            result["contact_sheet"] = self.paths.display(Path(result["contact_sheet"]))
        metadata = result.get("contact_sheet_metadata")
        if metadata and metadata.get("path"):
            metadata["path"] = self.paths.display(Path(metadata["path"]))
        return result

    @staticmethod
    def _visual_quality(
        manifest: dict[str, Any], criteria: dict[str, float]
    ) -> dict[str, float]:
        diagnostics = manifest.get("visual_diagnostics", [])
        views = list(manifest.get("evidence", {}).get("views", {}).values())
        midpoint = (criteria["min_occupancy"] + criteria["max_occupancy"]) * 0.5
        return {
            "error_count": float(
                sum(item.get("severity") == "error" for item in diagnostics)
            ),
            "warning_count": float(
                sum(item.get("severity") == "warning" for item in diagnostics)
            ),
            "occupancy_distance": sum(
                abs(float(view["occupancy"]) - midpoint) for view in views
            ),
            "clipped_fraction": sum(
                float(layer["projection"]["clipped_fraction"])
                for view in views
                for layer in view["layers"]
            ),
            "hidden_layer_count": float(
                sum(
                    layer["visible_pixel_count"] == 0 and not layer["empty"]
                    for view in views
                    for layer in view["layers"]
                )
            ),
            "contrast_deficit": sum(
                max(
                    criteria["min_contrast"]
                    - float(view["contrast"]["luminance_contrast"]),
                    0.0,
                )
                for view in views
                if view["contrast"] is not None
            ),
        }

    def observe_spec(
        self,
        spec: SpecInput,
        output_dir: str,
        views: list[str] | None = None,
        passes: list[str] | None = None,
        resolution: list[int] | None = None,
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
        visual_criteria: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Capture WebGL evidence, artifact metadata, and visual diagnostics."""
        try:
            parsed, runtime = self._runtime(spec, data_bindings)
            report = validate(runtime, backend="webgl", strict=strict)
            if not report.valid:
                return {"ok": False, "validation": report.to_dict()}
            directory = self.paths.resolve(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            selected_passes = passes or [
                "beauty",
                "depth",
                "normal",
                "element_id",
                "layer_id",
            ]
            size = tuple(resolution or [512, 512])
            if len(size) != 2:
                raise ValueError("resolution must contain width and height")
            criteria = self._visual_criteria(visual_criteria)
            observation = observe(
                runtime,
                views=cast(Any, views),
                passes=cast(Any, selected_passes),
                resolution=(int(size[0]), int(size[1])),
                output_dir=directory,
            )
            if visual_criteria is not None:
                observation.manifest["visual_diagnostics"] = [
                    item.to_dict()
                    for item in observation.visual_diagnostics(**criteria)
                ]
                (directory / "manifest.json").write_text(
                    json.dumps(observation.manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            manifest = self._client_observation_manifest(observation.manifest)
            return {
                "ok": True,
                "spec_id": self._store_spec(parsed),
                "validation": report.to_dict(),
                "output_dir": self.paths.display(directory),
                "evidence": manifest["evidence"],
                "visual_diagnostics": manifest["visual_diagnostics"],
                "manifest": manifest,
            }
        except Exception as exc:
            return self._error("observe.failed", exc, output_dir=output_dir)

    def evaluate_visual_patch(
        self,
        spec: SpecInput,
        operations: list[dict[str, Any]],
        output_dir: str,
        views: list[str] | None = None,
        resolution: list[int] | None = None,
        data_bindings: dict[str, str] | None = None,
        visual_criteria: dict[str, float] | None = None,
        max_operations: int = 3,
    ) -> dict[str, Any]:
        """Accept a visual patch only when measured evidence improves safely."""
        try:
            original_model = self._parse_spec(spec)
            original = original_model.to_dict()
            original_id = self._store_spec(original_model)
            if not operations:
                raise ValueError("At least one patch operation is required")
            if max_operations < 1 or len(operations) > max_operations:
                raise ValueError(
                    f"Visual patch has {len(operations)} operations; limit is {max_operations}."
                )
            criteria = self._visual_criteria(visual_criteria)
            root = self.paths.resolve(output_dir)
            before = self.observe_spec(
                original,
                self.paths.display(root / "before"),
                views=views,
                resolution=resolution,
                data_bindings=data_bindings,
                visual_criteria=criteria,
            )
            if not before["ok"]:
                return self._error(
                    "visual.baseline_failed",
                    ValueError("Baseline observation failed."),
                    baseline=before,
                )
            patched = self.apply_patch(
                original,
                operations,
                data_bindings=data_bindings,
                strict=True,
                semantic=True,
            )
            if not patched["ok"]:
                return {
                    "ok": True,
                    "accepted": False,
                    "rolled_back": True,
                    "reason": "candidate_validation_failed",
                    "spec": original,
                    "spec_id": original_id,
                    "patch": operations,
                    "before": before,
                    "candidate_validation": patched,
                }
            after = self.observe_spec(
                patched["spec"],
                self.paths.display(root / "after"),
                views=views,
                resolution=resolution,
                data_bindings=data_bindings,
                visual_criteria=criteria,
            )
            if not after["ok"]:
                return {
                    "ok": True,
                    "accepted": False,
                    "rolled_back": True,
                    "reason": "candidate_observation_failed",
                    "spec": original,
                    "spec_id": original_id,
                    "candidate_spec_id": after.get("spec_id", patched.get("spec_id")),
                    "candidate_spec": patched["spec"],
                    "patch": operations,
                    "before": before,
                    "after": after,
                }
            before_quality = self._visual_quality(before["manifest"], criteria)
            after_quality = self._visual_quality(after["manifest"], criteria)
            tolerance = 1e-9
            regressions = [
                key
                for key in before_quality
                if after_quality[key] > before_quality[key] + tolerance
            ]
            improvements = [
                key
                for key in before_quality
                if after_quality[key] < before_quality[key] - tolerance
            ]
            accepted = not regressions and bool(improvements)
            selected = FigureSpec.model_validate(
                patched["spec"] if accepted else original
            )
            return {
                "ok": True,
                "accepted": accepted,
                "rolled_back": not accepted,
                "reason": (
                    "measurable_improvement"
                    if accepted
                    else "visual_regression"
                    if regressions
                    else "no_measurable_improvement"
                ),
                "spec": selected.to_dict(),
                "spec_id": self._store_spec(selected),
                "candidate_spec": patched["spec"],
                "candidate_spec_id": after.get("spec_id", patched.get("spec_id")),
                "patch": operations,
                "comparison": {
                    "before": before_quality,
                    "after": after_quality,
                    "improvements": improvements,
                    "regressions": regressions,
                },
                "before": before,
                "after": after,
            }
        except Exception as exc:
            return self._error("visual.patch_failed", exc, output_dir=output_dir)

    def apply_patch(
        self,
        spec: SpecInput,
        operations: list[dict[str, Any]],
        backend: str = "webgl",
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
        semantic: bool = True,
    ) -> dict[str, Any]:
        """Atomically patch raw JSON, then validate its schema and semantics."""
        try:
            source: FigureSpec | dict[str, Any] = (
                self._parse_spec(spec) if isinstance(spec, str) else spec
            )
            patched = patch_spec(source, operations)
            self._check_resource_paths(patched.to_dict())
            result: dict[str, Any] = {
                "ok": True,
                "spec_id": self._store_spec(patched),
                "spec": patched.to_dict(),
            }
            if semantic:
                runtime = from_spec(
                    patched,
                    data_resolver=self._data_resolver(data_bindings),
                    base_dir=self.paths.root,
                )
                report = validate(
                    runtime, backend=cast(BackendName, backend), strict=strict
                )
                result["validation"] = report.to_dict()
                result["ok"] = report.valid
            return result
        except PatchError as exc:
            return {
                "ok": False,
                "error": {
                    "code": exc.failure.code,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "diagnostics": [exc.failure.to_dict()],
                },
            }
        except Exception as exc:
            return self._error("patch.failed", exc)

    def _fetch_remote_json(
        self,
        url: str,
        *,
        expected_sha256: str | None = None,
    ) -> Any:
        cached = self._remote_documents.get(url)
        if cached is not None and expected_sha256 is not None:
            if hashlib.sha256(cached.payload).hexdigest() == expected_sha256:
                return json.loads(cached.payload)

        headers = {
            "Accept": "application/json",
            "User-Agent": "hakowan-mcp",
        }
        if cached is not None:
            if cached.etag:
                headers["If-None-Match"] = cached.etag
            if cached.last_modified:
                headers["If-Modified-Since"] = cached.last_modified
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = response.read(_MAX_REMOTE_DOCUMENT_BYTES + 1)
                if len(payload) > _MAX_REMOTE_DOCUMENT_BYTES:
                    raise ValueError("Remote gallery document exceeds 8 MiB.")
                document = _RemoteDocument(
                    payload=payload,
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                )
                self._remote_documents[url] = document
        except urllib.error.HTTPError as exc:
            if exc.code != 304 or cached is None:
                raise
            document = cached
        except (OSError, urllib.error.URLError):
            if cached is None:
                raise
            document = cached
        if expected_sha256 is not None:
            actual_sha256 = hashlib.sha256(document.payload).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(f"Remote gallery digest mismatch for '{url}'.")
        return json.loads(document.payload)

    @staticmethod
    def _remote_artifact_url(index_url: str, relative: str) -> str:
        parsed = urllib.parse.urlparse(relative)
        if (
            parsed.scheme
            or parsed.netloc
            or relative.startswith("/")
            or ".." in parsed.path.split("/")
        ):
            raise ValueError(f"Unsafe remote gallery artifact path: '{relative}'.")
        return urllib.parse.urljoin(index_url, relative)

    def _remote_gallery_entries(self) -> tuple[list[dict[str, Any]], str]:
        assert self.gallery_url is not None
        index = self._fetch_remote_json(self.gallery_url)
        if not isinstance(index, dict) or not isinstance(index.get("recipes"), list):
            raise ValueError("Remote gallery index has an invalid structure.")
        digest = index.get("corpus_digest")
        if not isinstance(digest, str) or not digest:
            raise ValueError("Remote gallery index has no corpus digest.")
        entries = index["recipes"]
        if not all(isinstance(entry, dict) for entry in entries):
            raise ValueError("Remote gallery index contains an invalid recipe.")
        return entries, digest

    def search_gallery(
        self,
        query: str = "",
        features: list[str] | None = None,
        limit: int = 5,
        include_spec: bool = False,
    ) -> dict[str, Any]:
        """Search local or published canonical recipes without requiring either."""
        if limit < 1 or limit > 20:
            return self._error("gallery.limit", ValueError("limit must be in [1, 20]"))

        source_entries: list[dict[str, Any]]
        source: str
        corpus_digest: str | None = None
        if self.gallery is not None:
            source = str(self.gallery)
            source_entries = []
            for manifest_path in sorted(
                (self.gallery / "gallery").glob("*/recipe.toml")
            ):
                manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["_recipe_dir"] = manifest_path.parent
                source_entries.append(manifest)
        elif self.gallery_url is not None:
            source = self.gallery_url
            try:
                source_entries, corpus_digest = self._remote_gallery_entries()
            except (
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ):
                source_entries = []
        else:
            source = ""
            source_entries = []

        if not source_entries:
            return {
                "ok": True,
                "available": False,
                "gallery": None,
                "source": None,
                "matches": [],
                "message": (
                    "Gallery examples are unavailable; continue using data inspection, "
                    "the FigureSpec schema, validation, compilation, and rendering."
                ),
            }

        requested = {item.lower() for item in (features or [])}
        terms = set(re.findall(r"[a-z0-9_-]+", query.lower()))
        ranked: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for manifest in source_entries:
            tags = {str(item).lower() for item in manifest.get("features", [])}
            haystack = " ".join(
                [
                    str(manifest.get("id", "")),
                    str(manifest.get("title", "")),
                    str(manifest.get("summary", "")),
                    *tags,
                ]
            ).lower()
            if requested and not requested.issubset(tags):
                continue
            score = 3 * len(requested & tags) + sum(term in haystack for term in terms)
            if (requested or terms) and score == 0:
                continue
            recipe_id = str(manifest["id"])
            entry: dict[str, Any] = {
                "id": recipe_id,
                "title": manifest["title"],
                "summary": manifest["summary"],
                "features": manifest["features"],
                "backends": manifest["backends"],
                "inputs": manifest.get("inputs", []),
                "outputs": manifest.get("outputs", []),
                "folder": manifest.get("_recipe_dir", Path(recipe_id)).name,
                "score": score,
            }
            ranked.append((entry, manifest))
        ranked.sort(key=lambda item: (-item[0]["score"], item[0]["id"]))

        matches = []
        try:
            for entry, manifest in ranked[:limit]:
                if include_spec and self.gallery is not None:
                    recipe_dir = manifest["_recipe_dir"]
                    for artifact, key in (
                        ("figure", "spec"),
                        ("inspect", "inspection"),
                    ):
                        path = recipe_dir / "artifacts" / f"{artifact}.json"
                        if path.is_file():
                            entry[key] = json.loads(path.read_text(encoding="utf-8"))
                elif include_spec:
                    artifacts = manifest.get("artifacts", {})
                    hashes = manifest.get("sha256", {})
                    for artifact, key in (
                        ("figure", "spec"),
                        ("inspect", "inspection"),
                    ):
                        relative = artifacts.get(artifact)
                        expected = hashes.get(artifact)
                        if not isinstance(relative, str) or not isinstance(
                            expected, str
                        ):
                            raise ValueError(
                                f"Remote gallery recipe '{entry['id']}' lacks {artifact}."
                            )
                        url = self._remote_artifact_url(source, relative)
                        entry[key] = self._fetch_remote_json(
                            url, expected_sha256=expected
                        )
                matches.append(entry)
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ):
            return {
                "ok": True,
                "available": False,
                "gallery": None,
                "source": None,
                "matches": [],
                "message": (
                    "Gallery examples are unavailable; continue using data inspection, "
                    "the FigureSpec schema, validation, compilation, and rendering."
                ),
            }

        result: dict[str, Any] = {
            "ok": True,
            "available": True,
            "gallery": source,
            "source": "local" if self.gallery is not None else "remote",
            "matches": matches,
        }
        if corpus_digest is not None:
            result["corpus_digest"] = corpus_digest
        return result

    def agent_instructions(self) -> str:
        """Return the recommended deterministic agent workflow."""
        return (
            "Use Hakowan as a deterministic 3D visualization tool. First call "
            "inspect_data and never invent attribute names. Retrieve get_schema and, "
            "when available, relevant search_gallery examples. Author canonical "
            "FigureSpec JSON, then call validate_spec with strict=true. Repair schema "
            "or semantic failures with apply_patch. Use fit_camera when framing "
            "matters. Call render_spec and observe_spec for visual evidence. Use "
            "evaluate_visual_patch for bounded repairs and retain only an accepted "
            "candidate. Keep all paths inside the configured workspace root."
        )
