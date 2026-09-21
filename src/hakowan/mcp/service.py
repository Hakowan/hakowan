"""Provider-neutral, JSON-safe operations exposed by the Hakowan MCP server."""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..backends import BackendName, list_backend_capabilities
from ..compiler import compile as compile_layer
from ..grammar.figure import Figure
from ..grammar.layer import Layer
from ..inspection import inspect as inspect_data_source
from ..observation import observe
from ..render import render
from ..spec import FigureSpec, from_spec, json_schema, patch_spec
from ..validation import validate


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
        resolved = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
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
    ) -> None:
        self.paths = PathPolicy.create(root)
        self.gallery = self._find_gallery(gallery)

    def _find_gallery(self, gallery: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if gallery is not None:
            candidates.append(Path(gallery))
        if os.environ.get("HAKOWAN_GALLERY"):
            candidates.append(Path(os.environ["HAKOWAN_GALLERY"]))
        candidates.append(self.paths.root.parent / "hakowan-gallery")
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if (resolved / "gallery").is_dir():
                return resolved
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

    def get_schema(self) -> dict[str, Any]:
        """Return Hakowan's canonical FigureSpec JSON Schema."""
        return {"ok": True, "schema": json_schema()}

    def inspect_data(
        self, source: str, positions: list[str] | None = None
    ) -> dict[str, Any]:
        """Inspect a workspace data file and return geometry and attribute facts."""
        try:
            path = self.paths.resolve(source, must_exist=True)
            summary = inspect_data_source(path, positions=positions)
            return {"ok": True, "source": self.paths.display(path), "summary": summary.to_dict()}
        except Exception as exc:
            return self._error("inspect.failed", exc, source=source)

    def _parse_spec(self, spec: dict[str, Any]) -> FigureSpec:
        parsed = FigureSpec.model_validate(spec)
        self._check_resource_paths(parsed.to_dict())
        return parsed

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
        self, spec: dict[str, Any], data_bindings: dict[str, str] | None = None
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
        spec: dict[str, Any],
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
            return {
                "ok": True,
                "spec": parsed.to_dict(),
                "validation": report.to_dict(),
            }
        except Exception as exc:
            return self._error("validate.failed", exc)

    def compile_spec(
        self,
        spec: dict[str, Any],
        data_bindings: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compile a specification and summarize resolved views without rendering."""
        try:
            _, runtime = self._runtime(spec, data_bindings)
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
                        "mark": view.mark.name.lower() if view.mark is not None else None,
                        "vertex_count": int(mesh.num_vertices),
                        "facet_count": int(mesh.num_facets),
                        "bounds": bounds,
                    }
                )
            return {
                "ok": True,
                "validation": report.to_dict(),
                "scene": {
                    "views": views,
                    "legends": [item.to_dict() for item in scene.legends],
                    "annotations": [item.to_dict() for item in scene.annotations],
                },
            }
        except Exception as exc:
            return self._error("compile.failed", exc)

    def render_spec(
        self,
        spec: dict[str, Any],
        output: str,
        backend: str = "webgl",
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
        offline: bool = False,
    ) -> dict[str, Any]:
        """Validate and render a specification to a workspace output path."""
        try:
            _, runtime = self._runtime(spec, data_bindings)
            report = validate(runtime, backend=cast(BackendName, backend), strict=strict)
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

    def observe_spec(
        self,
        spec: dict[str, Any],
        output_dir: str,
        views: list[str] | None = None,
        passes: list[str] | None = None,
        resolution: list[int] | None = None,
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Capture WebGL evidence and structured visibility data to a directory."""
        try:
            _, runtime = self._runtime(spec, data_bindings)
            report = validate(runtime, backend="webgl", strict=strict)
            if not report.valid:
                return {"ok": False, "validation": report.to_dict()}
            directory = self.paths.resolve(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            selected_views = views or ["front", "right", "top", "isometric"]
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
            observation = observe(
                runtime,
                views=cast(Any, selected_views),
                passes=cast(Any, selected_passes),
                resolution=(int(size[0]), int(size[1])),
                output_dir=directory,
            )
            return {
                "ok": True,
                "validation": report.to_dict(),
                "output_dir": self.paths.display(directory),
                "manifest": observation.manifest,
            }
        except Exception as exc:
            return self._error("observe.failed", exc, output_dir=output_dir)

    def apply_patch(
        self,
        spec: dict[str, Any],
        operations: list[dict[str, Any]],
        backend: str = "webgl",
        data_bindings: dict[str, str] | None = None,
        strict: bool = True,
        semantic: bool = True,
    ) -> dict[str, Any]:
        """Atomically patch a specification and optionally validate semantics."""
        try:
            current = self._parse_spec(spec)
            patched = patch_spec(current, operations)
            self._check_resource_paths(patched.to_dict())
            result: dict[str, Any] = {"ok": True, "spec": patched.to_dict()}
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
        except Exception as exc:
            return self._error("patch.failed", exc)

    def search_gallery(
        self,
        query: str = "",
        features: list[str] | None = None,
        limit: int = 5,
        include_spec: bool = False,
    ) -> dict[str, Any]:
        """Search canonical gallery recipes by text and feature tags."""
        if self.gallery is None:
            return {
                "ok": False,
                "error": {
                    "code": "gallery.unavailable",
                    "message": (
                        "No gallery checkout found. Set HAKOWAN_GALLERY or pass --gallery."
                    ),
                },
            }
        if limit < 1 or limit > 20:
            return self._error("gallery.limit", ValueError("limit must be in [1, 20]"))
        requested = {item.lower() for item in (features or [])}
        terms = set(re.findall(r"[a-z0-9_-]+", query.lower()))
        matches = []
        for manifest_path in sorted((self.gallery / "gallery").glob("*/recipe.toml")):
            manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
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
            score = 3 * len(requested & tags) + sum(
                term in haystack for term in terms
            )
            if (requested or terms) and score == 0:
                continue
            entry: dict[str, Any] = {
                "id": manifest["id"],
                "title": manifest["title"],
                "summary": manifest["summary"],
                "features": manifest["features"],
                "backends": manifest["backends"],
                "inputs": manifest.get("inputs", []),
                "outputs": manifest.get("outputs", []),
                "folder": manifest_path.parent.name,
                "score": score,
            }
            if include_spec:
                figure_path = manifest_path.parent / "artifacts" / "figure.json"
                if figure_path.is_file():
                    entry["spec"] = json.loads(
                        figure_path.read_text(encoding="utf-8")
                    )
                inspect_path = manifest_path.parent / "artifacts" / "inspect.json"
                if inspect_path.is_file():
                    entry["inspection"] = json.loads(
                        inspect_path.read_text(encoding="utf-8")
                    )
            matches.append(entry)
        matches.sort(key=lambda item: (-item["score"], item["id"]))
        return {"ok": True, "gallery": str(self.gallery), "matches": matches[:limit]}

    def agent_instructions(self) -> str:
        """Return the recommended deterministic agent workflow."""
        return (
            "Use Hakowan as a deterministic 3D visualization tool. First call "
            "inspect_data and never invent attribute names. Retrieve get_schema and "
            "relevant search_gallery examples, author canonical FigureSpec JSON, then "
            "call validate_spec with strict=true. Repair only with apply_patch. Call "
            "render_spec and, when visual evidence matters, observe_spec before "
            "returning. Keep all paths inside the configured workspace root."
        )
