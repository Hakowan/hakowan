"""Schema, semantic, compile, render, intent, and camera scoring."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import hakowan as hkw
from pydantic import ValidationError as PydanticValidationError
from typing import cast

from hakowan.backends import BackendName
from hakowan.grammar.figure import Figure
from hakowan.grammar.layer import Layer

from .datasets import case_resolver
from .models import BenchmarkCase, CaseResult, StageResult


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _tokens(payload: dict[str, Any]) -> dict[str, Any]:
    kinds: set[str] = set()
    attributes: set[str] = set()
    colormaps: set[str] = set()
    legends: list[Any] = []
    for value in _walk(payload):
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("kind"), str):
            kinds.add(value["kind"])
        if (
            isinstance(value.get("name"), str)
            and "scales" in value
            and "unit" in value
        ):
            attributes.add(value["name"])
        if isinstance(value.get("colormap"), str):
            colormaps.add(value["colormap"])
        if "legend" in value:
            legends.append(value["legend"])
    scene = payload.get("scene") or {}
    output = scene.get("output") or {}
    camera = scene.get("camera") or {}
    return {
        "kinds": kinds,
        "attributes": attributes,
        "colormaps": colormaps,
        "legends": legends,
        "passes": set(output.get("passes", ())),
        "camera_kind": camera.get("kind"),
    }


def _fraction(required, actual) -> float:
    required_set = set(required)
    return (
        len(required_set & set(actual)) / len(required_set)
        if required_set
        else 1.0
    )


def _intent_scores(
    case: BenchmarkCase, payload: dict[str, Any], scene
) -> tuple[float, float, dict[str, Any]]:
    tokens = _tokens(payload)
    marks = {
        view.mark.name.lower()
        for view in scene
        if getattr(view, "mark", None) is not None
    }
    attribute_score = _fraction(case.expected.attributes, tokens["attributes"])
    checks = {
        "marks": _fraction(case.expected.marks, marks),
        "kinds": _fraction(case.expected.kinds, tokens["kinds"]),
        "colormaps": _fraction(case.expected.colormaps, tokens["colormaps"]),
        "passes": _fraction(case.expected.passes, tokens["passes"]),
        "views": 1.0 if len(scene) >= case.expected.minimum_views else 0.0,
    }
    if case.expected.legend is not None:
        enabled = any(value is True or isinstance(value, dict) for value in tokens["legends"])
        checks["legend"] = float(enabled is case.expected.legend)
    grammar_score = sum(checks.values()) / len(checks)
    return attribute_score, grammar_score, {
        "actual_marks": sorted(marks),
        "actual_attributes": sorted(tokens["attributes"]),
        "actual_kinds": sorted(tokens["kinds"]),
        "actual_colormaps": sorted(tokens["colormaps"]),
        "actual_passes": sorted(tokens["passes"]),
        "checks": checks,
    }

def _tuple3(values) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _camera_score(
    case: BenchmarkCase,
    runtime: Layer | Figure,
    diagnostics: tuple[dict[str, Any], ...],
    *,
    observe: bool,
) -> tuple[float, dict[str, Any]]:
    expected = case.expected
    if expected.camera_kind is None and expected.occupancy is None:
        return 1.0, {}
    figure = runtime if isinstance(runtime, Figure) else None
    camera = figure.scene.camera if figure is not None else None
    score = 1.0
    details: dict[str, Any] = {}
    if expected.camera_kind is not None:
        kind = (
            "orthographic"
            if isinstance(camera, hkw.OrthographicCamera)
            else "thin_lens"
            if isinstance(camera, hkw.ThinLensCamera)
            else "perspective"
            if isinstance(camera, hkw.PerspectiveCamera)
            else None
        )
        details["camera_kind"] = kind
        if kind != expected.camera_kind:
            score = 0.0
    camera_errors = [
        item for item in diagnostics if str(item.get("code", "")).startswith("camera.")
    ]
    details["camera_diagnostics"] = camera_errors
    if camera_errors:
        score = 0.0
    if observe and expected.occupancy is not None and camera is not None:
        state = hkw.CameraState(
            eye=_tuple3(camera.eye),
            target=_tuple3(camera.target),
            up=_tuple3(camera.up),
            fov=float(getattr(camera, "fov", 35.0)),
            near=float(camera.near),
            far=float(camera.far),
            mode=(
                "orthographic"
                if isinstance(camera, hkw.OrthographicCamera)
                else "perspective"
            ),
            scale=(
                float(camera.scale)
                if isinstance(camera, hkw.OrthographicCamera)
                else None
            ),
        )
        observation = hkw.observe(
            runtime,
            views=["front"],
            cameras={"front": state},
            passes=["element_id", "layer_id"],
            resolution=(128, 128),
        )
        summary = observation.region(0, 0, 128, 128, view="front")
        occupancy = 1.0 - summary.background_fraction
        low, high = expected.occupancy
        details["occupancy"] = occupancy
        details["occupancy_range"] = [low, high]
        if not low <= occupancy <= high:
            score = 0.0
    return score, details



def patch_minimality(
    operations: list[dict[str, Any]], allowed_prefixes: tuple[str, ...]
) -> float:
    """Score patch size and penalize paths unrelated to the expected repair."""
    if not operations:
        return 1.0
    unrelated = sum(
        not any(str(operation.get("path", "")).startswith(prefix) for prefix in allowed_prefixes)
        for operation in operations
    ) if allowed_prefixes else 0
    cost = max(0, len(operations) - 1) + 2 * unrelated
    return 1.0 / (1.0 + cost)


def evaluate_candidate(
    case: BenchmarkCase,
    candidate: dict[str, Any],
    *,
    observe: bool = False,
    patch_operations: list[dict[str, Any]] | None = None,
) -> CaseResult:
    """Evaluate one complete candidate through every benchmark stage."""
    diagnostics: tuple[dict[str, Any], ...] = ()
    try:
        spec = hkw.FigureSpec.model_validate(candidate)
        schema = StageResult(True)
    except (PydanticValidationError, TypeError, ValueError) as exc:
        return CaseResult(
            case_id=case.id,
            schema=StageResult(False, str(exc)),
            semantic=StageResult(False, "schema failed"),
            compile=StageResult(False, "schema failed"),
            render=StageResult(False, "schema failed"),
            attribute_score=0.0,
            grammar_score=0.0,
            camera_score=0.0,
            final_pass=False,
            patch_operations=len(patch_operations or ()),
            patch_minimality=patch_minimality(
                patch_operations or [], case.allowed_patch_prefixes
            ),
            candidate=candidate,
        )

    try:
        runtime = hkw.from_spec(spec, data_resolver=case_resolver(case.dataset))
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            schema=schema,
            semantic=StageResult(False, f"conversion failed: {exc}"),
            compile=StageResult(False, "conversion failed"),
            render=StageResult(False, "conversion failed"),
            attribute_score=0.0,
            grammar_score=0.0,
            camera_score=0.0,
            final_pass=False,
            patch_operations=len(patch_operations or ()),
            patch_minimality=patch_minimality(
                patch_operations or [], case.allowed_patch_prefixes
            ),
            candidate=candidate,
        )

    report = hkw.validate(runtime, backend=cast(BackendName, case.backend), strict=True)
    diagnostics = tuple(item.to_dict() for item in report.diagnostics)
    semantic = StageResult(
        report.valid,
        None if report.valid else "; ".join(item.message for item in report.errors),
        {"diagnostics": list(diagnostics)},
    )
    try:
        scene = hkw.compile(runtime, preserve_attributes=True)
        compile_stage = StageResult(True, details={"views": len(scene)})
        attribute_score, grammar_score, intent_details = _intent_scores(
            case, spec.to_dict(), scene
        )
    except Exception as exc:
        scene = None
        compile_stage = StageResult(False, str(exc))
        attribute_score = grammar_score = 0.0
        intent_details = {}

    if scene is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="hakowan-llm-eval-") as directory:
                suffix = ".html" if case.backend == "webgl" else ".png"
                path = Path(directory) / f"candidate{suffix}"
                result = hkw.render(
                    runtime, filename=path, backend=cast(BackendName, case.backend)
                )
                rendered = result.path is not None and result.path.is_file()
                render_stage = StageResult(
                    rendered,
                    None if rendered else "backend produced no output file",
                    {"artifact": result.path.name if result.path else None},
                )
        except Exception as exc:
            render_stage = StageResult(False, str(exc))
    else:
        render_stage = StageResult(False, "compile failed")

    camera_score, camera_details = _camera_score(
        case, runtime, diagnostics, observe=observe
    )
    compile_stage = StageResult(
        compile_stage.passed,
        compile_stage.message,
        {**compile_stage.details, "intent": intent_details, "camera": camera_details},
    )
    final_pass = all(
        (
            schema.passed,
            semantic.passed,
            compile_stage.passed,
            render_stage.passed,
            attribute_score == 1.0,
            grammar_score == 1.0,
            camera_score == 1.0,
        )
    )
    operations = patch_operations or []
    return CaseResult(
        case_id=case.id,
        schema=schema,
        semantic=semantic,
        compile=compile_stage,
        render=render_stage,
        attribute_score=attribute_score,
        grammar_score=grammar_score,
        camera_score=camera_score,
        final_pass=final_pass,
        patch_operations=len(operations),
        patch_minimality=patch_minimality(operations, case.allowed_patch_prefixes),
        diagnostics=diagnostics,
        candidate=spec.to_dict(),
    )
