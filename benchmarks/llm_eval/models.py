"""Typed benchmark cases, candidate responses, and stage-specific results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ExpectedIntent:
    """Observable semantic properties required by one benchmark case."""

    marks: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    colormaps: tuple[str, ...] = ()
    camera_kind: str | None = None
    passes: tuple[str, ...] = ()
    legend: bool | None = None
    minimum_views: int = 1
    occupancy: tuple[float, float] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExpectedIntent":
        """Parse tuple-valued expectations from JSON data."""
        return cls(
            marks=tuple(value.get("marks", ())),
            attributes=tuple(value.get("attributes", ())),
            kinds=tuple(value.get("kinds", ())),
            colormaps=tuple(value.get("colormaps", ())),
            camera_kind=value.get("camera_kind"),
            passes=tuple(value.get("passes", ())),
            legend=value.get("legend"),
            minimum_views=int(value.get("minimum_views", 1)),
            occupancy=tuple(value["occupancy"]) if value.get("occupancy") else None,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One natural-language visualization request and its observable contract."""

    id: str
    dataset: str
    prompt: str
    recipe: str
    backend: str = "webgl"
    initial_variant: str | None = None
    expected: ExpectedIntent = field(default_factory=ExpectedIntent)
    initial_patch: tuple[dict[str, Any], ...] = ()
    repair_patch: tuple[dict[str, Any], ...] = ()
    allowed_patch_prefixes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BenchmarkCase":
        """Parse one case from its JSON representation."""
        return cls(
            id=value["id"],
            dataset=value["dataset"],
            prompt=value["prompt"],
            recipe=value["recipe"],
            backend=value.get("backend", "webgl"),
            expected=ExpectedIntent.from_dict(value.get("expected", {})),
            initial_patch=tuple(value.get("initial_patch", ())),
            initial_variant=value.get("initial_variant"),
            repair_patch=tuple(value.get("repair_patch", ())),
            allowed_patch_prefixes=tuple(value.get("allowed_patch_prefixes", ())),
            tags=tuple(value.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class CandidateResponse:
    """Provider output containing a complete spec or patch operations."""

    kind: Literal["spec", "patch"]
    value: dict[str, Any] | list[dict[str, Any]]
    raw: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageResult:
    """Success state and evidence for one evaluation stage."""

    passed: bool
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Complete stage-specific result for one candidate and optional repair."""

    case_id: str
    schema: StageResult
    semantic: StageResult
    compile: StageResult
    render: StageResult
    attribute_score: float
    grammar_score: float
    camera_score: float
    final_pass: bool
    prompt: str = ""
    dataset: str = ""
    tags: tuple[str, ...] = ()
    repair_attempts: int = 0
    repair_success: bool = False
    patch_operations: int = 0
    trace: tuple[dict[str, Any], ...] = ()
    patch_minimality: float = 1.0
    diagnostics: tuple[dict[str, Any], ...] = ()
    candidate: dict[str, Any] | None = None
    response_format: StageResult = field(
        default_factory=lambda: StageResult(True, details={"status": "not_recorded"})
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate benchmark metadata and per-case results."""

    suite_version: str
    model: str
    prompt_version: str
    schema_version: str
    results: tuple[CaseResult, ...]

    @property
    def pass_rate(self) -> float:
        """Return the fraction of cases with a passing final candidate."""
        return (
            sum(result.final_pass for result in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    @property
    def format_pass_rate(self) -> float:
        """Return the fraction of responses obeying the direct JSON contract."""
        return (
            sum(result.response_format.passed for result in self.results)
            / len(self.results)
            if self.results
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report with aggregate and format counts."""
        formats: dict[str, int] = {}
        for result in self.results:
            status = str(result.response_format.details.get("status", "not_recorded"))
            formats[status] = formats.get(status, 0) + 1
        return {
            "suite_version": self.suite_version,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "case_count": len(self.results),
            "passed": sum(result.final_pass for result in self.results),
            "pass_rate": self.pass_rate,
            "format_passed": sum(
                result.response_format.passed for result in self.results
            ),
            "format_pass_rate": self.format_pass_rate,
            "response_formats": dict(sorted(formats.items())),
            "results": [result.to_dict() for result in self.results],
        }
