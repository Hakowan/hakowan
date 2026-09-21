"""Candidate-provider interfaces for deterministic replay and optional live models."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Protocol

from .models import BenchmarkCase, CandidateResponse
from .reference import reference_spec


class CandidateProvider(Protocol):
    """Generate an initial specification and an optional repair response."""

    name: str

    def generate(
        self,
        case: BenchmarkCase,
        context: dict[str, Any],
        *,
        attempt: int,
        candidate: dict[str, Any] | None = None,
        diagnostics: tuple[dict[str, Any], ...] = (),
    ) -> CandidateResponse:
        """Return a complete specification or patch operations."""


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def json_diff(current: Any, target: Any, path: str = "") -> list[dict[str, Any]]:
    """Return deterministic add/remove/replace operations from current to target."""
    if type(current) is not type(target):
        return [{"op": "replace", "path": path, "value": target}]
    if isinstance(current, dict):
        operations: list[dict[str, Any]] = []
        for key in sorted(current.keys() - target.keys()):
            operations.append({"op": "remove", "path": f"{path}/{_escape(key)}"})
        for key in sorted(target.keys() - current.keys()):
            operations.append(
                {"op": "add", "path": f"{path}/{_escape(key)}", "value": target[key]}
            )
        for key in sorted(current.keys() & target.keys()):
            operations.extend(
                json_diff(current[key], target[key], f"{path}/{_escape(key)}")
            )
        return operations
    if isinstance(current, list):
        return [] if current == target else [{"op": "replace", "path": path, "value": target}]
    return [] if current == target else [{"op": "replace", "path": path, "value": target}]


class ReferenceProvider:
    """Deterministic provider used to test the evaluator and repair loop."""

    name = "reference-fixture"

    def generate(
        self,
        case: BenchmarkCase,
        context: dict[str, Any],
        *,
        attempt: int,
        candidate: dict[str, Any] | None = None,
        diagnostics: tuple[dict[str, Any], ...] = (),
    ) -> CandidateResponse:
        """Return a controlled initial candidate or minimal patch to the reference."""
        target = reference_spec(case)
        if attempt == 0:
            return CandidateResponse("spec", reference_spec(case, faulty=True))
        if candidate is None:
            return CandidateResponse("spec", target)
        return CandidateResponse("patch", json_diff(candidate, target))


class ReplayProvider:
    """Replay previously captured model responses from a JSON mapping."""

    def __init__(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.name = payload.get("model", "replay")
        self._responses = payload["responses"]

    def generate(
        self,
        case: BenchmarkCase,
        context: dict[str, Any],
        *,
        attempt: int,
        candidate: dict[str, Any] | None = None,
        diagnostics: tuple[dict[str, Any], ...] = (),
    ) -> CandidateResponse:
        """Return the stored response for this case and attempt."""
        try:
            response = self._responses[case.id][attempt]
        except (KeyError, IndexError) as exc:
            raise KeyError(
                f"No replay response for {case.id!r} attempt {attempt}"
            ) from exc
        return CandidateResponse(
            kind=response.get("kind", "spec"),
            value=response["value"],
            raw=response.get("raw"),
            metadata=response.get("metadata", {}),
        )


class LiveProvider:
    """Adapt a user-supplied callable without depending on any model SDK."""

    def __init__(self, target: str) -> None:
        module_name, separator, attribute = target.partition(":")
        if not separator:
            raise ValueError("Live provider must use 'module:callable' syntax")
        function = getattr(importlib.import_module(module_name), attribute)
        if not callable(function):
            raise TypeError(f"Live provider target {target!r} is not callable")
        self._function = function
        self.name = target

    def generate(
        self,
        case: BenchmarkCase,
        context: dict[str, Any],
        *,
        attempt: int,
        candidate: dict[str, Any] | None = None,
        diagnostics: tuple[dict[str, Any], ...] = (),
    ) -> CandidateResponse:
        """Call the plugin with JSON-safe case, context, and repair evidence."""
        response = self._function(
            case={
                "id": case.id,
                "prompt": case.prompt,
                "dataset": case.dataset,
                "backend": case.backend,
            },
            context=context,
            attempt=attempt,
            candidate=candidate,
            diagnostics=list(diagnostics),
        )
        if isinstance(response, str):
            response = json.loads(response)
        if "kind" in response and "value" in response:
            return CandidateResponse(
                response["kind"],
                response["value"],
                response.get("raw"),
                response.get("metadata", {}),
            )
        return CandidateResponse("spec", response)
