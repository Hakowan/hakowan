"""End-to-end benchmark runner with bounded repair and JSON/HTML reports."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import hakowan as hkw

from hakowan.spec.model import SCHEMA_VERSION
from .datasets import dataset
from .gallery import few_shots, load_gallery
from .models import BenchmarkCase, BenchmarkReport, CandidateResponse
from .providers import CandidateProvider, LiveProvider, ReferenceProvider, ReplayProvider
from .scoring import evaluate_candidate

SUITE_VERSION = "1.0"
PROMPT_VERSION = "1.0"


def load_cases(path: str | Path | None = None) -> tuple[BenchmarkCase, ...]:
    """Load benchmark cases from JSON, defaulting to the bundled suite."""
    source = Path(path) if path is not None else Path(__file__).with_name("cases.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    cases = tuple(BenchmarkCase.from_dict(item) for item in payload)
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark case IDs must be unique")
    return cases


def _context(
    case: BenchmarkCase, gallery_recipes: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "schema": hkw.schema(),
        "data": hkw.inspect(dataset(case.dataset)).to_dict(),
        "examples": few_shots(case.tags, gallery_recipes),
    }


def _complete_spec(
    response: CandidateResponse, candidate: dict[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if response.kind == "spec":
        if not isinstance(response.value, dict):
            raise TypeError("spec response value must be an object")
        return response.value, []
    if candidate is None:
        raise ValueError("patch response requires an existing candidate")
    if not isinstance(response.value, list):
        raise TypeError("patch response value must be an array")
    operations = response.value
    return hkw.patch_spec(candidate, operations).to_dict(), operations


def run_case(
    case: BenchmarkCase,
    provider: CandidateProvider,
    gallery_recipes: tuple[dict[str, Any], ...],
    *,
    observe: bool = False,
    max_repairs: int = 1,
):
    """Generate, evaluate, and optionally repair one benchmark candidate."""
    context = _context(case, gallery_recipes)
    response = provider.generate(case, context, attempt=0)
    trace: list[dict[str, Any]] = [
        {
            "attempt": 0,
            "kind": response.kind,
            "raw": response.raw,
            "metadata": response.metadata,
            "diagnostics": [],
        }
    ]
    candidate, operations = _complete_spec(response, None)
    result = evaluate_candidate(
        case, candidate, observe=observe, patch_operations=operations
    )
    result = replace(
        result, prompt=case.prompt, dataset=case.dataset, tags=case.tags
    )
    if result.final_pass or max_repairs <= 0:
        return replace(result, trace=tuple(trace))

    current = candidate
    last = result
    for attempt in range(1, max_repairs + 1):
        response = provider.generate(
            case,
            context,
            attempt=attempt,
            candidate=current,
            diagnostics=last.diagnostics,
        )
        trace.append(
            {
                "attempt": attempt,
                "kind": response.kind,
                "raw": response.raw,
                "metadata": response.metadata,
                "diagnostics": list(last.diagnostics),
            }
        )
        try:
            current, operations = _complete_spec(response, current)
        except Exception as exc:
            return replace(
                last,
                repair_attempts=attempt,
                repair_success=False,
                render=replace(
                    last.render, message=f"repair application failed: {exc}"
                ),
                trace=tuple(trace),
            )
        repaired = evaluate_candidate(
            case, current, observe=observe, patch_operations=operations
        )
        repaired = replace(
            repaired, prompt=case.prompt, dataset=case.dataset, tags=case.tags
        )
        repaired = replace(
            repaired,
            repair_attempts=attempt,
            repair_success=repaired.final_pass,
            trace=tuple(trace),
        )
        if repaired.final_pass:
            return repaired
        last = repaired
    return replace(last, trace=tuple(trace))


def run_suite(
    provider: CandidateProvider,
    *,
    cases: tuple[BenchmarkCase, ...] | None = None,
    gallery_root: str | Path | None = None,
    observe: bool = False,
    max_repairs: int = 1,
) -> BenchmarkReport:
    """Run all selected cases and return a stage-specific benchmark report."""
    selected = cases if cases is not None else load_cases()
    gallery_recipes = load_gallery(gallery_root)
    results = tuple(
        run_case(
            case,
            provider,
            gallery_recipes,
            observe=observe,
            max_repairs=max_repairs,
        )
        for case in selected
    )
    return BenchmarkReport(
        suite_version=SUITE_VERSION,
        model=provider.name,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        results=results,
    )


def write_json_report(report: BenchmarkReport, path: str | Path) -> None:
    """Write a deterministic JSON benchmark report."""
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_html_report(report: BenchmarkReport, path: str | Path) -> None:
    """Write a standalone HTML table with stage and semantic scores."""
    rows = []
    for result in report.results:
        status = "pass" if result.final_pass else "fail"
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.case_id)}</td>"
            f"<td class='{status}'>{status}</td>"
            f"<td>{int(result.schema.passed)}</td>"
            f"<td>{int(result.semantic.passed)}</td>"
            f"<td>{int(result.compile.passed)}</td>"
            f"<td>{int(result.render.passed)}</td>"
            f"<td>{result.attribute_score:.2f}</td>"
            f"<td>{result.grammar_score:.2f}</td>"
            f"<td>{result.camera_score:.2f}</td>"
            f"<td>{result.repair_attempts}</td>"
            f"<td>{result.patch_minimality:.2f}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>Hakowan LLM evaluation</title>
<style>
body {{ font: 14px system-ui; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: .4rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.pass {{ color: #087830; font-weight: 700; }}
.fail {{ color: #b00020; font-weight: 700; }}
</style>
<h1>Hakowan LLM evaluation</h1>
<p>Model: <code>{html.escape(report.model)}</code> · Pass rate: {report.pass_rate:.1%}</p>
<table>
<thead><tr><th>Case</th><th>Final</th><th>Schema</th><th>Semantic</th><th>Compile</th><th>Render</th><th>Attribute</th><th>Grammar</th><th>Camera</th><th>Repairs</th><th>Patch</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""
    Path(path).write_text(document, encoding="utf-8")


def _provider(args) -> CandidateProvider:
    if args.provider == "reference":
        return ReferenceProvider()
    if args.provider == "replay":
        if args.responses is None:
            raise ValueError("--responses is required for replay provider")
        return ReplayProvider(args.responses)
    return LiveProvider(args.provider)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="reference", help="reference, replay, or module:callable")
    parser.add_argument("--responses", help="Replay response JSON")
    parser.add_argument("--cases", help="Case JSON; defaults to bundled cases.json")
    parser.add_argument("--case", action="append", dest="case_ids", help="Run one case ID; repeatable")
    parser.add_argument("--gallery", help="Path to hakowan-gallery checkout")
    parser.add_argument("--observe", action="store_true", help="Run browser-backed camera occupancy scoring")
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--json", default="llm-eval-report.json")
    parser.add_argument("--html", default="llm-eval-report.html")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = tuple(case for case in cases if case.id in wanted)
        missing = wanted - {case.id for case in cases}
        if missing:
            parser.error(f"unknown case IDs: {sorted(missing)}")
    report = run_suite(
        _provider(args),
        cases=cases,
        gallery_root=args.gallery,
        observe=args.observe,
        max_repairs=args.max_repairs,
    )
    write_json_report(report, args.json)
    write_html_report(report, args.html)
    print(
        f"{sum(result.final_pass for result in report.results)}/{len(report.results)} "
        f"passed ({report.pass_rate:.1%})"
    )
    return 0 if all(result.final_pass for result in report.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
