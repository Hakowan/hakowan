from __future__ import annotations

import json


from benchmarks.llm_eval.datasets import names
from benchmarks.llm_eval.gallery import few_shots, load_gallery
from benchmarks.llm_eval.models import CandidateResponse
from benchmarks.llm_eval.providers import ReferenceProvider, ReplayProvider, json_diff
from benchmarks.llm_eval.runner import (
    load_cases,
    run_case,
    run_suite,
    write_html_report,
    write_json_report,
)
from benchmarks.llm_eval.scoring import evaluate_candidate, patch_minimality


def test_bundled_suite_has_twenty_unique_cases_and_five_datasets():
    cases = load_cases()

    assert len(cases) == 20
    assert len({case.id for case in cases}) == 20
    assert len(names()) == 5
    assert {case.dataset for case in cases} == set(names())


def test_reference_suite_exercises_repairs_and_passes():
    report = run_suite(ReferenceProvider(), gallery_root="/missing")

    assert report.pass_rate == 1.0
    repairs = [result for result in report.results if result.repair_attempts]
    assert len(repairs) == 5
    assert all(result.repair_success for result in repairs)
    assert all(result.patch_operations > 0 for result in repairs)
    assert all(result.patch_minimality > 0 for result in repairs)
    assert all(result.trace for result in report.results)
    assert all(result.prompt and result.dataset for result in report.results)


def test_schema_failure_is_scored_without_running_later_stages():
    case = load_cases()[0]

    result = evaluate_candidate(case, {"version": "1.1", "root": {}})

    assert not result.schema.passed
    assert not result.semantic.passed
    assert not result.compile.passed
    assert not result.render.passed
    assert not result.final_pass


def test_json_diff_and_minimality_penalize_unrelated_repairs():
    operations = json_diff(
        {"scene": {"camera": {"eye": [0, 0, 5]}}, "version": "1.1"},
        {"scene": {"camera": {"eye": [1, 2, 3]}}, "version": "1.1"},
    )

    assert operations == [
        {"op": "replace", "path": "/scene/camera/eye", "value": [1, 2, 3]}
    ]
    assert patch_minimality(operations, ("/scene/camera",)) == 1.0
    assert patch_minimality(operations, ("/root",)) < 1.0


def test_reports_are_json_safe_and_html_is_standalone(tmp_path):
    report = run_suite(
        ReferenceProvider(), cases=(load_cases()[0],), gallery_root="/missing"
    )
    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"

    write_json_report(report, json_path)
    write_html_report(report, html_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["case_count"] == 1
    assert payload["passed"] == 1
    assert "<table>" in html_path.read_text(encoding="utf-8")


def test_replay_provider_returns_stored_attempt(tmp_path):
    path = tmp_path / "responses.json"
    path.write_text(
        json.dumps(
            {
                "model": "stored-model",
                "responses": {
                    "scalar-temperature": [
                        {"kind": "spec", "value": {"version": "1.1"}}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    provider = ReplayProvider(path)

    response = provider.generate(load_cases()[0], {}, attempt=0)

    assert provider.name == "stored-model"
    assert response == CandidateResponse("spec", {"version": "1.1"})


def test_gallery_loader_and_feature_matching(tmp_path):
    recipe = tmp_path / "gallery" / "Example"
    artifacts = recipe / "artifacts"
    artifacts.mkdir(parents=True)
    (tmp_path / "gallery" / "README.md").write_text("# gallery\n", encoding="utf-8")
    (recipe / "recipe.toml").write_text(
        '\n'.join([
            'id = "example"',
            'title = "Example"',
            'script = "example.py"',
            'summary = "Scalar field."',
            'features = ["scalar"]',
            'backends = ["webgl"]',
            'inputs = []',
            'outputs = []',
        ]),
        encoding="utf-8",
    )
    (artifacts / "figure.json").write_text(
        json.dumps({"version": "1.1"}), encoding="utf-8"
    )

    recipes = load_gallery(tmp_path)
    examples = few_shots(("scalar",), recipes, limit=1)

    assert len(recipes) == 1
    assert examples[0]["id"] == "example"


def test_browser_backed_camera_occupancy():
    case = next(case for case in load_cases() if case.id == "isometric-fit-camera")

    result = run_case(
        case,
        ReferenceProvider(),
        (),
        observe=True,
        max_repairs=0,
    )

    assert result.final_pass
    occupancy = result.compile.details["camera"]["occupancy"]
    assert case.expected.occupancy[0] <= occupancy <= case.expected.occupancy[1]
