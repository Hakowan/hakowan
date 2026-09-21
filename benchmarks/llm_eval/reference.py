"""Deterministic reference candidates and controlled faulty variants."""

from __future__ import annotations

import copy
from typing import Any

import hakowan as hkw

from .datasets import dataset
from .models import BenchmarkCase


def _figure(case: BenchmarkCase) -> hkw.Figure:
    mesh = dataset(case.dataset)
    layer = hkw.layer(mesh)
    recipe = case.recipe

    if recipe == "scalar_color":
        figure = hkw.figure(layer.color_by("temperature", colormap="viridis"))
    elif recipe == "log_color":
        attribute = hkw.attribute("pressure", scale=hkw.scale.Log())
        figure = hkw.figure(layer.color_by(attribute, colormap="viridis"))
    elif recipe == "signed_color":
        figure = hkw.figure(layer.color_by("signed", colormap="coolwarm"))
    elif recipe == "finite_quality":
        figure = hkw.figure(layer.color_by("signed", colormap="viridis"))
    elif recipe == "categorical_components":
        figure = hkw.figure(layer.color_by("region", colormap="set1", categories=True))
    elif recipe == "wireframe":
        figure = hkw.figure(layer.show_edges(width=0.02))
    elif recipe == "vector_glyphs":
        figure = hkw.figure(layer.glyph_vectors("velocity", scale=0.2, overlay=False))
    elif recipe == "point_color":
        figure = hkw.figure(layer.mark("point").color_by("pressure", colormap="fire"))
    elif recipe == "point_size":
        figure = hkw.figure(layer.mark("point").channel(size=hkw.norm("velocity")))
    elif recipe == "slice":
        figure = hkw.figure(layer.slice((1, 0, 0), offset=0))
    elif recipe == "isolate_component":
        figure = hkw.figure(layer.isolate_component(1, attribute="region", compute=False))
    elif recipe == "comparison":
        figure = hkw.figure(
            layer.compare(layer.color_by("temperature"), labels=("raw", "colored"))
        )
    elif recipe == "fit_camera":
        figure = hkw.figure(layer).camera("fit", direction="isometric")
    elif recipe == "section_camera":
        figure = hkw.figure(layer).camera(
            "section", normal=(0, 0, 1), projection="orthographic"
        )
    elif recipe == "extremum_camera":
        figure = hkw.figure(layer).camera(
            "attribute_extremum", attribute="stress", extremum="max"
        )
    elif recipe == "render_passes":
        figure = hkw.figure(layer).output(passes=("beauty", "depth", "normal"))
    elif recipe == "backend_passes":
        figure = hkw.figure(layer).output(passes=("beauty", "depth"))
    elif recipe == "occlusion_layers":
        front = layer.name("front").translate((0, 0, 0.2))
        back = layer.name("back")
        figure = hkw.figure(front + back).camera("fit", direction="front")
    elif recipe == "constant_field":
        figure = hkw.figure(layer.color_by("constant", colormap="viridis"))
    elif recipe == "elongated_camera":
        figure = hkw.figure(layer).camera("fit", direction="front", margin=0.1)
    else:
        raise KeyError(f"Unknown reference recipe {recipe!r}")

    if figure.scene.camera is None and recipe not in {"render_passes", "backend_passes"}:
        figure = figure.camera("fit", direction="isometric")
    return figure


def _attribute_nodes(value: Any):
    if isinstance(value, dict):
        if "name" in value and "scales" in value and "unit" in value:
            yield value
        for child in value.values():
            yield from _attribute_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _attribute_nodes(child)


def _fault(payload: dict[str, Any], variant: str | None) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    if variant == "schema_extra":
        result["unexpected"] = True
    elif variant in {"wrong_attribute", "nonfinite"}:
        source = "temperature" if variant == "wrong_attribute" else "signed"
        target = "missing" if variant == "wrong_attribute" else "quality"
        for attribute in _attribute_nodes(result):
            if attribute["name"] == source:
                attribute["name"] = target
                break
    elif variant == "log_nonpositive":
        for attribute in _attribute_nodes(result):
            if attribute["name"] == "signed":
                attribute["scales"] = [{"kind": "log", "base": 10.0}]
                break
    elif variant == "unsupported_pass":
        result["scene"]["output"]["passes"] = ["beauty", "facet_id"]
    elif variant == "bad_camera":
        result["scene"]["camera"].update(
            {"eye": [0, 0, 5], "target": [0, 0, 10], "fov": 20}
        )
    return result


def reference_spec(case: BenchmarkCase, *, faulty: bool = False) -> dict[str, Any]:
    """Return a canonical reference or controlled faulty candidate mapping."""
    figure = _figure(case)
    payload = hkw.to_spec(figure, data_ids=lambda _mesh: "data").to_dict()
    return _fault(payload, case.initial_variant) if faulty else payload
