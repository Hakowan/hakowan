# Hakowan LLM evaluation

This suite evaluates natural-language-to-`FigureSpec` systems by observable visualization behavior rather than textual similarity.

## Stages

Every candidate is scored independently for:

1. JSON Schema validity
2. runtime reconstruction and strict semantic validation
3. compilation
4. backend rendering
5. attribute grounding
6. grammar intent
7. camera framing
8. bounded diagnostic repair
9. repair patch minimality

The bundled suite contains 20 requests over five deterministic datasets, including missing fields, non-positive log input, NaNs, constant fields, extreme aspect ratios, backend limitations, point clouds, disconnected components, comparisons, and occlusion.

## Deterministic run

```sh
python -m benchmarks.llm_eval --provider reference
```

This exercises the evaluator with controlled faulty first attempts and one repair. Reports are written to `llm-eval-report.json` and `llm-eval-report.html`.

Add browser-backed occupancy scoring:

```sh
python -m benchmarks.llm_eval --provider reference --observe
```

## Replay real responses

Store provider output as:

```json
{
  "model": "provider/model-id",
  "responses": {
    "scalar-temperature": [
      {"kind": "spec", "value": {"$schema": "...", "version": "1.1", "root": {}}, "metadata": {"latency_ms": 420, "input_tokens": 1200, "output_tokens": 350}}
    ]
  }
}
```

Then run:

```sh
python -m benchmarks.llm_eval --provider replay --responses responses.json
```

Each case may contain a second response with `kind: "patch"` for bounded repair.

## Live provider plugin

Pass `--provider package.module:callable`. The callable receives JSON-safe `case`, `context`, `attempt`, `candidate`, and `diagnostics` keyword arguments. It returns a complete spec mapping, JSON text, or `{ "kind": "patch", "value": [...] }`.

The context contains `hkw.schema()`, `hkw.inspect()` output for the case dataset, and up to three feature-matched canonical examples loaded from `HAKOWAN_GALLERY` or a sibling `hakowan-gallery` checkout.

Live runs are explicit and never part of ordinary CI.

Provider `raw` responses and arbitrary `metadata` (for example model ID, token
usage, latency, temperature, and cache statistics) are preserved in each case's
attempt trace.
