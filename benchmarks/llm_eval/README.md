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

## Strict MCP host benchmark

Run the suite through fresh Oh My Pi sessions with an enforced MCP-only tool
policy:

```sh
python -m benchmarks.llm_eval.mcp_harness \
  --model anthropic/claude-sonnet-4-6 \
  --gallery ../hakowan-gallery \
  --output /tmp/sonnet.responses.json \
  --timeout 300 \
  --max-tool-calls 32 \
  --keep-events
```

The harness creates deterministic datasets and an isolated MCP configuration in
a temporary workspace. Each case runs in a fresh host process. Wall-clock and
tool-call budgets are enforced; timeouts, nonzero exits, missing required tools,
non-MCP calls, provider retries, token usage, and reported cost are retained in
the response metadata.

Output-envelope compliance is reported independently from FigureSpec behavior:
`direct_json` passes the format contract, `normalized_envelope` records JSON
recovered deterministically from Markdown or surrounding prose, and `unusable`
records output with no complete response object. Normalized specifications still
receive ordinary schema, semantic, compile, render, intent, and camera scores.

## Recorded MCP harness runs

| Date | Host and model | Isolation | Result |
|---|---|---|---:|
| 2026-09-21 | Oh My Pi with `github-copilot/gpt-5.6` | MCP-assisted; repository tools available | 18/20 (90%) |
| 2026-09-21 | Oh My Pi with `github-copilot/gpt-5-mini` | Strict MCP-only | 11/20 (55%) |
| 2026-09-21 | Oh My Pi with `ollama/gpt-oss:20b` | Strict MCP-only, fully local | 0/20 (0%) |
| 2026-09-21 | Oh My Pi with `anthropic/claude-sonnet-4-6` | Strict MCP-only, pre-fix baseline | 14/20 (70%) |
| 2026-09-21 | Oh My Pi with `anthropic/claude-sonnet-4-6` | Strict MCP-only, post-fix rerun | 17/20 (85%) |
| 2026-09-21 | Oh My Pi with `github-copilot/claude-opus-5` | Strict MCP-only, pre-fix baseline | 18/20 (90%) |
| 2026-09-21 | Oh My Pi with `github-copilot/claude-opus-5` | Strict MCP-only, post-fix rerun | 20/20 (100%) |

The strict run used GPT-5 mini, a model available on the GitHub Copilot Free
plan. Each case ran in a fresh non-interactive OMP session with extensions,
skills, rules, session persistence, and LSP disabled. MCP calls may appear as
direct `mcp__hakowan_*` events or as OMP `xd://mcp__hakowan_*` device transport;
the harness normalizes both forms. Any filesystem, shell, browser, or other
non-MCP call fails the host-compliance result. The captured event streams
contained 139 normalized Hakowan MCP calls; every case called `inspect_data`,
`get_schema`, `get_backends`, `search_gallery`, and `validate_spec`. No model
response was retried or manually repaired.

The nine strict-run failures comprised three invalid or malformed specs, two
semantic/camera-validation failures, one camera-occupancy failure, and three
otherwise valid results that missed an exact requested grammar constraint. The
malformed JSON response for `wireframe-overlay` is preserved as an invalid empty
candidate rather than replaced by a retry.

Replay the strict run deterministically:

```sh
python -m benchmarks.llm_eval \
  --provider replay \
  --responses benchmarks/llm_eval/results/omp-github-copilot-gpt-5-mini-mcp-only-2026-09-21.responses.json \
  --gallery ../hakowan-gallery \
  --observe \
  --max-repairs 0 \
  --json /tmp/gpt-5-mini-mcp-only.report.json \
  --html /tmp/gpt-5-mini-mcp-only.report.html
```

The recorded response and report files retain model identity, isolation flags,
per-case MCP tool names, elapsed time, raw model output, aggregate usage, and
the scored results. The OMP provider reported 20 premium requests for the run;
"Free" describes model availability on Copilot Free, not unmetered usage.

### Local Ollama run

`gpt-oss:20b` was the only installed Ollama model that successfully completed
the MCP tool-call probe. Its full 20-case run made 122 Hakowan MCP calls and no
non-MCP calls. Eleven responses were empty or malformed JSON. Deterministic
envelope recovery extracted five additional canonical JSON payloads from prose
or Markdown without a model retry, yielding nine parseable responses. Three
passed schema, semantic, compilation, and rendering stages but missed required
behavioral constraints; six failed schema validation. The final result remained
**0/20**.

OMP's Ollama transport performed 115 automatic provider retries across 19
cases, primarily while handling tool-call continuations. The benchmark harness
did not retry or replace any case response. This distinction is retained in the
response metadata.

Three other installed local models were rejected after capability probes:

- `llama3.2:latest` fabricated a `render_spec` call instead of invoking MCP;
- `deepseek-r1:latest` described a tool call but emitted no tool execution;
- `mistral:latest` returned the requested success text without calling MCP.

They did not receive full suites because those runs would measure fabricated
tool use rather than Hakowan authoring. Replay the completed local run with:

```sh
python -m benchmarks.llm_eval \
  --provider replay \
  --responses benchmarks/llm_eval/results/omp-ollama-gpt-oss-20b-mcp-only-2026-09-21.responses.json \
  --gallery ../hakowan-gallery \
  --observe \
  --max-repairs 0 \
  --json /tmp/ollama-gpt-oss-20b.report.json \
  --html /tmp/ollama-gpt-oss-20b.report.html
```

### Anthropic Sonnet run

`claude-sonnet-4-6` completed the strict MCP-only suite at **14/20 (70%)**. It
made 140 Hakowan MCP calls and no non-MCP calls; every case completed the full
required inspect, schema, backend, gallery, and validation sequence. Seven
responses were direct JSON. Thirteen wrapped canonical JSON in Markdown or
brief prose, which the host recovered deterministically without a model retry.

Two failures were invalid specs despite MCP validation attempts. Four otherwise
valid and renderable specs failed camera occupancy, with one also missing an
exact grammar constraint. OMP reported no provider auto-retries and an estimated
direct Anthropic cost of **$4.53** for the 20 independent sessions.

Replay the Sonnet run with:

```sh
python -m benchmarks.llm_eval \
  --provider replay \
  --responses benchmarks/llm_eval/results/omp-anthropic-claude-sonnet-4-6-mcp-only-2026-09-21.responses.json \
  --gallery ../hakowan-gallery \
  --observe \
  --max-repairs 0 \
  --json /tmp/anthropic-sonnet-4-6.report.json \
  --html /tmp/anthropic-sonnet-4-6.report.html
```

After the Hakowan fixes, the same strict Sonnet harness improved to **17/20
(85%)**. The rerun made 168 Hakowan MCP calls, including 18 `fit_camera` calls,
with no non-MCP calls or provider retries. Fifteen responses were direct JSON;
three JSON payloads were recovered deterministically from their envelopes. Two
cases exhausted the case time limit without a final response. The remaining
failure requested beauty and depth passes but returned albedo and depth.

OMP reported an estimated direct Anthropic cost of **$4.97**. Replay the
post-fix run with:

```sh
python -m benchmarks.llm_eval \
  --provider replay \
  --responses benchmarks/llm_eval/results/omp-anthropic-claude-sonnet-4-6-post-fixes-2026-09-21.responses.json \
  --gallery ../hakowan-gallery \
  --observe \
  --max-repairs 0 \
  --json /tmp/anthropic-sonnet-4-6-post-fixes.report.json \
  --html /tmp/anthropic-sonnet-4-6-post-fixes.report.html
```

### Anthropic Opus 5 run

The direct `anthropic/claude-opus-5` route returned HTTP 404 because the model
was not available from the configured Anthropic API account. The same Claude
Opus 5 model was available through GitHub Copilot, so the recorded run uses
`github-copilot/claude-opus-5` and preserves that routing distinction in its
metadata.

Opus 5 completed the strict MCP-only suite at **18/20 (90%)**, the strongest
strict result recorded here. It made 136 Hakowan MCP calls and no non-MCP calls.
Nineteen responses were direct JSON; `point-pressure` exhausted the case time
limit without a final response and was retained as an invalid candidate.

The remaining behavioral failure was `isolate-region`, which computed and
filtered a replacement component field instead of grounding the filter in the
existing `region` attribute. OMP reported no provider auto-retries, 20 Copilot
premium requests, and an estimated model cost of **$5.07**.

Replay the Opus run with:

```sh
python -m benchmarks.llm_eval \
  --provider replay \
  --responses benchmarks/llm_eval/results/omp-github-copilot-claude-opus-5-mcp-only-2026-09-21.responses.json \
  --gallery ../hakowan-gallery \
  --observe \
  --max-repairs 0 \
  --json /tmp/github-copilot-claude-opus-5.report.json \
  --html /tmp/github-copilot-claude-opus-5.report.html
```

After the point-cloud inspection fix, deterministic `fit_camera` tool,
schema-invalid patch support, structured diagnostics, and benchmark correction,
the same strict harness was run again in fresh sessions. The post-fix result was
**20/20 (100%)**.

The rerun made 143 Hakowan MCP calls, including 19 `fit_camera` calls, with no
non-MCP calls or provider retries. All 20 cases inspected data and validated a
spec. Eighteen responses were direct JSON; two JSON payloads were recovered
deterministically from Markdown envelopes without a model retry. OMP reported
20 Copilot premium requests and an estimated model cost of **$5.12**.

Replay the post-fix result with:

```sh
python -m benchmarks.llm_eval \
  --provider replay \
  --responses benchmarks/llm_eval/results/omp-github-copilot-claude-opus-5-post-fixes-2026-09-21.responses.json \
  --gallery ../hakowan-gallery \
  --observe \
  --max-repairs 0 \
  --json /tmp/github-copilot-claude-opus-5-post-fixes.report.json \
  --html /tmp/github-copilot-claude-opus-5-post-fixes.report.html
```
