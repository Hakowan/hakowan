# Hakowan Improvement Plan

Status: draft, 2026-06-14, against v0.5.2 (branch `dev/0.5.2`).

## Goals

1. **Make hakowan more useful as a 3D visualization tool** — close the gaps that block it from being the default tool for publication-quality 3D mesh figures.
2. **Engage with LLMs directly** — let users describe a figure in natural language and get back an editable, reproducible hakowan spec.

## Guiding principles

- **The LLM amplifies the grammar; it cannot exceed it.** A model can only emit specs the grammar can express. Building NL on top of a grammar that lacks colorbars/legends produces confident specs that render *incomplete figures*. **Therefore foundation (Track A) is sequenced before the LLM layer (Track B).**
- **Preserve "one spec = one image."** Everything that determines the figure lives in the declarative spec; only cosmetic global defaults live in `config`. (Camera/lighting currently violate this — A5 fixes it.)
- **Stay in the niche.** 3D mesh + photoreal rendering. Do not chase text-to-3D / scene generation (that's 3D-GPT's lane). Frame the LLM work as "NL → editable spec," never "AI generates your viz."
- **Hakowan is a complement, not a replacement.** Workflow: polyscope/pyvista to debug interactively → hakowan for the final figure. Lean into this in docs and positioning.

The structural bet: a *declarative* spec is a far more reliable LLM target than imperative pyvista/polyscope code. Track B is where the grammar design pays off competitively — but only once Track A makes the grammar complete.

---

## Track A — Make hakowan more useful

### A1. Colorbar / legend  ⭐ top gap

A scientific figure without a colorbar is barely publishable. This is the single highest-leverage item.

- **Design**: 2D overlay (not in-scene geometry). HTML/SVG panel for WebGL; PIL raster composite for Mitsuba/Blender. Implicit-by-default legend on `ScalarField` (Vega-Lite model). Forward-sample the scale chain for ticks (handles `Log` + `Custom`). Reuse the `categories` flag for discrete vs continuous. Full design in the saved memory `hakowan-colorbar-plan.md`.
- **Files**: `compiler/color.py` (capture domain min/max during `apply_colormap`), `grammar/texture/texture.py` (`legend` field on `ScalarField` + new `Legend` dataclass), `backends/webgl/template.py` (overlay), new `backends/<x>/legend.py` per backend, `compiler/scene.py` (collect/dedupe `LegendSpec`).
- **Phases**: (1) continuous colorbar, WebGL, auto-emit, linear+log ticks. (2) Mitsuba/Blender PIL composite. (3) categorical legend + `Legend` customization + multi-legend stacking. (4) vector/size legends.
- **Effort**: ~4–6 days through phase 3.
- **Done when**: `layer.channel(material=ScalarField("curvature", colormap="viridis"))` renders a labeled colorbar in all three backends with no extra code; `legend=False` suppresses it.

### A2. JSON-safe spec round-trip

`Filter.condition` is a `Callable` (`grammar/transform/transform.py:75`) → blocks serialization, caching, reproducibility, and is a hard blocker for Track B.

- **Fix**: replace the callable with a restricted string-expression DSL (`"x > 0.5 and y < 1"`), parsed via a safe AST walker (comparison/boolean/arithmetic/attribute access only — no calls). Keep the callable path as a deprecated escape hatch if needed, but exclude it from serialization.
- **Add**: `layer.to_json()` / `hkw.from_json(...)` round-trip.
- **Files**: `grammar/transform/transform.py`, new `grammar/expr.py` (DSL parser), `compiler/transform.py`.
- **Effort**: ~3 days.
- **Done when**: every spec in `examples/` round-trips through JSON and renders identically.

### A3. Schema export (Pydantic mirror)

The grammar lives in dataclasses with no machine-readable schema. Needed for validation, IDE hints, web forms, docs, and LLM grounding (Track B).

- **Fix**: auto-generate a Pydantic mirror from the dataclass tree (one introspection script, not hand-maintained — avoids drift). Publish `hkw.schema()` → JSON Schema.
- **Files**: new `grammar/schema.py`.
- **Effort**: ~2–3 days. **Shared deliverable with B1.**
- **Done when**: `hkw.schema()` returns valid JSON Schema covering marks/channels/transforms/scales/materials; regenerates automatically from the dataclasses.

### A4. Broader data inputs — shipped

`hkw.layer(...)`, `Layer.data(...)`, `hkw.inspect(...)`, and
`hkw.dataframe.to_dataframe(...)` accept numeric NumPy-compatible point arrays,
pandas DataFrames, xarray Datasets, PyVista datasets, and Trimesh meshes. Tables
infer `x`, `y`, and optional `z` positions or accept explicit `positions=` names.
Numeric fields become domain-correct Lagrange attributes; polygonal adapters
preserve topology and vertex/point plus facet/cell attributes.

### A5. Camera / lighting / environment in the grammar — shipped

`Figure` is the serializable scene envelope around a layer tree. It carries
perspective, orthographic, or thin-lens camera intent; ordered point and
directional lights; environment lighting and visibility; and semantic output
settings (resolution, background, passes, and sampler seed). Schema version
1.1 round-trips these settings while continuing to load layer-only 1.0 specs.
Explicit `Config` and backend keyword arguments retain invocation-time
precedence.
High-level framing resolves fitted, principal-axis, attribute-extremum, section,
named-layer, component, bounding-box, and turntable requests to concrete
serializable cameras.

### A6. Examples gallery website  (discoverability)

Niche name + sparse docs = low discovery. A Vega-Lite-style gallery is the best adoption lever per unit effort. `mkdocs-material` is already configured.

- **Fix**: auto-render each `examples/` spec → thumbnail + spec + interactive WebGL viewer on a gallery page. Indexable, copy-pasteable.
- **Effort**: ~3–4 days.
- **Done when**: `hakowan.github.io/hakowan/gallery` shows every example with code + live viewer.

### A7. Grammar breadth  (stretch)

- More marks: `Text`/`Label` (annotation, axis ticks — also needed for polished legends), `Glyph` (parameterized arrows/crosses), `Volume` (volumetric scalar fields; Mitsuba supports). Marks are still only `Point`/`Curve`/`Surface`.
- `Time` channel + `hkw.render(layer, "out.mp4", frames=N)` for time-varying data.
- **Effort**: large; defer until A1–A6 land.

**Note — already shipped, do not re-plan:** fast preview backend (WebGL default), notebook inline display (`Layer._repr_html_`, `layer.py:554`), multi-view composition (`&`/`|`).

---

## Track B — Engage with LLMs directly

Builds on the existing saved NL plan (`hakowan-nl-plan.md`, Path B). **Hard dependency: A1 (colorbar), A2 (serialization), A3 (schema) must land first** — otherwise the model emits specs that can't express complete figures or can't be validated.

### B0. Module + packaging

- New `src/hakowan/nl/`, import-guarded like backends (mirror `register_backend_loader` in `__init__.py`).
- `pip install hakowan[nl]` extra → `anthropic` + `pydantic`.
- Target API: `layer = hkw.nl.generate("color by curvature, log scale, glass material", data="mesh.obj")`.
- **Effort**: ~0.5 day.

### B1. LLM-facing schema  (= A3)

Reuse the Pydantic mirror from A3. Do **not** expose internal dataclasses to the LLM (internal plumbing leaks; `SurfaceMesh` isn't JSON-able → path-only at the boundary; discriminated unions on a `kind` field are cleaner). Add `nl/compile.py::spec_to_layer`.

### B2. Data introspection

The model needs the mesh's actual attributes or it hallucinates (`velocity` when only `curvature` exists). `nl/introspect.py`: load via lagrange, return attribute names/types/ranges + vertex count + bounds; inject into the system prompt.

- **Effort**: ~1 day.

### B3. LLM backend abstraction

`nl/llm.py` interface `complete(messages, schema) -> dict`. `ClaudeBackend` (Anthropic SDK, tool-use with the JSON schema, prompt caching on the system prompt). `OllamaBackend` (OpenAI-compat, json-schema mode) for offline/privacy. Select via arg or `HAKOWAN_NL_BACKEND`.

- Honest expectations: Claude ~95% on moderate prompts; 32B-class local models ~70–80%; below that, structured output degrades on nested schema.
- **Follow the `claude-api` skill for model IDs, caching, and tool-use specifics before coding this.**
- **Effort**: ~1 day.

### B4. Prompt + few-shot library

Hardest part — "hakowan" is not in training data, so few-shot carries the weight. `nl/prompts/system.md` (mark/channel/transform/material menus) + `nl/prompts/examples/*.json` (15–20 curated NL→spec pairs derived from `examples/`, easy→hard). Cache the system prompt + examples on the Claude path.

- **Effort**: ~2 days.

### B5. Validation + retry loop

LLM output → Pydantic parse → `spec_to_layer` → optional `compile()` dry-run. Pydantic error → re-prompt with the exact message (max 3 tries). Compile error (bad attribute) → re-prompt with the valid attribute list. Render error → bubble up.

- **Effort**: ~1 day.

### B6. Vision feedback loop

Now practical: WebGL default gives sub-second renders, so render→look→critique→refine is cheap (the old "Mitsuba too slow for eval" risk is gone). Gated by `hkw.nl.generate(..., iterate=True, max_iters=3)`. Each iteration renders a low-res preview, feeds the PNG back, requests diff edits. Claude path only.

- **Effort**: ~1.5 days. Optional for v1.

### B7. UX

CLI `python -m hakowan.nl "prompt" --data mesh.obj --out fig.png`. Notebook: `hkw.nl.generate(...)` returns a `Layer`, which already renders inline via the existing `_repr_html_` — so generated specs preview automatically. No new display code needed.

- **Effort**: ~1 day.

### B8. Eval harness

`tests/nl/eval.py`: 30 NL prompts with reference specs. Metrics: schema-valid %, compile %, render %, semantic match (LLM-judge or manual). Run vs Claude + a local 32B model; publish the numbers to justify model floors.

- **Effort**: ~1 day.

---

## Sequencing

| Milestone | Theme | Items | Outcome |
|-----------|-------|-------|---------|
| **M1** (~2 wk) | Publishable figures | A1 colorbar, A2 serialization | Figures gain colorbars; specs round-trip — the two biggest blockers gone |
| **M2** (~2 wk) | Reach + reproducibility | A3 schema, A4 data inputs, A5 camera/lighting | Non-mesh users onboard; full spec reproduces the figure |
| **M3** (~1 wk) | Discoverability | A6 gallery | People can actually find and copy examples |
| **M4** (~2 wk) | LLM layer | B0–B5, B7, B8 | `hkw.nl.generate(...)` ships; specs are editable + reproducible |
| **M5** (optional) | LLM polish | B6 vision loop, A7 marks/animation | Self-correcting generation; broader grammar |

**Critical path for the LLM goal**: A2 + A3 → B1 → B3 → B4 → B5. A1 should land before any NL release so generated figures aren't missing colorbars.

## What to resist

- Interactivity beyond preview (selection/brushing) — leave to deck.gl/polyscope.
- 2D charts — plenty of tools exist; stay in 3D mesh.
- Generative/scene-creation features and "text-to-3D" framing — wrong lane, wrong expectations.
- Merging marks/channels into one config blob — the separation is the grammar's strength.
- Hand-maintaining the LLM schema — auto-generate it from the dataclasses (A3) so it can't drift.

## References

- Colorbar design detail: memory `hakowan-colorbar-plan.md`.
- NL agent detail: memory `hakowan-nl-plan.md`.
- Full tiered roadmap + competitive landscape: memory `hakowan-roadmap.md`.
