# Hakowan Improvement Plan

Status: draft, 2026-06-14, against v0.5.2 (branch `dev/0.5.2`).

## Goals

1. **Make hakowan more useful as a 3D visualization tool** — close the gaps that block it from being the default tool for publication-quality 3D mesh figures.
2. **Make Hakowan easy for external agents to use** — expose deterministic visualization tools through MCP while leaving provider communication and reasoning to the host.

## Guiding principles

- **The agent amplifies the grammar; it cannot exceed it.** A model can only emit specs the grammar can express. Foundation work remains the prerequisite for reliable agent use.
- **Preserve "one spec = one image."** Everything determining the figure belongs in the declarative FigureSpec; invocation policy remains outside it.
- **Stay in the niche.** Hakowan owns 3D data semantics, validation, rendering, observation, and safe edits—not model providers, authentication, or conversation state.
- **Use a standard agent boundary.** MCP exposes inspect → author → validate → render → observe → patch to Copilot, Oh My Pi, and other hosts.
- **Hakowan is a complement, not a replacement.** Use interactive tools to debug data and Hakowan to produce reproducible figures.

The structural bet remains a declarative spec, but external agent harnesses own natural-language interpretation and provider integrations.

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

## Track B — Agent integration

Hakowan remains provider-neutral. It does not maintain OpenAI, Anthropic,
Ollama, or other model clients. External MCP hosts own credentials, model
selection, language understanding, conversation state, and retry reasoning.

### B1. Agent-facing schema and inspection — shipped

Canonical FigureSpec JSON Schema, `hkw.inspect()`, strict semantic validation,
backend capability reports, and the canonical gallery provide grounded context.

### B2. Deterministic authoring tools — shipped

Compilation, high-level framing, rendering, observation queries, and atomic
JSON Pointer patches provide the complete deterministic authoring loop.

### B3. MCP server — shipped

`pip install "hakowan[mcp]"` installs the provider-neutral MCP server. It
exposes schema retrieval, data inspection, gallery search, validation,
compilation, rendering, observation, and patching over stdio or Streamable
HTTP. All agent-controlled paths are confined to an explicit workspace root.

### B4. Agent workflow guidance — shipped

The MCP server publishes a reusable authoring prompt and workflow resource:
inspect data, retrieve schema/examples, author FigureSpec JSON, validate
strictly, render/observe, and repair with minimal patches.

### B5. Evaluation harness — shipped

`benchmarks/llm_eval` provides 20 natural-language visualization cases over
five deterministic scientific datasets. It records schema, semantic, compile,
render, grounding, grammar, camera, repair, and patch-minimality scores; emits
JSON and HTML reports; consumes canonical gallery examples; and supports
deterministic reference, stored replay, and plugin-based live providers.
Browser-backed occupancy scoring is optional via `--observe`.

### B6. Vision refinement — host-owned

External agents may render, observe structured visibility/occlusion evidence,
and apply bounded patches. Hakowan supplies the deterministic primitives but
does not own the reasoning loop.

---

## Sequencing

| Milestone | Theme | Items | Outcome |
|-----------|-------|-------|---------|
| **M1** (~2 wk) | Publishable figures | A1 colorbar, A2 serialization | Figures gain colorbars; specs round-trip — the two biggest blockers gone |
| **M2** (~2 wk) | Reach + reproducibility | A3 schema, A4 data inputs, A5 camera/lighting | Non-mesh users onboard; full spec reproduces the figure |
| **M3** (~1 wk) | Discoverability | A6 gallery | People can actually find and copy examples |
| **M4** (shipped) | Agent integration | B1–B5 | MCP hosts can inspect, author, validate, render, observe, patch, and evaluate specs |
| **M5** (host-owned) | Agent refinement | B6, A7 | External agents iterate on visual evidence; Hakowan broadens visualization grammar |

**Agent critical path**: schema + inspection → strict validation → render/observe → atomic patch. MCP exposes this path without provider-specific code.

## What to resist

- Interactivity beyond preview (selection/brushing) — leave to deck.gl/polyscope.
- 2D charts — plenty of tools exist; stay in 3D mesh.
- Generative/scene-creation features and "text-to-3D" framing — wrong lane, wrong expectations.
- Merging marks/channels into one config blob — the separation is the grammar's strength.
- Hand-maintaining the LLM schema — auto-generate it from the dataclasses (A3) so it can't drift.
- Provider SDKs, model routing, credentials, and conversation orchestration — keep them in MCP hosts, not Hakowan core.

## References

- Colorbar design detail: memory `hakowan-colorbar-plan.md`.
- NL agent detail: memory `hakowan-nl-plan.md`.
- Full tiered roadmap + competitive landscape: memory `hakowan-roadmap.md`.
