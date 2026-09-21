# Agent integration with MCP

Hakowan exposes its deterministic visualization workflow through the Model
Context Protocol (MCP). The external host—such as a coding agent or IDE—owns
language understanding, model selection, credentials, and conversation state.
Hakowan owns data inspection, schema validation, compilation, rendering,
observation, and safe patches. The server never calls an LLM provider.

## Install

```sh
pip install "hakowan[mcp]"
```

The MCP integration uses the official Python MCP SDK v2.

## Start a local server

For local agent hosts, use the default stdio transport:

```sh
hakowan-mcp \
  --root /path/to/project \
  --gallery /path/to/hakowan-gallery
```

The equivalent module command is:

```sh
python -m hakowan.mcp --root /path/to/project
```

`--root` is the security boundary for all agent-controlled reads and writes.
Relative paths resolve under it. Absolute paths, `..` traversal, and symlinks
that resolve outside it are rejected.

For a local Streamable HTTP endpoint:

```sh
hakowan-mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000 \
  --root /path/to/project
```

Clients connect to `http://127.0.0.1:8000/mcp`. Keep the default loopback host
unless authentication and transport security are configured outside Hakowan.

## Host configuration

A generic stdio MCP configuration is:

```json
{
  "servers": {
    "hakowan": {
      "type": "stdio",
      "command": "hakowan-mcp",
      "args": [
        "--root", "/path/to/project",
        "--gallery", "/path/to/hakowan-gallery"
      ]
    }
  }
}
```

GitHub Copilot in VS Code can place this server entry in `.vscode/mcp.json`.
Other hosts use the same command and arguments under their MCP-server
configuration format. For repository-aware harnesses such as Oh My Pi, register
the command as a project MCP server and set the project directory as `--root`.

## Tools

| Tool | Purpose |
|---|---|
| `get_schema` | Return the canonical `FigureSpec` JSON Schema. |
| `inspect_data` | Inspect geometry, topology, attributes, ranges, and non-finite values. |
| `search_gallery` | Retrieve feature-matched canonical gallery recipes. |
| `validate_spec` | Validate schema, resources, semantics, backend support, and compilation. |
| `compile_spec` | Return resolved views, bounds, counts, legends, and annotations. |
| `render_spec` | Validate and render a specification inside the workspace. |
| `observe_spec` | Capture deterministic views, passes, visibility, and occlusion evidence. |
| `get_backends` | Return declared backend features, passes, and limitations. |
| `apply_patch` | Atomically apply JSON Pointer patches and revalidate. |

The server also exposes:

- `hakowan://schema` — canonical JSON Schema resource;
- `hakowan://agent-instructions` — the recommended safe workflow;
- `author_figure` — a reusable prompt containing that workflow.

## Recommended agent loop

```text
inspect_data
    ↓
search_gallery + get_schema
    ↓
author canonical FigureSpec JSON
    ↓
validate_spec(strict=true)
    ↓
render_spec
    ↓
observe_spec when visual evidence matters
    ↓
apply_patch for minimal repairs
```

The agent should never invent attributes. Validation failures are returned as
structured data with stable diagnostic codes, paths, messages, and hints.

## Data references

Mesh-file specifications may use paths relative to `--root`. In-memory-style
external IDs use explicit workspace bindings:

```json
{
  "spec": {
    "$schema": "https://hakowan.github.io/hakowan/schema/v1.json",
    "version": "1.0",
    "root": {
      "kind": "layer",
      "spec": {
        "data": {"kind": "external", "id": "input"}
      }
    }
  },
  "data_bindings": {
    "input": "data/simulation.ply"
  }
}
```

Arbitrary Python function references are intentionally unsupported through MCP.
Use Hakowan's restricted expression specification for serializable filters and
custom scales.

## Gallery discovery

`search_gallery` looks for recipes in this order:

1. the `--gallery` argument;
2. `HAKOWAN_GALLERY`;
3. a sibling `hakowan-gallery` checkout beside the workspace.

Set `include_spec=true` only when the full canonical example is needed; compact
metadata is cheaper for discovery.

## Development and inspection

Run with the official MCP Inspector:

```sh
mcp dev src/hakowan/mcp/app.py --with-editable .
```

Or exercise the installed server from any MCP v2 client using the
`hakowan-mcp` command. The default stdio transport reserves stdout for protocol
messages; Hakowan logs are emitted through Python logging rather than prints.
