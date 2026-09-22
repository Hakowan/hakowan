"""Run the LLM evaluation through an isolated Oh My Pi MCP host."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Literal, TextIO

import lagrange

from .datasets import dataset, names as dataset_names
from .runner import load_cases

EnvelopeStatus = Literal["direct_json", "normalized_envelope", "unusable"]
_REQUIRED_TOOLS = {
    "mcp__hakowan_get_backends",
    "mcp__hakowan_get_schema",
    "mcp__hakowan_get_spec",
    "mcp__hakowan_get_spec_template",
    "mcp__hakowan_inspect_data",
    "mcp__hakowan_search_gallery",
    "mcp__hakowan_validate_spec",
}


@dataclass(frozen=True, slots=True)
class ParsedEnvelope:
    """One model response separated from its output-envelope compliance."""

    status: EnvelopeStatus
    response: dict[str, Any] | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class HostRun:
    """Captured output and bounded-execution state for one host invocation."""

    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    timed_out: bool
    tool_budget_exceeded: bool


def _response_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if "kind" in value and "value" in value:
        return value
    if "root" in value:
        return {"kind": "spec", "value": value}
    return None


def parse_response_envelope(raw: str) -> ParsedEnvelope:
    """Parse direct JSON first, then deterministic fenced/prose JSON envelopes."""
    text = raw.strip()
    try:
        direct = _response_object(json.loads(text))
    except json.JSONDecodeError:
        direct = None
    if direct is not None:
        return ParsedEnvelope("direct_json", direct)

    candidates = re.findall(
        r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = _response_object(json.loads(candidate))
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            return ParsedEnvelope("normalized_envelope", parsed)
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            parsed = _response_object(value)
            if parsed is not None:
                return ParsedEnvelope("normalized_envelope", parsed)
    return ParsedEnvelope("unusable", None, "No complete JSON response object found.")


def parse_event_stream(output: str) -> tuple[dict[str, Any], ...]:
    """Parse JSON event lines while ignoring host progress output."""
    events = []
    for line in output.splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return tuple(events)


def final_assistant_text(events: tuple[dict[str, Any], ...]) -> str:
    """Return the last assistant text payload from an OMP event stream."""
    texts = []
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if text:
            texts.append(text)
    return texts[-1].strip() if texts else ""


def _tool_identity(event: dict[str, Any]) -> str:
    """Return the logical tool behind a direct or xdev-transported call."""
    name = str(event.get("toolName", ""))
    arguments = event.get("args")
    if name in {"read", "write"} and isinstance(arguments, dict):
        path = arguments.get("path")
        if isinstance(path, str) and path.startswith("xd://mcp__hakowan_"):
            return path.removeprefix("xd://")
        if isinstance(path, str) and path.startswith("mcp://"):
            return path
    return name


def summarize_events(events: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Summarize tool usage, provider retries, and token/cost telemetry."""
    tools = [
        _tool_identity(event)
        for event in events
        if event.get("type") == "tool_execution_start"
    ]
    retry_attempts = [
        int(event.get("attempt", 0))
        for event in events
        if event.get("type") == "auto_retry_end"
    ]
    usage: Counter[str] = Counter()
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        values = message.get("usage")
        if not isinstance(values, dict):
            continue
        usage.update(
            {
                "input": values.get("input", 0),
                "output": values.get("output", 0),
                "cache_read": values.get("cacheRead", 0),
                "cache_write": values.get("cacheWrite", 0),
                "premium_requests": values.get("premiumRequests", 0),
            }
        )
        usage["reported_cost"] += values.get("cost", {}).get("total", 0)
    return {
        "tools": tools,
        "tool_count": len(tools),
        "non_mcp_tools": [
            name for name in tools if not name.startswith(("mcp__hakowan_", "mcp://"))
        ],
        "missing_required_tools": sorted(_REQUIRED_TOOLS - set(tools)),
        "provider_auto_retry_attempts": max(retry_attempts, default=0),
        "usage": dict(usage),
    }


def benchmark_prompt(case: dict[str, Any]) -> str:
    """Build the fixed strict-MCP prompt for one benchmark case."""
    source = f"data/{case['dataset']}.ply"
    return f'''You are being benchmarked as a Hakowan FigureSpec author in strict MCP-only isolation. Use only Hakowan MCP tools.
Case ID: {case["id"]}
Source file: {source}
Backend: webgl
Request: {case["prompt"]}

Required procedure:
1. Call inspect_data on the source. Never invent attributes.
2. Use get_spec_template and focused get_schema fragments. Call get_backends and search_gallery.
3. Author one complete canonical FigureSpec whose geometry source is exactly {{"kind":"external","id":"data"}}.
4. Call validate_spec with backend="webgl", strict=true, compile_check=true, and data_bindings={{"data":"{source}"}}. Continue with the returned spec_id.
5. Repair failures with minimal apply_patch operations against the spec_id, at most twice.
6. When framing matters, call fit_camera and use its returned spec_id.
7. Call get_spec once at the end and return that exact canonical specification.

Return only one JSON object, without Markdown or prose:
{{"kind":"spec","value":<FigureSpec>,"metadata":{{"notes":"brief facts"}}}}'''


def prepare_workspace(path: Path, *, gallery: Path | None, repository: Path) -> None:
    """Write deterministic datasets and an isolated project MCP configuration."""
    data_dir = path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in dataset_names():
        lagrange.io.save_mesh(data_dir / f"{name}.ply", dataset(name))
    config_path = path / ".mcp.json"
    arguments = ["-m", "hakowan.mcp", "--root", str(path)]
    if gallery is not None:
        arguments.extend(("--gallery", str(gallery.resolve())))
    config = {
        "mcpServers": {
            "hakowan": {
                "command": sys.executable,
                "args": arguments,
                "cwd": str(repository.resolve()),
            }
        }
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def _reader(
    label: str, stream: TextIO, output: queue.Queue[tuple[str, str | None]]
) -> None:
    try:
        for line in iter(stream.readline, ""):
            output.put((label, line))
    finally:
        output.put((label, None))


def run_host(
    command: list[str], *, cwd: Path, timeout: int, max_tool_calls: int
) -> HostRun:
    """Run one host process with hard wall-clock and MCP tool-call limits."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    assert process.stdout is not None and process.stderr is not None
    output: queue.Queue[tuple[str, str | None]] = queue.Queue()
    threads = [
        threading.Thread(
            target=_reader, args=("stdout", process.stdout, output), daemon=True
        ),
        threading.Thread(
            target=_reader, args=("stderr", process.stderr, output), daemon=True
        ),
    ]
    for thread in threads:
        thread.start()

    stdout: list[str] = []
    stderr: list[str] = []
    closed: set[str] = set()
    tool_calls = 0
    timed_out = False
    tool_budget_exceeded = False
    deadline = started + timeout
    while len(closed) < 2:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            process.terminate()
            break
        try:
            label, line = output.get(timeout=min(0.25, remaining))
        except queue.Empty:
            continue
        if line is None:
            closed.add(label)
            continue
        (stdout if label == "stdout" else stderr).append(line)
        if label == "stdout" and line.startswith("{"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict) and event.get("type") == "tool_execution_start":
                tool_calls += 1
                if tool_calls > max_tool_calls:
                    tool_budget_exceeded = True
                    process.terminate()
                    break

    try:
        return_code = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return_code = process.wait()
    for thread in threads:
        thread.join(timeout=1)
    while not output.empty():
        label, line = output.get_nowait()
        if line is not None:
            (stdout if label == "stdout" else stderr).append(line)
    return HostRun(
        stdout="".join(stdout),
        stderr="".join(stderr),
        return_code=return_code,
        duration_seconds=round(time.monotonic() - started, 3),
        timed_out=timed_out,
        tool_budget_exceeded=tool_budget_exceeded,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run selected cases and return a replay-compatible response bundle."""
    repository = Path(__file__).resolve().parents[2]
    selected = load_cases(args.cases)
    if args.case_ids:
        wanted = set(args.case_ids)
        selected = tuple(case for case in selected if case.id in wanted)
        missing = wanted - {case.id for case in selected}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    if shutil.which(args.omp_command) is None:
        raise FileNotFoundError(f"Host command not found: {args.omp_command}")

    with tempfile.TemporaryDirectory(prefix="hakowan-mcp-benchmark-") as directory:
        workspace = Path(directory)
        prepare_workspace(workspace, gallery=args.gallery, repository=repository)
        events_dir = args.output.with_suffix(".events")
        if args.keep_events:
            events_dir.mkdir(parents=True, exist_ok=True)
        responses: dict[str, list[dict[str, Any]]] = {}
        runs: dict[str, dict[str, Any]] = {}
        formats: Counter[str] = Counter()
        all_tools: Counter[str] = Counter()
        all_usage: Counter[str] = Counter()
        provider_retry_cases: dict[str, int] = {}
        for index, case in enumerate(selected, 1):
            command = [
                args.omp_command,
                "--model",
                args.model,
                "--thinking",
                args.thinking,
                "--no-lsp",
                "--no-extensions",
                "--no-skills",
                "--no-rules",
                "--no-session",
                "--no-title",
                "--mode",
                "json",
                "--max-time",
                str(args.timeout),
                "-p",
                benchmark_prompt(
                    {"id": case.id, "dataset": case.dataset, "prompt": case.prompt}
                ),
            ]
            host = run_host(
                command,
                cwd=workspace,
                timeout=args.timeout + 30,
                max_tool_calls=args.max_tool_calls,
            )
            events = parse_event_stream(host.stdout)
            summary = summarize_events(events)
            raw = final_assistant_text(events)
            envelope = parse_response_envelope(raw)
            timed_out = host.timed_out or (
                host.return_code != 0
                and host.duration_seconds >= max(args.timeout * 0.9, 1.0)
            )
            formats[envelope.status] += 1
            all_tools.update(summary["tools"])
            all_usage.update(summary["usage"])
            if summary["provider_auto_retry_attempts"]:
                provider_retry_cases[case.id] = summary["provider_auto_retry_attempts"]
            diagnostics = []
            if timed_out:
                diagnostics.append(
                    {
                        "code": "harness.timeout",
                        "message": f"Case exceeded {args.timeout} seconds.",
                    }
                )
            if host.tool_budget_exceeded:
                diagnostics.append(
                    {
                        "code": "harness.tool_budget",
                        "message": f"Tool-call limit {args.max_tool_calls} was exceeded.",
                    }
                )
            if summary["non_mcp_tools"]:
                diagnostics.append(
                    {
                        "code": "harness.non_mcp_tool",
                        "message": f"Non-MCP tools used: {summary['non_mcp_tools']}",
                    }
                )
            if summary["missing_required_tools"]:
                diagnostics.append(
                    {
                        "code": "harness.missing_required_tool",
                        "message": (
                            "Required tools not called: "
                            f"{summary['missing_required_tools']}"
                        ),
                    }
                )

            if (
                host.return_code != 0
                and not timed_out
                and not host.tool_budget_exceeded
            ):
                diagnostics.append(
                    {
                        "code": "harness.process_exit",
                        "message": f"Host exited with status {host.return_code}.",
                    }
                )
            response = envelope.response
            if response is None or diagnostics:
                response = {"kind": "spec", "value": {}}
            metadata = response.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"model_metadata": metadata}
            metadata.update(
                {
                    "host": "oh-my-pi",
                    "model": args.model,
                    "strict_mcp_only": True,
                    "response_format": {
                        "status": envelope.status,
                        "error": envelope.error,
                    },
                    "stderr": host.stderr[-4000:] or None,
                    "host_diagnostics": diagnostics,
                    "exit_code": host.return_code,
                    "timed_out": timed_out,
                    "duration_seconds": host.duration_seconds,
                    **summary,
                    "harness_passed": not diagnostics,
                }
            )
            responses[case.id] = [
                {
                    "kind": response.get("kind", "spec"),
                    "value": response.get("value", {}),
                    "raw": raw,
                    "metadata": metadata,
                }
            ]
            runs[case.id] = metadata
            if args.keep_events:
                (events_dir / f"{case.id}.jsonl").write_text(host.stdout)
            print(
                f"[{index:02d}/{len(selected)}] {case.id}: "
                f"format={envelope.status} tools={summary['tool_count']} "
                f"timeout={timed_out}",
                flush=True,
            )
    return {
        "model": args.model,
        "responses": responses,
        "run_metadata": {
            "host": "oh-my-pi",
            "model": args.model,
            "isolation": "strict-mcp-only",
            "case_count": len(selected),
            "case_timeout_seconds": args.timeout,
            "max_tool_calls": args.max_tool_calls,
            "usage": dict(all_usage),
            "provider_auto_retry_attempts": sum(provider_retry_cases.values()),
            "provider_auto_retry_cases": dict(sorted(provider_retry_cases.items())),
            "non_mcp_tool_cases": sorted(
                cid for cid, run in runs.items() if run["non_mcp_tools"]
            ),
            "missing_required_tool_cases": {
                cid: run["missing_required_tools"]
                for cid, run in sorted(runs.items())
                if run["missing_required_tools"]
            },
            "failed_host_cases": sorted(
                cid for cid, run in runs.items() if not run["harness_passed"]
            ),
            "response_formats": dict(sorted(formats.items())),
            "tool_calls": dict(sorted(all_tools.items())),
            "timed_out_cases": sorted(
                cid for cid, run in runs.items() if run["timed_out"]
            ),
            "tool_budget_exceeded_cases": sorted(
                cid
                for cid, run in runs.items()
                if any(
                    item["code"] == "harness.tool_budget"
                    for item in run["host_diagnostics"]
                )
            ),
            "runs": runs,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse strict MCP harness options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="OMP provider/model identifier")
    parser.add_argument(
        "--output", type=Path, required=True, help="Replay JSON output path"
    )
    parser.add_argument("--cases", type=Path, help="Alternate benchmark case file")
    parser.add_argument(
        "--case", action="append", dest="case_ids", help="Case ID; repeatable"
    )
    parser.add_argument("--gallery", type=Path, help="Optional local gallery checkout")
    parser.add_argument("--omp-command", default="omp")
    parser.add_argument("--thinking", default="low")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tool-calls", type=int, default=32)
    parser.add_argument("--keep-events", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the strict MCP harness and write replay-compatible responses."""
    args = parse_args(argv)
    payload = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
