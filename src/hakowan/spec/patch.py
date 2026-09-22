"""Atomic JSON Pointer patches for canonical and runtime Hakowan specifications."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import ValidationError as PydanticValidationError

from ..backends import BackendName
from ..grammar.figure import Figure
from ..grammar.layer import Layer
from ..workflow.validation import ValidationReport, validate
from .codec import (
    DataIds,
    DataResolver,
    FunctionIds,
    FunctionResolver,
    SpecConversionError,
    from_spec,
    to_spec,
)
from .model import FigureSpec


class PatchOperation(TypedDict):
    """One supported JSON Patch operation."""

    op: Literal["add", "remove", "replace"]
    path: str
    value: NotRequired[Any]


@dataclass(frozen=True, slots=True)
class PatchFailure:
    """Machine-readable cause of a rejected atomic patch."""

    code: str
    path: str
    message: str
    operation_index: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        """Return this failure as a JSON-safe mapping."""
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "operation_index": self.operation_index,
        }


class PatchError(ValueError):
    """Raised when a patch operation, schema, or semantic check fails."""

    def __init__(
        self,
        failure: PatchFailure,
        *,
        validation_report: ValidationReport | None = None,
    ) -> None:
        """Initialize an error with its structured failure and validation report."""
        self.failure = failure
        self.validation_report = validation_report
        location = (
            f"operation {failure.operation_index} at {failure.path}"
            if failure.operation_index is not None
            else failure.path
        )
        super().__init__(f"{failure.code}: {location}: {failure.message}")


def _tokens(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must be empty or start with '/'")
    result: list[str] = []
    for raw in pointer[1:].split("/"):
        index = 0
        while index < len(raw):
            if raw[index] == "~" and (
                index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}
            ):
                raise ValueError("JSON Pointer contains an invalid '~' escape")
            index += 2 if raw[index] == "~" else 1
        result.append(raw.replace("~1", "/").replace("~0", "~"))
    return result


def _list_index(token: str, length: int, *, append: bool = False) -> int:
    if append and token == "-":
        return length
    if not token or (token.startswith("0") and token != "0") or not token.isdigit():
        raise ValueError(f"Invalid array index {token!r}")
    return int(token)


def _parent(document: Any, tokens: list[str]) -> tuple[Any, str]:
    if not tokens:
        raise ValueError("The document root has no parent")
    current = document
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"Object member {token!r} does not exist")
            current = current[token]
        elif isinstance(current, list):
            index = _list_index(token, len(current))
            if index >= len(current):
                raise IndexError(f"Array index {index} is outside [0, {len(current)})")
            current = current[index]
        else:
            raise TypeError(f"Cannot traverse into {type(current).__name__}")
    return current, tokens[-1]


def _apply_one(document: Any, operation: Mapping[str, Any]) -> Any:
    if not isinstance(operation, Mapping):
        raise TypeError("Patch operation must be an object")
    op = operation.get("op")
    path = operation.get("path")
    if op not in {"add", "remove", "replace"}:
        raise ValueError("op must be 'add', 'remove', or 'replace'")
    if not isinstance(path, str):
        raise TypeError("path must be a JSON Pointer string")
    if op in {"add", "replace"} and "value" not in operation:
        raise ValueError(f"{op} requires a value")
    tokens = _tokens(path)
    if not tokens:
        if op == "remove":
            raise ValueError("Cannot remove the document root")
        return copy.deepcopy(operation["value"])

    parent, token = _parent(document, tokens)
    if isinstance(parent, dict):
        if op in {"remove", "replace"} and token not in parent:
            raise KeyError(f"Object member {token!r} does not exist")
        if op == "remove":
            del parent[token]
        else:
            parent[token] = copy.deepcopy(operation["value"])
        return document
    if isinstance(parent, list):
        index = _list_index(token, len(parent), append=op == "add")
        if op == "add":
            if index > len(parent):
                raise IndexError(
                    f"Array insertion index {index} exceeds length {len(parent)}"
                )
            parent.insert(index, copy.deepcopy(operation["value"]))
        else:
            if index >= len(parent):
                raise IndexError(f"Array index {index} is outside [0, {len(parent)})")
            if op == "remove":
                del parent[index]
            else:
                parent[index] = copy.deepcopy(operation["value"])
        return document
    raise TypeError(f"Cannot modify a member of {type(parent).__name__}")


def _schema_pointer(location: tuple[Any, ...]) -> str:
    return "".join(
        "/" + str(item).replace("~", "~0").replace("/", "~1") for item in location
    )


def patch_spec(
    spec: FigureSpec | Mapping[str, Any],
    operations: Iterable[PatchOperation | Mapping[str, Any]],
) -> FigureSpec:
    """Apply atomic JSON Pointer operations and validate the resulting schema.

    Supported operations are ``add``, ``remove``, and ``replace``. All
    operations run against a private deep copy; failure leaves ``spec`` and
    supplied values unchanged. ``-`` appends to an array.

    Args:
        spec: Immutable FigureSpec or a canonical specification mapping.
        operations: Ordered patch operations using RFC 6901 pointer paths.

    Returns:
        A newly validated immutable FigureSpec.

    Raises:
        PatchError: If an operation or final schema is invalid.

    """
    document = copy.deepcopy(
        spec.to_dict() if isinstance(spec, FigureSpec) else dict(spec)
    )
    for index, operation in enumerate(operations):
        path = operation.get("path") if isinstance(operation, Mapping) else None
        try:
            document = _apply_one(document, operation)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise PatchError(
                PatchFailure(
                    code="patch.operation",
                    path=path if isinstance(path, str) else "",
                    message=str(exc),
                    operation_index=index,
                )
            ) from exc
    try:
        return FigureSpec.model_validate(document)
    except PydanticValidationError as exc:
        error = exc.errors(include_url=False)[0]
        raise PatchError(
            PatchFailure(
                code="patch.schema",
                path=_schema_pointer(error["loc"]),
                message=error["msg"],
            )
        ) from exc


def _external_identifier(value: Any, provider: Any, prefix: str, index: int) -> str:
    if provider is None:
        return f"@hakowan/{prefix}/{index}"
    if callable(provider):
        identifier = provider(value)
    else:
        try:
            identifier = provider[value]
        except (KeyError, TypeError):
            identifier = provider[id(value)]
    if not identifier:
        raise SpecConversionError(f"External {prefix} id must not be empty")
    return str(identifier)


def _runtime_spec_and_resolvers(
    value: Layer | Figure,
    data_ids: DataIds | None,
    function_ids: FunctionIds | None,
    data_resolver: DataResolver | None,
    function_resolver: FunctionResolver | None,
) -> tuple[FigureSpec, DataResolver, FunctionResolver]:
    data_values: dict[str, Any] = {}
    function_values: dict[str, Callable[..., Any]] = {}

    def identify_data(mesh: Any) -> str:
        identifier = _external_identifier(mesh, data_ids, "data", len(data_values))
        if identifier in data_values and data_values[identifier] is not mesh:
            raise SpecConversionError(
                f"External data id {identifier!r} refers to multiple meshes"
            )
        data_values[identifier] = mesh
        return identifier

    def identify_function(function: Callable[..., Any]) -> str:
        identifier = _external_identifier(
            function, function_ids, "function", len(function_values)
        )
        if (
            identifier in function_values
            and function_values[identifier] is not function
        ):
            raise SpecConversionError(
                f"External function id {identifier!r} refers to multiple callables"
            )
        function_values[identifier] = function
        return identifier

    spec = to_spec(value, data_ids=identify_data, function_ids=identify_function)

    def resolve_data(identifier: str) -> Any:
        if identifier in data_values:
            return data_values[identifier]
        if data_resolver is None:
            raise KeyError(identifier)
        return (
            data_resolver(identifier)
            if callable(data_resolver)
            else data_resolver[identifier]
        )

    def resolve_function(identifier: str) -> Callable[..., Any]:
        if identifier in function_values:
            return function_values[identifier]
        if function_resolver is None:
            raise KeyError(identifier)
        return (
            function_resolver(identifier)
            if callable(function_resolver)
            else function_resolver[identifier]
        )

    return spec, resolve_data, resolve_function


def patch(
    value: Layer | Figure,
    operations: Iterable[PatchOperation | Mapping[str, Any]],
    *,
    data_ids: DataIds | None = None,
    function_ids: FunctionIds | None = None,
    data_resolver: DataResolver | None = None,
    function_resolver: FunctionResolver | None = None,
    base_dir: str | Path | None = None,
    backend: BackendName | None = None,
    strict: bool = False,
    semantic: bool = True,
) -> Layer | Figure:
    """Atomically patch a runtime Layer or Figure through its canonical form.

    In-memory meshes and callable references are rebound automatically.
    Explicit identifier and resolver hooks support references introduced by a
    patch. Schema validation always runs; semantic validation runs by default.

    Args:
        value: Runtime Layer or Figure to patch without mutation.
        operations: Ordered add, remove, or replace operations.
        data_ids: Optional stable IDs for existing in-memory meshes.
        function_ids: Optional stable IDs for existing callables.
        data_resolver: Resolver for new external data IDs.
        function_resolver: Resolver for new external function IDs.
        base_dir: Base directory for relative resource paths.
        backend: Backend used by semantic validation.
        strict: Promote backend degradations to semantic errors.
        semantic: Run semantic and compile validation when true.

    Returns:
        A reconstructed Layer or Figure containing all patch operations.

    Raises:
        PatchError: If conversion, an operation, schema validation, or semantic
            validation fails.

    """
    if not isinstance(value, (Layer, Figure)):
        raise TypeError(f"Expected Layer or Figure, got {type(value)!r}")
    try:
        spec, runtime_data, runtime_functions = _runtime_spec_and_resolvers(
            value, data_ids, function_ids, data_resolver, function_resolver
        )
        patched_spec = patch_spec(spec, operations)
        result = from_spec(
            patched_spec,
            data_resolver=runtime_data,
            function_resolver=runtime_functions,
            base_dir=base_dir,
        )
    except PatchError:
        raise
    except (KeyError, SpecConversionError, TypeError, ValueError) as exc:
        raise PatchError(PatchFailure("patch.conversion", "", str(exc))) from exc
    if semantic:
        report = validate(result, backend=backend, strict=strict)
        if not report.valid:
            first = report.errors[0]
            raise PatchError(
                PatchFailure(
                    code="patch.semantic",
                    path=first.path,
                    message=first.message,
                ),
                validation_report=report,
            )
    return result


__all__ = [
    "PatchError",
    "PatchFailure",
    "PatchOperation",
    "patch",
    "patch_spec",
]
