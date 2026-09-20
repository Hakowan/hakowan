"""Restricted expression evaluator used by serializable transforms and scales."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

import numpy as np


class ExpressionError(ValueError):
    """Raised when an expression uses syntax outside the safe allowlist."""


_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}
_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "isfinite": np.isfinite,
    "norm": np.linalg.norm,
}
_NAMES = frozenset({"value", "x", "y", "z", "true", "false", "null"})


def _context(value: Any) -> dict[str, Any]:
    array = np.asarray(value)

    def component(index: int) -> Any:
        return array.reshape(-1)[index] if array.size > index else None

    return {
        "value": value,
        "x": component(0),
        "y": component(1),
        "z": component(2),
        "true": True,
        "false": False,
        "null": None,
    }


def _evaluate(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, context)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool, type(None))):
            return node.value
        raise ExpressionError(f"Unsupported literal: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in _NAMES:
            raise ExpressionError(f"Unknown name: {node.id!r}")
        return context[node.id]
    if isinstance(node, ast.List):
        return [_evaluate(item, context) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(item, context) for item in node.elts)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](
            _evaluate(node.left, context), _evaluate(node.right, context)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand, context))
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for item in node.values:
                result = _evaluate(item, context)
                if not bool(result):
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for item in node.values:
                result = _evaluate(item, context)
                if bool(result):
                    return result
            return result
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, context)
        for op, comparator in zip(node.ops, node.comparators):
            function = _COMPARE.get(type(op))
            if function is None:
                raise ExpressionError(f"Unsupported comparison: {type(op).__name__}")
            right = _evaluate(comparator, context)
            if not bool(function(left, right)):
                return False
            left = right
        return True
    if isinstance(node, ast.Subscript):
        value = _evaluate(node.value, context)
        index = _evaluate(node.slice, context)
        if not isinstance(index, int):
            raise ExpressionError("Only integer subscripts are allowed.")
        return value[index]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ExpressionError(
                "Only abs, min, max, isfinite, and norm calls are allowed."
            )
        if node.keywords:
            raise ExpressionError("Keyword arguments are not allowed.")
        return _FUNCTIONS[node.func.id](*[_evaluate(arg, context) for arg in node.args])
    raise ExpressionError(f"Unsupported syntax: {type(node).__name__}")


def compile_expression(source: str) -> Callable[[Any], Any]:
    """Compile a safe one-argument expression into a callable.

    Available names are ``value`` and vector aliases ``x``, ``y``, ``z``.
    Allowed functions are ``abs``, ``min``, ``max``, ``isfinite``, and ``norm``.
    """
    if len(source) > 1024:
        raise ExpressionError("Expression exceeds the 1024-character limit.")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid expression: {exc.msg}") from exc

    # Validate immediately so malformed specifications fail before execution.
    _validate(tree)

    def expression(value: Any) -> Any:
        return _evaluate(tree, _context(value))

    expression.__name__ = "hakowan_expression"
    expression.__doc__ = source
    return expression


def _validate(node: ast.AST) -> None:
    """Walk all nodes once, rejecting syntax that could execute arbitrary code."""
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.List,
        ast.Tuple,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Subscript,
        ast.Call,
        *tuple(_BINARY),
        *tuple(_UNARY),
        *tuple(_COMPARE),
        ast.And,
        ast.Or,
        ast.Load,
    )
    nodes = list(ast.walk(node))
    if len(nodes) > 128:
        raise ExpressionError("Expression exceeds the 128-node limit.")
    for child in nodes:
        if not isinstance(child, allowed):
            raise ExpressionError(f"Unsupported syntax: {type(child).__name__}")
        if isinstance(child, ast.Name) and child.id not in _NAMES | _FUNCTIONS.keys():
            raise ExpressionError(f"Unknown name: {child.id!r}")
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Pow):
            if not isinstance(child.right, ast.Constant) or not isinstance(
                child.right.value, (int, float)
            ):
                raise ExpressionError("Exponent must be a numeric literal.")
            if abs(child.right.value) > 16:
                raise ExpressionError("Exponent magnitude must not exceed 16.")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in _FUNCTIONS:
                raise ExpressionError(
                    "Only abs, min, max, isfinite, and norm calls are allowed."
                )


__all__ = ["ExpressionError", "compile_expression"]
