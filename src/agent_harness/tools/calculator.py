"""Arithmetic limited to numbers and the four basic operations."""

from __future__ import annotations

import ast
from decimal import ROUND_HALF_UP, Decimal

from agent_harness.core.errors import ToolError

_MAX_EXPRESSION_LENGTH = 200
_BINARIES = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
}


def format_decimal(value: Decimal) -> str:
    quantized = value.quantize(Decimal("0.0000000001"), rounding=ROUND_HALF_UP)
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _number(node: ast.Constant) -> Decimal:
    if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
        raise ToolError("expression may contain only numbers and + - * /")
    raw = str(node.value) if isinstance(node.value, float) else str(int(node.value))
    return Decimal(raw)


def _eval(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant):
        return _number(node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARIES:
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ToolError("division by zero")
        return _BINARIES[type(node.op)](left, right)
    raise ToolError("expression may contain only numbers and + - * /")


def evaluate_expression(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("expression is empty")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ToolError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError("expression could not be parsed") from exc
    return format_decimal(_eval(tree.body))
