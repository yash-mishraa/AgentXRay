"""Hermetic mock tools with AST-safe execution and zero external access."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from agentxray.providers.base import ToolDefinition

# Safeguard thresholds for AST arithmetic evaluation
_MAX_EXPRESSION_LENGTH = 250
_MAX_AST_DEPTH = 10
_MAX_POWER_BASE = 10000
_MAX_POWER_EXPONENT = 100

# Supported safe arithmetic operators for calculator
_SAFE_OPERATORS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _check_ast_depth(node: ast.AST, current_depth: int = 1) -> None:
    """Ensure expression does not exceed nesting recursion limit."""
    if current_depth > _MAX_AST_DEPTH:
        raise ValueError(f"Expression exceeds maximum nesting depth of {_MAX_AST_DEPTH}")
    for child in ast.iter_child_nodes(node):
        _check_ast_depth(child, current_depth + 1)


def _safe_eval_math(node: ast.AST) -> float | int:
    """Safely evaluate an arithmetic AST expression without using eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if not math.isfinite(node.value):
            raise ValueError("Non-finite numeric constant (inf or nan) is not permitted")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = -_safe_eval_math(node.operand)
        if not math.isfinite(val):
            raise ValueError("Result is non-finite (inf or nan)")
        return val
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _safe_eval_math(node.operand)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported arithmetic operator: {op_type.__name__}")
        left = _safe_eval_math(node.left)
        right = _safe_eval_math(node.right)
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero")
        if op_type is ast.Pow:
            if abs(left) > _MAX_POWER_BASE or abs(right) > _MAX_POWER_EXPONENT:
                raise ValueError("Exponentiation operands exceed safe limits")
        res = _SAFE_OPERATORS[op_type](left, right)
        if isinstance(res, (int, float)) and not math.isfinite(res):
            raise ValueError("Calculation resulted in non-finite value (inf or nan)")
        return res
    raise ValueError(f"Unsafe or invalid syntax in mathematical expression: {type(node).__name__}")


def safe_calculator(expression: str) -> str:
    """Evaluate an arithmetic expression using strict AST inspection.

    Zero shell execution, zero code evaluation, zero file access.
    """
    if not isinstance(expression, str):
        return "Error: Expression must be a string"
    expr_clean = expression.strip()
    if not expr_clean:
        return "Error: Empty expression"
    if len(expr_clean) > _MAX_EXPRESSION_LENGTH:
        return f"Error: Expression exceeds maximum length of {_MAX_EXPRESSION_LENGTH} characters"
    try:
        parsed = ast.parse(expr_clean, mode="eval")
        _check_ast_depth(parsed)
        result = _safe_eval_math(parsed.body)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 6))
    except (ValueError, ZeroDivisionError, SyntaxError, OverflowError) as err:
        return f"Error: {err}"


# In-memory mock database records
_MOCK_DATABASE: dict[str, dict[str, Any]] = {
    "user_42": {"status": "active", "balance": 450.0, "role": "admin", "department": "finance"},
    "user_99": {"status": "suspended", "balance": 0.0, "role": "guest", "department": "marketing"},
    "order_101": {"status": "shipped", "total": 129.99, "customer": "user_42", "items": 3},
    "order_102": {"status": "pending", "total": 49.50, "customer": "user_42", "items": 1},
}


def mock_database_lookup(entity: str, field: str) -> str:
    """Query an in-memory dictionary table. Completely sandboxed."""
    ent = entity.strip().lower()
    fld = field.strip().lower()

    if ent not in _MOCK_DATABASE:
        return f"Error: Entity '{entity}' not found in sandboxed database"

    record = _MOCK_DATABASE[ent]
    if fld not in record:
        return f"Error: Field '{field}' not found for entity '{entity}'. Available fields: {list(record.keys())}"

    return str(record[fld])


def mock_unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between basic units using in-memory arithmetic."""
    u_from = from_unit.strip().lower()
    u_to = to_unit.strip().lower()

    # Length: km, mi, m
    length_to_m = {"m": 1.0, "km": 1000.0, "mi": 1609.344}
    if u_from in length_to_m and u_to in length_to_m:
        meters = value * length_to_m[u_from]
        res = meters / length_to_m[u_to]
        return str(round(res, 4))

    # Mass: kg, lb, g
    mass_to_g = {"g": 1.0, "kg": 1000.0, "lb": 453.59237}
    if u_from in mass_to_g and u_to in mass_to_g:
        grams = value * mass_to_g[u_from]
        res = grams / mass_to_g[u_to]
        return str(round(res, 4))

    return f"Error: Incompatible or unsupported units: '{from_unit}' to '{to_unit}'"


# Tool Definitions schemas
CALCULATOR_TOOL_DEF = ToolDefinition(
    name="calculator",
    description="Perform basic arithmetic calculations. Accepts standard arithmetic expressions containing numbers, +, -, *, /, and parentheses.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate, e.g. '150 + 250' or '42 * 10'.",
            }
        },
        "required": ["expression"],
    },
)

DATABASE_LOOKUP_TOOL_DEF = ToolDefinition(
    name="database_lookup",
    description="Query a sandboxed in-memory database table for entity properties (e.g. users, orders).",
    parameters={
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Entity identifier, e.g. 'user_42' or 'order_101'.",
            },
            "field": {
                "type": "string",
                "description": "Field name to inspect (e.g. 'status', 'balance', 'role', 'total').",
            },
        },
        "required": ["entity", "field"],
    },
)

UNIT_CONVERTER_TOOL_DEF = ToolDefinition(
    name="unit_converter",
    description="Convert quantities between standard units of length or mass.",
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "Numerical quantity."},
            "from_unit": {"type": "string", "description": "Source unit (e.g. 'km', 'mi', 'kg', 'lb')."},
            "to_unit": {"type": "string", "description": "Target unit (e.g. 'km', 'mi', 'kg', 'lb')."},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
)


class MockToolEnvironment:
    """Hermetic execution environment for registered mock tools."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, Callable[..., str]]] = {
            "calculator": (CALCULATOR_TOOL_DEF, safe_calculator),
            "database_lookup": (DATABASE_LOOKUP_TOOL_DEF, mock_database_lookup),
            "unit_converter": (UNIT_CONVERTER_TOOL_DEF, mock_unit_converter),
        }

    def get_definitions(self, tool_names: Sequence[str] | None = None) -> list[ToolDefinition]:
        """Return ToolDefinitions for the requested tool names, or all if None."""
        if tool_names is None:
            return [defn for defn, _ in self._tools.values()]
        return [self._tools[name][0] for name in tool_names if name in self._tools]

    def execute(self, tool_name: str, arguments: Any) -> str:
        """Safely execute a tool by name in the hermetic environment."""
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' is not recognized in mock environment."

        if not isinstance(arguments, dict):
            return f"Error: Arguments for tool '{tool_name}' must be a dictionary, got {type(arguments).__name__}."

        defn, fn = self._tools[tool_name]

        # Validate required parameters from definition schema
        required_params = defn.parameters.get("required", [])
        missing = [p for p in required_params if p not in arguments]
        if missing:
            return f"Error: Invalid arguments for '{tool_name}': Missing required argument(s): {', '.join(missing)}"

        # Validate parameter types where declared
        properties = defn.parameters.get("properties", {})
        for param, val in arguments.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type == "number" and not isinstance(val, (int, float)):
                    return f"Error: Invalid arguments for '{tool_name}': Parameter '{param}' must be a number, got {type(val).__name__}."
                elif expected_type == "string" and not isinstance(val, str):
                    return f"Error: Invalid arguments for '{tool_name}': Parameter '{param}' must be a string, got {type(val).__name__}."

        try:
            return fn(**arguments)
        except TypeError as err:
            return f"Error: Invalid arguments for '{tool_name}': {err}"
        except Exception as exc:
            return f"Error: Tool execution failed: {exc}"
