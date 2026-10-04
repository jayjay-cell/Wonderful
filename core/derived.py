"""Restricted expression evaluator for derived values and trigger conditions.

Used for two things authored in mission files:

    derived:    expr: "fuel_lb / 22 * 60"          -> a number
    thresholds: when: "fuel_lb <= 45"               -> a boolean

WHY NOT eval(): mission files are the natural "import a mission someone
sent me" path. Python's eval() on a YAML field is a remote-code-execution
hole, and "we only load our own files" stops being true the first time a
mission is shared between units. So this module walks Python's AST and
permits an explicit ALLOWLIST of node types -- no attribute access, no
subscripting, no calls except a handful of named maths functions, no
names except declared parameters.

An allowlist rather than a denylist, for the same reason the logger uses
one: a new Python AST node type is automatically rejected rather than
automatically permitted.

Derived values are recomputed on every read and never stored, which is
what makes FR-D2 structural: endurance cannot contradict fuel because it
has no independent existence to drift out of sync.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Mapping

# Only these AST nodes may appear. Notably absent: ast.Attribute (blocks
# `x.__class__`), ast.Subscript, ast.Lambda, ast.ListComp, ast.Await,
# anything statement-like.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.UAdd, ast.USub, ast.Not,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.IfExp,
    ast.Call,
    ast.Name, ast.Load,
    ast.Constant,
)

# Callable helpers. Deliberately pure, total, and side-effect free.
_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
}

# Bare words that are values, not parameter references.
_CONSTANTS: dict[str, Any] = {
    "true": True, "false": False,
    "True": True, "False": False,
    "none": None, "None": None,
}

_MAX_EXPRESSION_LENGTH = 500


class ExpressionError(ValueError):
    """Raised for a malformed or disallowed expression.

    Carries the offending expression so the mission loader can report the
    field path alongside it -- a typo should take seconds to find, not a
    debugging session (FR-E8).
    """

    def __init__(self, message: str, expression: str | None = None) -> None:
        super().__init__(message)
        self.expression = expression


def _validate_node(node: ast.AST, expression: str, allowed_names: set[str] | None,
                   function_positions: frozenset[int] = frozenset()) -> None:
    """Depth-first allowlist check over the whole tree.

    `function_positions` holds the id()s of Name nodes sitting in a call's
    function slot, computed once by the caller. An earlier version
    re-parsed the whole expression for every Name encountered, which was
    quadratic and -- worse -- compared nodes from a different parse tree by
    identity, so the check never actually matched.
    """
    if not isinstance(node, _ALLOWED_NODES):
        raise ExpressionError(
            f"disallowed syntax {type(node).__name__!r} in expression {expression!r}",
            expression,
        )

    if isinstance(node, ast.Call):
        # Only a bare name may be called, and only one from the allowlist.
        # This is what blocks `(1).__class__` style escapes: an attribute
        # is not a Name, so it never becomes callable.
        if not isinstance(node.func, ast.Name):
            raise ExpressionError(
                f"only simple function calls are allowed in expression {expression!r}", expression
            )
        if node.func.id not in _ALLOWED_FUNCTIONS:
            raise ExpressionError(
                f"unknown function {node.func.id!r} in expression {expression!r}; "
                f"allowed: {sorted(_ALLOWED_FUNCTIONS)}",
                expression,
            )
        if node.keywords:
            raise ExpressionError(
                f"keyword arguments are not allowed in expression {expression!r}", expression
            )

    if isinstance(node, ast.Name):
        name = node.id
        is_function_position = id(node) in function_positions
        if not is_function_position and name not in _CONSTANTS:
            if allowed_names is not None and name not in allowed_names:
                raise ExpressionError(
                    f"unknown reference {name!r} in expression {expression!r}; "
                    f"not a declared parameter or derived value",
                    expression,
                )

    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool, str, type(None))):
        raise ExpressionError(
            f"disallowed constant type in expression {expression!r}", expression
        )

    for child in ast.iter_child_nodes(node):
        _validate_node(child, expression, allowed_names, function_positions)


def _function_name_positions(tree: ast.AST) -> frozenset[int]:
    """id()s of Name nodes used as a call's function, e.g. the `min` in
    `min(a, b)`. Those are checked against _ALLOWED_FUNCTIONS by the Call
    branch, so they must not also be required to be declared parameters."""
    return frozenset(
        id(node.func) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )


def validate_expression(expression: str, allowed_names: set[str] | None = None) -> None:
    """Check an expression without evaluating it.

    Called by the mission loader at load time, so a bad expression fails
    when the file is read rather than mid-session (FR-E8). Pass
    `allowed_names` to also verify every reference is a declared parameter
    or derived value.
    """
    if not expression or not expression.strip():
        raise ExpressionError("expression is empty", expression)
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"expression exceeds {_MAX_EXPRESSION_LENGTH} characters", expression
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as err:
        raise ExpressionError(f"syntax error in expression {expression!r}: {err.msg}", expression) from err
    _validate_node(tree, expression, allowed_names, _function_name_positions(tree))


def evaluate(expression: str, values: Mapping[str, Any]) -> Any:
    """Evaluate a validated expression against current state.

    `values` maps parameter and derived ids to current values. A reference
    with no value raises rather than defaulting to zero -- silently
    treating a missing fuel reading as 0 would make the counterpart report
    an emergency that is not happening, which is worse than failing.
    """
    validate_expression(expression, allowed_names=set(values))
    tree = ast.parse(expression, mode="eval")

    # Evaluation namespace: only the allowlisted functions, the literal
    # constants, and the supplied state. __builtins__ is emptied so no
    # name resolution can reach Python's globals.
    namespace: dict[str, Any] = {"__builtins__": {}}
    namespace.update(_ALLOWED_FUNCTIONS)
    namespace.update(_CONSTANTS)
    namespace.update(values)

    try:
        # Safe despite the name: the AST was allowlist-validated above and
        # __builtins__ is empty, so the reachable surface is exactly the
        # namespace assembled here.
        return eval(compile(tree, "<mission-expression>", "eval"), namespace)  # noqa: S307
    except ZeroDivisionError as err:
        raise ExpressionError(
            f"division by zero evaluating {expression!r}", expression
        ) from err
    except TypeError as err:
        raise ExpressionError(
            f"type error evaluating {expression!r}: {err}", expression
        ) from err


def evaluate_bool(expression: str, values: Mapping[str, Any]) -> bool:
    """Evaluate a trigger condition.

    Requires a genuine boolean rather than coercing: `when: "fuel_lb"` is
    almost certainly an author mistake (a forgotten comparison), and
    treating 180 as truthy would fire the trigger immediately and silently.
    """
    result = evaluate(expression, values)
    if not isinstance(result, bool):
        raise ExpressionError(
            f"condition {expression!r} evaluated to {type(result).__name__} "
            f"({result!r}), not true/false -- did you mean a comparison?",
            expression,
        )
    return result


def referenced_names(expression: str) -> set[str]:
    """Every parameter/derived id an expression refers to.

    Used by the loader to check references resolve, and to order derived
    values so one may depend on another.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as err:
        raise ExpressionError(f"syntax error in expression {expression!r}: {err.msg}", expression) from err

    called: set[str] = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id not in called
        and node.id not in _ALLOWED_FUNCTIONS
        and node.id not in _CONSTANTS
    }
