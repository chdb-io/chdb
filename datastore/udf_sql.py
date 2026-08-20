"""Turn the Python a UDF actually contains into SQL, when that is safe.

Most rules people write as a UDF are arithmetic and comparisons wearing Python
syntax: a threshold, a discount, a bit of rounding. Sent to a server they become
a per-row call into an external process; expressed as SQL they become part of
the query and cost nothing. So the first question about a UDF is not where to
run it - it is whether it needs to be a UDF at all.

Translation is opt-in and narrow, because an AST whitelist cannot prove that
Python and ClickHouse agree on types, NULLs, exceptions or truthiness. The
stable set is arithmetic, comparison and conditional returns over declared
numeric arguments - nothing that depends on any of those four - and a function
that steps outside it is refused rather than approximated:

    def recognized_revenue(value):
        if value >= 150:
            return value * 0.92
        if value >= 75:
            return value * 0.95
        return value * 0.97

    multiIf(v >= 150, v * 0.92, v >= 75, v * 0.95, v * 0.97)

Deliberately outside it, each for a reason rather than for lack of time:

    and / or        Python returns an operand, SQL returns a boolean, and the
                    two disagree the moment an operand is not a bool
    calls           every builtin needs its own proof; round() alone differs
                    between Python versions and ClickHouse settings
    str / int / float  an implicit conversion is where type semantics diverge
    // and %        Python floors towards negative infinity, ClickHouse
                    truncates towards zero
    x / y           unless y is a non-zero literal: Python raises on zero and
                    ClickHouse returns inf
    strings, loops, assignments, attribute access, imports

Everything refused returns None, and the caller keeps the function as a
function - deployed as a UDF, or run in pandas.
"""

import ast
import inspect
import textwrap
from typing import Callable, List, Optional

from .functions import Function, format_alias

__all__ = [
    "RewrittenCall",
    "SqlRewrite",
    "numeric_arg_types",
    "rewritten_calls_in",
    "sql_rewrite_for",
]


class Unsupported(Exception):
    """Raised while translating, and caught here: it means "leave it alone"."""


# Python operators that mean the same thing in ClickHouse. Integer division and
# modulo are missing on purpose: Python rounds towards negative infinity and
# ClickHouse truncates towards zero, so they agree only for positive operands.
_BINARY_OPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",  # only over a non-zero literal divisor; see _translate
}

_COMPARISONS = {
    ast.Eq: "=",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

# Numeric ClickHouse types a rewritten argument may be declared as. The
# translation reasons about arithmetic, so it has to know it is arithmetic.
_NUMERIC_TYPES = (
    "int8", "int16", "int32", "int64", "int128", "int256",
    "uint8", "uint16", "uint32", "uint64", "uint128", "uint256",
    "float32", "float64", "decimal", "decimal32", "decimal64",
    "decimal128", "decimal256",
)


def numeric_arg_types(arg_types) -> bool:
    """Whether every declared argument is a number this translation can reason about."""
    if not arg_types:
        return False
    for declared in arg_types:
        text = str(declared or "").strip().lower()
        if text.startswith("nullable(") and text.endswith(")"):
            # NULL semantics are one of the things this set stays clear of.
            return False
        if not text.split("(", 1)[0] in _NUMERIC_TYPES:
            return False
    return True


class SqlRewrite:
    """A function's body as SQL, waiting for its arguments."""

    def __init__(self, name: str, parameters: List[str], render: Callable):
        self.name = name
        self.parameters = parameters
        self._render = render

    def sql(self, arguments: List[str]) -> str:
        """The expression, with each parameter replaced by the caller's SQL."""
        if len(arguments) != len(self.parameters):
            raise ValueError(
                f"{self.name} takes {len(self.parameters)} argument(s), "
                f"given {len(arguments)}"
            )
        return self._render(dict(zip(self.parameters, arguments)))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        slots = {name: name for name in self.parameters}
        return f"SqlRewrite({self.name}: {self._render(slots)})"


def sql_rewrite_for(fn) -> Optional[SqlRewrite]:
    """The SQL this function is equivalent to, or None when it is not.

    None is the common answer and not a failure: it means the function keeps
    being a function.
    """
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        # A function whose source is unavailable - a builtin, a lambda typed
        # into a REPL - cannot be read, let alone translated.
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    definition = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)), None
    )
    if definition is None:
        return None
    if definition.args.vararg or definition.args.kwarg or definition.args.kwonlyargs:
        return None
    if definition.args.defaults:
        # A default only shows up when the caller omits an argument, which the
        # SQL form has no way to express.
        return None
    if any(
        isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom, ast.Global))
        for node in ast.walk(definition)
    ):
        return None

    parameters = [argument.arg for argument in definition.args.args]
    try:
        render = _translate_body(definition.body, set(parameters))
    except Unsupported:
        return None
    return SqlRewrite(definition.name, parameters, render)


def _translate_body(body, parameters) -> Callable:
    """A function body of guarded returns, as one conditional expression."""
    body = [node for node in body if not _is_docstring(node)]
    if not body:
        raise Unsupported("empty body")

    branches = []  # (condition, value) pairs, in order
    default = None
    for index, node in enumerate(body):
        if isinstance(node, ast.Return):
            if index != len(body) - 1:
                raise Unsupported("code after a return")
            default = _translate(node.value, parameters)
        elif isinstance(node, ast.If):
            branches.append(_translate_if(node, parameters))
        else:
            raise Unsupported(type(node).__name__)

    if default is None:
        # Without a trailing return the function falls off the end and yields
        # None for the remaining rows, which SQL says as NULL.
        default = lambda _slots: "NULL"  # noqa: E731

    if not branches:
        return default

    def render(slots):
        pieces = []
        for condition, value in branches:
            pieces.append(condition(slots))
            pieces.append(value(slots))
        pieces.append(default(slots))
        return "multiIf(" + ", ".join(pieces) + ")"

    return render


def _translate_if(node: ast.If, parameters):
    """``if <test>: return <value>`` - the only branch shape allowed."""
    if node.orelse:
        # else / elif would nest another body; the guard-clause form is the one
        # people write for these rules, and it is unambiguous.
        raise Unsupported("if/else")
    body = [statement for statement in node.body if not _is_docstring(statement)]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        raise Unsupported("branch that does more than return")
    return _translate(node.test, parameters), _translate(body[0].value, parameters)


def _is_docstring(node) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
        node.value.value, str
    )


def _translate(node, parameters) -> Callable:
    """One expression, as a function from parameter SQL to expression SQL."""
    if node is None:
        raise Unsupported("bare return")

    if isinstance(node, ast.Constant):
        literal = _literal(node.value)
        return lambda _slots: literal

    if isinstance(node, ast.Name):
        if node.id not in parameters:
            raise Unsupported(f"name {node.id}")
        name = node.id
        return lambda slots: slots[name]

    if isinstance(node, ast.BinOp):
        operator = _BINARY_OPS.get(type(node.op))
        if operator is None:
            raise Unsupported(type(node.op).__name__)
        if isinstance(node.op, ast.Div) and not _is_nonzero_number(node.right):
            # Python raises on a zero divisor and ClickHouse returns inf, so
            # the two only agree when the divisor cannot be zero.
            raise Unsupported("division by a value that could be zero")
        left = _translate(node.left, parameters)
        right = _translate(node.right, parameters)
        return lambda slots: f"({left(slots)} {operator} {right(slots)})"

    if isinstance(node, ast.UnaryOp):
        operand = _translate(node.operand, parameters)
        if isinstance(node.op, ast.USub):
            return lambda slots: f"(-{operand(slots)})"
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Not):
            return lambda slots: f"(NOT {operand(slots)})"
        raise Unsupported(type(node.op).__name__)

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            # a < b < c is two comparisons sharing a subexpression; translating
            # it would evaluate b twice, which is only safe for pure b.
            raise Unsupported("chained comparison")
        operator = _COMPARISONS.get(type(node.ops[0]))
        if operator is None:
            raise Unsupported(type(node.ops[0]).__name__)
        left = _translate(node.left, parameters)
        right = _translate(node.comparators[0], parameters)
        return lambda slots: f"({left(slots)} {operator} {right(slots)})"

    if isinstance(node, ast.IfExp):
        test = _translate(node.test, parameters)
        body = _translate(node.body, parameters)
        orelse = _translate(node.orelse, parameters)
        return lambda slots: f"if({test(slots)}, {body(slots)}, {orelse(slots)})"

    raise Unsupported(type(node).__name__)


def _is_nonzero_number(node) -> bool:
    """Whether this is a literal number that is definitely not zero."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_nonzero_number(node.operand)
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value != 0
    )


def _literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    raise Unsupported(type(value).__name__)


class RewrittenCall(Function):
    """A translated function, rendered as the SQL it turned out to be.

    It is a Function so the planner treats it like any other expression. There
    is no call at execution time and no engine has to have heard of the
    function - the rule became part of the query.
    """

    def __init__(self, rewrite: SqlRewrite, *args, alias=None, arg_types=None):
        self._rewrite = rewrite
        self._arg_types = list(arg_types or [])
        super().__init__(rewrite.name, *args, alias=alias)

    @property
    def rewrite(self) -> SqlRewrite:
        return self._rewrite

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        rendered = []
        for index, argument in enumerate(self.args):
            piece = argument.to_sql(quote_char=quote_char, **kwargs)
            declared = self._arg_types[index] if index < len(self._arg_types) else None
            if declared:
                # The Python this replaces would have been handed the declared
                # type, so the arithmetic has to happen in it too.
                piece = f"CAST({piece} AS {declared})"
            rendered.append(piece)
        sql = self._rewrite.sql(rendered)
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)
        return sql

    def __copy__(self):
        from copy import copy

        return RewrittenCall(
            self._rewrite,
            *[copy(argument) for argument in self.args],
            alias=self.alias,
            arg_types=self._arg_types,
        )

    def rebuild_with_args(self, args):
        """A copy over new arguments, still the same translated rule."""
        return RewrittenCall(
            self._rewrite, *args, alias=self.alias, arg_types=self._arg_types
        )


def rewritten_calls_in(ops) -> list:
    """Every translated rule inside these operations.

    The same walk the UDF calls use, so a rule that became an expression is
    still reported by the name its author gave it.
    """
    from .udf import udf_calls_in

    return udf_calls_in(ops, node_type=RewrittenCall)
