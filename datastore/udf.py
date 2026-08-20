"""Where a Python UDF can run, and the name it answers to there.

A UDF registered with chDB exists under one name in the local engine and, once
deployed, under another on the server that owns the data - a session-scoped name
unless the deployment was made permanent. A compiled plan therefore cannot carry
the function's name as text: the same expression has to render ``recognized(x)``
for the local engine and ``chdb_udf_2f9c_ab12(x)`` for the server, decided when
the SQL is built rather than when the chain is written.

This module holds that mapping, and the expression node that reads it. Nothing
here deploys or registers anything; chdb.deploy owns the lifecycle and reports
the names it produced.
"""

from typing import Any, Dict, Optional

from .functions import Function
from .pushdown import LOCAL_CHDB, REMOTE_CLICKHOUSE, current_compile_target

__all__ = [
    "UdfBinding",
    "UdfCall",
    "udf_calls_in",
    "unresolvable_udfs",
    "binding_for",
    "binding_named",
    "bind_local",
    "bind_remote",
    "clear_bindings",
    "known_bindings",
]

# The attribute a bound callable carries, so a chain can recognise a UDF it is
# handed without a lookup table keyed by object identity.
_BINDING_ATTR = "__datastore_udf_binding__"

_BINDINGS: Dict[str, "UdfBinding"] = {}


class UdfBinding:
    """One Python function, and the name each engine knows it by.

    Mutable on purpose: a function is registered locally when it is decorated
    and deployed some time later, and the callable the user holds must see the
    remote name appear without being rebound.
    """

    def __init__(
        self,
        logical_name: str,
        arity: int,
        local_name: Optional[str] = None,
        arg_types: Optional[list] = None,
    ):
        self.logical_name = logical_name
        self.arity = arity
        self.local_name = local_name
        # The types the function was declared with. A deployed UDF is called
        # through them - the server converts the column to the declared type -
        # so the local call has to declare the same conversion or the same chain
        # works on one engine and fails on the other.
        self.arg_types = list(arg_types or [])
        # The SQL this function turned out to be, when it could be translated.
        # A rule that becomes an expression needs no engine to host it.
        self.rewrite = None
        # connection name -> the name the function was deployed under there
        self.remote_names: Dict[str, str] = {}

    def name_for(self, target: str, connection: Optional[str] = None) -> Optional[str]:
        """The name to emit for ``target``, or None when it cannot run there.

        None is the answer that matters: it is what stops a planner from
        compiling a call the target could not resolve.
        """
        if target == LOCAL_CHDB:
            return self.local_name
        if target != REMOTE_CLICKHOUSE:
            return None
        if connection is not None:
            return self.remote_names.get(connection)
        if len(self.remote_names) == 1:
            # One deployment, no ambiguity to resolve. With several, the caller
            # has to say which connection it is compiling for.
            return next(iter(self.remote_names.values()))
        return None

    def runs_on(self, target: str, connection: Optional[str] = None) -> bool:
        return self.name_for(target, connection) is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"UdfBinding({self.logical_name!r}, local={self.local_name!r}, "
            f"remote={self.remote_names!r})"
        )


def bind_local(
    fn: Any,
    logical_name: str,
    arity: int,
    local_name: Optional[str] = None,
    arg_types: Optional[list] = None,
):
    """Record that ``fn`` is registered in the local engine, and return its binding."""
    binding = _BINDINGS.get(logical_name)
    if binding is None:
        binding = UdfBinding(logical_name, arity)
        _BINDINGS[logical_name] = binding
    binding.arity = arity
    binding.local_name = local_name or logical_name
    if arg_types:
        binding.arg_types = list(arg_types)
    if binding.rewrite is None and fn is not None:
        from .udf_sql import sql_rewrite_for

        binding.rewrite = sql_rewrite_for(fn)
    _attach(fn, binding)
    return binding


def bind_remote(fn: Any, logical_name: str, connection: str, remote_name: str):
    """Record the name ``logical_name`` was deployed under on ``connection``."""
    binding = _BINDINGS.get(logical_name)
    if binding is None:
        binding = UdfBinding(logical_name, 1)
        _BINDINGS[logical_name] = binding
    binding.remote_names[connection] = remote_name
    _attach(fn, binding)
    return binding


def _attach(fn: Any, binding: "UdfBinding") -> None:
    """Mark a callable with its binding, tolerating objects that reject writes."""
    if fn is None:
        return
    try:
        setattr(fn, _BINDING_ATTR, binding)
    except (AttributeError, TypeError):
        # A builtin or a slotted object cannot carry the mark; the registry
        # still holds the binding, so name resolution keeps working.
        pass


def binding_for(fn: Any) -> Optional["UdfBinding"]:
    """The binding behind a callable, or None for an ordinary function."""
    binding = getattr(fn, _BINDING_ATTR, None)
    if isinstance(binding, UdfBinding):
        return binding
    name = getattr(fn, "__name__", None)
    if isinstance(name, str):
        return _BINDINGS.get(name)
    return None


def binding_named(logical_name: str) -> Optional["UdfBinding"]:
    """The binding registered under ``logical_name``, if any."""
    return _BINDINGS.get(logical_name)


def known_bindings() -> Dict[str, "UdfBinding"]:
    """Every binding registered in this process, by logical name."""
    return dict(_BINDINGS)


def clear_bindings() -> None:
    """Forget every binding. For tests that register their own UDFs."""
    _BINDINGS.clear()


class UdfCall(Function):
    """A call to a Python UDF, named for whichever engine compiles it.

    The name is resolved at ``to_sql()`` time rather than stored, because the
    same expression object is compiled twice: once to show the user what the
    local engine would run, once for the server that owns the table.
    """

    def __init__(self, binding: UdfBinding, *args, alias: Optional[str] = None, connection: Optional[str] = None):
        self._binding = binding
        self._connection = connection
        super().__init__(binding.logical_name, *args, alias=alias)

    @property
    def binding(self) -> UdfBinding:
        return self._binding

    @property
    def name(self) -> str:
        resolved = self._binding.name_for(current_compile_target(), self._connection)
        # Falling back to the logical name keeps a preview readable; a target
        # that cannot resolve the function is refused by the planner, not here.
        return resolved or self._logical_name

    @name.setter
    def name(self, value: str) -> None:
        self._logical_name = value

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Render the call, converting arguments to the declared types.

        A deployed UDF is called through its declared types: the server converts
        the column on the way in. Without the same conversion here, a chain over
        a Decimal column would run on the server and fail locally against a
        Float64 declaration - the same code, two answers, decided by placement.
        """
        from .functions import format_alias

        pieces = []
        for index, argument in enumerate(self.args):
            rendered = argument.to_sql(quote_char=quote_char, **kwargs)
            declared = (
                self._binding.arg_types[index]
                if index < len(self._binding.arg_types)
                else None
            )
            if declared:
                rendered = f"CAST({rendered} AS {declared})"
            pieces.append(rendered)
        sql = f"{self.name}({','.join(pieces)})"
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)
        return sql

    def __copy__(self):
        # Function.__copy__ rebuilds a plain Function, which would freeze the
        # name resolved for whichever engine happened to be compiling. Aliasing
        # an expression copies it, so without this a UDF call would lose its
        # binding on the way into the plan.
        from copy import copy

        return UdfCall(
            self._binding,
            *[copy(arg) for arg in self.args],
            alias=self.alias,
            connection=self._connection,
        )

    def rebuild_with_args(self, args):
        """A copy of this call over ``args``, still bound to the same UDF."""
        return UdfCall(
            self._binding,
            *args,
            alias=self.alias,
            connection=self._connection,
        )


def udf_calls_in(ops) -> list:
    """Every UDF call inside these operations, however deeply nested.

    A UDF reaches a plan as one node in an expression that itself sits in an
    operation's field list, so finding it means walking the expression tree. The
    walk stays inside expressions on purpose: an operation also holds frames and
    engine handles, and none of those can contain a call.
    """
    from .expressions import Expression

    found = []
    seen = set()

    def visit(node, depth=0):
        if depth > 24 or node is None:
            return
        marker = id(node)
        if marker in seen:
            return
        seen.add(marker)
        if isinstance(node, UdfCall):
            found.append(node)
        if isinstance(node, (list, tuple, set)):
            for item in node:
                visit(item, depth + 1)
            return
        if isinstance(node, dict):
            for item in node.values():
                visit(item, depth + 1)
            return
        inner = getattr(node, "_expr", None)
        if isinstance(inner, Expression):
            visit(inner, depth + 1)
        if isinstance(node, Expression):
            for value in vars(node).values():
                visit(value, depth + 1)

    for op in ops or ():
        try:
            attributes = vars(op).values()
        except TypeError:  # pragma: no cover - ops without a __dict__
            continue
        for value in attributes:
            visit(value)
    return found


def unresolvable_udfs(ops, target: str, connection: Optional[str] = None) -> list:
    """Logical names of UDFs in ``ops`` that ``target`` has no name for.

    An empty list means every call in the segment can be written for that
    engine. A non-empty one is a reason to keep the segment where the functions
    actually exist, rather than send a statement the server would reject.
    """
    names = []
    for call in udf_calls_in(ops):
        binding = call.binding
        if binding.name_for(target, connection) is None:
            if binding.logical_name not in names:
                names.append(binding.logical_name)
    return names
