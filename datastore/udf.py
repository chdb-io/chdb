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

    def __init__(self, logical_name: str, arity: int, local_name: Optional[str] = None):
        self.logical_name = logical_name
        self.arity = arity
        self.local_name = local_name
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


def bind_local(fn: Any, logical_name: str, arity: int, local_name: Optional[str] = None):
    """Record that ``fn`` is registered in the local engine, and return its binding."""
    binding = _BINDINGS.get(logical_name)
    if binding is None:
        binding = UdfBinding(logical_name, arity)
        _BINDINGS[logical_name] = binding
    binding.arity = arity
    binding.local_name = local_name or logical_name
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
