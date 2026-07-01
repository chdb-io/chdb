"""Shared safety primitives for chdb.agents.

These are the *identifier* helpers (values go through server-side parameter
binding, never through here). They are deliberately tiny and dependency-free so
the same rules can be mirrored verbatim by other chdb-io language bindings — the
CONTRACT.md points every language at this behavior to stop each one hand-rolling
its own (subtly different) quoting.
"""

__all__ = ["quote_ident", "InvalidIdentifier"]


class InvalidIdentifier(ValueError):
    """Raised when an identifier cannot be safely quoted."""


def quote_ident(name):
    """Backtick-quote a ClickHouse identifier (db / table / column).

    Identifiers cannot be passed as bound parameters, so agent-supplied names
    are quoted here. Embedded backticks are doubled (ClickHouse escaping); a NUL
    byte is rejected outright since it cannot appear in a valid identifier and is
    a classic truncation-smuggling vector.
    """
    if not isinstance(name, str) or name == "":
        raise InvalidIdentifier("identifier must be a non-empty string")
    if "\x00" in name:
        raise InvalidIdentifier("identifier must not contain a NUL byte")
    return "`" + name.replace("`", "``") + "`"
