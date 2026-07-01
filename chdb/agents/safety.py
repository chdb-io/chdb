"""Shared safety primitives for chdb.agents.

Values go through server-side parameter binding, never through here. The only
exceptions are places where the engine cannot bind (identifiers, and string
literals baked into a stored `CREATE VIEW` definition) — those use `quote_ident`
/ `quote_string`. These helpers are deliberately tiny and dependency-free so the
same rules can be mirrored verbatim by other chdb-io language bindings; the
CONTRACT.md points every language here to stop each one hand-rolling its own
(subtly different) quoting or path-allowlist logic.
"""

import re

__all__ = [
    "quote_ident",
    "quote_string",
    "InvalidIdentifier",
    "path_allowed",
    "scan_file_paths",
]


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


def quote_string(value):
    """Escape a value as a ClickHouse single-quoted string literal.

    Prefer bound parameters (`{name:Type}`) for values — this is only for the few
    spots the engine cannot bind, e.g. a path/format literal baked into a stored
    `CREATE VIEW` definition. Backslashes and single quotes are escaped; a NUL
    byte is rejected (it cannot appear in a valid path/literal and is a smuggling
    vector).
    """
    if not isinstance(value, str):
        value = str(value)
    if "\x00" in value:
        raise InvalidIdentifier("string literal must not contain a NUL byte")
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def path_allowed(path, allowlist):
    """True if `path` starts with one of the allowlist prefixes.

    `allowlist` is an iterable of path prefixes; an empty/None allowlist means
    "no allowlist configured" and returns True (caller decides the default).
    """
    if not allowlist:
        return True
    return any(str(path).startswith(str(p)) for p in allowlist)


# Literal first-argument of file()/s3()/url()/etc. table functions. Best-effort:
# it catches the common `file('<path>' ...)` literal form used by agents, not
# computed/concatenated arguments — the real write backstop is readonly=2.
_FILE_FN_RE = re.compile(
    r"\b(?P<fn>file|s3|url|hdfs|azureBlobStorage)\s*\(\s*(?P<q>['\"])(?P<path>.*?)(?P=q)",
    re.I | re.S,
)


def scan_file_paths(sql):
    """Return the (fn, path) literals of file-like table functions in `sql`.

    Heuristic — used only to enforce a configured allowlist as defense in depth;
    documented as best-effort (won't see computed arguments).
    """
    return [(m.group("fn").lower(), m.group("path")) for m in _FILE_FN_RE.finditer(sql)]
