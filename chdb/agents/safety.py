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
    "find_source_calls",
    "SAFE_TABLE_FUNCTIONS",
    "FALLBACK_KNOWN_TABLE_FUNCTIONS",
    "NETWORK_TABLE_FUNCTIONS",
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


# --- Table-function source scanning -----------------------------------------
#
# chDB exposes table functions that reach outside the process (file, url, s3,
# remote, postgresql, the RCE-class executable / python, ...). When a
# file_allowlist is configured, raw SQL is scanned for those calls; the scanner
# below is shared verbatim (in semantics) with the TypeScript binding.
#
# Table functions that are safe by construction: they consume only literal or
# synthetic arguments and never reach outside the chDB process. Lowercase so
# matching against system.table_functions is case-insensitive. Anything the
# engine exposes that is NOT in this set is treated as a potential external
# source when the allowlist is configured — an allowlist over safe functions,
# not a denylist of dangerous ones, so new source functions in future engines
# are gated by default instead of silently allowed.
#
# view / merge / dictionary can *contain* nested table-function calls
# (e.g. view(SELECT * FROM file(...))), but the text-level scanner sees the
# inner file( directly, so allowlisting the wrappers is safe.
SAFE_TABLE_FUNCTIONS = frozenset(
    {
        "numbers",
        "numbers_mt",
        "zeros",
        "zeros_mt",
        "null",
        "values",
        "format",
        "input",
        "generaterandom",
        "generateseries",
        "generate_series",
        "primes",
        "loop",
        "fuzzquery",
        "fuzzjson",
        "view",
        "viewexplain",
        "viewifpermitted",
        "dictionary",
        "merge",
        "mergetreeindex",
        "mergetreeprojection",
        "mergetreeanalyzeindexes",
        "mergetreeanalyzeindexesuuid",
        "mergetreetextindex",
        "timeseriesdata",
        "timeseriesmetrics",
        "timeseriesselector",
        "timeseriestags",
    }
)

# Conservative fallback when system.table_functions can't be queried (older
# chDB / stripped build). Covers the table functions that reach outside the
# process, including the RCE-class executable / python. Lowercase.
FALLBACK_KNOWN_TABLE_FUNCTIONS = frozenset(
    {
        "file",
        "filecluster",
        "url",
        "urlcluster",
        "urlwithheaders",
        "s3",
        "s3cluster",
        "remote",
        "remotesecure",
        "cluster",
        "clusterallreplicas",
        "hdfs",
        "hdfscluster",
        "mongodb",
        "postgresql",
        "mysql",
        "redis",
        "sqlite",
        "odbc",
        "jdbc",
        "iceberg",
        "iceberglocal",
        "iceberglocalcluster",
        "icebergs3",
        "icebergs3cluster",
        "icebergazure",
        "icebergazurecluster",
        "iceberghdfs",
        "iceberghdfscluster",
        "deltalake",
        "deltalakelocal",
        "deltalakeazure",
        "deltalakeazurecluster",
        "deltalakes3",
        "deltalakes3cluster",
        "hudi",
        "hudicluster",
        "paimon",
        "paimonlocal",
        "paimonazure",
        "paimonazurecluster",
        "paimonhdfs",
        "paimonhdfscluster",
        "paimons3",
        "paimons3cluster",
        "paimoncluster",
        "azureblobstorage",
        "azureblobstoragecluster",
        "gcs",
        "cosn",
        "oss",
        "ytsaurus",
        "executable",
        "python",
        "prometheusquery",
        "prometheusqueryrange",
    }
)

# Single pass over the SQL: a token is either a string literal, a line comment,
# or a block comment. Left-to-right alternation guarantees that once a construct
# opens, its body is consumed up to the matching close before any other rule can
# fire — so a call smuggled between two strings whose contents *look* like a
# block comment cannot mislead the scanner. The string sub-pattern accepts the
# escape forms ClickHouse honours: '' doubling and backslash escapes.
_MASK_RE = re.compile(
    r"'(?:[^'\\]|\\.|'')*'"  # single-quoted string with \. and '' escapes
    r"|--[^\n]*"  # line comment
    r"|/\*.*?\*/",  # block comment
    re.DOTALL,
)

# A function-call token in masked SQL: a bare word, or a backtick/double-quote
# wrapped word (`file`( / "file"( — quoting a function name must not bypass the
# scan), followed by optional whitespace and '('. Masking already blanked
# comments between name and paren, so \s covers them.
_CALL_RE = re.compile(r"(?:\b(?P<bare>\w+)|(?P<q>[`\"])(?P<quoted>\w+)(?P=q))\s*\(")

# A single-quoted string literal immediately at the start of an argument list.
_LEADING_STRING_RE = re.compile(r"\s*'(?P<body>(?:[^'\\]|\\.|'')*)'")

_UNESCAPE_RE = re.compile(r"''|\\(.)")


def _mask(sql):
    """Blank out string literals and comments, preserving every position."""
    return _MASK_RE.sub(lambda m: " " * len(m.group(0)), sql)


def _literal_first_arg(sql, args_start):
    """The unescaped leading string-literal argument at sql[args_start:], or
    None when the first argument is anything else (identifier, call, number)."""
    m = _LEADING_STRING_RE.match(sql, args_start)
    if not m:
        return None
    return _UNESCAPE_RE.sub(lambda e: "'" if e.group(0) == "''" else e.group(1), m.group("body"))


# Network-reaching table functions — these get the watchdog deadline
# (CONTRACT P5). file()/sqlite and *local* lake variants deliberately absent.
NETWORK_TABLE_FUNCTIONS = frozenset(
    {
        "url",
        "urlcluster",
        "urlwithheaders",
        "s3",
        "s3cluster",
        "gcs",
        "azureblobstorage",
        "azureblobstoragecluster",
        "remote",
        "remotesecure",
        "hdfs",
        "hdfscluster",
        "mongodb",
        "postgresql",
        "mysql",
        "redis",
        "odbc",
        "jdbc",
        "icebergs3",
        "icebergs3cluster",
        "icebergazure",
        "icebergazurecluster",
        "iceberghdfs",
        "iceberghdfscluster",
        "deltalake",
        "deltalakeazure",
        "hudi",
    }
)


def find_source_calls(sql, known):
    """Return [(name, literal_first_arg_or_None), ...] for every table-function
    call in `sql` whose lowercase name is in `known` but not SAFE.

    Scans a position-preserving MASKED copy (string literals and comments
    blanked), so a path-looking string literal never false-positives and a
    comment between name and paren never hides a call; quoted function names
    (`file`( / "file"() are matched as calls. The literal argument, when the
    call has one, is extracted from the ORIGINAL text and unescaped. Scalar
    functions (sum, length, ...) are not table functions, so they are not in
    `known` and never flag.
    """
    masked = _mask(sql)
    out = []
    for m in _CALL_RE.finditer(masked):
        name = (m.group("bare") or m.group("quoted")).lower()
        if name in known and name not in SAFE_TABLE_FUNCTIONS:
            out.append((name, _literal_first_arg(sql, m.end())))
    return out


def scan_file_paths(sql):
    """Return the (fn, path) literal pairs of source table functions in `sql`.

    Kept for compatibility with earlier callers; implemented on the masked
    scanner with the fallback function set, returning only calls that carry a
    leading string-literal argument. Allowlist enforcement should use
    `find_source_calls` (it also surfaces non-literal calls, which must be
    denied rather than skipped).
    """
    return [
        (name, arg)
        for name, arg in find_source_calls(sql, FALLBACK_KNOWN_TABLE_FUNCTIONS)
        if arg is not None
    ]
