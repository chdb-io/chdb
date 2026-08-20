"""Deploy chdb Python UDFs to a remote ClickHouse server.

This module extends :func:`chdb.udf.func` with two extra parameters:

- ``deploy``: ``False`` (default, local-only), ``True`` (deploy to the default
  connection), or a connection name registered via
  :func:`datastore.config.register_connection`.
- ``permanent``: controls lifetime only. ``False`` (default) drops the
  function when the process exits; ``True`` keeps it. Either way the remote
  function is registered under the function's own name (or an explicit
  ``name=`` override), so it stays callable by a name the user can predict.
  Concurrent sessions deploying the same name share it: identical code is
  reused, changed code replaces the artifacts in place.

The local registration is delegated to :func:`chdb.udf.func` unchanged, so a
deployed function keeps working in local chdb queries exactly as before.

Deployment translates the decorated function into ClickHouse's executable UDF
artifacts — an executable script under the server's ``user_scripts_path`` and
an XML config under ``user_defined_executable_functions_config`` — then runs
``SYSTEM RELOAD FUNCTIONS``. The server must therefore share a filesystem
channel with this process (e.g. docker bind mounts); the channel paths are
part of the registered connection. ClickHouse Cloud, which only accepts UDF
uploads through its console today, is not yet supported as a deploy target.

No feasibility validation is performed: the function source is shipped as-is
and incompatibilities (closures over local state, unavailable imports, ...)
surface as ClickHouse-side errors at query time.

Type declarations mirror local chdb registration: the same annotation
inference table (including numpy scalars), loud errors for unknown
annotations and missing return types, and every declared type wrapped in
Nullable like the engine's makeNullable, and any type outside local
registration's whitelist (integers, floats, Bool, String, Date/DateTime —
isSupportedUDFType) is rejected with the same error shape.

DateTime values arrive as aware datetimes: a type with an explicit timezone
(``DateTime('UTC')``) attaches that zone, a plain DateTime attaches the host
zone of the server the wrapper runs on. Two known distortions: a
``<timezone>`` override in the server's config.xml makes the server render
in a zone the wrapper cannot see, and a per-query ``session_timezone``
setting likewise. Other divergences, inherent to the text protocol:
parameters without annotations are declared String (locally they stay
dynamic) and bytes/bytearray arguments arrive as str.

Example:
    >>> import datastore.config as dsconfig
    >>> dsconfig.register_connection(
    ...     "demo", host="localhost", port=8123,
    ...     udf_scripts_dir="/srv/ch/user_scripts",
    ...     udf_config_dir="/srv/ch/udf_config",
    ... )
    >>> from chdb import func
    >>> @func(deploy="demo", permanent=True)
    ... def add_tax(price: float, rate: float) -> float:
    ...     return price * (1 + rate)
"""

import atexit
import functools
import inspect
import os
import re
import secrets
import sys
import textwrap
import threading
import time
import types
import typing
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import chdb

__all__ = [
    "func",
    "deploy",
    "undeploy",
    "cleanup_session",
    "session_id",
    "DeployedFunction",
]

# Session identity: one per process, keyed by PID. Remote names no longer
# embed it (deployments use the function's own name), but the identity still
# guards the cleanup ledger across forks: a forked child must NOT inherit the
# parent's ledger, or the child's atexit would tear down functions the parent
# still uses.

_session_deployments: List["DeployedFunction"] = []
_session_lock = threading.Lock()
_atexit_registered = False
_session_state: Dict[str, Any] = {"pid": None, "id": None}


def session_id() -> str:
    """This process's deploy session id (fork guard for the cleanup ledger)."""
    with _session_lock:
        pid = os.getpid()
        if _session_state["pid"] != pid:
            _session_state["pid"] = pid
            _session_state["id"] = secrets.token_hex(3)
            # entries inherited across fork belong to the parent process
            _session_deployments.clear()
        return _session_state["id"]


# ---------------------------------------------------------------------------
# Type resolution: decorator specs / annotations -> ClickHouse type names
# ---------------------------------------------------------------------------

# Mirrors chdb-core's annotationToDataType (PythonScalarUDF.cpp), so a
# function deploys with the same declared types it registers with locally.
_PY_TO_CLICKHOUSE = {
    int: "Int64",
    float: "Float64",
    str: "String",
    bytes: "String",
    bytearray: "String",
    bool: "Bool",
    date: "Date",
    datetime: "DateTime64(6)",
}

# numpy scalar annotations, keyed by dtype name — same table as chdb-core's
# fromNumpyType (note float16 -> Float32, matching the engine).
_NUMPY_DTYPE_TO_CLICKHOUSE = {
    "bool": "Bool",
    "int8": "Int8",
    "uint8": "UInt8",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "float16": "Float32",
    "float32": "Float32",
    "float64": "Float64",
}

_UNION_TYPE = getattr(types, "UnionType", None)  # PEP 604 `X | None` (3.10+)

# SQL-compat aliases, resolved case-insensitively like DataTypeFactory does
# when local chdb parses a type string. This is the complete set of
# registerAlias() entries (src/DataTypes/*.cpp) whose canonical target is in
# the UDF whitelist; aliases of non-whitelisted types (NUMERIC -> Decimal,
# ENUM, BINARY -> FixedString, INET4/6, ...) are left out on purpose — both
# sides reject those either way, so mapping them would only change the type
# spelling inside the error message. String-family aliases may carry a
# length parameter (VARCHAR(255)), which ClickHouse ignores — the canonical
# form drops it.
_TYPE_ALIASES = {
    "TINYINT": "Int8", "INT1": "Int8", "BYTE": "Int8",
    "TINYINT SIGNED": "Int8", "INT1 SIGNED": "Int8",
    "SMALLINT": "Int16", "SMALLINT SIGNED": "Int16",
    "INT": "Int32", "INT4": "Int32", "INTEGER": "Int32", "MEDIUMINT": "Int32",
    "INT SIGNED": "Int32", "INTEGER SIGNED": "Int32",
    "MEDIUMINT SIGNED": "Int32",
    "BIGINT": "Int64", "SIGNED": "Int64", "BIGINT SIGNED": "Int64",
    "TINYINT UNSIGNED": "UInt8", "INT1 UNSIGNED": "UInt8",
    "SMALLINT UNSIGNED": "UInt16", "YEAR": "UInt16",
    "INT UNSIGNED": "UInt32", "INTEGER UNSIGNED": "UInt32",
    "MEDIUMINT UNSIGNED": "UInt32",
    "UNSIGNED": "UInt64", "BIGINT UNSIGNED": "UInt64",
    "BIT": "UInt64", "SET": "UInt64",
    "FLOAT": "Float32", "REAL": "Float32", "SINGLE": "Float32",
    "DOUBLE": "Float64", "DOUBLE PRECISION": "Float64",
    "BOOL": "Bool", "BOOLEAN": "Bool",
    "TEXT": "String", "TINYTEXT": "String", "MEDIUMTEXT": "String",
    "LONGTEXT": "String", "BLOB": "String", "TINYBLOB": "String",
    "MEDIUMBLOB": "String", "LONGBLOB": "String",
    "CHAR": "String", "CHAR VARYING": "String",
    "CHAR LARGE OBJECT": "String",
    "CHARACTER": "String", "CHARACTER VARYING": "String",
    "CHARACTER LARGE OBJECT": "String",
    "NCHAR": "String", "NCHAR VARYING": "String",
    "NCHAR LARGE OBJECT": "String",
    "NATIONAL CHAR": "String", "NATIONAL CHAR VARYING": "String",
    "NATIONAL CHARACTER": "String",
    "NATIONAL CHARACTER VARYING": "String",
    "NATIONAL CHARACTER LARGE OBJECT": "String",
    "VARCHAR": "String", "VARCHAR2": "String", "NVARCHAR": "String",
    "VARBINARY": "String", "BINARY VARYING": "String",
    "BINARY LARGE OBJECT": "String",
    "CLOB": "String", "BYTEA": "String",
    "TIMESTAMP": "DateTime",
}


def _canonical_type(ch_type: str) -> str:
    """Resolve SQL-compat aliases to canonical names, like DataTypeFactory."""
    stripped = ch_type.strip()
    head, sep, params = stripped.partition("(")
    canonical = _TYPE_ALIASES.get(head.strip().upper())
    if canonical is None:
        return stripped
    if canonical == "String":
        return canonical  # length parameters are ignored by ClickHouse
    return canonical + sep + params


def _numpy_clickhouse_type(spec: type) -> Optional[str]:
    """dtype-based mapping for numpy scalar annotations, like chdb-core."""
    try:
        dtype = str(spec().dtype)
    except Exception:
        return None
    return _NUMPY_DTYPE_TO_CLICKHOUSE.get(dtype)


def _clickhouse_type(spec: Any) -> str:
    """Map a type spec to a ClickHouse type name, mirroring local chdb.

    Accepts ClickHouse type strings ("Int64", "DateTime64(6)"), chdb.sqltypes
    objects (via their ``name`` attribute), Python types (int/float/str/
    bytes/bytearray/bool/date/datetime), numpy scalar types (np.int32, ...),
    and ``Optional[X]`` / ``X | None`` (translated as X — the declaration is
    made Nullable regardless, matching the engine's makeNullable behavior).
    ``None`` (a parameter without annotation) falls back to String: locally
    such parameters are dynamic, but an executable UDF must declare a type.
    Anything else raises, matching local registration errors.
    """
    if spec is None:
        return "String"
    # Optional[X] / X | None — beyond local chdb, which currently rejects
    # these annotations outright (see the chdb-core issue); the inner type is
    # what matters since declarations are Nullable either way.
    origin = typing.get_origin(spec)
    if origin is typing.Union or (_UNION_TYPE is not None and origin is _UNION_TYPE):
        members = [m for m in typing.get_args(spec) if m is not type(None)]
        if len(members) == 1:
            return _clickhouse_type(members[0])
        raise ValueError(f"Unknown Python UDF type annotation: {spec!r}")
    if isinstance(spec, str):
        return _canonical_type(spec)
    name = getattr(spec, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(spec, type):
        for py_type, ch_type in _PY_TO_CLICKHOUSE.items():
            if spec is py_type:
                return ch_type
        numpy_type = _numpy_clickhouse_type(spec)
        if numpy_type is not None:
            return numpy_type
        raise ValueError(
            f"Cannot convert Python type {spec!r} to a ClickHouse type"
        )
    raise ValueError(f"Unknown Python UDF type annotation: {spec!r}")


# Types we must not wrap in Nullable. Ground truth is IDataType::
# canBeInsideNullable (default false, per-type overrides); the set below is
# the conservative union across server versions we may deploy to — newer
# engines allow Nullable(Tuple)/Nullable(Object), but e.g. 24.8 rejects
# them, and an unwrapped declaration is always legal.
_NOT_INSIDE_NULLABLE = {
    "Nullable",
    "Array",
    "Map",
    "Tuple",
    "Nested",
    "LowCardinality",
    "AggregateFunction",
    "SimpleAggregateFunction",
    "Variant",
    "Dynamic",
    "JSON",
    "Object",
}

# The exact whitelist local chdb UDF registration enforces
# (isSupportedUDFType in PythonScalarUDF.cpp); Bool is the engine's alias
# for UInt8 and passes the same check. Everything else — Decimal, UUID,
# FixedString, Enum, LowCardinality, IPv4/6, Array/Map/Tuple, explicit
# Nullable(...) argument declarations — is rejected there, so deployment
# rejects it the same way. Nullability is applied implicitly on both sides.
_SUPPORTED_BASE_TYPES = {
    "UInt8", "UInt16", "UInt32", "UInt64", "UInt128", "UInt256",
    "Int8", "Int16", "Int32", "Int64", "Int128", "Int256",
    "Float32", "Float64", "Bool",
    "String", "Date", "Date32", "DateTime", "DateTime64",
}


def _reject_unsupported(fn_name: str, what: str, ch_type: str) -> None:
    base = ch_type.split("(")[0].strip()
    if base not in _SUPPORTED_BASE_TYPES:
        raise ValueError(
            f"Cannot deploy {fn_name}(): unsupported {what} type "
            f"'{ch_type}' — matching local chdb UDF registration, which "
            "supports integers, floats, Bool, String and Date/DateTime "
            "types (nullability is applied implicitly)"
        )


def _make_nullable(ch_type: str) -> str:
    """Wrap a type in Nullable(...) like the engine does for local UDFs."""
    base = ch_type.split("(")[0].strip()
    if base in _NOT_INSIDE_NULLABLE:
        return ch_type
    return f"Nullable({ch_type})"


def _resolve_types(
    fn, arg_types: Optional[Sequence[Any]], return_type: Any
) -> Tuple[List[Tuple[str, str]], str]:
    """Resolve ((arg_name, ch_type), ...) and the return ClickHouse type."""
    signature = inspect.signature(fn)
    params = list(signature.parameters.values())
    # The generated wrapper calls fn(*args): only positional parameters can
    # be mapped to ClickHouse UDF arguments.
    unsupported = [
        param.name
        for param in params
        if param.kind
        not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if unsupported:
        raise ValueError(
            f"Cannot deploy {fn.__name__}(): parameters "
            f"{', '.join(unsupported)} are keyword-only/*args/**kwargs, "
            "which cannot be mapped to positional ClickHouse UDF arguments"
        )
    # Under `from __future__ import annotations` (or quoted annotations) the
    # signature holds strings like "int", which _clickhouse_type would treat
    # as ClickHouse type names. Resolve them to real objects first.
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    def annotation_of(param):
        if param.name in hints:
            return hints[param.name]
        if param.annotation is inspect.Parameter.empty:
            return None
        return param.annotation

    if arg_types is not None:
        if len(arg_types) != len(params):
            raise ValueError(
                f"arg_types has {len(arg_types)} entries but "
                f"{fn.__name__}() takes {len(params)} parameters"
            )
        resolved = [
            (param.name, _clickhouse_type(spec))
            for param, spec in zip(params, arg_types)
        ]
    else:
        resolved = [
            (param.name, _clickhouse_type(annotation_of(param)))
            for param in params
        ]
    if return_type is not None:
        ch_return = _clickhouse_type(return_type)
    elif "return" in hints:
        ch_return = _clickhouse_type(hints["return"])
    elif signature.return_annotation is not inspect.Signature.empty:
        ch_return = _clickhouse_type(signature.return_annotation)
    else:
        # Matches local registration: "return type not specified".
        raise ValueError(
            f"Cannot deploy {fn.__name__}(): return type not specified — "
            "annotate the return or pass return_type=..."
        )
    # Strict local alignment: reject everything isSupportedUDFType rejects.
    # Arguments are validated as declared (local rejects explicit
    # Nullable(...) there); the return type is validated after stripping
    # Nullable (local removeNullable's it first).
    for index, (_, arg_type) in enumerate(resolved, start=1):
        _reject_unsupported(fn.__name__, f"argument {index}", arg_type)
    _reject_unsupported(fn.__name__, "return", _strip_nullable(ch_return))
    return resolved, ch_return


# ---------------------------------------------------------------------------
# Source extraction and artifact generation
# ---------------------------------------------------------------------------


def _function_source(fn) -> str:
    """The function's own source with any decorator lines stripped."""
    source = textwrap.dedent(inspect.getsource(fn))
    lines = source.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("async def "):
            return "\n".join(lines[index:])
    raise ValueError(
        f"Could not locate the `def` line in the source of {fn.__name__}()"
    )


_TZ_IN_TYPE = re.compile(r"'([^']+)'")


def _timezone_of(ch_type: str) -> Optional[str]:
    """The explicit timezone in a DateTime/DateTime64 type string, if any."""
    match = _TZ_IN_TYPE.search(ch_type)
    return match.group(1) if match else None


def _strip_nullable(ch_type: str) -> str:
    if ch_type.split("(")[0] == "Nullable":
        return ch_type[len("Nullable(") : -1]
    return ch_type


def _converter_expr(ch_type: str) -> str:
    """The converter expression baked into the generated script."""
    inner = _strip_nullable(ch_type)
    base = inner.split("(")[0]
    if base.startswith("Int") or base.startswith("UInt"):
        return "int"
    if base.startswith("Float"):
        return "float"
    if base == "Bool":
        return "_parse_bool"
    if base in ("Date", "Date32"):
        return "_parse_date"
    if base in ("DateTime", "DateTime64"):
        # DateTime('UTC') -> parser that attaches that zone; plain DateTime ->
        # parser that attaches the host zone (== the server default unless
        # config.xml overrides it; see the module docstring)
        return f"_make_datetime_parser({_timezone_of(inner)!r})"
    return "_identity"


_SCRIPT_HELPERS = '''
def _identity(value):
    return value


def _parse_bool(value):
    return value in ("true", "1", "True")


def _parse_date(value):
    import datetime
    return datetime.date.fromisoformat(value)


def _make_datetime_parser(tz_name):
    # The pipe carries wall-clock text with no offset. It is rendered in the
    # column's declared timezone when the type carries one, else in the
    # server default; attach the matching tzinfo so the function receives an
    # aware datetime, like local chdb UDFs do.
    import datetime
    tz = None
    if tz_name is not None:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)

    def _parse(value):
        parsed = datetime.datetime.fromisoformat(value)
        if tz is not None:
            return parsed.replace(tzinfo=tz)
        # naive server-default wall time -> aware, using the host zone (the
        # wrapper runs on the server host)
        return parsed.astimezone()

    return _parse


def _unescape(value):
    if "\\\\" not in value:
        return value
    out = []
    escapes = {"\\\\": "\\\\", "t": "\\t", "n": "\\n", "r": "\\r",
               "0": "\\0", "b": "\\b", "f": "\\f", "'": "'"}
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\\\" and index + 1 < len(value):
            nxt = value[index + 1]
            out.append(escapes.get(nxt, nxt))
            index += 2
        else:
            out.append(char)
            index += 1
    return "".join(out)


def _escape(value):
    return (value.replace("\\\\", "\\\\\\\\").replace("\\t", "\\\\t")
            .replace("\\n", "\\\\n").replace("\\r", "\\\\r"))


def _format_result(result):
    if result is None:
        return "\\\\N"
    if isinstance(result, bool):
        return "true" if result else "false"
    if isinstance(result, (bytes, bytearray)):
        # protocol text, not Python repr (str(b"x") would emit "b'x'")
        return _escape(bytes(result).decode("utf-8"))
    import datetime
    if isinstance(result, datetime.datetime):
        if result.tzinfo is not None:
            # The server parses the output as wall-clock text in the return
            # type's timezone (or the server default): convert, then strip
            # the offset — TSV carries none.
            if _RETURN_TZ is not None:
                from zoneinfo import ZoneInfo
                result = result.astimezone(ZoneInfo(_RETURN_TZ))
            else:
                result = result.astimezone()
            result = result.replace(tzinfo=None)
        return result.isoformat(sep=" ")
    return _escape(str(result))
'''


def _on_null_skips(on_null: Any) -> bool:
    """True when NULL inputs return NULL without calling the function.

    Mirrors chdb.udf.func's on_null: "skip" (default) or "pass"; accepts the
    chdb.NullHandling enums as well.
    """
    if on_null is None:
        return True
    return "pass" not in str(on_null).lower()


def _on_error_ignores(on_error: Any) -> bool:
    """True when a raising row returns NULL instead of failing the query.

    Mirrors chdb.udf.func's on_error: "propagate" (default) or "ignore";
    accepts the chdb.ExceptionHandling enums as well.
    """
    if on_error is None:
        return False
    return "ignore" in str(on_error).lower()


def _generate_script(
    fn_name: str,
    source: str,
    arg_ch_types: List[str],
    *,
    return_ch_type: str = "String",
    null_skip: bool = True,
    error_ignore: bool = False,
) -> str:
    """Generate the executable stdin/stdout wrapper script for the server."""
    converters = ", ".join(_converter_expr(t) for t in arg_ch_types)
    return_tz = _timezone_of(_strip_nullable(return_ch_type))
    return (
        "#!/usr/bin/env python3\n"
        "# Generated by chdb.deploy — do not edit by hand.\n"
        "\n"
        "# Deferred annotation evaluation: the function source keeps its type\n"
        "# annotations (e.g. `d: date`) whose names are not imported here; on\n"
        "# Python < 3.14 they would otherwise be evaluated at definition time.\n"
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "\n"
        f"{source.rstrip()}\n"
        "\n"
        f"{_SCRIPT_HELPERS.strip()}\n"
        "\n"
        f"_CONVERTERS = [{converters}]\n"
        f"_NULL_SKIP = {null_skip!r}\n"
        f"_ERROR_IGNORE = {error_ignore!r}\n"
        f"_RETURN_TZ = {return_tz!r}\n"
        "\n"
        "\n"
        "def _main():\n"
        "    for line in sys.stdin:\n"
        '        fields = line.rstrip("\\n").split("\\t")\n'
        "        args = []\n"
        "        for convert, field in zip(_CONVERTERS, fields):\n"
        '            # \\N is the TSV representation of NULL\n'
        '            args.append(None if field == "\\\\N"\n'
        "                        else convert(_unescape(field)))\n"
        "        # on_null='skip': NULL input returns NULL without calling\n"
        "        if _NULL_SKIP and any(arg is None for arg in args):\n"
        '            sys.stdout.write("\\\\N\\n")\n'
        "            sys.stdout.flush()\n"
        "            continue\n"
        "        try:\n"
        f"            result = {fn_name}(*args)\n"
        "        except Exception:\n"
        "            # on_error='ignore': a raising row returns NULL\n"
        "            if not _ERROR_IGNORE:\n"
        "                raise\n"
        "            result = None\n"
        '        sys.stdout.write(_format_result(result) + "\\n")\n'
        "        sys.stdout.flush()\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    _main()\n"
    )


def _generate_config_xml(
    remote_name: str,
    script_filename: str,
    arg_specs: List[Tuple[str, str]],
    return_type: str,
) -> bytes:
    """Generate the ClickHouse executable-UDF XML config for one function."""
    root = ET.Element("functions")
    function = ET.SubElement(root, "function")
    ET.SubElement(function, "type").text = "executable"
    ET.SubElement(function, "execute_direct").text = "1"
    ET.SubElement(function, "name").text = remote_name
    ET.SubElement(function, "return_type").text = return_type
    ET.SubElement(function, "format").text = "TabSeparated"
    ET.SubElement(function, "command").text = script_filename
    for arg_name, arg_type in arg_specs:
        argument = ET.SubElement(function, "argument")
        ET.SubElement(argument, "type").text = arg_type
        ET.SubElement(argument, "name").text = arg_name
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Server communication (stdlib-only HTTP client)
# ---------------------------------------------------------------------------


def _http_query(connection, query: str, timeout: float = 30.0) -> str:
    """Run a query over the ClickHouse HTTP interface and return its body."""
    url = connection.http_url + "/?" + urllib.parse.urlencode(
        {"database": connection.database}
    )
    request = urllib.request.Request(url, data=query.encode("utf-8"), method="POST")
    request.add_header("X-ClickHouse-User", connection.username)
    if connection.password:
        request.add_header("X-ClickHouse-Key", connection.password)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"ClickHouse HTTP error {error.code} from {connection.http_url}: {body}"
        ) from None
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach ClickHouse at {connection.http_url}: {error.reason}"
        ) from None


def _artifact_paths(connection, name: str) -> Tuple[str, str]:
    """(script_path, config_path) for a UDF name on this connection."""
    return (
        os.path.join(connection.udf_scripts_dir, f"{name}.py"),
        os.path.join(connection.udf_config_dir, f"{name}_function.xml"),
    )


def _artifacts_exist(connection, name: str) -> bool:
    """Whether this channel holds deploy artifacts for the given name."""
    if not connection.supports_udf_deploy():
        return False
    script_path, config_path = _artifact_paths(connection, name)
    return os.path.exists(script_path) and os.path.exists(config_path)


def _artifacts_match(
    connection, name: str, script_body: str, config_xml: bytes
) -> bool:
    """Whether the on-disk artifacts equal the freshly generated ones."""
    script_path, config_path = _artifact_paths(connection, name)
    try:
        with open(script_path, "rb") as stream:
            if stream.read() != script_body.encode("utf-8"):
                return False
        with open(config_path, "rb") as stream:
            return stream.read() == config_xml
    except OSError:
        return False


def _function_exists(connection, name: str) -> bool:
    result = _http_query(
        connection,
        f"SELECT count() FROM system.functions WHERE name = '{name}'",
    )
    return result.strip() == "1"


def _reload_functions(connection) -> None:
    _http_query(connection, "SYSTEM RELOAD FUNCTIONS")


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


@dataclass
class DeployedFunction:
    """Handle for a UDF deployed to a ClickHouse server."""

    remote_name: str
    connection: str
    permanent: bool
    skipped: bool = False
    artifact_paths: List[str] = field(default_factory=list)

    def undeploy(self) -> None:
        """Remove the deployed artifacts and reload server functions."""
        _remove_deployment(self)


def _write_atomic(path: str, data: bytes, mode: Optional[int] = None) -> None:
    """Write a file via a temp sibling + os.replace (never seen half-written)."""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as stream:
            stream.write(data)
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _resolve_connection(to: Any):
    # NOTE: `from datastore import config` would yield the DataStoreConfig
    # facade instance (it shadows the module in the package namespace); the
    # from-import of the function below reliably resolves the module.
    from datastore.config import get_connection

    name = to if isinstance(to, str) else None
    return get_connection(name)


def _remove_deployment(deployment: DeployedFunction) -> None:
    for path in deployment.artifact_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        from datastore.config import get_connection

        _reload_functions(get_connection(deployment.connection))
    except Exception:
        pass
    with _session_lock:
        if deployment in _session_deployments:
            _session_deployments.remove(deployment)


def cleanup_session() -> None:
    """Drop every non-permanent UDF this process deployed (best effort)."""
    with _session_lock:
        deployments = list(_session_deployments)
    for deployment in deployments:
        _remove_deployment(deployment)


def _register_session_cleanup(deployment: DeployedFunction) -> None:
    global _atexit_registered
    with _session_lock:
        _session_deployments.append(deployment)
        if not _atexit_registered:
            atexit.register(cleanup_session)
            _atexit_registered = True


def _deploy_impl(
    fn,
    to: Any = None,
    *,
    permanent: bool = False,
    arg_types: Optional[Sequence[Any]] = None,
    return_type: Any = None,
    name: Optional[str] = None,
    on_null: Any = None,
    on_error: Any = None,
) -> DeployedFunction:
    if inspect.iscoroutinefunction(fn):
        raise ValueError(
            f"Cannot deploy {fn.__name__}(): async functions are not "
            "supported as ClickHouse executable UDFs"
        )
    arg_specs, ch_return = _resolve_types(fn, arg_types, return_type)
    # Local chdb wraps every UDF argument and return type in Nullable (the
    # engine's makeNullable); declare the same remotely so NULL flows in (the
    # wrapper's on_null handling needs to see it) and NULL results parse as
    # NULL instead of being coerced to the type's default value.
    arg_specs = [
        (arg_name, _make_nullable(arg_type)) for arg_name, arg_type in arg_specs
    ]
    ch_return = _make_nullable(ch_return)
    source = _function_source(fn)
    null_skip = _on_null_skips(on_null)
    error_ignore = _on_error_ignores(on_error)
    connection = _resolve_connection(to)

    # One predictable name whether or not the deployment is permanent: the
    # function's own, or the caller's name= override. Session-scoped
    # chdb_nb_* names guaranteed isolation between concurrent processes but
    # made a temporary UDF uncallable by any name a user could type.
    # session_id() is still consulted for its side effect: it resets the
    # cleanup ledger after a fork.
    session_id()
    remote_name = name or fn.__name__
    if not remote_name.isidentifier():
        raise ValueError(f"Invalid UDF name: {remote_name!r}")

    script_filename = f"{remote_name}.py"
    script_body = _generate_script(
        fn.__name__,
        source,
        [arg_type for _, arg_type in arg_specs],
        return_ch_type=ch_return,
        null_skip=null_skip,
        error_ignore=error_ignore,
    )
    config_xml = _generate_config_xml(
        remote_name, script_filename, arg_specs, ch_return
    )

    # An existing function with the same name is reused (idempotent re-runs)
    # only when this channel's artifacts match what we would deploy now:
    # - identical artifacts -> skip;
    # - our artifacts but different content (a permanent function whose code
    #   changed) -> fall through and replace them in place;
    # - no artifacts -> it is a ClickHouse built-in or an unrelated UDF;
    #   silently skipping would leave queries running the wrong function.
    if _function_exists(connection, remote_name):
        if _artifacts_exist(connection, remote_name):
            if _artifacts_match(connection, remote_name, script_body, config_xml):
                deployment = DeployedFunction(
                    remote_name=remote_name,
                    connection=connection.name,
                    permanent=permanent,
                    skipped=True,
                    artifact_paths=(
                        list(_artifact_paths(connection, remote_name))
                        if connection.supports_udf_deploy()
                        else []
                    ),
                )
                if not permanent:
                    # ensure the skipped handle is covered by session cleanup
                    # without double-tracking the original deployment
                    with _session_lock:
                        tracked = any(
                            d.remote_name == remote_name
                            and d.connection == connection.name
                            for d in _session_deployments
                        )
                    if not tracked:
                        _register_session_cleanup(deployment)
                return deployment
        else:
            raise ValueError(
                f"A function named {remote_name!r} already exists on "
                f"connection {connection.name!r} but was not deployed through "
                "this channel (it may be a ClickHouse built-in or an "
                "unrelated UDF). Pick a different name via name=..."
            )

    if not connection.supports_udf_deploy():
        raise RuntimeError(
            f"Connection {connection.name!r} has no UDF delivery channel. "
            "Register it with udf_scripts_dir=... and udf_config_dir=... "
            "(local paths the server reads as user_scripts_path and "
            "user_defined_executable_functions_config). ClickHouse Cloud "
            "currently only accepts UDF uploads through its console and is "
            "not yet supported as a deploy target."
        )

    script_path, config_path = _artifact_paths(connection, remote_name)

    # Everything from the first write onward is guarded: a failure at any
    # point must not leave partial artifacts, and files that already existed
    # for this name are restored rather than deleted. Writes go through a
    # temporary file + os.replace so the server's config watcher can never
    # observe a half-written artifact.
    # {path: (contents, permission bits)} — restored verbatim on failure
    previous_artifacts: Dict[str, Tuple[bytes, int]] = {}
    for path in (script_path, config_path):
        if os.path.exists(path):
            with open(path, "rb") as stream:
                data = stream.read()
            previous_artifacts[path] = (data, os.stat(path).st_mode & 0o7777)

    try:
        _write_atomic(script_path, script_body.encode("utf-8"), mode=0o755)
        _write_atomic(config_path, config_xml)

        _reload_functions(connection)
        deadline = time.monotonic() + 10.0
        while not _function_exists(connection, remote_name):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"UDF {remote_name!r} did not appear in system.functions "
                    f"after SYSTEM RELOAD FUNCTIONS on {connection.http_url}. "
                    "Check the server's user_defined_executable_functions_config "
                    "glob matches the configured udf_config_dir."
                )
            time.sleep(0.2)
    except Exception:
        for path in (script_path, config_path):
            try:
                if path in previous_artifacts:
                    data, mode = previous_artifacts[path]
                    _write_atomic(path, data, mode=mode)
                else:
                    os.remove(path)
            except OSError:
                pass
        # The failed attempt may have happened after a successful reload;
        # reload again so the server does not keep serving a UDF whose
        # artifacts were just removed or restored.
        try:
            _reload_functions(connection)
        except Exception:
            pass
        raise

    deployment = DeployedFunction(
        remote_name=remote_name,
        connection=connection.name,
        permanent=permanent,
        artifact_paths=[script_path, config_path],
    )
    if not permanent:
        _register_session_cleanup(deployment)
    return deployment


def deploy(
    fn,
    to: Any = None,
    *,
    permanent: bool = False,
    arg_types: Optional[Sequence[Any]] = None,
    return_type: Any = None,
    name: Optional[str] = None,
    on_null: Any = None,
    on_error: Any = None,
) -> DeployedFunction:
    """Deploy an already-defined Python function as a ClickHouse UDF.

    Args:
        fn: The function to deploy (does not need the ``@func`` decorator).
        to: Connection name registered in datastore.config; None uses the
            default connection.
        permanent: Keep the function after this session ends. Defaults to
            False: the deployment is dropped at process exit. Either way it
            registers under the function's own name (or ``name=``).
        arg_types: Optional explicit argument types (else annotations, else
            String).
        return_type: Optional explicit return type (else annotation, else
            String).
        name: Override the remote name (any deployment).
        on_null: "skip" (default — NULL input returns NULL without calling
            the function) or "pass" (call with None), as in chdb.udf.func;
            emulated in the generated wrapper.
        on_error: "propagate" (default — a raising row fails the query) or
            "ignore" (return NULL for that row), as in chdb.udf.func;
            emulated in the generated wrapper.

    Returns:
        DeployedFunction handle. ``skipped=True`` means the identical
        artifacts were already deployed and nothing was written; a permanent
        redeploy whose code or types changed replaces the artifacts in place.
    """
    return _deploy_impl(
        fn,
        to,
        permanent=permanent,
        arg_types=arg_types,
        return_type=return_type,
        name=name,
        on_null=on_null,
        on_error=on_error,
    )


def undeploy(name: str, to: Any = None) -> bool:
    """Remove a deployed UDF's artifacts by name and reload functions.

    Returns True when at least one artifact file was removed.
    """
    connection = _resolve_connection(to)
    if not connection.supports_udf_deploy():
        raise RuntimeError(
            f"Connection {connection.name!r} has no UDF delivery channel."
        )
    # A non-identifier name could escape the configured directories through
    # os.path.join (absolute paths, ../ segments).
    if not name.isidentifier():
        raise ValueError(f"Invalid UDF name: {name!r}")
    removed = False
    script_path, config_path = _artifact_paths(connection, name)
    for path in (config_path, script_path):
        try:
            os.remove(path)
            removed = True
        except OSError:
            pass
    if removed:
        _reload_functions(connection)
    return removed


# ---------------------------------------------------------------------------
# The extended decorator
# ---------------------------------------------------------------------------

# chdb.udf's genuine decorator. _install() rebinds chdb.udf.func to this
# module's func, so the delegation below must hold on to the original the
# first time it is seen — otherwise the decorator would delegate to itself.
_original_local_func = None


def _remember_original_local_func(candidate) -> None:
    global _original_local_func
    if (
        candidate is not None
        and candidate is not func
        and _original_local_func is None
    ):
        _original_local_func = candidate


def _fallback_local_func(arg_types=None, return_type=None, **kwargs):
    """Replicate chdb.udf.func when the original was lost to patching."""

    def decorator(fn):
        create = getattr(chdb, "create_function", None) or getattr(
            chdb._chdb, "create_function", None
        )
        if create is None:
            raise RuntimeError("chdb.create_function is unavailable")
        create(fn.__name__, fn, arg_types, return_type, **kwargs)

        @functools.wraps(fn)
        def wrapper(*args, **call_kwargs):
            return fn(*args, **call_kwargs)

        return wrapper

    return decorator


def _resolve_local_func():
    global _original_local_func
    if _original_local_func is None:
        from chdb.udf import func as candidate

        _remember_original_local_func(candidate)
    return _original_local_func or _fallback_local_func


def _register_expression_method(fn) -> None:
    """Expose a locally-registered UDF as a DataStore expression method.

    After this, ``ds["price"].tax(0.13)`` builds the same SQL the built-in
    function methods do, executed by the local chdb engine where the UDF is
    registered. Fail-open: expression sugar must never break registration.
    """
    try:
        from datastore.function_registry import FunctionRegistry

        register = getattr(FunctionRegistry, "register_udf", None)
        if register is None:
            return
        arity = len(inspect.signature(fn).parameters)
        register(fn.__name__, arity, doc=fn.__doc__ or "")
    except Exception:
        pass


def func(
    arg_types=None,
    return_type=None,
    *,
    on_null=None,
    on_error=None,
    deploy=False,
    permanent=False,
):
    """Drop-in replacement for :func:`chdb.udf.func` with deploy support.

    Local registration is delegated to ``chdb.udf.func`` unchanged. With
    ``deploy`` set, the function is additionally translated to a ClickHouse
    executable UDF and shipped to the target server (see the module
    docstring).

    Args:
        arg_types: As in chdb.udf.func.
        return_type: As in chdb.udf.func.
        on_null: As in chdb.udf.func.
        on_error: As in chdb.udf.func.
        deploy: False (local only), True (default connection), or the name of
            a connection registered via datastore.config.register_connection.
        permanent: Keep the deployed UDF after this session ends, under the
            function's own name. Requires ``deploy``.
    """
    if permanent and not deploy:
        raise ValueError("permanent=True requires deploy=True or deploy='<name>'")

    def decorator(fn):
        passthrough = {}
        if on_null is not None:
            passthrough["on_null"] = on_null
        if on_error is not None:
            passthrough["on_error"] = on_error
        wrapped = _resolve_local_func()(arg_types, return_type, **passthrough)(fn)
        _register_expression_method(fn)

        if deploy:
            deployment = _deploy_impl(
                fn,
                deploy if isinstance(deploy, str) else None,
                permanent=permanent,
                arg_types=arg_types,
                return_type=return_type,
                on_null=on_null,
                on_error=on_error,
            )
            wrapped.chdb_deployment = deployment
        return wrapped

    return decorator


# ---------------------------------------------------------------------------
# Import-time self-repair
# ---------------------------------------------------------------------------
#
# chdb/__init__.py prefers this module's `func` via its own hook, but the
# on-disk __init__.py may be an older chdb-core copy without the hook (the
# two wheels share that file path and the later install wins). Importing
# chdb.deploy — directly or via `import datastore` — repairs the entry points
# regardless of which __init__.py won.


def _install() -> None:
    chdb.func = func
    udf_module = sys.modules.get("chdb.udf")
    if udf_module is not None:
        _remember_original_local_func(getattr(udf_module, "func", None))
        udf_module.func = func


_install()
