"""Deploy chdb Python UDFs to a remote ClickHouse server.

This module extends :func:`chdb.udf.func` with two extra parameters:

- ``deploy``: ``False`` (default, local-only), ``True`` (deploy to the default
  connection), or a connection name registered via
  :func:`datastore.config.register_connection`.
- ``permanent``: ``False`` (default) deploys under a session-scoped name
  ``chdb_nb_{session}_{hash}`` that is dropped when the process exits;
  ``True`` deploys under the function's own name and survives the session.

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
import hashlib
import inspect
import os
import secrets
import sys
import textwrap
import threading
import time
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

# Session identity: one per process, embedded in non-permanent remote names so
# concurrent notebooks cannot collide and leaked functions stay identifiable.
_SESSION_ID = secrets.token_hex(3)
_SESSION_NAME_PREFIX = "chdb_nb_"

_session_deployments: List["DeployedFunction"] = []
_session_lock = threading.Lock()
_atexit_registered = False


def session_id() -> str:
    """This process's deploy session id (embedded in temporary UDF names)."""
    return _SESSION_ID


# ---------------------------------------------------------------------------
# Type resolution: decorator specs / annotations -> ClickHouse type names
# ---------------------------------------------------------------------------

_PY_TO_CLICKHOUSE = {
    int: "Int64",
    float: "Float64",
    str: "String",
    bool: "Bool",
    date: "Date32",
    datetime: "DateTime64(3)",
}


def _clickhouse_type(spec: Any) -> str:
    """Best-effort mapping of a type spec to a ClickHouse type name.

    Accepts ClickHouse type strings ("Int64"), chdb.sqltypes objects (via
    their ``name`` attribute), Python types (int/float/str/bool/date/
    datetime), or None. Anything unrecognized falls back to String, matching
    the legacy ``@chdb_udf`` behavior.
    """
    if spec is None:
        return "String"
    if isinstance(spec, str):
        return spec
    name = getattr(spec, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(spec, type):
        for py_type, ch_type in _PY_TO_CLICKHOUSE.items():
            if spec is py_type:
                return ch_type
    return "String"


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
        ch_return = "String"
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


def _converter_name(ch_type: str) -> str:
    base = ch_type.split("(")[0]
    if base == "Nullable":
        inner = ch_type[len("Nullable(") : -1]
        return _converter_name(inner)
    if base.startswith("Int") or base.startswith("UInt"):
        return "int"
    if base.startswith("Float"):
        return "float"
    if base.startswith("Decimal"):
        return "_parse_decimal"
    if base == "Bool":
        return "_parse_bool"
    if base in ("Date", "Date32"):
        return "_parse_date"
    if base in ("DateTime", "DateTime64"):
        return "_parse_datetime"
    return "_identity"


_SCRIPT_HELPERS = '''
def _identity(value):
    return value


def _parse_bool(value):
    return value in ("true", "1", "True")


def _parse_decimal(value):
    from decimal import Decimal
    return Decimal(value)


def _parse_date(value):
    import datetime
    return datetime.date.fromisoformat(value)


def _parse_datetime(value):
    import datetime
    return datetime.datetime.fromisoformat(value)


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
    return _escape(str(result))
'''


def _generate_script(fn_name: str, source: str, arg_ch_types: List[str]) -> str:
    """Generate the executable stdin/stdout wrapper script for the server."""
    converters = ", ".join(_converter_name(t) for t in arg_ch_types)
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
        f"        result = {fn_name}(*args)\n"
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
    """Drop every session-scoped UDF this process deployed (best effort)."""
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
) -> DeployedFunction:
    if inspect.iscoroutinefunction(fn):
        raise ValueError(
            f"Cannot deploy {fn.__name__}(): async functions are not "
            "supported as ClickHouse executable UDFs"
        )
    arg_specs, ch_return = _resolve_types(fn, arg_types, return_type)
    source = _function_source(fn)
    connection = _resolve_connection(to)

    if permanent:
        remote_name = name or fn.__name__
    else:
        digest_input = source + repr(arg_specs) + ch_return + connection.name
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:8]
        remote_name = f"{_SESSION_NAME_PREFIX}{_SESSION_ID}_{digest}"
    if not remote_name.isidentifier():
        raise ValueError(f"Invalid UDF name: {remote_name!r}")

    # An existing function with the same name is reused as-is (idempotent
    # re-runs) — but only when this channel's artifacts are what defined it.
    # system.functions also lists built-ins and unrelated UDFs; silently
    # skipping for those would leave queries running the wrong function.
    if _function_exists(connection, remote_name):
        if _artifacts_exist(connection, remote_name):
            return DeployedFunction(
                remote_name=remote_name,
                connection=connection.name,
                permanent=permanent,
                skipped=True,
            )
        raise ValueError(
            f"A function named {remote_name!r} already exists on connection "
            f"{connection.name!r} but was not deployed through this channel "
            "(it may be a ClickHouse built-in or an unrelated UDF). Pick a "
            "different name via name=..."
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
    script_filename = os.path.basename(script_path)

    script_body = _generate_script(
        fn.__name__, source, [arg_type for _, arg_type in arg_specs]
    )
    config_xml = _generate_config_xml(
        remote_name, script_filename, arg_specs, ch_return
    )

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
) -> DeployedFunction:
    """Deploy an already-defined Python function as a ClickHouse UDF.

    Args:
        fn: The function to deploy (does not need the ``@func`` decorator).
        to: Connection name registered in datastore.config; None uses the
            default connection.
        permanent: Deploy under the function's own name and keep it after the
            session ends. Defaults to False (session-scoped name, cleaned up
            at process exit).
        arg_types: Optional explicit argument types (else annotations, else
            String).
        return_type: Optional explicit return type (else annotation, else
            String).
        name: Override the remote name (permanent deployments only).

    Returns:
        DeployedFunction handle (``skipped=True`` when the name already
        existed on the server and nothing was written).
    """
    return _deploy_impl(
        fn,
        to,
        permanent=permanent,
        arg_types=arg_types,
        return_type=return_type,
        name=name,
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

        if deploy:
            deployment = _deploy_impl(
                fn,
                deploy if isinstance(deploy, str) else None,
                permanent=permanent,
                arg_types=arg_types,
                return_type=return_type,
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
