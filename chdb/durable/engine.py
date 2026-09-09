"""The managed chdb-core connection a durable object owns.

Three things live here, and nothing else does:

* the **capability check** — this binding needs the Durable V1 ABI
  (`backup_database`, `restore_database`, `classify_query`) that chdb-core
  exports from `programs/local/chdb.h`. An older engine is refused at open
  with a message that says what to install, not a missing-attribute traceback;
* the **query analysis** the entry gates are built on, as a value type instead
  of the raw dict the extension returns;
* the **one place** that reaches through `chdb.state.sqlitelike.Connection` to
  the extension object, because the three ABI methods are bound on it and not
  on the Python wrapper.

Contract §3 forbids a binding from building `BACKUP` / `RESTORE` SQL or
guessing at a statement's class with a regex, so no SQL text for those three
operations is written anywhere in `chdb.durable` — the engine quotes its own
identifiers and paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .errors import EngineError, EngineIncompatible

#: The three methods the Durable V1 ABI adds to a connection. Named here so the
#: refusal can list what is missing rather than fail on first use.
_REQUIRED_ABI = ("backup_database", "restore_database", "classify_query")

#: Where the wheels that carry the ABI actually are. Not PyPI: a chdb-core
#: release is about half a gigabyte of wheels against a project quota that only
#: moves one way, so releases are published to GitHub and only some of them go
#: on to PyPI. `pip install -U chdb-core` therefore does not reliably get you a
#: newer engine, and a hint that says it does sends people in a circle.
_RELEASES_URL = "https://github.com/chdb-io/chdb-core/releases"

#: The first chdb-core that exports backup / restore / classify.
_ABI_SINCE = "26.7.2-rc.2"


def _abi_hint(missing=()) -> str:
    """Why the engine was refused, which engine it was, and what to install.

    The version is in the message because without it the two failures read
    identically: an engine too old to have the ABI, and a new engine that the
    wrapper is not actually loading because something else owns `chdb/`.
    """
    try:
        loaded = engine_version() or "unknown"
    except Exception:
        loaded = "unknown"

    lines = [
        "chdb.durable needs a chdb-core with the Durable V1 ABI "
        "(backup_database / restore_database / classify_query), added in "
        f"{_ABI_SINCE}.",
        f"The engine loaded here reports {loaded}.",
    ]
    if missing:
        lines.append("Missing: " + ", ".join(missing) + ".")
    lines += [
        "",
        "chdb-core wheels are published as release assets, and the newest one "
        "on PyPI may be older than the newest release, so `pip install -U "
        "chdb-core` is not enough. Install the wheel for your platform from "
        f"{_RELEASES_URL}/latest in the same command as chdb, so the wrapper's "
        "chdb/__init__.py lands on top of the engine's:",
        "",
        '    pip install "chdb[durable]" \\',
        '      "chdb-core @ <the chdb_core-*.whl URL for your platform>"',
    ]
    return "\n".join(lines)

#: Settings pinned on the managed connection. A durable object promises that a
#: statement `execute()` returned from has actually been applied locally, so
#: nothing may complete asynchronously behind the call (§5.3). The public entry
#: gate refuses statement settings that would relax these, because chdb-core
#: classifies them as CONTROL.
_SYNCHRONOUS_SETTINGS = {
    "async_insert": "0",
    "wait_for_async_insert": "1",
    "mutations_sync": "2",
    "alter_sync": "2",
}


def engine_has_v1_abi() -> bool:
    """Does the loaded chdb-core bind the Durable V1 ABI?

    Answered from the extension's connection *class*, so it costs nothing: no
    engine is started, and on a process that already holds a data path this
    cannot conflict with it.
    """
    try:
        import chdb

        raw = getattr(chdb, "_chdb", None)
        connection = getattr(raw, "connect", None)
        return connection is not None and all(hasattr(connection, n) for n in _REQUIRED_ABI)
    except Exception:
        return False


def require_v1_abi() -> None:
    """Refuse early, with a message that names the fix."""
    if not engine_has_v1_abi():
        raise EngineIncompatible(_abi_hint())


#: `chdb_version()` is a compile-time constant, so one reading holds for the
#: life of the process. Cached because the cheapest way to read it needs a
#: connection, and a durable object should not open one just to ask.
_ENGINE_VERSION: Optional[str] = None

#: How many managed connections are live. chdb-core allows one active data path
#: per process (§3.6), so the throwaway connection below must not be opened
#: while an object holds one.
_LIVE_CONNECTIONS = 0


def _select_chdb_version(query) -> Optional[str]:
    """`SELECT chdb()` returns `CHDB_VERSION` — the same compile-time string
    the C `chdb_version()` returns, reachable without a Python binding for it.

    Note it is *not* `SELECT version()`, which is the ClickHouse version
    (chDB 26.7.2-rc.2 carries ClickHouse 26.7.2.1). The contract's engine
    identity is the chDB one.
    """
    try:
        result = query("SELECT chdb()", "CSV")
        text = result.data() if hasattr(result, "data") else str(result)
    except Exception:
        return None
    text = text.strip().strip('"').strip()
    return text or None


def _probe_engine_version() -> Optional[str]:
    """Read the version from a throwaway in-memory engine.

    Only reached before any durable object has started its own engine — with
    one live, a second data path would be refused, so the probe is skipped
    rather than allowed to fail.
    """
    if _LIVE_CONNECTIONS:
        return None
    try:
        import chdb

        connection = getattr(getattr(chdb, "_chdb", None), "connect", None)
        if connection is None:
            return None
        conn = connection()
    except Exception:
        return None
    try:
        return _select_chdb_version(conn.query)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def engine_version() -> str:
    """The running engine's version, for `head.engine.version` and the
    `min_reader` gate.

    `chdb_version()` is the contract's source of truth (§3.1). The C ABI
    exports it and both export allow-lists carry it, but no Python binding
    calls it, so this reads the same constant by the routes that exist, in
    order of how directly they answer the question:

    1. `_chdb.chdb_version()`, if a later chdb-core binds it — the C answer;
    2. `SELECT chdb()`, which returns that same `CHDB_VERSION` constant. Taken
       from a managed connection when one is open, otherwise from a throwaway
       in-memory engine, and cached either way;
    3. the ClickHouse version the engine reports. A different numbering, but
       measured from the engine actually loaded and sharing the chDB tag's
       numeric prefix, so it still orders against a `min_reader`;
    4. the chdb-core distribution version, which is the tag for a released
       wheel but a placeholder in a source checkout — last, for that reason.

    Every form orders under `protocol.parse_version`.
    """
    global _ENGINE_VERSION
    import chdb

    raw = getattr(chdb, "_chdb", None)
    fn = getattr(raw, "chdb_version", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    if _ENGINE_VERSION is None:
        _ENGINE_VERSION = _probe_engine_version()
    if _ENGINE_VERSION:
        return _ENGINE_VERSION
    reported = str(getattr(chdb, "engine_version", "") or "")
    from .protocol import parse_version

    if parse_version(reported) is not None:
        return reported
    try:
        import importlib.metadata

        return importlib.metadata.version("chdb-core")
    except Exception:
        pass
    return reported


@dataclass(frozen=True)
class QueryAnalysis:
    """What chdb-core says a statement would do, without running it.

    `query_class` is the enum member's name rather than the enum itself, so a
    conformance runner can compare it to the same string every other binding
    reports.
    """

    query_class: str
    statement_count: int
    has_secrets: bool
    writes_only_target_database: bool
    changes_database_lifecycle: bool


class ManagedConnection:
    """A chDB connection scoped to one durable object's scratch directory.

    The connection is not handed to callers (§5.3): its current database is
    pinned to the object's, its synchronous-write settings are part of the
    durability promise, and its backup path is confined to the object's own
    scratch.
    """

    def __init__(self, data_path: str, backups_path: str):
        global _ENGINE_VERSION, _LIVE_CONNECTIONS
        from chdb.state import sqlitelike

        self._counted = False
        args = dict(_SYNCHRONOUS_SETTINGS)
        # `backups.allowed_path` is what makes backup/restore possible at all:
        # chdb-core refuses an archive path outside it, and a connection that
        # never set it cannot back anything up. Scoping it to this object's
        # scratch also means a restore cannot read an archive some other
        # object left on disk.
        args["backups.allowed_path"] = backups_path
        conn_str = data_path + "?" + "&".join(f"{k}={v}" for k, v in args.items())
        try:
            self._conn = sqlitelike.Connection(conn_str)
        except Exception as exc:
            raise EngineError(f"could not start the chDB engine: {exc}") from exc
        _LIVE_CONNECTIONS += 1
        self._counted = True
        self._raw = getattr(self._conn, "_conn", None)
        missing = [n for n in _REQUIRED_ABI if not hasattr(self._raw, n)]
        if missing:
            self.close()
            raise EngineIncompatible(_abi_hint(missing))
        # Read the engine identity off the connection we already have, which is
        # both cheaper and more accurate than the fallbacks in engine_version().
        if _ENGINE_VERSION is None:
            _ENGINE_VERSION = _select_chdb_version(self._conn.query)

    def engine_version(self) -> str:
        """This engine's version, measured from this connection where possible."""
        global _ENGINE_VERSION
        if _ENGINE_VERSION is None:
            _ENGINE_VERSION = _select_chdb_version(self._conn.query)
        return _ENGINE_VERSION or engine_version()

    # -- statements -------------------------------------------------------
    def query(self, sql: str, fmt: str = "CSV", *, redact: bool = False):
        """Run `sql`. `redact=True` keeps a failed statement's text out of the
        exception, for SQL the analysis flagged as carrying a credential."""
        try:
            return self._conn.query(sql, fmt)
        except Exception as exc:
            if redact:
                raise EngineError("statement failed (text withheld: it carries a credential)") from None
            raise EngineError(str(exc)) from exc

    def use_database(self, database: str) -> None:
        """Pin the current database, so unqualified names in a caller's SQL
        resolve to the object's database — both when chdb-core resolves them
        during analysis and when the statement runs."""
        # The identifier goes through a parameter rather than into the SQL
        # text, so a database name holding a backtick cannot change what runs.
        try:
            self._conn.query("USE {db:Identifier}", "CSV", params={"db": database})
        except Exception as exc:
            raise EngineError(f"could not select database {database!r}: {exc}") from exc

    def create_database(self, database: str) -> None:
        try:
            self._conn.query(
                "CREATE DATABASE IF NOT EXISTS {db:Identifier}", "CSV", params={"db": database}
            )
        except Exception as exc:
            raise EngineError(f"could not create database {database!r}: {exc}") from exc

    # -- Durable V1 ABI ---------------------------------------------------
    def analyze(self, sql: str, target_database: str) -> QueryAnalysis:
        try:
            report = self._raw.classify_query(sql, target_database)
        except Exception as exc:
            raise EngineError(f"query analysis failed: {exc}") from exc
        return QueryAnalysis(
            query_class=getattr(report["query_class"], "name", str(report["query_class"])),
            statement_count=int(report["statement_count"]),
            has_secrets=bool(report["has_secrets"]),
            writes_only_target_database=bool(report["writes_only_target_database"]),
            changes_database_lifecycle=bool(report["changes_database_lifecycle"]),
        )

    def backup(self, database: str, file_path: str) -> None:
        """Full backup of `database` into `file_path`.

        `base_file_path` is deliberately never passed: an incremental archive
        records the base's *local path*, which does not survive the trip
        through object storage, so V1 publishes full checkpoints only (§3.2).
        """
        if not os.path.isabs(file_path):
            raise EngineError(f"backup path must be absolute: {file_path!r}")
        try:
            self._raw.backup_database(database, file_path)
        except Exception as exc:
            raise EngineError(f"backup of database {database!r} failed: {exc}") from exc

    def restore(self, database: str, file_path: str) -> None:
        if not os.path.isabs(file_path):
            raise EngineError(f"restore path must be absolute: {file_path!r}")
        try:
            self._raw.restore_database(database, file_path)
        except Exception as exc:
            raise EngineError(f"restore of database {database!r} failed: {exc}") from exc

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        global _LIVE_CONNECTIONS
        conn, self._conn, self._raw = getattr(self, "_conn", None), None, None
        if getattr(self, "_counted", False):
            _LIVE_CONNECTIONS -= 1
            self._counted = False
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def require_engine_compatible(
    *, engine_name: str, backup_format: int, min_reader: str, running: Optional[str] = None
) -> None:
    """The §4.3 open gate: format generation first, then reader version.

    Note what is *not* compared: whether `head.engine.version` equals the
    running version. An object written by 26.7.2-rc.2 stays open to every
    later core that supports the same archive generation — that is the whole
    point of carrying `backup_format` and `min_reader` separately.
    """
    from .protocol import BACKUP_FORMAT, ENGINE_NAME, version_lt

    if engine_name != ENGINE_NAME:
        raise EngineIncompatible(
            f"object was written by engine {engine_name!r}, not {ENGINE_NAME!r}")
    if backup_format > BACKUP_FORMAT:
        raise EngineIncompatible(
            f"object uses backup format {backup_format}; this engine restores up to {BACKUP_FORMAT}")
    running = engine_version() if running is None else running
    older = version_lt(running, min_reader)
    if older is None:
        raise EngineIncompatible(
            f"cannot order running engine {running!r} against the object's min_reader "
            f"{min_reader!r}; refusing rather than assuming compatibility")
    if older:
        raise EngineIncompatible(
            f"object needs chDB {min_reader} or newer to restore; this engine is {running}")
