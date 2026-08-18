"""Run a compiled SQL segment on the server that already holds the data.

A DataStore built on a remote ClickHouse table normally reads that table through
the ``remote()`` table function: chDB opens the connection, pulls rows to the
caller and computes locally.  Binding a segment executor sends the compiled SQL
to that server instead, so only the result travels back.

Two rules keep the behaviour predictable:

* The data source is rendered for the chosen execution target while the SQL is
  compiled.  SQL that will run on the server itself reads ``database.table``
  directly and never contains ``remote()``.  Rewriting a finished ``remote()``
  query by string substitution would break joins, subqueries and quoting.
* A failed remote query is raised to the caller.  Re-running it on the local
  engine would hide authentication and dialect errors, and can pay for the same
  scan twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from .exceptions import DataStoreError

# Execution targets a compiled segment can be rendered for.
LOCAL_CHDB = "local_chdb"
REMOTE_CLICKHOUSE = "remote_clickhouse"


class SegmentExecutorError(DataStoreError):
    """A bound segment executor violated its contract."""


@dataclass(frozen=True)
class RemoteSource:
    """Where a DataStore's rows live.  Deliberately carries no credentials."""

    host: str
    database: str
    table: str
    secure: bool = False

    def qualified_name(self) -> str:
        """``database.table``, or just the table when no database is bound."""
        return f"{self.database}.{self.table}" if self.database else self.table


@dataclass(frozen=True)
class SegmentResult:
    """Rows returned by an executor, plus whatever metrics it can report.

    ``metrics`` is opaque here: the executor owns query IDs, scanned rows and
    server timings, and passes them through for tracing and UI display.
    """

    frame: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PushdownTrace:
    """Evidence that one segment ran somewhere other than the local engine."""

    target: str
    sql: str
    source: RemoteSource
    result_rows: int
    metrics: Mapping[str, Any] = field(default_factory=dict)


class SqlSegmentExecutor(ABC):
    """Executes one already-compiled SQL segment against a remote server.

    Implementations own connections, credentials, execution limits and query
    IDs.  DataStore only guarantees that the SQL it hands over is valid for
    ``target`` - for ``REMOTE_CLICKHOUSE`` that means the source is written as a
    plain table reference the server can resolve itself.
    """

    target = REMOTE_CLICKHOUSE

    @abstractmethod
    def accepts(self, source: RemoteSource) -> bool:
        """Whether this executor is connected to the server holding ``source``."""

    @abstractmethod
    def execute(self, sql: str, source: RemoteSource) -> SegmentResult:
        """Run ``sql`` and return its rows.  Raise on failure; never fall back."""


def normalize_segment_result(value: Any) -> SegmentResult:
    """Accept either a ``SegmentResult`` or a bare DataFrame from an executor."""
    import pandas as pd

    if isinstance(value, SegmentResult):
        if not isinstance(value.frame, pd.DataFrame):
            raise SegmentExecutorError(
                "segment executor returned SegmentResult without a DataFrame: "
                f"{type(value.frame).__name__}"
            )
        return value
    if isinstance(value, pd.DataFrame):
        return SegmentResult(frame=value)
    raise SegmentExecutorError(
        "segment executor must return a SegmentResult or a pandas DataFrame, "
        f"got {type(value).__name__}"
    )
