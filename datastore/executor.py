"""
Query executor for DataStore.

Provides a high-level interface for query execution, delegating all chDB
operations to the centralized Connection class.

This is a facade that:
- Manages connection lifecycle (explicit or per-thread ':memory:' connections)
- Provides convenience methods for common operations
- Ensures connection is established before queries
"""

from collections import deque
from contextlib import contextmanager
import itertools
import threading
import weakref
from typing import Dict, Iterator, Optional
import pandas as pd

from .connection import Connection, QueryResult
from .exceptions import ExecutionError
from .config import get_logger


_slot_id_counter = itertools.count(1)


class _ThreadConnectionSlot:
    """Thread-local wrapper holding one ':memory:' Connection for a worker thread."""

    def __init__(self, slot_id: int, conn: Connection):
        self.slot_id = slot_id
        self.conn = conn
        self.retired = False


class Executor:
    """
    High-level query executor for DataStore.

    When initialized without an explicit Connection (such as the global executor),
    maintains a dedicated ':memory:' Connection per thread via threading.local()
    backed by chDB's shared EmbeddedServer. This prevents concurrent threads from
    cancelling each other's active send_query('DataFrame') streams or aliasing
    Python(__df__) bindings on a single ChdbClient, while tracking every created
    connection so close() and reset_executor() release all handles deterministically.
    """

    def __init__(self, connection: Optional[Connection] = None):
        """
        Initialize executor.

        Args:
            connection: chdb Connection object (creates per-thread ':memory:' connections if None)
        """
        self._owns_connection = connection is None
        self._explicit_connection: Optional[Connection] = connection
        self._local = threading.local()
        self._lock = threading.Lock()
        self._thread_slots: Dict[int, _ThreadConnectionSlot] = {}
        self._retired_slot_ids: "deque[int]" = deque()
        self._closed = False
        self._retired_global = False
        self._logger = get_logger()

    def _drain_retired_slots(self) -> None:
        """Close connections belonging to threads that have already exited."""
        if not self._retired_slot_ids:
            return
        to_close = []
        with self._lock:
            while self._retired_slot_ids:
                try:
                    slot_id = self._retired_slot_ids.popleft()
                except IndexError:
                    break
                slot = self._thread_slots.pop(slot_id, None)
                if slot is not None:
                    to_close.append(slot)
        for slot in to_close:
            with slot.conn._lock:
                slot.retired = True
                slot.conn.close()

    def _get_or_create_slot(self) -> _ThreadConnectionSlot:
        """Return the current thread's connection slot, creating it if needed."""
        slot: Optional[_ThreadConnectionSlot] = getattr(self._local, "slot", None)
        if slot is not None and not slot.retired and not self._closed:
            return slot

        self._drain_retired_slots()

        with self._lock:
            if self._closed:
                raise ExecutionError("No connection available")
            slot = getattr(self._local, "slot", None)
            if slot is not None and not slot.retired:
                return slot
            slot_id = next(_slot_id_counter)
            conn = Connection(":memory:")
            slot = _ThreadConnectionSlot(slot_id, conn)
            self._thread_slots[slot_id] = slot
            self._local.slot = slot
            # Queue slot_id lock-free when the thread exits and its local dict is
            # collected; actual conn.close() runs outside GC on normal boundaries.
            weakref.finalize(slot, self._retired_slot_ids.append, slot_id)
            return slot

    @property
    def connection(self) -> Optional[Connection]:
        """Return the explicit Connection or the calling thread's ':memory:' Connection."""
        if not self._owns_connection:
            return self._explicit_connection
        if self._retired_global:
            return get_executor().connection
        if self._closed:
            return None
        return self._get_or_create_slot().conn

    @connection.setter
    def connection(self, value: Optional[Connection]) -> None:
        """Allow setting an explicit connection reference."""
        self._explicit_connection = value
        if value is not None:
            self._owns_connection = False

    def _ensure_connected(self):
        """Ensure the calling thread's connection is established."""
        if self._owns_connection and self._retired_global:
            get_executor()._ensure_connected()
            return
        conn = self.connection
        if conn is None:
            raise ExecutionError("No connection available")
        with conn._lock:
            if conn._conn is None:
                conn.connect()

    @contextmanager
    def _acquire_connection(self) -> Iterator[Connection]:
        """Acquire a connected Connection locked for the duration of an operation."""
        if not self._owns_connection:
            conn = self._explicit_connection
            if conn is None:
                raise ExecutionError("No connection available")
            with conn._lock:
                if conn._conn is None:
                    conn.connect()
                yield conn
            return

        while True:
            if self._retired_global:
                with get_executor()._acquire_connection() as conn:
                    yield conn
                return
            slot = self._get_or_create_slot()
            with slot.conn._lock:
                if slot.retired:
                    # Closed/retired concurrently before we acquired slot.conn._lock.
                    continue
                if slot.conn._conn is None:
                    slot.conn.connect()
                yield slot.conn
                return

    def execute(self, sql: str) -> QueryResult:
        """
        Execute a SQL query via Connection.

        Args:
            sql: SQL query string

        Returns:
            QueryResult with data and metadata
        """
        with self._acquire_connection() as conn:
            return conn.execute(sql)

    def query_dataframe(
        self,
        sql: str,
        df: pd.DataFrame,
        df_name: str = "__df__",
        preserve_order: bool = True,
    ) -> pd.DataFrame:
        """
        Execute a SQL query on a DataFrame using chDB's Python() table function.

        Args:
            sql: SQL query string. Reference DataFrame using Python(df_name) or df_name.
            df: The DataFrame to query
            df_name: Name for the DataFrame in the query (default: '__df__')
            preserve_order: If True, preserve original row order (default: True)

        Returns:
            Result DataFrame
        """
        with self._acquire_connection() as conn:
            return conn.query_df(sql, df, df_name, preserve_order=preserve_order)

    def query_arrow(self, sql: str):
        """
        Execute a SQL query via Connection and return a ``pyarrow.Table``.

        Asks chDB for its native ``ArrowTable`` output, so the result keeps
        ClickHouse's Arrow types with no pandas round-trip.

        Args:
            sql: SQL query string

        Returns:
            pyarrow.Table
        """
        with self._acquire_connection() as conn:
            return conn.query_arrow(sql)

    def execute_expression(
        self, expr_sql: str, df: pd.DataFrame, result_column: str = "__result__"
    ) -> pd.Series:
        """
        Evaluate a SQL expression on a DataFrame and return the result as a Series.

        Args:
            expr_sql: SQL expression to evaluate (e.g., "CAST(value AS Float64)")
            df: DataFrame to operate on
            result_column: Name for the result column (default: '__result__')

        Returns:
            Result Series with the original DataFrame's index
        """
        with self._acquire_connection() as conn:
            return conn.eval_expression(expr_sql, df, result_column)

    def close(self):
        """Close all owned connections across threads."""
        if not self._owns_connection:
            return
        with self._lock:
            self._closed = True
            slots = list(self._thread_slots.values())
            self._thread_slots.clear()
            self._retired_slot_ids.clear()
        if hasattr(self._local, "slot"):
            try:
                delattr(self._local, "slot")
            except AttributeError:
                pass
        for slot in slots:
            with slot.conn._lock:
                slot.retired = True
                slot.conn.close()

    def __del__(self):
        """No-op: avoid running C++ connection teardown inside cyclic GC."""
        pass


# Global executor instance for convenience
# Protected by _executor_lock so concurrent threads safely share one Executor
# (which internally isolates per-thread ':memory:' connections).
_global_executor: Optional[Executor] = None
_executor_lock = threading.Lock()


def get_executor() -> Executor:
    """
    Get the global Executor instance.

    Creates one if it doesn't exist.

    Returns:
        Global Executor instance
    """
    global _global_executor
    executor = _global_executor
    if executor is not None:
        return executor
    with _executor_lock:
        if _global_executor is None:
            _global_executor = Executor()
        return _global_executor


def reset_executor():
    """Reset the global executor and deterministically close all thread connections."""
    global _global_executor
    with _executor_lock:
        executor = _global_executor
        _global_executor = None
        if executor is not None:
            executor._retired_global = True
    if executor is not None:
        executor.close()
