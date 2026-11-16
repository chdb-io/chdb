# Direct PyArrow queries in chDB (experimental)

The `chdb.arrow` module provides a native Arrow C Data result path. It does not
serialize an Arrow IPC file or stream between chDB and PyArrow. Conversion
between ClickHouse columns and Arrow buffers may still copy data. Existing
`chdb.query(sql, "ArrowTable")` and `FORMAT Arrow` retain their IPC behavior.

This API needs a `chdb-core` build with the `query_arrow_reader`,
`send_query_arrow`, and `streaming_fetch_arrow` Python extension methods. Older
builds raise a clear error; the current package minimum requirement is not a
claim that every compatible wheel already has these methods.

Validation status (2026-11-10): the #323-repaired C `libchdb.so` passes the
registration example's 155 checks and 11 C Data→PyArrow tests. That build
used `ENABLE_PYTHON=0`. A separate Python-enabled build of this branch now
provides `_chdb.abi3.so`; with Python 3.9.25 and PyArrow 21.0.0, all four
core direct-output tests and the main wrapper's public end-to-end test pass.
The latter covers `register_arrow`, direct whole-result/batch output,
deferred source cleanup, insert, and module-level helpers. This does not
validate other Python/Arrow versions, release wheels, or the build script's
later attempts to generate every Python shim (it stopped at missing pyenv
Python 3.10 after producing the working Python 3.9 extension).

```python
import chdb
import pyarrow as pa

tbl = pa.table({"id": [1, 2], "name": ["é", "你好"]})

# `Python(tbl)` scans the in-memory Arrow table. The result is produced
# through the native C Data path, then collected into a PyArrow Table.
result = chdb.query_arrow(
    "SELECT id, name FROM Python(tbl) ORDER BY id",
    options={"string_as_string": True},
)
print(result)

# For large results, keep one session and close the batch stream even when
# exiting early. Each value yielded is a pyarrow.RecordBatch.
with chdb.arrow.connect() as session:
    with session.register_arrow(tbl) as source:
        summary = session.query_arrow(f"SELECT count() AS n FROM {source.sql_expr}")
        print(summary)
    with session.query_arrow_batches("SELECT number FROM numbers(1000000)") as batches:
        for batch in batches:
            consume(batch)
            break
```

`query_arrow` materializes the whole result; `query_arrow_batches` fetches
incrementally and supports early close/cancellation. A session accepts only
one active batch stream at a time. Do not concurrently use the same connection
from multiple threads. The module-level batch iterator retains the usual
single-active-path lock until it is exhausted or explicitly closed; prefer a
`with` block. Returned batches can outlive the iterator after their data has
been imported into PyArrow.

`register_arrow` returns a closable, opaque SQL source expression. It uses the
existing Python `Python(...)` table function and PyArrow C Stream conversion,
not the separate core `chdb_arrow_scan` registry. The table or reader stays
alive for the query; closing a source during an active batch stream defers
cleanup until that stream closes. A one-shot reader cannot be scanned twice.
Use the same session's `query` method when requesting a non-Arrow output from
a registered source; the Python table function resolves its source in that
session method's Python frame.

`ArrowSession.insert_arrow` accepts a `pyarrow.Table`, `RecordBatch`, or a
producer implementing `__arrow_c_stream__` and exposing `schema`. It performs
an `INSERT ... SELECT ... FROM Python(_source)` into an **existing** table,
without IPC or a temporary file. Readers are one-shot; do not reuse them after
an insert or expect a second scan to replay data. Column names and destination
identifiers are quoted; the destination may be `(database, table)`.

```python
with chdb.arrow.connect() as session:
    session.execute("CREATE TABLE dst (id Int64, name String) ENGINE=Memory")
    session.insert_arrow("dst", tbl)
```

Default C Data output options match the core ABI:
`unsupported_as_binary=False`, `low_cardinality_as_dictionary=False`,
`string_as_string=True`. `DateTime` exports as Arrow `uint32` Unix seconds;
use an explicit SQL `DateTime64` conversion for a timezone-tagged timestamp.
