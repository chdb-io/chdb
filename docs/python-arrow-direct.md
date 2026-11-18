# Direct PyArrow queries in chDB

The `chdb.arrow` module provides a native Arrow C Data result path. It does not
serialize an Arrow IPC file or stream between chDB and PyArrow. Conversion
between ClickHouse columns and Arrow buffers may still copy data. Existing
`chdb.query(sql, "ArrowTable")` and `FORMAT Arrow` retain their IPC behavior.

This API requires `chdb-core>=26.11.1`, whose Python extension exposes
`query_arrow_reader`, `send_query_arrow`, and `streaming_fetch_arrow`. The
package dependency enforces that minimum; calling a direct API against an
incompatible manually installed core raises a clear error.

The published `chdb-core` 26.11.1 macOS arm64 wheel was validated with Python
3.9.25 and PyArrow 21.0.0: four native Arrow tests and twelve wrapper tests
passed, including in-memory input, direct output, early cancellation, and
insertion. Other Python, PyArrow, and platform combinations rely on their
own CI and should not be inferred from this single local check.

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
            print(batch.num_rows)
            break
```

`query_arrow` materializes the whole result; `query_arrow_batches` fetches
incrementally and supports early close/cancellation. A session accepts only
one active batch stream at a time. Do not concurrently use the same connection
from multiple threads. While a module-level batch iterator is active, other
module-level queries fail immediately instead of waiting for that iterator to
close; prefer a `with` block. Returned batches can outlive the iterator after
their data has been imported into PyArrow.

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
