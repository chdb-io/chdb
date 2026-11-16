---
name: chdb-arrow
description: Choose chDB Python Arrow input, result, insert, or standard Arrow database-driver paths when a task uses PyArrow tables or readers. Not for ordinary SQL or DataStore work.
---

# Python Arrow path selection

Arrow Database Connectivity (ADBC) is a separate standardized driver API. `Python(tbl)` is chDB's table function for a Python object already in the calling application; a `RecordBatchReader` is a one-shot source of batches.

- In-process SQL over an existing `pyarrow.Table`: `chdb.query("SELECT ... FROM Python(tbl)")`. For an incoming one-shot `pyarrow.RecordBatchReader`, use `Python(reader)`; do not promise a second scan. Neither requires constructing a separate session or using ADBC.
- Bulk insert into an existing table: `chdb.query("INSERT INTO dst SELECT ... FROM Python(tbl)", path=data_path)`; keep the source alive through the query. For several operations on one in-memory database, keep a session rather than opening one per call.
- Typed results: `chdb.query_arrow(sql)` returns a `pyarrow.Table`; `with chdb.query_arrow_batches(sql) as batches:` yields incremental `RecordBatch` values and closes a partial result.
- If the application already uses an ADBC framework for standard Arrow results or bulk ingest, `chdb[adbc]` is the appropriate separate driver route. Its `BindStream` on a general `SELECT` binds parameters, not a temporary table in `FROM`; prefer `Python(tbl)` / `Python(reader)` for a query over Python objects. Start with `docs/guides/querying-apache-arrow.mdx` for the Python Arrow documentation map and examples.
