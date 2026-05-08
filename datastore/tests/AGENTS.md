# AGENTS.md — datastore/tests/

This directory holds the chdb-ds test suite — 267 test files. The
*design philosophy* for chdb-ds is in [`../../CLAUDE.md`](../../CLAUDE.md);
the *contributor process* is in [`../../AGENTS.md`](../../AGENTS.md). This
file covers test-execution mechanics specific to working *inside* this
directory.

## Run targeted, not full

A targeted run is seconds. The full run is tens of minutes and
exercises the chdb-core engine extensively. Don't run the full suite
as a sanity check.

```bash
# from the repo root, not from inside tests/
cd datastore && python -m pytest tests/test_<file>.py -v --tb=short
cd datastore && python -m pytest tests/test_<file>.py::TestClass::test_method -v
cd datastore && python -m pytest tests/ -k "groupby and not slow" -v --tb=short
```

The full suite is for pre-push verification or CI:

```bash
cd datastore && python -m pytest tests/ -v --tb=short -x   # -x mirrors CI fail-fast
```

## What lives where in this directory

- `conftest.py` — pytest configuration and shared fixtures
- `test_utils.py` — assertion helpers (see below — use these)
- `dataset/` — small reference datasets used by tests
- `__init__.py` — empty; makes `tests/` a package
- `setup_clickhouse_server.sh` / `stop_clickhouse_server.sh` —
  helpers for tests that need a real ClickHouse server

## Use the existing assertion helpers

`test_utils.py` provides the canonical comparison helpers. Don't
hand-roll equivalents:

```python
from tests.test_utils import (
    assert_datastore_equals_pandas,   # columns + values + row order
    assert_column_values_equal,
    assert_columns_match,
    assert_row_count_match,
)
```

For unordered results (e.g. `groupby` without `sort=True`):

```python
assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
```

## Mirror pandas and DataStore in every test

Hard rule: every test writes the pandas form *and* the DataStore form
of the same operation, then compares.

```python
# pandas oracle
pd_df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
pd_result = pd_df[pd_df['age'] > 20].sort_values('name')

# DataStore mirror — syntactically as similar as possible
ds_df = DataStore({'name': ['Alice', 'Bob'], 'age': [25, 30]})
ds_result = ds_df[ds_df['age'] > 20].sort_values('name')

assert_datastore_equals_pandas(ds_result, pd_result)
```

Don't write a DataStore test that only checks DataStore output
without a pandas oracle. Without a pandas reference, regressions go
undetected.

## Forbidden patterns

These are real anti-patterns the project has fought, not generic
advice:

- ❌ `print()` or `logging` without a corresponding assertion
- ❌ Asserting only `len(result)` and skipping the actual values
- ❌ "Verify X" / "TODO: verify later" comments — verify, or delete
- ❌ `reset_index()` to make a test pass — that masks a DataStore
  index bug; file the bug instead
- ❌ Marking a crashing test `xfail` / `skip` to make CI green — see
  [`../../AGENTS.md`](../../AGENTS.md) §2.6 for the crash-diagnosis flow

## Test naming

Names should reflect *what is being verified*, not what is being
exercised:

```python
# bad
def test_complex_pipeline(self): ...

# good
def test_sql_pandas_sql_segments_exact_values_and_structure(self): ...
```

## SQL execution verification (when relevant)

For tests that care about which SQL the DataStore actually emits,
capture logs and assert specific clauses:

```python
log_output = log_capture.getvalue()
self.assertIn('WHERE "value" > 30', log_output)
self.assertIn('Segment 1/', log_output)
self.assertIn('[Pandas]', log_output)
```

## When a test crashes

Always capture the trace before changing anything. The diagnosis
flow is in [`../../AGENTS.md`](../../AGENTS.md) §2.6. Don't paper
over a `SIGSEGV` with `xfail` — it usually points at a real
chdb-core engine bug.
