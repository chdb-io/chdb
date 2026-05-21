# AGENTS.md — chDB

This file captures the **design and testing principles** of the
chdb-ds (DataStore) layer for AI coding agents working on this
repository and follows the [agents.md](https://agents.md) open
standard.

For setup, build commands, the modify-then-test workflow, PR
conventions, CI matrix, security, and other contributor mechanics,
see [`CONTRIBUTING.md`](./CONTRIBUTING.md).

---

## 1. Fully lazy execution architecture

- All methods returning DataFrame or Series should return Lazy
  objects (such as `DataStore`, `LazySeries`).
- Defer execution until results are actually needed.
- Preserve the ability to select the optimal execution engine
  (pandas vs chDB / SQL) at execution stage.
- **API style does not determine execution engine** — pandas-style,
  Pythonic, SQL terminology should all compile to the same optimised
  backend execution.
- Final execute stage selects pandas or chDB ExecutionEngine based
  on the config system.

## 2. Natural execution triggering (explicit calls prohibited)

- **Prohibit explicit calls to `_execute()`.**
- **Avoid explicit conversions** like `to_df()`, `to_list()`,
  `to_pandas()` as much as possible.
- Execution is triggered through natural means:
  - `.columns` — get column names
  - `len()` — get length
  - `.index` — get index
  - `repr()` / `print()` — display results
  - `__iter__` — iteration
  - `.equals()` — comparison

## 3. Unified architecture, simplicity first

- **Do not consider backward compatibility** — first priority is
  architectural simplicity and elegance.
- Don't create split class hierarchies for different execution
  engines.
- `ColumnExpr` uniformly wraps all expression types.
- Handle lazy execution through unified `LazySeries`, `LazyGroupBy`,
  etc.
- `LazyOp` uniformly manages all lazy operations.
- Avoid duplicate definitions; keep code structure clear with single
  responsibility.

## 4. Testing principles

**Philosophy:**

- Discovered problems are opportunities to improve the library.
- **Analyse problems from an architectural perspective** — don't
  easily modify tests just to pass them.
- Using `reset_index()` in tests to mask problems = **DataStore
  bug**, not correct test writing.
- Don't obsess over container type differences between DataFrame
  and DataStore.

**FORBIDDEN behaviours:**

- ❌ Using comments to describe expected behaviour without actual
  assertions
- ❌ Using `print()` / `logging` without corresponding assertions
- ❌ Only verifying `len()` without verifying actual values
- ❌ Writing "verify X" comments but not actually verifying X
- ❌ Using `# TODO: verify later` or similar postponement

**REQUIRED:**

1. **Mirror code pattern (DataStore ↔ pandas)** — test code must
   mirror DataStore and pandas operations for easy comparison.

   ```python
   # pandas operations
   pd_df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
   pd_result = pd_df[pd_df['age'] > 20].sort_values('name')

   # DataStore operations (mirror of pandas)
   ds_df = DataStore({'name': ['Alice', 'Bob'], 'age': [25, 30]})
   ds_result = ds_df[ds_df['age'] > 20].sort_values('name')

   assert_datastore_equals_pandas(ds_result, pd_result)
   ```

2. **Complete output comparison (columns + data + order)** —
   comparison must be complete: column names, data values, row
   order (if the pandas operation preserves or defines order).

   ```python
   from tests.test_utils import assert_datastore_equals_pandas

   # Full comparison (columns + values + order)
   assert_datastore_equals_pandas(ds_result, pd_result)

   # For unordered results (e.g. groupby without sort)
   assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

   # Column-level helpers
   # assert_column_values_equal(ds_result, pd_result, 'col_name')
   # assert_columns_match(ds_result, pd_result)
   # assert_row_count_match(ds_result, pd_result)
   ```

3. **SQL execution verification via logs:**

   ```python
   log_output = log_capture.getvalue()
   self.assertIn('WHERE "value" > 30', log_output)  # exact SQL clause
   self.assertIn('Segment 1/', log_output)           # segment execution
   self.assertIn('[Pandas]', log_output)             # Pandas ops logged
   ```

4. **Test name must reflect what is being verified.**

   ```python
   # BAD
   def test_complex_pipeline(self):
       ...
   # GOOD
   def test_sql_pandas_sql_segments_exact_values_and_structure(self):
       ...
   ```

**Self-check before submitting a test:**

- [ ] Does every "expected behaviour" comment have a corresponding
      assertion?
- [ ] Are actual data values verified, not just lengths?
- [ ] Are DataStore and pandas code mirrored (same operations, same
      style)?
- [ ] Is column order verified (not just column names as a set)?
- [ ] Is row order verified for order-preserving operations?
- [ ] Is segment structure (type, ops, `is_first_segment`) fully
      verified?
- [ ] Are error messages descriptive for debugging?
- [ ] Would this test catch a real bug, or just pass trivially?

---

**Core philosophy: users write familiar pandas-style code; the
backend automatically selects the optimal execution engine.**

---

*Standard followed: [agents.md](https://agents.md).*
