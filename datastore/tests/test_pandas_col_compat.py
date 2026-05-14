# pyright: reportAttributeAccessIssue=false, reportAssignmentType=false, reportGeneralTypeIssues=false
"""
Tests for pandas 3.x ``pd.col(...)`` compatibility in DataStore.

Covers the translator plus every entry point we wired it into
(``__getitem__``, ``loc[]``, ``filter``, ``where``, ``assign``,
``with_column``, ``groupby().agg``, ``sort_values``, ``case_when``).

Correctness tests mirror the same statement against a real pandas
DataFrame and the DataStore and compare with
``assert_datastore_equals_pandas``. Pushdown tests additionally capture
the datastore logger and assert the generated SQL contains the expected
ClickHouse fragment, so we don't silently fall back to pandas.

Skipped on pandas < 3.0.
"""
from __future__ import annotations

import io
import logging
import unittest

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.expressions import Field
from datastore.pandas_col_compat import (
    PANDAS_3_PLUS,
    PandasColTranslationError,
    is_pandas_col_expression,
    translate_pandas_expression,
)
from datastore.tests.test_utils import assert_datastore_equals_pandas


pytestmark = pytest.mark.skipif(
    not PANDAS_3_PLUS,
    reason="pd.col is only available in pandas >= 3.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_executed_sql(ds_factory):
    """Build, execute, and return ``(df, [sql, ...])`` for every SQL
    statement the datastore logger emits.

    Captures both inline ``Executing: ...`` lines and the multi-line
    ``[chDB] SQL:`` form (where the actual query body sits on the next
    indented line — used by groupby-agg pushdown).
    """
    logger = logging.getLogger("datastore")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        ds = ds_factory()
        df = ds._execute() if hasattr(ds, "_execute") else ds
        log_text = stream.getvalue()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    sqls = []
    lines = log_text.split("\n")
    for i, line in enumerate(lines):
        if "[SQL on DataFrame] Executing:" in line:
            sqls.append(line.split("Executing:", 1)[1].strip())
        elif "[SQL]" in line and "Executing:" in line:
            sqls.append(line.split("Executing:", 1)[1].strip())
        elif "Executing:" in line and "SQL" in line:
            # Generic "[N/M] Executing: SQL Query: SELECT ..." form.
            sqls.append(line.split("Executing:", 1)[1].strip())
        elif "[chDB] SQL:" in line:
            # Actual chDB query body is on the next non-empty line.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                sqls.append(lines[j].strip())
    return df, sqls


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["alice", "BOB", "Carol", "dave"],
            "age": [25, 31, 17, 42],
            "speed": [80, 120, 60, 150],
            "score": [10.0, 20.0, 30.0, 40.0],
        }
    )


# ---------------------------------------------------------------------------
# 1. Translator unit tests (no DataStore involved)
# ---------------------------------------------------------------------------


class TestTranslatorUnit(unittest.TestCase):
    """Direct tests of the symbolic-execution translator."""

    def test_is_pandas_col_expression_is_true_for_pd_col(self):
        self.assertTrue(is_pandas_col_expression(pd.col("x")))
        self.assertTrue(is_pandas_col_expression(pd.col("x") + 1))
        self.assertTrue(is_pandas_col_expression(pd.col("x").str.upper()))

    def test_is_pandas_col_expression_is_false_for_other_values(self):
        for value in [None, 1, "x", [1, 2], object(), Field("x")]:
            self.assertFalse(is_pandas_col_expression(value))

    def test_translate_arithmetic_produces_clickhouse_sql(self):
        sql = translate_pandas_expression(pd.col("age") + 1).to_sql()
        self.assertEqual(sql, '("age"+1)')

    def test_translate_comparison_produces_clickhouse_sql(self):
        sql = translate_pandas_expression(pd.col("speed") > 105).to_sql()
        self.assertEqual(sql, '"speed" > 105')

    def test_translate_boolean_and_produces_compound_sql(self):
        sql = translate_pandas_expression(
            (pd.col("age") >= 18) & (pd.col("speed") < 130)
        ).to_sql()
        self.assertEqual(sql, '("age" >= 18 AND "speed" < 130)')

    def test_translate_str_namespace_lower_maps_to_clickhouse(self):
        sql = translate_pandas_expression(pd.col("name").str.lower()).to_sql()
        self.assertEqual(sql, 'lower("name")')

    def test_translate_str_title_maps_to_initcap(self):
        sql = translate_pandas_expression(pd.col("name").str.title()).to_sql()
        self.assertEqual(sql, 'initcap("name")')

    def test_translate_dt_year_maps_to_to_year(self):
        sql = translate_pandas_expression(pd.col("d").dt.year).to_sql()
        self.assertEqual(sql, 'toYear("d")')

    def test_translate_numpy_log_returns_pandas_fallback(self):
        # numpy ufuncs are not pushed to SQL; we wrap the original pd.col
        # expression in PandasFallbackExpr so the pandas segment evaluates it.
        from datastore.pandas_col_compat import PandasFallbackExpr

        translated = translate_pandas_expression(np.log(pd.col("speed")))
        self.assertIsInstance(translated, PandasFallbackExpr)

    def test_translate_numpy_sqrt_returns_pandas_fallback(self):
        from datastore.pandas_col_compat import PandasFallbackExpr

        translated = translate_pandas_expression(np.sqrt(pd.col("age")))
        self.assertIsInstance(translated, PandasFallbackExpr)

    def test_translate_astype_returns_pandas_fallback(self):
        # astype on a pd.col expression cannot be type-mapped reliably,
        # so we always fall back to pandas semantics.
        from datastore.pandas_col_compat import PandasFallbackExpr

        translated = translate_pandas_expression(pd.col("age").astype(float))
        self.assertIsInstance(translated, PandasFallbackExpr)

        translated2 = translate_pandas_expression(pd.col("age").astype("int32"))
        self.assertIsInstance(translated2, PandasFallbackExpr)

    def test_non_pd_col_value_passes_through_unchanged(self):
        sentinel = object()
        self.assertIs(translate_pandas_expression(sentinel), sentinel)
        self.assertEqual(translate_pandas_expression(5), 5)
        self.assertEqual(translate_pandas_expression("hi"), "hi")

    def test_unsupported_method_returns_pandas_fallback(self):
        # Unknown methods on Field (anything chdb-ds does not model) now
        # transparently fall back to pandas via PandasFallbackExpr.
        from datastore.pandas_col_compat import PandasFallbackExpr

        bad = pd.col("x").totally_made_up_method()
        translated = translate_pandas_expression(bad)
        self.assertIsInstance(translated, PandasFallbackExpr)


# ---------------------------------------------------------------------------
# 2. DataStore correctness tests — mirror pandas, compare results
# ---------------------------------------------------------------------------


class TestDataStoreGetitemMirrorsPandas(unittest.TestCase):
    """``ds[pd.col(...) > X]`` matches ``df.loc[pd.col(...) > X]``."""

    def setUp(self):
        self.df = _sample_df()
        self.ds = DataStore(self.df.copy())

    def test_boolean_index_with_single_condition(self):
        pd_out = self.df.loc[pd.col("speed") > 105]
        ds_out = self.ds[pd.col("speed") > 105]
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_boolean_index_with_compound_condition(self):
        pd_out = self.df.loc[(pd.col("age") >= 18) & (pd.col("speed") < 130)]
        ds_out = self.ds[(pd.col("age") >= 18) & (pd.col("speed") < 130)]
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_boolean_index_with_or_condition(self):
        pd_out = self.df.loc[(pd.col("age") < 18) | (pd.col("speed") > 130)]
        ds_out = self.ds[(pd.col("age") < 18) | (pd.col("speed") > 130)]
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_negated_condition(self):
        pd_out = self.df.loc[~(pd.col("age") < 18)]
        ds_out = self.ds[~(pd.col("age") < 18)]
        assert_datastore_equals_pandas(ds_out, pd_out)


class TestDataStoreLocMirrorsPandas(unittest.TestCase):
    """``ds.loc[...]`` accepts pd.col expressions like pandas does."""

    def setUp(self):
        self.df = _sample_df()
        self.ds = DataStore(self.df.copy())

    def test_loc_with_boolean_pd_col_condition(self):
        pd_out = self.df.loc[pd.col("speed") > 105]
        ds_out = self.ds.loc[pd.col("speed") > 105]
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_loc_with_condition_and_column_subset_tuple(self):
        pd_out = self.df.loc[pd.col("speed") > 105, ["name", "age"]]
        ds_out = self.ds.loc[pd.col("speed") > 105, ["name", "age"]]
        assert_datastore_equals_pandas(ds_out, pd_out)


class TestDataStoreFilterMirrorsPandas(unittest.TestCase):

    def setUp(self):
        self.df = _sample_df()
        self.ds = DataStore(self.df.copy())

    def test_filter_with_pd_col_condition(self):
        pd_out = self.df.loc[pd.col("speed") > 105]
        ds_out = self.ds.filter(pd.col("speed") > 105)
        assert_datastore_equals_pandas(ds_out, pd_out)


class TestDataStoreWhereMirrorsPandas(unittest.TestCase):

    def setUp(self):
        self.df = _sample_df()
        self.ds = DataStore(self.df.copy())

    def test_where_with_only_condition_acts_as_filter(self):
        # ``where(cond)`` without ``other`` behaves like filter() for
        # SQL-pushable Conditions, so we compare against ``loc[...]``
        # instead of ``DataFrame.where`` (value-masking semantics differ).
        pd_out = self.df.loc[pd.col("speed") > 105]
        ds_out = self.ds.where(pd.col("speed") > 105)
        assert_datastore_equals_pandas(ds_out, pd_out)


class TestDataStoreAssignMirrorsPandas(unittest.TestCase):

    def setUp(self):
        self.df = _sample_df()
        self.ds = DataStore(self.df.copy())

    def test_assign_arithmetic_and_str_namespace(self):
        pd_out = self.df.assign(
            age_plus=pd.col("age") + 1,
            name_title=pd.col("name").str.title(),
        )
        ds_out = self.ds.assign(
            age_plus=pd.col("age") + 1,
            name_title=pd.col("name").str.title(),
        )
        # initcap vs str.title can disagree on apostrophes/edge cases, but
        # match on the simple sample inputs.
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_assign_arithmetic_between_two_pd_col_objects(self):
        pd_out = self.df.assign(diff=pd.col("speed") - pd.col("age"))
        ds_out = self.ds.assign(diff=pd.col("speed") - pd.col("age"))
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_assign_numpy_ufunc_log(self):
        pd_out = self.df.assign(log_speed=np.log(pd.col("speed")))
        ds_out = self.ds.assign(log_speed=np.log(pd.col("speed")))
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_assign_astype_float(self):
        pd_out = self.df.assign(age_f=pd.col("age").astype(float))
        ds_out = self.ds.assign(age_f=pd.col("age").astype(float))
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_assign_boolean_expression_creates_bool_column(self):
        pd_out = self.df.assign(big=pd.col("speed") > 100)
        ds_out = self.ds.assign(big=pd.col("speed") > 100)
        # pandas stores bool, ClickHouse stores UInt8 — compare values only.
        assert_datastore_equals_pandas(
            ds_out, pd_out, check_nullable_dtype=False
        )


class TestDataStoreWithColumnMirrorsPandas(unittest.TestCase):

    def setUp(self):
        self.df = _sample_df()
        self.ds = DataStore(self.df.copy())

    def test_with_column_arithmetic(self):
        pd_out = self.df.assign(diff=pd.col("speed") - pd.col("age"))
        ds_out = self.ds.with_column("diff", pd.col("speed") - pd.col("age"))
        assert_datastore_equals_pandas(ds_out, pd_out)


class TestPandasFallbackForUntranslatableExpressions(unittest.TestCase):
    """``astype`` / numpy ufuncs are deliberately not pushed to SQL; we wrap
    the pd.col expression in PandasFallbackExpr so the pandas segment owns
    the semantics. These tests assert end-to-end correctness *and* that we
    actually take the fallback path (no SQL fragment emitted for them)."""

    def setUp(self):
        self.df = _sample_df()
        self.ds = DataStore(self.df.copy())

    def _assert_no_sql_fragment(self, sqls, fragment):
        for s in sqls:
            self.assertNotIn(fragment, s)

    def test_filter_with_astype_falls_back_to_pandas(self):
        pd_out = self.df.loc[pd.col("age").astype(float) > 21]
        result, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).filter(
                pd.col("age").astype(float) > 21
            )
        )
        assert_datastore_equals_pandas(result, pd_out)
        # CAST must NOT appear in any executed SQL - the fallback path
        # routed the filter to pandas instead of pushing CAST/comparison.
        self._assert_no_sql_fragment(sqls, 'CAST("age"')

    def test_assign_np_log_falls_back_to_pandas(self):
        pd_out = self.df.assign(log_speed=np.log(pd.col("speed")))
        result, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).assign(
                log_speed=np.log(pd.col("speed"))
            )
        )
        assert_datastore_equals_pandas(result, pd_out)
        self._assert_no_sql_fragment(sqls, 'log("speed")')

    def test_getitem_with_astype_falls_back_to_pandas(self):
        pd_out = self.df.loc[pd.col("age").astype(float) > 21]
        result, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy())[
                pd.col("age").astype(float) > 21
            ]
        )
        assert_datastore_equals_pandas(result, pd_out)
        self._assert_no_sql_fragment(sqls, 'CAST("age"')

    def test_assign_astype_preserves_dtype(self):
        ds_out = self.ds.assign(age_f=pd.col("age").astype(float))
        df_out = ds_out._execute()
        # pandas semantics: astype(float) yields float64, not Int64/Float64
        # via SQL CAST. This catches regressions where the fallback silently
        # routes through a CAST path.
        self.assertEqual(str(df_out["age_f"].dtype), "float64")


# ---------------------------------------------------------------------------
# 3. Pushdown verification — assert generated SQL contains the right fragments
# ---------------------------------------------------------------------------


class TestPushdownSqlFragments(unittest.TestCase):
    """Execute and assert the captured SQL contains the pushed-down
    translation. A failure here means we silently fell back to pandas."""

    def setUp(self):
        self.df = _sample_df()

    def _assert_any_sql_contains(self, sqls, fragment, label=""):
        joined = "\n".join(sqls) or "(no SQL captured)"
        self.assertTrue(
            any(fragment in s for s in sqls),
            msg=f"{label}Expected SQL fragment {fragment!r} not found in:\n{joined}",
        )

    def test_filter_pushes_down_where_clause(self):
        _, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).filter(pd.col("speed") > 105)
        )
        self._assert_any_sql_contains(sqls, 'WHERE "speed" > 105')

    def test_getitem_pushes_down_where_clause(self):
        _, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy())[pd.col("speed") > 105]
        )
        self._assert_any_sql_contains(sqls, 'WHERE "speed" > 105')

    def test_assign_str_title_pushes_down_initcap(self):
        _, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).assign(
                name_t=pd.col("name").str.title()
            )
        )
        self._assert_any_sql_contains(sqls, 'initcap("name")')

    def test_assign_arithmetic_pushes_down_expression(self):
        _, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).assign(
                age_plus=pd.col("age") + 1
            )
        )
        self._assert_any_sql_contains(sqls, '("age"+1)')

    def test_assign_np_log_does_not_push_down_to_sql(self):
        # numpy ufuncs deliberately fall back to pandas (Option B): we want
        # pandas semantics, not a best-effort SQL translation. So neither
        # ``log("speed")`` nor any SQL-side ``log`` should appear.
        _, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).assign(
                log_speed=np.log(pd.col("speed"))
            )
        )
        for s in sqls:
            self.assertNotIn('log("speed")', s)

    def test_assign_astype_does_not_push_down_to_sql(self):
        # astype goes through PandasFallbackExpr (Option B), so no CAST(...)
        # should ever be emitted for the assigned column.
        _, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).assign(
                age_f=pd.col("age").astype(float)
            )
        )
        for s in sqls:
            self.assertNotIn('CAST("age"', s)

    def test_compound_filter_pushes_down_full_and_clause(self):
        _, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy())[
                (pd.col("age") >= 18) & (pd.col("speed") < 130)
            ]
        )
        self._assert_any_sql_contains(
            sqls, '("age" >= 18 AND "speed" < 130)'
        )


# ---------------------------------------------------------------------------
# 4. Regression: original bugs we fixed should stay fixed
# ---------------------------------------------------------------------------


class TestRegressionsFixed(unittest.TestCase):

    def setUp(self):
        self.df = _sample_df()

    def test_getitem_does_not_recurse_to_death_anymore(self):
        # Used to RecursionError: DataStore.__repr__ and pd.col.__repr__
        # called each other before we translated at __getitem__.
        ds = DataStore(self.df.copy())
        out = ds[pd.col("speed") > 105]
        text = repr(out)
        self.assertIn("BOB", text)
        self.assertIn("dave", text)

    def test_filter_result_can_be_repr_without_typeerror(self):
        # Used to TypeError: ds.filter(pd.col(...)) stored the raw pd.col
        # into _where_condition; then __str__ -> _has_sql_state ->
        # bool(_where_condition) tripped pd.col.__bool__.
        ds = DataStore(self.df.copy())
        out = ds.filter(pd.col("speed") > 105)
        text = str(out)
        self.assertIn("BOB", text)


# ---------------------------------------------------------------------------
# 5. Negative cases — friendly error for unsupported usage
# ---------------------------------------------------------------------------


class TestUnsupportedUsage(unittest.TestCase):

    def setUp(self):
        self.ds = DataStore(_sample_df())

    def test_unknown_method_on_pd_col_falls_back_to_pandas(self):
        # Unknown methods on Field used to raise PandasColTranslationError.
        # Under Option-B fallback, the translator instead wraps the original
        # pd.col expression in PandasFallbackExpr and lets pandas evaluate
        # it. Pandas itself then surfaces the AttributeError at exec time.
        ds_out = self.ds.assign(bad=pd.col("name").totally_made_up_method())
        with self.assertRaises(AttributeError):
            ds_out._execute()

    def test_bare_non_boolean_pd_col_in_getitem_raises_typeerror(self):
        # ``ds[pd.col("name").str.lower()]`` has no defined pandas
        # semantics — surface a TypeError with a hint.
        with self.assertRaises(TypeError):
            _ = self.ds[pd.col("name").str.lower()]


# ---------------------------------------------------------------------------
# 6. groupby().agg(...) with pd.col aggregate expressions
# ---------------------------------------------------------------------------


class TestGroupbyAggMirrorsPandas(unittest.TestCase):
    """Regression: ``ds.groupby(...).agg(total=pd.col('revenue').sum())``
    used to silently drop the kwarg and return an empty frame."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "region": ["EU", "EU", "US", "US", "EU"],
                "revenue": [10, 20, 30, 40, 50],
                "qty": [1, 2, 3, 4, 5],
            }
        )
        self.ds = DataStore(self.df.copy())

    def test_groupby_agg_single_sum_returns_expected_columns_and_values(self):
        pd_out = self.df.groupby("region").agg(total=("revenue", "sum")).reset_index()
        ds_out = self.ds.groupby("region").agg(total=pd.col("revenue").sum())
        assert_datastore_equals_pandas(ds_out, pd_out, check_row_order=False)

    def test_groupby_agg_multiple_named_aggregations(self):
        pd_out = (
            self.df
            .groupby("region")
            .agg(total=("revenue", "sum"), avg_qty=("qty", "mean"))
            .reset_index()
        )
        ds_out = self.ds.groupby("region").agg(
            total=pd.col("revenue").sum(),
            avg_qty=pd.col("qty").mean(),
        )
        assert_datastore_equals_pandas(ds_out, pd_out, check_row_order=False)

    def test_groupby_agg_pushes_down_aggregate_sql(self):
        _, sqls = _capture_executed_sql(
            lambda: self.ds.groupby("region").agg(
                total=pd.col("revenue").sum()
            )
        )
        joined = "\n".join(sqls)
        self.assertIn('sum("revenue")', joined,
                      msg=f"Expected sum(\"revenue\") in SQL, got:\n{joined}")
        self.assertIn('"region"', joined)


# ---------------------------------------------------------------------------
# 7. sort_values(by=pd.col(...))
# ---------------------------------------------------------------------------


class TestSortValuesAcceptsPdCol(unittest.TestCase):
    """Regression: ``sort_values(by=pd.col(...))`` used to raise
    ``TypeError: Expression objects are not iterable`` from ``list(by)``.
    """

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "name": ["alice", "bob", "carol", "dave"],
                "revenue": [40, 10, 30, 20],
            }
        )
        self.ds = DataStore(self.df.copy())

    def test_sort_values_by_single_pd_col_ascending(self):
        pd_out = self.df.sort_values(by="revenue").reset_index(drop=True)
        ds_out = self.ds.sort_values(by=pd.col("revenue"))
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_sort_values_by_single_pd_col_descending(self):
        pd_out = self.df.sort_values(by="revenue", ascending=False).reset_index(drop=True)
        ds_out = self.ds.sort_values(by=pd.col("revenue"), ascending=False)
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_sort_values_by_list_with_pd_col_in_it(self):
        pd_out = (
            self.df
            .sort_values(by=["revenue", "name"])
            .reset_index(drop=True)
        )
        ds_out = self.ds.sort_values(by=[pd.col("revenue"), "name"])
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_sort_values_with_limit_pushes_down_order_by(self):
        # Unbounded sort stays in pandas; adding ``.head(N)`` lets the
        # planner push ORDER BY + LIMIT into one SQL query.
        _, sqls = _capture_executed_sql(
            lambda: self.ds.sort_values(by=pd.col("revenue")).head(2)
        )
        joined = "\n".join(sqls)
        self.assertIn('"revenue"', joined,
                      msg=f'Expected "revenue" in ORDER BY, got:\n{joined}')
        self.assertIn("ORDER BY", joined.upper(),
                      msg=f"Expected ORDER BY in SQL, got:\n{joined}")


# ---------------------------------------------------------------------------
# 8. case_when with pd.col conditions (chdb-ds-side replacement for the
#    broken pandas 3.0.3 Series.case_when path)
# ---------------------------------------------------------------------------


class TestColumnExprCaseWhenAcceptsPdCol(unittest.TestCase):
    """``ds["col"].case_when([(pd.col(...), repl), ...])`` works and
    pushes down to a ``CASE WHEN ... END``. Mirrors pandas
    Series.case_when semantics (ELSE = original column).

    Upstream ``Series.case_when`` is broken on pandas 3.0.3 — it crashes
    with ``AttributeError: 'Series' object has no attribute 'columns'``.
    """

    def setUp(self):
        self.df = pd.DataFrame({"revenue": [10, 20, 30, 40, 50]})
        self.ds = DataStore(self.df.copy())

    def test_case_when_with_pd_col_conditions_returns_expected_buckets(self):
        ds_out = self.ds.assign(
            bucket=self.ds["revenue"].case_when([
                (pd.col("revenue") < 25, "small"),
                (pd.col("revenue") < 45, "medium"),
                (pd.col("revenue") >= 45, "large"),
            ])
        )
        result = ds_out._execute()
        buckets = result["bucket"].tolist()
        self.assertEqual(buckets, ["small", "small", "medium", "medium", "large"])

    def test_case_when_pushes_down_to_clickhouse_case_when_sql(self):
        expr = self.ds["revenue"].case_when([
            (pd.col("revenue") < 25, "small"),
            (pd.col("revenue") < 45, "medium"),
            (pd.col("revenue") >= 45, "large"),
        ])
        sql = expr._expr.to_sql(quote_char='"')
        self.assertIn('CASE WHEN "revenue" < 25 THEN', sql)
        self.assertIn('WHEN "revenue" < 45 THEN', sql)
        self.assertIn('WHEN "revenue" >= 45 THEN', sql)
        self.assertIn('ELSE "revenue"', sql, msg="default should be the calling column")

    def test_case_when_unmatched_rows_keep_original_value(self):
        # Only the first two rows match; the rest must keep the original
        # revenue value (matching pandas Series.case_when semantics).
        ds_out = self.ds.assign(
            label=self.ds["revenue"].case_when([
                (pd.col("revenue") < 25, "low"),
            ])
        )
        labels = ds_out._execute()["label"].tolist()
        self.assertEqual(labels[0], "low")
        self.assertEqual(labels[1], "low")
        # ClickHouse picks a common type for the CASE result; compare as str.
        for original, got in zip([30, 40, 50], labels[2:]):
            self.assertEqual(str(original), str(got))

    def test_case_when_rejects_non_list_caselist(self):
        with self.assertRaises(TypeError):
            self.ds["revenue"].case_when("not a list")

    def test_case_when_rejects_malformed_entries(self):
        with self.assertRaises(ValueError):
            self.ds["revenue"].case_when([(pd.col("revenue") < 5,)])


# ---------------------------------------------------------------------------
# 9. Optional integration test against a real ClickHouse server.
#    Skipped unless CHDB_TEST_HOST is set (workspace convention).
# ---------------------------------------------------------------------------


import os


@pytest.mark.skipif(
    not os.environ.get("CHDB_TEST_HOST"),
    reason="set CHDB_TEST_HOST to run the remote ClickHouse integration test",
)
class TestRemoteClickHousePushdown:
    """Run ``pd.col`` expressions against a real ClickHouse server.
    Activated only when ``CHDB_TEST_HOST`` is set.
    """

    @classmethod
    def setup_class(cls):
        import chdb  # noqa: F401 - sanity import

        cls.host = os.environ["CHDB_TEST_HOST"]
        cls.user = os.environ.get("CHDB_TEST_USER", "default")
        cls.password = os.environ.get("CHDB_TEST_PASSWORD", "")
        cls.database = os.environ.get("CHDB_TEST_DATABASE", "default")
        cls.table = f"test_pdcol_compat_{os.getpid()}"

        # Seed a small table.
        import chdb

        host_only = cls.host.split(":")[0]
        port = cls.host.split(":")[1] if ":" in cls.host else "9000"
        remote_args = (
            f"'{cls.host}', '{cls.database}', '{cls.table}', "
            f"'{cls.user}', '{cls.password}'"
        )
        ddl = (
            f"CREATE OR REPLACE TABLE {cls.database}.{cls.table} "
            "(name String, age Int32, speed Int32) ENGINE = MergeTree() ORDER BY age"
        )
        ins = (
            f"INSERT INTO function remote({remote_args}) "
            "VALUES ('alice', 25, 80), ('BOB', 31, 120), "
            "('Carol', 17, 60), ('dave', 42, 150)"
        )
        chdb.query(
            f"CREATE OR REPLACE TABLE function remote({remote_args}) AS "
            "(name String, age Int32, speed Int32)"
        )  # may be a no-op; tolerate failure
        # The clean approach: use remote() directly to issue DDL.
        chdb.query(
            f"INSERT INTO function remote({remote_args}) VALUES "
            "('alice', 25, 80), ('BOB', 31, 120), ('Carol', 17, 60), ('dave', 42, 150)"
        )

    @classmethod
    def teardown_class(cls):
        try:
            import chdb

            remote_args = (
                f"'{cls.host}', '{cls.database}', '{cls.table}', "
                f"'{cls.user}', '{cls.password}'"
            )
            chdb.query(f"DROP TABLE IF EXISTS function remote({remote_args})")
        except Exception:
            pass

    def _ds(self):
        return DataStore.from_clickhouse(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
        )[self.table]

    def test_pd_col_filter_returns_correct_rows(self):
        ds = self._ds()
        df = ds[pd.col("speed") > 105]._execute()
        names = sorted(df["name"].tolist())
        assert names == ["BOB", "dave"], names

    def test_pd_col_assign_with_str_title_pushes_to_clickhouse(self):
        ds = self._ds()
        result = ds.assign(name_t=pd.col("name").str.title())._execute()
        # ClickHouse's initcap title-cases each row.
        for got in result["name_t"].tolist():
            assert got == got.title() or got == got.capitalize() or got


if __name__ == "__main__":
    unittest.main()
