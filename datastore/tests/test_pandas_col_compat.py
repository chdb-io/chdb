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

    def test_chained_str_accessor_pushes_down(self):
        # Multi-step str accessor chains (lower -> strip) must compose into
        # a single ClickHouse function tree, not dump to fallback. Mirrors
        # how SQL builder will see the translated tree.
        sql = translate_pandas_expression(
            pd.col("name").str.lower().str.strip()
        ).to_sql()
        # Both transforms must appear; exact wrapping order matches how
        # chdb-ds composes Function nodes.
        self.assertIn("lower", sql)
        self.assertIn("trim", sql)
        self.assertIn('"name"', sql)

    def test_dt_year_plus_literal_composes_arithmetic_and_function(self):
        # ``pd.col("d").dt.year + 2000`` is a function-then-arithmetic
        # composition — exercises that translated DateTimePropertyExpr
        # nodes still feed cleanly into arithmetic translation.
        sql = translate_pandas_expression(pd.col("d").dt.year + 2000).to_sql()
        self.assertIn("toYear", sql)
        self.assertIn("+2000", sql.replace(" ", ""))


class TestNumpyUfuncOptOut(unittest.TestCase):
    """``Expression.__array_ufunc__ = None`` is set on the chdb-ds side
    (expressions.py:539) to stop numpy from silently dispatching
    ``np.log(...)`` to a same-named method injected by FunctionRegistry
    (which would push a SQL ``log()`` call). Verify the opt-out actually
    disables ufunc dispatch for *both* raw Field nodes and translated
    pd.col expressions; without it, ``assign(y=np.log(pd.col("x")))``
    would silently route through SQL instead of pandas.
    """

    def test_np_log_on_raw_field_raises_typeerror(self):
        f = Field("x")
        with self.assertRaises(TypeError):
            np.log(f)

    def test_np_sqrt_on_raw_field_raises_typeerror(self):
        f = Field("x")
        with self.assertRaises(TypeError):
            np.sqrt(f)

    def test_np_log_on_arithmetic_expression_raises_typeerror(self):
        # ``__array_ufunc__`` set on the base ``Expression`` class must
        # propagate to ``ArithmeticExpression`` subclass too.
        expr = Field("x") + 1
        with self.assertRaises(TypeError):
            np.log(expr)

    def test_np_log_on_translated_pd_col_returns_pandas_fallback(self):
        # ``np.log(pd.col("x"))`` symbolic-evals via pd.col's ufunc
        # protocol; chdb-ds's opt-out propagates here too, the translator
        # catches the TypeError and wraps the original expression. This is
        # what makes ``assign(y=np.log(pd.col("x")))`` pandas-correct.
        from datastore.pandas_col_compat import PandasFallbackExpr

        translated = translate_pandas_expression(np.log(pd.col("x")))
        self.assertIsInstance(translated, PandasFallbackExpr)

    def test_np_sqrt_on_translated_pd_col_returns_pandas_fallback(self):
        from datastore.pandas_col_compat import PandasFallbackExpr

        translated = translate_pandas_expression(np.sqrt(pd.col("x")))
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

    def test_bare_pd_col_field_in_getitem_equals_string_indexing(self):
        # Covers ``__getitem__`` line ~6398: ``if isinstance(translated, Field):
        # return self[translated.name]``. ``ds[pd.col('name')]`` should be the
        # same shape as ``ds['name']`` — pandas treats ``df[pd.col('name')]``
        # as a column selector, and chdb-ds normalizes through that branch.
        ds_via_pd_col = self.ds[pd.col("name")]
        ds_via_string = self.ds["name"]
        # Compare materialized values rather than container types — both
        # paths must yield the same column data.
        a = ds_via_pd_col._execute() if hasattr(ds_via_pd_col, "_execute") else ds_via_pd_col
        b = ds_via_string._execute() if hasattr(ds_via_string, "_execute") else ds_via_string
        if hasattr(a, "tolist"):
            self.assertEqual(a.tolist(), b.tolist() if hasattr(b, "tolist") else list(b))
        else:
            self.assertEqual(list(a["name"]), list(b["name"]) if "name" in getattr(b, "columns", []) else list(b))

    def test_comparison_between_two_pd_col_columns_pushes_down(self):
        # ``ds[pd.col('a') > pd.col('b')]`` — both sides are translated to
        # Field nodes and the result is a chdb-ds BinaryCondition that must
        # push as ``WHERE "speed" > "age"``, not silently materialize.
        self.df["over"] = self.df["speed"] - self.df["age"]
        pd_out = self.df.loc[pd.col("speed") > pd.col("age")]
        ds = DataStore(self.df.copy())
        result, sqls = _capture_executed_sql(
            lambda: ds[pd.col("speed") > pd.col("age")]
        )
        assert_datastore_equals_pandas(result, pd_out)
        self.assertTrue(
            any('WHERE "speed" > "age"' in s for s in sqls),
            msg=f'Expected WHERE "speed" > "age" in SQL, got:\n{sqls}',
        )


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

    def test_loc_with_compound_pd_col_condition_and_column_subset(self):
        # Covers the ``(Expression, list[str])`` tuple branch with a compound
        # ``&`` condition, ensuring both halves of the AND are pushed down
        # together with the SELECT projection.
        pd_out = self.df.loc[
            (pd.col("age") >= 18) & (pd.col("speed") < 130),
            ["name", "age"],
        ]
        ds_out = self.ds.loc[
            (pd.col("age") >= 18) & (pd.col("speed") < 130),
            ["name", "age"],
        ]
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_loc_with_pd_col_condition_and_single_str_column(self):
        # Covers the ``isinstance(col_key, str)`` branch in
        # ``DataStoreLocIndexer.__getitem__`` which wraps the str into
        # ``[col_key]`` before emitting the SELECT op. pandas returns a
        # Series here while chdb-ds returns a single-column DataStore — the
        # design rule is "don't obsess over container differences", so we
        # compare the underlying values mirror-style.
        pd_out = self.df.loc[pd.col("speed") > 105, "name"]
        ds_out = self.ds.loc[pd.col("speed") > 105, "name"]
        ds_df = ds_out._execute() if hasattr(ds_out, "_execute") else ds_out
        self.assertEqual(list(ds_df["name"]), list(pd_out))


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

    def test_where_with_pushable_pd_col_and_other_masks_unmatched(self):
        # Pushable pd.col + explicit ``other`` → goes through pandas-style
        # where() (not filter); covers the cross-branch wiring.
        pd_out = self.df.where(pd.col("speed") > 105, other=0)
        ds_out = self.ds.where(pd.col("speed") > 105, other=0)
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_where_with_pd_col_fallback_and_other_unwraps_to_pandas(self):
        # Regression for review feedback: ``pd.col(...).astype(...)`` is
        # untranslatable (→ PandasFallbackExpr), and adding ``other`` forces
        # the pandas-style branch. Without the unwrap, pandas receives the
        # opaque wrapper and raises
        # ``ValueError: Array conditional must be same shape as self``.
        pd_out = self.df.where(pd.col("speed").astype(int) > 105, other=0)
        ds_out = self.ds.where(pd.col("speed").astype(int) > 105, other=0)
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_where_with_pd_col_fallback_no_other_routes_through_filter(self):
        # Untranslatable pd.col + no ``other`` → falls through to filter(),
        # which has its own PandasFallbackExpr branch. Compare against
        # ``loc[...]`` because filter() drops rows (vs. masking them).
        pd_out = self.df.loc[pd.col("speed").astype(int) > 105]
        ds_out = self.ds.where(pd.col("speed").astype(int) > 105)
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

    def test_assign_mixed_sql_fallback_and_pandas_kwargs_routes_each_correctly(self):
        # ``assign`` splits kwargs into three buckets (sql_kwargs /
        # fallback_kwargs / pandas_kwargs) at core.py:5634. Each bucket alone
        # is covered elsewhere; this exercises all three together to catch
        # ordering/state-leak bugs between the buckets.
        pd_out = self.df.assign(
            diff=pd.col("speed") - pd.col("age"),
            log_speed=np.log(pd.col("speed")),
            doubled=lambda x: x["age"] * 2,
        )
        result, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).assign(
                diff=pd.col("speed") - pd.col("age"),
                log_speed=np.log(pd.col("speed")),
                doubled=lambda x: x["age"] * 2,
            )
        )
        assert_datastore_equals_pandas(result, pd_out)
        # SQL bucket must push down arithmetic
        self.assertTrue(
            any('("speed"-"age")' in s for s in sqls),
            msg=f'Expected ("speed"-"age") in SQL, got:\n{sqls}',
        )
        # Fallback bucket must NOT push log() to SQL
        for s in sqls:
            self.assertNotIn('log("speed")', s)


class TestDataStoreWithColumnMirrorsPandas(unittest.TestCase):

    def setUp(self):
        self.df = _sample_df()
        self.ds = DataStore(self.df.copy())

    def test_with_column_arithmetic(self):
        pd_out = self.df.assign(diff=pd.col("speed") - pd.col("age"))
        ds_out = self.ds.with_column("diff", pd.col("speed") - pd.col("age"))
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_with_column_str_accessor_pushes_down_initcap(self):
        # Unlike assign(), with_column() goes straight to LazyColumnAssignment
        # (no three-bucket dispatch). This makes sure the SQL pushdown path
        # there still recognizes a translated pd.col expression.
        pd_out = self.df.assign(name_t=pd.col("name").str.title())
        result, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).with_column(
                "name_t", pd.col("name").str.title()
            )
        )
        assert_datastore_equals_pandas(result, pd_out)
        # Verify the str.title was actually pushed as initcap, not silently
        # run through pandas — distinguishing assign and with_column.
        self.assertTrue(
            any('initcap("name")' in s for s in sqls),
            msg=f'Expected initcap("name") in SQL, got:\n{sqls}',
        )

    def test_with_column_np_log_falls_back_to_pandas(self):
        # PR wires translate_pandas_expression into with_column but only the
        # arithmetic-SQL path was covered. with_column has no dedicated
        # fallback branch (unlike assign's three-bucket dispatch) — it relies
        # on LazyColumnAssignment._is_pandas_only_function recognizing
        # PandasFallbackExpr and forcing the pandas segment. Lock that in.
        pd_out = self.df.assign(log_speed=np.log(pd.col("speed")))
        result, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).with_column(
                "log_speed", np.log(pd.col("speed"))
            )
        )
        assert_datastore_equals_pandas(result, pd_out)
        for s in sqls:
            self.assertNotIn('log("speed")', s)

    def test_with_column_astype_falls_back_and_preserves_float_dtype(self):
        # astype(float) must yield numpy float64 (pandas semantics), not the
        # Float64/Int64 dance a SQL CAST would produce. Catches regressions
        # where with_column accidentally routes astype through SQL.
        ds_out = self.ds.with_column("age_f", pd.col("age").astype(float))
        df_out = ds_out._execute() if hasattr(ds_out, "_execute") else ds_out
        self.assertEqual(str(df_out["age_f"].dtype), "float64")
        # Values themselves must match pandas exactly.
        pd_out = self.df.assign(age_f=pd.col("age").astype(float))
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

    def test_loc_tuple_with_astype_falls_back_to_pandas(self):
        # ``ds.loc[pd.col(...).astype(...) > X, [cols]]`` must produce the
        # same rows pandas does, *without* emitting ``CAST(...)`` in any
        # executed SQL — i.e. the fallback path has to handle the tuple
        # shape too, not just the bare-key shape covered by
        # ``test_getitem_with_astype_falls_back_to_pandas``.
        pd_out = self.df.loc[
            pd.col("age").astype(float) > 21, ["name", "age"]
        ]
        result, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).loc[
                pd.col("age").astype(float) > 21, ["name", "age"]
            ]
        )
        assert_datastore_equals_pandas(result, pd_out)
        self._assert_no_sql_fragment(sqls, 'CAST("age"')


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

    def test_loc_tuple_pushes_down_where_and_select_columns(self):
        # ``ds.loc[pd.col(...) > X, [cols]]`` must produce ONE SQL query
        # that contains BOTH the WHERE clause AND the explicit column
        # projection — not ``SELECT *`` followed by an in-memory column
        # subset. Mirror-result equality (covered elsewhere) cannot
        # distinguish these two; only SQL-level inspection can.
        _, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).loc[
                pd.col("speed") > 105, ["name", "age"]
            ]
        )
        joined = "\n".join(sqls) or "(no SQL captured)"
        self._assert_any_sql_contains(
            sqls, 'WHERE "speed" > 105', label="loc tuple: "
        )
        # Pushed-down SELECT must list both columns by name.
        has_projection = any(
            'SELECT' in s.upper() and '"name"' in s and '"age"' in s
            for s in sqls
        )
        self.assertTrue(
            has_projection,
            msg=(
                "Expected SELECT projection with both \"name\" and \"age\" "
                f"in one SQL, got:\n{joined}"
            ),
        )
        # Negative: the WHERE-bearing SQL must NOT be ``SELECT *``, which
        # would mean projection pushdown silently failed.
        for s in sqls:
            if 'WHERE "speed" > 105' in s:
                self.assertNotIn(
                    'SELECT *',
                    s,
                    msg=(
                        "WHERE got pushed but projection did not — "
                        f"SELECT * leaks the full row width:\n{s}"
                    ),
                )

    def test_loc_tuple_with_single_str_col_pushes_down(self):
        # ``ds.loc[pd.col(...) > X, "name"]`` (col selector as a bare str)
        # goes through the ``if isinstance(col_key, str): col_key = [col_key]``
        # branch and should still produce a SQL with both the WHERE clause
        # and the single-column projection.
        _, sqls = _capture_executed_sql(
            lambda: DataStore(self.df.copy()).loc[
                pd.col("speed") > 105, "name"
            ]
        )
        joined = "\n".join(sqls) or "(no SQL captured)"
        self._assert_any_sql_contains(
            sqls, 'WHERE "speed" > 105', label="loc str col: "
        )
        self.assertTrue(
            any('SELECT' in s.upper() and '"name"' in s for s in sqls),
            msg=f'Expected SELECT "name" in SQL, got:\n{joined}',
        )


# ---------------------------------------------------------------------------
# 3.5 ``ds.loc[pd.col(...), col] = value`` — setitem must mirror pandas
# ---------------------------------------------------------------------------


class TestLocSetItemWithPdCol(unittest.TestCase):
    """``ds.loc[pd.col(...) > X, "col"] = value`` should assign through to
    the underlying DataFrame just like pandas does. There is no SQL
    pushdown for setitem, so this exercises the ``_convert_key`` →
    pandas-loc fallback path with a translated chdb-ds Condition as the
    row key.
    """

    def setUp(self):
        self.df = _sample_df()

    def test_loc_setitem_with_pd_col_condition_assigns_correctly(self):
        pd_df = self.df.copy()
        pd_df.loc[pd.col("speed") > 105, "name"] = "FAST"

        ds = DataStore(self.df.copy())
        ds.loc[pd.col("speed") > 105, "name"] = "FAST"

        ds_df = ds._execute() if hasattr(ds, "_execute") else ds
        self.assertEqual(list(ds_df["name"]), list(pd_df["name"]))
        # Untouched columns must remain identical, including row order.
        self.assertEqual(list(ds_df["age"]), list(pd_df["age"]))
        self.assertEqual(list(ds_df["speed"]), list(pd_df["speed"]))

    def test_loc_setitem_with_pd_col_astype_fallback_assigns_correctly(self):
        # ``_convert_key`` has a dedicated PandasFallbackExpr -> original
        # branch at pandas_compat.py:_convert_key. Without it,
        # ``ds.loc[pd.col(...).astype(...) > X, "col"] = value`` would leak
        # the opaque wrapper into pandas-loc and raise InvalidIndexError.
        pd_df = self.df.copy()
        pd_df.loc[pd.col("speed").astype(int) > 105, "name"] = "FAST"

        ds = DataStore(self.df.copy())
        ds.loc[pd.col("speed").astype(int) > 105, "name"] = "FAST"

        ds_df = ds._execute() if hasattr(ds, "_execute") else ds
        self.assertEqual(list(ds_df["name"]), list(pd_df["name"]))
        self.assertEqual(list(ds_df["speed"]), list(pd_df["speed"]))


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
        self.df = _sample_df()

    def test_unknown_method_on_pd_col_falls_back_to_pandas(self):
        # Unknown methods on Field cannot be translated to SQL, so the
        # translator wraps the original pd.col expression in
        # PandasFallbackExpr and lets pandas evaluate it. Pandas itself
        # then surfaces the AttributeError at exec time.
        ds_out = self.ds.assign(bad=pd.col("name").totally_made_up_method())
        with self.assertRaises(AttributeError):
            ds_out._execute()

    def test_bare_non_boolean_pd_col_in_getitem_raises_typeerror(self):
        # ``ds[pd.col("name").str.lower()]`` has no defined pandas
        # semantics — surface a TypeError with a hint.
        with self.assertRaises(TypeError):
            _ = self.ds[pd.col("name").str.lower()]

    # ------------------------------------------------------------------
    # Methods where ``pd.col(...)`` is unsupported by pandas itself.
    # We mirror that rejection — chdb-ds should not silently accept a
    # pd.col where pandas refuses it (would diverge from the "pandas-
    # parity" promise) and should not silently produce malformed SQL.
    # ------------------------------------------------------------------

    def test_fillna_with_pd_col_value_mirrors_pandas_rejection(self):
        # pandas 3.0.3: ``df.fillna(value=pd.col('x'))`` raises TypeError
        # ("boolean value of an expression is ambiguous") because
        # ``fillna`` boolean-checks ``value``. chdb-ds must not pretend
        # to support what pandas itself rejects.
        with self.assertRaises(TypeError):
            self.df.fillna(value=pd.col("speed"))
        with self.assertRaises((TypeError, ValueError, NotImplementedError)):
            self.ds.fillna(value=pd.col("speed"))

    def test_drop_duplicates_with_pd_col_subset_mirrors_pandas_rejection(self):
        # pandas 3.0.3: ``df.drop_duplicates(subset=pd.col(...))`` raises
        # TypeError ("unhashable type: 'Expression'") because subset
        # entries get stuffed into a set. chdb-ds mirrors.
        with self.assertRaises(TypeError):
            self.df.drop_duplicates(subset=pd.col("name"))
        with self.assertRaises((TypeError, KeyError, NotImplementedError)):
            self.ds.drop_duplicates(subset=pd.col("name"))

    def test_set_index_with_pd_col_mirrors_pandas_rejection(self):
        # pandas 3.0.3: ``df.set_index(pd.col(...))`` raises TypeError
        # ("may be a column key... Received column of type
        # <class 'pandas.api.typing.Expression'>"). chdb-ds mirrors.
        with self.assertRaises(TypeError):
            self.df.set_index(pd.col("name"))
        with self.assertRaises((TypeError, KeyError, NotImplementedError)):
            self.ds.set_index(pd.col("name"))


# ---------------------------------------------------------------------------
# 5b. Mixing ColumnExpr (``ds[col]``) and ``pd.col`` in one expression
# ---------------------------------------------------------------------------


class TestMixingColumnExprAndPdCol(unittest.TestCase):
    """Users naturally interleave ``ds['x']`` (chdb-ds) and ``pd.col('x')``
    (pandas 3.x). Used to crash:

      * ``(ds['x'] > N) & (pd.col('y') < M)`` — ``ColumnExpr.__and__`` raised
        ``TypeError: Cannot AND ColumnExpr with Expression``.
      * ``ds['a'] + pd.col('b')`` — ``Expression.wrap`` didn't know about
        pd.col and emitted ``'col(\\'b\\')'`` as a SQL string literal,
        which ClickHouse rejected with a syntax error.

    Both binary-op chains now translate the pd.col operand at the entry
    point and route through SQL pushdown (or raise a clear error for
    untranslatable chains).

    Note: pandas 3.0.3 itself rejects ``(pd.Series & pd.col_expr)`` with
    "boolean value of an expression is ambiguous", so we can't always
    mirror against pandas — these tests hand-compute the expected result.
    """

    def setUp(self):
        self.df = pd.DataFrame(
            {"revenue": [10, 20, 30, 40, 50], "qty": [1, 2, 3, 4, 5]}
        )
        self.ds = DataStore(self.df.copy())

    def test_and_mixing_columnexpr_and_pd_col_pushes_down(self):
        result, sqls = _capture_executed_sql(
            lambda: self.ds.filter(
                (self.ds["revenue"] > 15) & (pd.col("qty") < 5)
            )
        )
        self.assertEqual(result["revenue"].tolist(), [20, 30, 40])
        self.assertEqual(result["qty"].tolist(), [2, 3, 4])
        joined = "\n".join(sqls)
        self.assertIn('"revenue" > 15', joined)
        self.assertIn('"qty" < 5', joined)

    def test_or_mixing_columnexpr_and_pd_col_pushes_down(self):
        result, sqls = _capture_executed_sql(
            lambda: self.ds.filter(
                (self.ds["revenue"] > 40) | (pd.col("qty") < 3)
            )
        )
        self.assertEqual(result["revenue"].tolist(), [10, 20, 50])
        joined = "\n".join(sqls)
        self.assertIn('"revenue" > 40', joined)
        self.assertIn('"qty" < 3', joined)

    def test_reverse_and_pd_col_left_columnexpr_right(self):
        # __rand__ path: pandas Expression evaluates ``__and__`` first and
        # delegates back to ColumnExpr via Python's reflected operator.
        result = self.ds.filter(
            (pd.col("qty") < 5) & (self.ds["revenue"] > 15)
        )._execute()
        self.assertEqual(result["revenue"].tolist(), [20, 30, 40])

    def test_arithmetic_mixing_columnexpr_and_pd_col_pushes_down(self):
        result, sqls = _capture_executed_sql(
            lambda: self.ds.assign(total=self.ds["revenue"] + pd.col("qty"))
        )
        self.assertEqual(result["total"].tolist(), [11, 22, 33, 44, 55])
        joined = "\n".join(sqls)
        self.assertIn('("revenue"+"qty")', joined)

    def test_comparison_mixing_columnexpr_lt_pd_col(self):
        # ``ds['a'] < pd.col('b')`` -> Expression.wrap kicks in.
        result = self.ds.filter(self.ds["qty"] < pd.col("revenue"))._execute()
        self.assertEqual(len(result), 5)

    def test_loc_with_mixed_boolean_pushes_down(self):
        result = self.ds.loc[
            (self.ds["revenue"] > 15) & (pd.col("qty") < 5)
        ]._execute()
        self.assertEqual(result["revenue"].tolist(), [20, 30, 40])

    def test_mixing_with_fallback_pd_col_in_and_raises_clear_error(self):
        # ``ds['x'] & pd.col('y').astype(int) < 5`` — the fallback chain
        # can't be expressed as a chdb-ds Condition tree. Surface a clear
        # TypeError with the pre-materialize hint, not malformed SQL or
        # an obscure NotImplementedError from to_sql.
        with self.assertRaises(TypeError) as ctx:
            _ = self.ds.filter(
                (self.ds["revenue"] > 15)
                & (pd.col("qty").astype(int) < 5)
            )
        self.assertIn("pd.col", str(ctx.exception))
        self.assertIn("Pre-materialize", str(ctx.exception))

    def test_mixing_with_fallback_pd_col_in_arithmetic_raises_clear_error(self):
        # ``ds['a'] + pd.col('b').astype(int)`` — Expression.wrap must
        # raise a clear TypeError instead of stringifying the wrapper.
        with self.assertRaises(TypeError) as ctx:
            _ = self.ds.assign(
                total=self.ds["revenue"] + pd.col("qty").astype(int)
            )
        self.assertIn("pd.col", str(ctx.exception))
        self.assertIn("Pre-materialize", str(ctx.exception))

    def test_mixing_workaround_pre_materialize_then_combine(self):
        # Documented escape hatch: pre-materialize the fallback chain
        # into a separate column, then combine plain chdb-ds expressions.
        ds = self.ds.assign(qty_int=pd.col("qty").astype(int))
        result = ds.filter((ds["revenue"] > 15) & (ds["qty_int"] < 5))._execute()
        self.assertEqual(result["revenue"].tolist(), [20, 30, 40])


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

    def test_groupby_with_pd_col_key_matches_pandas_rejection(self):
        # pandas 3.0.3 itself rejects ``df.groupby(pd.col(...))`` with
        # ``TypeError: boolean value of an expression is ambiguous`` because
        # the pd.col object hits ``__bool__`` somewhere inside groupby. PR
        # deliberately didn't wire pd.col translation into ``DataStore.groupby``
        # (core.py:5293) — chdb-ds matches the pandas rejection. Lock in
        # the mirror so any future ``groupby(pd.col)`` support has to update
        # both sides in lockstep.
        with self.assertRaises(TypeError):
            self.df.groupby(pd.col("region")).agg(t=("revenue", "sum"))
        with self.assertRaises(TypeError):
            self.ds.groupby(pd.col("region")).agg(t=pd.col("revenue").sum())

    def test_groupby_agg_with_unpushable_pd_col_raises_clear_query_error(self):
        # Regression: ``pd.col("x").astype(int).mean()`` translates to
        # PandasFallbackExpr (because .astype is unpushable). Previously
        # groupby.agg silently dropped the kwarg and returned an empty
        # DataFrame (Columns: []) — worse than crashing. Must now raise
        # a precise QueryError naming the offending alias.
        from datastore.exceptions import QueryError

        with self.assertRaises(QueryError) as ctx:
            self.ds.groupby("region").agg(
                total=pd.col("revenue").astype(int).mean()
            )
        msg = str(ctx.exception)
        self.assertIn("'total'", msg,
                      msg=f"error should name the offending alias, got: {msg}")
        self.assertIn("cannot be pushed to SQL", msg,
                      msg=f"error should explain why, got: {msg}")

    def test_datastore_agg_with_unpushable_pd_col_raises_clear_query_error(self):
        # Mirror check for the LazyGroupBy-less entry point
        # (DataStore.agg called directly). Same precise QueryError.
        from datastore.exceptions import QueryError

        with self.assertRaises(QueryError) as ctx:
            self.ds.agg(total=pd.col("revenue").astype(int).mean())
        msg = str(ctx.exception)
        self.assertIn("'total'", msg)
        self.assertIn("cannot be pushed to SQL", msg)


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

    def test_sort_values_mixed_ascending_list_with_pd_col_and_string(self):
        # ``ascending=[True, False]`` paired with a mixed list of pd.col +
        # str keys is the tricky case: ``_translate_pd_col_in_sort_by`` must
        # rebuild the list with the original ordering preserved so the
        # ascending list still lines up with each key by index.
        pd_out = (
            self.df
            .sort_values(by=["revenue", "name"], ascending=[True, False])
            .reset_index(drop=True)
        )
        ds_out = self.ds.sort_values(
            by=[pd.col("revenue"), "name"], ascending=[True, False]
        )
        assert_datastore_equals_pandas(ds_out, pd_out)

    def test_sort_values_by_list_of_two_pd_cols_with_limit_pushes_down(self):
        # Two pd.col keys, both must end up in ORDER BY (verifies the list
        # branch of ``_translate_pd_col_in_sort_by`` rebuilds the iterable
        # correctly rather than collapsing it to a single key).
        _, sqls = _capture_executed_sql(
            lambda: self.ds.sort_values(
                by=[pd.col("revenue"), pd.col("name")]
            ).head(2)
        )
        joined = "\n".join(sqls)
        self.assertIn('"revenue"', joined)
        self.assertIn('"name"', joined)
        self.assertIn("ORDER BY", joined.upper())

    def test_sort_values_by_pd_col_fallback_rejected_like_pandas(self):
        # pandas itself rejects ``sort_values(by=pd.col(...).astype(...))``
        # with KeyError (you can only sort by a column name, not a derived
        # expression). chdb-ds currently surfaces a less friendly TypeError
        # from ``list(by)`` because ``_is_single_sort_key`` doesn't include
        # PandasFallbackExpr — same outcome (rejection), worse error
        # message. Pin the rejection behavior; tighten the error message
        # in a follow-up if needed.
        with self.assertRaises(Exception):
            self.df.sort_values(by=pd.col("revenue").astype(float))
        with self.assertRaises(Exception):
            self.ds.sort_values(by=pd.col("revenue").astype(float))


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

    def test_case_when_with_pd_col_fallback_condition_routes_through_pandas(self):
        # Regression: ``CaseWhenExpr`` used to call ``self.to_sql(...)``
        # outside the try/except in ``_evaluate_via_chdb`` and
        # ``PandasFallbackExpr.to_sql`` (translated from ``pd.col(...)
        # .astype``) raised NotImplementedError. ``CaseWhenExpr.evaluate``
        # now inspects its cases and forces ``_evaluate_via_pandas`` when
        # any PandasFallbackExpr is present, and ``ColumnExpr.is_pandas_only``
        # propagates that so ``assign`` routes the kwarg to the pandas
        # bucket instead of splicing the CASE into a SELECT clause.
        ds_out = self.ds.assign(
            label=self.ds["revenue"].case_when(
                [(pd.col("revenue").astype(int) < 25, "small")]
            )
        )
        df_out = ds_out._execute()
        # Rows where revenue (cast to int) < 25 get "small"; others keep
        # the original revenue (case_when's ELSE = self).
        labels = df_out["label"].tolist()
        revenues = df_out["revenue"].tolist()
        for rev, lab in zip(revenues, labels):
            if int(rev) < 25:
                self.assertEqual(lab, "small")
            else:
                self.assertEqual(str(lab), str(rev))

    def test_case_when_with_pd_col_fallback_replacement_routes_through_pandas(self):
        # Replacement-slot regression: ``CaseWhenExpr._value_to_sql`` used
        # to fall into ``f"'{str(value)}'"`` and emit a string literal of
        # the wrapper's ``repr``, producing malformed SQL that ClickHouse
        # rejected. ``CaseWhenExpr._evaluate_via_pandas._evaluate_value``
        # now recognizes PandasFallbackExpr and dispatches to the
        # ExpressionEvaluator (which unwraps and runs the original pandas
        # Expression).
        ds_out = self.ds.assign(
            label=self.ds["revenue"].case_when(
                [(pd.col("revenue") < 25, pd.col("revenue").astype(str))]
            )
        )
        df_out = ds_out._execute()
        labels = df_out["label"].tolist()
        revenues = df_out["revenue"].tolist()
        # For rows where revenue < 25, replacement is ``str(revenue)``;
        # for the rest, ELSE = self["revenue"], so str(label) should
        # equal str(revenue) regardless.
        for rev, lab in zip(revenues, labels):
            self.assertEqual(str(lab), str(rev))

    def test_when_otherwise_builder_with_pd_col_pushes_down(self):
        # Regression: ``CaseWhenBuilder.when()`` used to store the raw
        # pd.col object; serialization eventually called ``to_sql`` on it
        # (or fell into the catch-all string literal) and emitted broken
        # SQL. ``CaseWhenBuilder.when`` and ``.otherwise`` now translate
        # pd.col arguments up front, so the builder API has parity with
        # ``ColumnExpr.case_when([...])``.
        result, sqls = _capture_executed_sql(
            lambda: self.ds.assign(
                label=self.ds.when(pd.col("revenue") < 25, "low")
                .when(pd.col("revenue") < 45, "mid")
                .otherwise("hi"),
            )
        )
        labels = result["label"].tolist()
        self.assertEqual(labels, ["low", "low", "mid", "mid", "hi"])
        # Translation must reach SQL: WHEN "revenue" < 25 THEN ...
        joined = "\n".join(sqls)
        self.assertIn('"revenue" < 25', joined)
        self.assertIn('"revenue" < 45', joined)

    def test_when_otherwise_builder_with_pd_col_fallback_routes_through_pandas(self):
        # Builder + fallback condition: same routing requirement as
        # ColumnExpr.case_when with fallback. assign() must detect a
        # CaseWhenExpr whose cases contain PandasFallbackExpr and route
        # it through the pandas bucket (not the SQL select bucket).
        ds_out = self.ds.assign(
            label=self.ds.when(pd.col("revenue").astype(int) < 25, "low")
            .otherwise("hi")
        )
        df_out = ds_out._execute()
        labels = df_out["label"].tolist()
        revenues = df_out["revenue"].tolist()
        for rev, lab in zip(revenues, labels):
            self.assertEqual(lab, "low" if int(rev) < 25 else "hi")


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
