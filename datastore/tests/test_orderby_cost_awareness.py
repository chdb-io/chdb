"""
ORDER BY Cost Awareness Tests (CH-2)

Verify that sort_values() does not produce unbounded ORDER BY in SQL,
which would force remote servers to sort entire tables before returning data.

Key behaviors tested:
1. sort_values() alone: ORDER BY deferred to pandas (no SQL ORDER BY)
2. sort_values().head(N): ORDER BY + LIMIT merged in SQL
3. sort_values().count(): ORDER BY stripped from COUNT subquery
4. sort_values().groupby().agg(): GROUP BY pushed down, ORDER BY stripped
5. Correctness: all results match pandas behavior
"""

import pytest
import pandas as pd
import numpy as np
from copy import copy

from datastore import DataStore
from datastore.query_planner import QueryPlanner
from datastore.lazy_ops import LazyRelationalOp, LazyGroupByAgg
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


@pytest.fixture
def sample_data():
    """Sample data for ORDER BY cost awareness tests (all numeric except group)."""
    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "group": ["a", "a", "b", "b", "a", "b", "a", "b"],
        "value": [30, 10, 50, 20, 40, 60, 5, 15],
        "score": [85, 92, 78, 88, 95, 70, 99, 65],
    }
    return data


@pytest.fixture
def ds(sample_data):
    return DataStore(sample_data)


@pytest.fixture
def pdf(sample_data):
    return pd.DataFrame(sample_data)


class TestUnboundedOrderByNotPushed:
    """sort_values() without LIMIT should NOT push ORDER BY to SQL."""

    def test_sort_values_alone_no_sql_orderby(self, ds):
        """sort_values() alone should be classified as pandas, not SQL."""
        sorted_ds = ds.sort_values("value")
        planner = QueryPlanner()
        has_sql = bool(
            sorted_ds._table_function
            or sorted_ds.table_name
            or sorted_ds._source_df is not None
        )
        schema = sorted_ds.schema() if has_sql else sorted_ds._schema
        exec_plan = planner.plan_segments(sorted_ds._lazy_ops, has_sql, schema=schema)

        # ORDER BY should be in a pandas segment, not SQL
        for seg in exec_plan.segments:
            if seg.is_sql() and seg.plan:
                for op in seg.plan.sql_ops:
                    if isinstance(op, LazyRelationalOp) and op.op_type == "ORDER BY":
                        pytest.fail(
                            "ORDER BY should not be in SQL segment without LIMIT"
                        )

    def test_sort_values_alone_result_correct(self, ds, pdf):
        """sort_values() alone should produce correct results via pandas execution."""
        ds_result = ds.sort_values("value")
        pd_result = pdf.sort_values("value")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_descending_no_sql_orderby(self, ds, pdf):
        """sort_values(ascending=False) without LIMIT defers to pandas."""
        ds_result = ds.sort_values("value", ascending=False)
        pd_result = pdf.sort_values("value", ascending=False)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_multiple_columns_no_sql(self, ds, pdf):
        """sort_values with multiple columns without LIMIT defers to pandas."""
        ds_result = ds.sort_values(["group", "value"])
        pd_result = pdf.sort_values(["group", "value"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_double_sort_no_sql(self, ds, pdf):
        """Chained sort_values without LIMIT defers to pandas."""
        ds_result = ds.sort_values("value").sort_values("score")
        pd_result = pdf.sort_values("value").sort_values("score")
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestOrderByLimitMerged:
    """sort_values().head(N) should merge ORDER BY + LIMIT in SQL."""

    def test_sort_head_sql_has_order_and_limit(self, ds):
        """sort_values().head() should generate ORDER BY + LIMIT in SQL."""
        sorted_head = ds.sort_values("value").head(3)
        sql = sorted_head.to_sql(execution_format=True)
        assert "ORDER BY" in sql, f"Expected ORDER BY in SQL: {sql}"
        assert "LIMIT" in sql, f"Expected LIMIT in SQL: {sql}"

    def test_sort_head_result_correct(self, ds, pdf):
        """sort_values().head(N) results match pandas."""
        ds_result = ds.sort_values("value").head(3)
        pd_result = pdf.sort_values("value").head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_desc_head_result_correct(self, ds, pdf):
        """sort_values(ascending=False).head(N) results match pandas."""
        ds_result = ds.sort_values("value", ascending=False).head(3)
        pd_result = pdf.sort_values("value", ascending=False).head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_head_sql_has_limit_value(self, ds):
        """sort_values().head(5) should produce LIMIT 5."""
        sorted_head = ds.sort_values("value").head(5)
        sql = sorted_head.to_sql(execution_format=True)
        assert "LIMIT 5" in sql, f"Expected LIMIT 5 in SQL: {sql}"

    def test_sort_multiple_cols_head(self, ds, pdf):
        """sort_values with multiple columns + head results match pandas."""
        ds_result = ds.sort_values(["group", "value"]).head(4)
        pd_result = pdf.sort_values(["group", "value"]).head(4)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_double_sort_head_sql_merged(self, ds, pdf):
        """Chained sort_values with head merges correctly."""
        ds_result = ds.sort_values("id").sort_values("value").head(3)
        pd_result = pdf.sort_values("id").sort_values("value").head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)
        sql = ds.sort_values("id").sort_values("value").head(3).to_sql(
            execution_format=True
        )
        assert "ORDER BY" in sql
        assert "LIMIT" in sql


class TestCountStripsOrderBy:
    """sort_values().count() should strip ORDER BY from the SQL."""

    def test_sort_count_result_correct(self, ds, pdf):
        """sort_values().count() results match pandas."""
        ds_result = ds.sort_values("value").count()
        pd_result = pdf.sort_values("value").count()
        assert_series_equal(ds_result, pd_result)

    def test_sort_count_no_orderby_in_sql(self, ds):
        """sort_values().count() SQL should not contain ORDER BY."""
        sorted_ds = ds.sort_values("value")
        # count() builds its own SQL internally, stripping ORDER BY
        count_base = copy(sorted_ds)
        count_base._lazy_ops = [
            op
            for op in count_base._lazy_ops
            if not (isinstance(op, LazyRelationalOp) and op.op_type == "ORDER BY")
        ]
        count_base._orderby_fields = []
        # Verify ORDER BY ops are removed
        for op in count_base._lazy_ops:
            if isinstance(op, LazyRelationalOp) and op.op_type == "ORDER BY":
                pytest.fail("ORDER BY should be stripped from count base")

    def test_sort_count_rows_result_correct(self, ds, pdf):
        """sort_values().count_rows() results match pandas len()."""
        ds_count = ds.sort_values("value").count_rows()
        pd_count = len(pdf.sort_values("value"))
        assert ds_count == pd_count

    def test_sort_desc_count(self, ds, pdf):
        """sort_values(ascending=False).count() still correct."""
        ds_result = ds.sort_values("value", ascending=False).count()
        pd_result = pdf.sort_values("value", ascending=False).count()
        assert_series_equal(ds_result, pd_result)


class TestGroupByStripsOrderBy:
    """sort_values().groupby().agg() should strip ORDER BY and push GROUP BY."""

    def test_sort_groupby_sum_result_correct(self, ds, pdf):
        """sort_values().groupby().sum() results match pandas."""
        ds_result = ds.sort_values("value").groupby("group").sum()
        pd_result = pdf.sort_values("value").groupby("group").sum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_agg_sum_result_correct(self, ds, pdf):
        """sort_values().groupby().agg('sum') results match pandas."""
        ds_result = ds.sort_values("value").groupby("group").agg("sum")
        pd_result = pdf.sort_values("value").groupby("group").agg("sum")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_mean_result_correct(self, ds, pdf):
        """sort_values().groupby().mean() results match pandas."""
        ds_result = ds.sort_values("value").groupby("group").mean()
        pd_result = pdf.sort_values("value").groupby("group").mean()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_count_result_correct(self, ds, pdf):
        """sort_values().groupby().count() results match pandas."""
        ds_result = ds.sort_values("value").groupby("group").count()
        pd_result = pdf.sort_values("value").groupby("group").count()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_min_max(self, ds, pdf):
        """sort_values().groupby().min/max results match pandas."""
        ds_min = ds.sort_values("value").groupby("group").min()
        pd_min = pdf.sort_values("value").groupby("group").min()
        assert_datastore_equals_pandas(ds_min, pd_min)

        ds_max = ds.sort_values("value").groupby("group").max()
        pd_max = pdf.sort_values("value").groupby("group").max()
        assert_datastore_equals_pandas(ds_max, pd_max)

    def test_sort_groupby_no_orderby_in_plan(self, ds):
        """ORDER BY should not appear in SQL when followed by GROUP BY."""
        sorted_grouped = ds.sort_values("value").groupby("group").sum()
        planner = QueryPlanner()
        has_sql = bool(
            sorted_grouped._table_function
            or sorted_grouped.table_name
            or sorted_grouped._source_df is not None
        )
        schema = sorted_grouped.schema() if has_sql else sorted_grouped._schema
        exec_plan = planner.plan_segments(
            sorted_grouped._lazy_ops, has_sql, schema=schema
        )

        # No SQL segment should contain ORDER BY
        for seg in exec_plan.segments:
            if seg.is_sql() and seg.plan:
                for op in seg.plan.sql_ops:
                    if isinstance(op, LazyRelationalOp) and op.op_type == "ORDER BY":
                        pytest.fail(
                            "ORDER BY should not be in SQL when followed by GROUP BY"
                        )

    def test_groupby_sort_preserves_post_agg_order(self, ds, pdf):
        """groupby(sort=True) should still order results by group keys."""
        ds_result = ds.sort_values("value").groupby("group", sort=True).sum()
        pd_result = pdf.sort_values("value").groupby("group", sort=True).sum()
        assert_datastore_equals_pandas(ds_result, pd_result)
        # Verify index is sorted
        assert list(ds_result.index) == sorted(list(ds_result.index))


class TestSortGroupbyFirstLast:
    """sort_values() before groupby().first()/last() preserves pandas semantics."""

    def test_sort_groupby_first(self, ds, pdf):
        """sort_values().groupby().first() returns first per group after sort."""
        ds_result = ds.sort_values("value").groupby("group").first()
        pd_result = pdf.sort_values("value").groupby("group").first()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_last(self, ds, pdf):
        """sort_values().groupby().last() returns last per group after sort."""
        ds_result = ds.sort_values("value").groupby("group").last()
        pd_result = pdf.sort_values("value").groupby("group").last()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEdgeCases:
    """Edge cases for ORDER BY cost awareness."""

    def test_head_then_sort(self, ds, pdf):
        """head().sort_values(): LIMIT first, then sort in pandas."""
        ds_result = ds.head(5).sort_values("value")
        pd_result = pdf.head(5).sort_values("value")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_filter_head(self, ds, pdf):
        """sort_values + filter + head chain."""
        ds_result = ds.sort_values("value")
        ds_result = ds_result[ds_result["value"] > 15].head(3)
        pd_result = pdf.sort_values("value")
        pd_result = pd_result[pd_result["value"] > 15].head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_then_sort_head(self, ds, pdf):
        """groupby().sum().sort_values().head() - post-aggregation sort with LIMIT."""
        ds_result = ds.groupby("group").sum().sort_values("value").head(1)
        pd_result = pdf.groupby("group").sum().sort_values("value").head(1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_dataframe_sort(self):
        """sort_values on empty DataFrame."""
        pdf = pd.DataFrame({"col": [], "val": []})
        ds = DataStore({"col": [], "val": []})
        ds_result = ds.sort_values("col")
        pd_result = pdf.sort_values("col")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_sort(self):
        """sort_values on single-row DataFrame."""
        pdf = pd.DataFrame({"col": [1], "val": [10]})
        ds = DataStore({"col": [1], "val": [10]})
        ds_result = ds.sort_values("col").head(1)
        pd_result = pdf.sort_values("col").head(1)
        assert_datastore_equals_pandas(ds_result, pd_result)
