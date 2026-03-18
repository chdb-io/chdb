"""
Tests for large-scale data row order determinism (P1).

Verifies that multi-step operation chains on 100k+ row DataFrames produce
deterministic, correct row order across multiple executions. This catches
issues caused by chDB's parallel execution that may be hidden with smaller datasets.

Test scenarios:
1. filter -> assign -> sort -> head: repeated 5x for consistency
2. dropna (non-contiguous index) -> assign + groupby: row order correctness
3. rank(method='first') on large data with ties: tie-breaking matches pandas
4. PythonTableFunction path vs direct execution path: result comparison
"""

import unittest
import numpy as np
import pandas as pd
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestFilterAssignSortHeadDeterminism(unittest.TestCase):
    """Verify filter -> assign -> sort -> head produces identical results on repeated runs."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n = 100_000
        cls.data = {
            "id": np.arange(cls.n),
            "value": np.random.randn(cls.n),
            "category": np.random.choice(["A", "B", "C", "D", "E"], size=cls.n),
            "score": np.random.randint(0, 100, size=cls.n).astype(float),
        }

    def test_filter_assign_sort_head_repeated_5x(self):
        """Run filter -> assign -> sort_values -> head 5 times, verify all results identical."""
        pd_df = pd.DataFrame(self.data)

        # Compute expected result once with pandas
        pd_filtered = pd_df[pd_df["value"] > 0]
        pd_assigned = pd_filtered.assign(
            weighted=pd_filtered["value"] * pd_filtered["score"]
        )
        # Use kind='stable' so pandas tie-breaking matches chDB's stable ORDER BY
        pd_sorted = pd_assigned.sort_values("weighted", ascending=False, kind="stable")
        pd_result = pd_sorted.head(1000)

        for i in range(5):
            ds_df = DataStore(self.data)
            ds_filtered = ds_df[ds_df["value"] > 0]
            ds_assigned = ds_filtered.assign(
                weighted=ds_filtered["value"] * ds_filtered["score"]
            )
            ds_sorted = ds_assigned.sort_values("weighted", ascending=False, kind="stable")
            ds_result = ds_sorted.head(1000)

            assert_datastore_equals_pandas(
                ds_result, pd_result,
                msg=f"Iteration {i}: filter->assign->sort->head result differs"
            )

    def test_filter_assign_sort_head_stable_sort_repeated_5x(self):
        """Same chain but with kind='stable' sort, repeated 5 times."""
        pd_df = pd.DataFrame(self.data)

        pd_filtered = pd_df[pd_df["score"] > 50]
        pd_assigned = pd_filtered.assign(
            combo=pd_filtered["value"] + pd_filtered["score"]
        )
        pd_sorted = pd_assigned.sort_values("combo", ascending=True, kind="stable")
        pd_result = pd_sorted.head(2000)

        for i in range(5):
            ds_df = DataStore(self.data)
            ds_filtered = ds_df[ds_df["score"] > 50]
            ds_assigned = ds_filtered.assign(
                combo=ds_filtered["value"] + ds_filtered["score"]
            )
            ds_sorted = ds_assigned.sort_values("combo", ascending=True, kind="stable")
            ds_result = ds_sorted.head(2000)

            assert_datastore_equals_pandas(
                ds_result, pd_result,
                msg=f"Iteration {i}: stable sort chain differs"
            )

    def test_multi_column_sort_determinism(self):
        """Sort by multiple columns with ties, verify deterministic order across 5 runs."""
        np.random.seed(99)
        n = 100_000
        # Intentionally create many ties in category and bucket
        data = {
            "category": np.random.choice(["X", "Y", "Z"], size=n),
            "bucket": np.random.choice([1, 2, 3, 4, 5], size=n),
            "amount": np.random.randn(n),
        }
        pd_df = pd.DataFrame(data)

        pd_filtered = pd_df[pd_df["amount"] > -0.5]
        pd_sorted = pd_filtered.sort_values(
            ["category", "bucket", "amount"], ascending=[True, False, True],
            kind="stable"
        )
        pd_result = pd_sorted.head(5000)

        for i in range(5):
            ds_df = DataStore(data)
            ds_filtered = ds_df[ds_df["amount"] > -0.5]
            ds_sorted = ds_filtered.sort_values(
                ["category", "bucket", "amount"], ascending=[True, False, True],
                kind="stable"
            )
            ds_result = ds_sorted.head(5000)

            assert_datastore_equals_pandas(
                ds_result, pd_result,
                msg=f"Iteration {i}: multi-column sort determinism differs"
            )


class TestDropnaAssignGroupbyRowOrder(unittest.TestCase):
    """Verify row order after dropna (non-contiguous index) + assign + groupby."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n = 100_000
        # ~20% NaN in key column to create non-contiguous index after dropna
        values = np.random.randn(cls.n)
        mask = np.random.random(cls.n) < 0.2
        values[mask] = np.nan

        cls.data = {
            "key": values,
            "group": np.random.choice(["alpha", "beta", "gamma", "delta"], size=cls.n),
            "metric": np.random.randint(1, 100, size=cls.n).astype(float),
        }

    def test_dropna_assign_groupby_sum(self):
        """dropna -> assign -> groupby -> sum: verify grouped results match pandas."""
        pd_df = pd.DataFrame(self.data)
        pd_dropped = pd_df.dropna(subset=["key"])
        pd_assigned = pd_dropped.assign(
            product=pd_dropped["key"] * pd_dropped["metric"]
        )
        pd_result = pd_assigned.groupby("group")["product"].sum()

        ds_df = DataStore(self.data)
        ds_dropped = ds_df.dropna(subset=["key"])
        ds_assigned = ds_dropped.assign(
            product=ds_dropped["key"] * ds_dropped["metric"]
        )
        ds_result = ds_assigned.groupby("group")["product"].sum()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_dropna_assign_groupby_agg_multiple(self):
        """dropna -> assign -> groupby -> agg with multiple functions.

        Uses Series-level groupby to avoid reset_index column name issues.
        """
        pd_df = pd.DataFrame(self.data)
        pd_dropped = pd_df.dropna(subset=["key"])
        pd_assigned = pd_dropped.assign(
            scaled_key=pd_dropped["key"] * 10
        )

        ds_df = DataStore(self.data)
        ds_dropped = ds_df.dropna(subset=["key"])
        ds_assigned = ds_dropped.assign(
            scaled_key=ds_dropped["key"] * 10
        )

        # Compare individual aggregations to avoid reset_index column name issues
        pd_mean = pd_assigned.groupby("group")["scaled_key"].mean()
        ds_mean = ds_assigned.groupby("group")["scaled_key"].mean()
        assert_series_equal(ds_mean, pd_mean, check_dtype=False)

        pd_sum = pd_assigned.groupby("group")["metric"].sum()
        ds_sum = ds_assigned.groupby("group")["metric"].sum()
        assert_series_equal(ds_sum, pd_sum, check_dtype=False)

        pd_count = pd_assigned.groupby("group")["metric"].count()
        ds_count = ds_assigned.groupby("group")["metric"].count()
        assert_series_equal(ds_count, pd_count, check_dtype=False)

    def test_dropna_assign_preserves_row_alignment(self):
        """After dropna, assigned column values must align with original rows.

        Verifies that assign on a non-contiguous-index DataFrame doesn't
        shuffle row data - each row's assigned value must correspond to
        the correct original row values.
        """
        pd_df = pd.DataFrame(self.data)
        pd_dropped = pd_df.dropna(subset=["key"])
        pd_result = pd_dropped.assign(
            sum_val=pd_dropped["key"] + pd_dropped["metric"]
        )

        ds_df = DataStore(self.data)
        ds_dropped = ds_df.dropna(subset=["key"])
        ds_result = ds_dropped.assign(
            sum_val=ds_dropped["key"] + ds_dropped["metric"]
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_multiple_assigns_then_groupby(self):
        """dropna -> multiple assigns -> groupby: stress test row alignment.

        Uses Series-level groupby to avoid reset_index column name issues.
        """
        pd_df = pd.DataFrame(self.data)
        pd_dropped = pd_df.dropna(subset=["key"])
        pd_assigned = pd_dropped.assign(
            abs_key=pd_dropped["key"].abs(),
            metric_sq=pd_dropped["metric"] ** 2,
            combo=pd_dropped["key"] * pd_dropped["metric"],
        )

        ds_df = DataStore(self.data)
        ds_dropped = ds_df.dropna(subset=["key"])
        ds_assigned = ds_dropped.assign(
            abs_key=ds_dropped["key"].abs(),
            metric_sq=ds_dropped["metric"] ** 2,
            combo=ds_dropped["key"] * ds_dropped["metric"],
        )

        pd_abs_sum = pd_assigned.groupby("group")["abs_key"].sum()
        ds_abs_sum = ds_assigned.groupby("group")["abs_key"].sum()
        assert_series_equal(ds_abs_sum, pd_abs_sum, check_dtype=False)

        pd_metric_mean = pd_assigned.groupby("group")["metric_sq"].mean()
        ds_metric_mean = ds_assigned.groupby("group")["metric_sq"].mean()
        assert_series_equal(ds_metric_mean, pd_metric_mean, check_dtype=False)


class TestLargeDataRankDeterminism(unittest.TestCase):
    """Verify rank(method='first') on large data with ties matches pandas exactly."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n = 100_000
        # Create data with many ties: only 10 distinct values for 100k rows
        cls.data_with_ties = {
            "val": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0,
                                     6.0, 7.0, 8.0, 9.0, 10.0], size=cls.n),
            "group": np.random.choice(["A", "B", "C"], size=cls.n),
        }
        # Mix of unique and tied values
        cls.data_mixed = {
            "score": np.concatenate([
                np.random.randn(cls.n // 2),        # unique values
                np.repeat([1.0, 2.0, 3.0], cls.n // 6),  # tied values
                np.random.randn(cls.n - cls.n // 2 - 3 * (cls.n // 6)),  # fill remainder
            ]),
        }

    def test_rank_method_first_with_heavy_ties(self):
        """rank(method='first') must break ties by row position, matching pandas."""
        pd_df = pd.DataFrame(self.data_with_ties)
        ds_df = DataStore(self.data_with_ties)

        pd_result = pd_df["val"].rank(method="first")
        ds_result = ds_df["val"].rank(method="first")

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rank_method_first_mixed_ties(self):
        """rank(method='first') with a mix of unique and tied values."""
        pd_df = pd.DataFrame(self.data_mixed)
        ds_df = DataStore(self.data_mixed)

        pd_result = pd_df["score"].rank(method="first")
        ds_result = ds_df["score"].rank(method="first")

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rank_method_first_ascending_false(self):
        """rank(method='first', ascending=False) with ties on large data."""
        pd_df = pd.DataFrame(self.data_with_ties)
        ds_df = DataStore(self.data_with_ties)

        pd_result = pd_df["val"].rank(method="first", ascending=False)
        ds_result = ds_df["val"].rank(method="first", ascending=False)

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_dataframe_rank_method_first(self):
        """DataFrame-level rank(method='first') on large data with ties."""
        pd_df = pd.DataFrame(self.data_with_ties)
        ds_df = DataStore(self.data_with_ties)

        pd_result = pd_df[["val"]].rank(method="first")
        ds_result = ds_df[["val"]].rank(method="first")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_after_filter_preserves_tie_order(self):
        """Filter then rank(method='first') - row order must be preserved through filter."""
        pd_df = pd.DataFrame(self.data_with_ties)
        ds_df = DataStore(self.data_with_ties)

        pd_filtered = pd_df[pd_df["group"] == "A"]
        ds_filtered = ds_df[ds_df["group"] == "A"]

        pd_result = pd_filtered["val"].rank(method="first")
        ds_result = ds_filtered["val"].rank(method="first")

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rank_determinism_repeated_5x(self):
        """Run rank(method='first') 5 times, verify all results identical."""
        pd_df = pd.DataFrame(self.data_with_ties)
        pd_expected = pd_df["val"].rank(method="first")

        for i in range(5):
            ds_df = DataStore(self.data_with_ties)
            ds_result = ds_df["val"].rank(method="first")

            assert_series_equal(
                ds_result, pd_expected, check_dtype=False,
                obj=f"Iteration {i}"
            )


class TestPythonTableFunctionVsDirectExecution(unittest.TestCase):
    """Compare PythonTableFunction path vs direct execution path on large data.

    After operations like dropna/rank that execute via pandas, the result
    gets wrapped back into a DataStore with a PythonTableFunction source.
    Subsequent SQL operations on that DataStore go through the PythonTableFunction
    path. This test verifies that this path produces the same results as
    direct pandas execution.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n = 100_000
        cls.data = {
            "id": np.arange(cls.n),
            "value": np.random.randn(cls.n),
            "category": np.random.choice(["A", "B", "C", "D"], size=cls.n),
            "amount": np.random.randint(1, 1000, size=cls.n).astype(float),
        }

    def test_dropna_then_sql_filter(self):
        """dropna (pandas path) -> filter (SQL path via PythonTableFunction).

        After dropna executes via pandas, the result is wrapped with a
        PythonTableFunction. The subsequent filter should work correctly
        through this path.
        """
        # Add some NaN values
        data = {k: v.copy() for k, v in self.data.items()}
        mask = np.random.random(self.n) < 0.15
        data["value"] = data["value"].astype(float)
        data["value"][mask] = np.nan

        pd_df = pd.DataFrame(data)
        pd_result = pd_df.dropna(subset=["value"])
        pd_result = pd_result[pd_result["value"] > 0]

        ds_df = DataStore(data)
        ds_result = ds_df.dropna(subset=["value"])
        ds_result = ds_result[ds_result["value"] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_then_sort_then_head(self):
        """dropna (pandas) -> sort_values (SQL via PythonTableFunction) -> head."""
        data = {k: v.copy() for k, v in self.data.items()}
        mask = np.random.random(self.n) < 0.15
        data["value"] = data["value"].astype(float)
        data["value"][mask] = np.nan

        pd_df = pd.DataFrame(data)
        pd_result = pd_df.dropna(subset=["value"])
        pd_result = pd_result.sort_values("value", ascending=False)
        pd_result = pd_result.head(500)

        ds_df = DataStore(data)
        ds_result = ds_df.dropna(subset=["value"])
        ds_result = ds_result.sort_values("value", ascending=False)
        ds_result = ds_result.head(500)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_then_filter_then_rank(self):
        """dropna (pandas) -> filter -> rank: chained pandas-path operations."""
        data = {k: v.copy() for k, v in self.data.items()}
        mask = np.random.random(self.n) < 0.15
        data["value"] = data["value"].astype(float)
        data["value"][mask] = np.nan

        pd_df = pd.DataFrame(data)
        pd_dropped = pd_df.dropna(subset=["value"])
        pd_filtered = pd_dropped[pd_dropped["value"] > 0]
        pd_result = pd_filtered["amount"].rank(method="first")

        ds_df = DataStore(data)
        ds_dropped = ds_df.dropna(subset=["value"])
        ds_filtered = ds_dropped[ds_dropped["value"] > 0]
        ds_result = ds_filtered["amount"].rank(method="first")

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_direct_vs_ptf_filter_assign_chain(self):
        """Compare direct DataStore chain vs one that forces PythonTableFunction intermediate.

        Path A (direct): DataStore -> filter -> assign -> sort(stable) -> head
        Path B (PTF intermediate): DataStore -> dropna(no-op) -> filter -> assign -> sort(stable) -> head

        Both paths should produce identical results. dropna on data without NaN
        is effectively a no-op but forces execution through PythonTableFunction.
        """
        # Use data without NaN so dropna is effectively a no-op
        pd_df = pd.DataFrame(self.data)

        # Path A: direct
        pd_a = pd_df[pd_df["value"] > 0]
        pd_a = pd_a.assign(doubled=pd_a["amount"] * 2)
        pd_a = pd_a.sort_values("doubled", ascending=False, kind="stable")
        pd_result_a = pd_a.head(1000)

        ds_a = DataStore(self.data)
        ds_a = ds_a[ds_a["value"] > 0]
        ds_a = ds_a.assign(doubled=ds_a["amount"] * 2)
        ds_a = ds_a.sort_values("doubled", ascending=False, kind="stable")
        ds_result_a = ds_a.head(1000)

        assert_datastore_equals_pandas(
            ds_result_a, pd_result_a,
            msg="Direct path"
        )

        # Path B: force PythonTableFunction via dropna (no NaN in data => no-op)
        pd_b = pd_df.dropna()  # no-op: no NaN values
        pd_b = pd_b[pd_b["value"] > 0]
        pd_b = pd_b.assign(doubled=pd_b["amount"] * 2)
        pd_b = pd_b.sort_values("doubled", ascending=False, kind="stable")
        pd_result_b = pd_b.head(1000)

        ds_b = DataStore(self.data)
        ds_b = ds_b.dropna()  # forces PythonTableFunction path
        ds_b = ds_b[ds_b["value"] > 0]
        ds_b = ds_b.assign(doubled=ds_b["amount"] * 2)
        ds_b = ds_b.sort_values("doubled", ascending=False, kind="stable")
        ds_result_b = ds_b.head(1000)

        assert_datastore_equals_pandas(
            ds_result_b, pd_result_b,
            msg="PythonTableFunction path"
        )

        # Both pandas paths produce identical results
        pd.testing.assert_frame_equal(
            pd_result_a.reset_index(drop=True),
            pd_result_b.reset_index(drop=True),
            obj="Pandas paths A vs B should match"
        )

    def test_ptf_chain_multiple_pandas_ops(self):
        """Chain multiple pandas-path operations, each creating a PythonTableFunction.

        dropna -> sort(stable) -> head
        Each pandas-path op wraps result in new DataStore with PythonTableFunction.
        """
        data = {k: v.copy() for k, v in self.data.items()}
        mask = np.random.random(self.n) < 0.1
        data["value"] = data["value"].astype(float)
        data["value"][mask] = np.nan

        pd_df = pd.DataFrame(data)

        # Step 1: dropna (pandas path)
        pd_step1 = pd_df.dropna(subset=["value"])
        # Step 2: sort with stable kind to ensure deterministic tie-breaking, then head
        pd_step2 = pd_step1.sort_values("amount", ascending=False, kind="stable")
        pd_result = pd_step2.head(1000)

        ds_df = DataStore(data)
        ds_step1 = ds_df.dropna(subset=["value"])
        ds_step2 = ds_step1.sort_values("amount", ascending=False, kind="stable")
        ds_result = ds_step2.head(1000)

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == "__main__":
    unittest.main()
