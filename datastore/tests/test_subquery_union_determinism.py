"""
Tests for Subquery / UNION ALL result determinism (P1 Scenario 9).

Verifies that:
1. JSON accessor chains (extract + filter) produce deterministic results
2. Nested subqueries (filter referencing computed columns) produce correct results
3. concat of multiple DataStores produces order-consistent results matching pandas
4. Repeated execution of the same query yields identical results (non-flaky)

Root cause: UNION ALL in ClickHouse does not guarantee order. Tests that depend
on implicit ordering from UNION ALL data sources are inherently flaky. We verify
that DataStore handles this correctly by either:
- Using ORDER BY in the source query
- Preserving insertion order through the lazy execution engine
- Matching pandas behavior for concat operations
"""

import numpy as np
import pandas as pd
import pytest
import chdb

import datastore as ds
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


# =============================================================================
# Section 1: JSON Accessor Chain Determinism
# =============================================================================


class TestJsonAccessorChainDeterminism:
    """Verify JSON extract + filter chains produce deterministic, correct results."""

    @pytest.fixture
    def ds_json_ordered(self):
        """Create DataStore from UNION ALL query with explicit ORDER BY."""
        df = chdb.query(
            """
            SELECT * FROM (
                SELECT 1 as id, '{"name": "Alice", "score": 85}' as data
                UNION ALL
                SELECT 2 as id, '{"name": "Bob", "score": 45}' as data
                UNION ALL
                SELECT 3 as id, '{"name": "Charlie", "score": 95}' as data
                UNION ALL
                SELECT 4 as id, '{"name": "Diana", "score": 70}' as data
                UNION ALL
                SELECT 5 as id, '{"name": "Eve", "score": 60}' as data
            ) ORDER BY id
            """,
            "DataFrame",
        )
        return DataStore.from_df(df)

    def test_json_extract_filter_deterministic_5x(self, ds_json_ordered):
        """JSON extract + filter must produce identical results across 5 runs."""
        expected_ids = None
        expected_names = None

        for i in range(5):
            ds_copy = DataStore.from_df(ds_json_ordered.to_df())
            ds_copy["score"] = ds_copy["data"].json.json_extract_int("score")
            result = ds_copy[ds_copy["score"] > 60]
            result_df = result.to_df()

            ids = list(result_df["id"])
            # Use tolist() to get names from ColumnExpr
            names = list(result_df["data"].apply(
                lambda x: x.split('"name": "')[1].split('"')[0]
            ))

            if expected_ids is None:
                expected_ids = ids
                expected_names = names
                assert len(ids) == 3, f"Expected 3 rows (scores 85, 95, 70), got {len(ids)}"
                assert set(ids) == {1, 3, 4}
            else:
                assert ids == expected_ids, (
                    f"Run {i}: IDs differ. Expected {expected_ids}, got {ids}"
                )
                assert names == expected_names, (
                    f"Run {i}: Names differ. Expected {expected_names}, got {names}"
                )

    def test_json_extract_multiple_fields_filter(self, ds_json_ordered):
        """Extract multiple JSON fields then filter - results must be deterministic."""
        ds_json_ordered["name"] = ds_json_ordered["data"].json.json_extract_string("name")
        ds_json_ordered["score"] = ds_json_ordered["data"].json.json_extract_int("score")

        result = ds_json_ordered[ds_json_ordered["score"] >= 70]
        result_df = result.to_df()

        assert len(result_df) == 3
        assert list(result_df["id"]) == [1, 3, 4]
        assert list(result_df["name"]) == ["Alice", "Charlie", "Diana"]

    def test_json_extract_assign_sort(self, ds_json_ordered):
        """JSON extract -> assign -> sort must match expected order."""
        ds_json_ordered["score"] = ds_json_ordered["data"].json.json_extract_int("score")
        result = ds_json_ordered.sort_values("score", ascending=False)
        result_df = result.to_df()

        assert list(result_df["id"]) == [3, 1, 4, 5, 2]


# =============================================================================
# Section 2: Nested Subquery Determinism
# =============================================================================


class TestNestedSubqueryDeterminism:
    """Verify nested subqueries (filter on computed column) produce correct results."""

    def test_filter_on_computed_column(self):
        """Filter referencing a computed column should produce correct results."""
        data = {
            "id": [1, 2, 3, 4, 5],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "quantity": [5, 3, 2, 4, 1],
        }

        pd_df = pd.DataFrame(data)
        pd_df["total"] = pd_df["price"] * pd_df["quantity"]
        pd_result = pd_df[pd_df["total"] > 60]

        ds_df = DataStore(data)
        ds_df["total"] = ds_df["price"] * ds_df["quantity"]
        ds_result = ds_df[ds_df["total"] > 60]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_on_computed_column_repeated_5x(self):
        """Filter on computed column must be deterministic across 5 runs."""
        data = {
            "id": list(range(100)),
            "val_a": list(range(0, 200, 2)),
            "val_b": list(range(100, 0, -1)),
        }

        pd_df = pd.DataFrame(data)
        pd_df["combined"] = pd_df["val_a"] + pd_df["val_b"]
        pd_result = pd_df[pd_df["combined"] > 150]

        for i in range(5):
            ds_df = DataStore(data)
            ds_df["combined"] = ds_df["val_a"] + ds_df["val_b"]
            ds_result = ds_df[ds_df["combined"] > 150]

            assert_datastore_equals_pandas(
                ds_result, pd_result,
                msg=f"Iteration {i}: filter on computed column differs"
            )

    def test_chained_filters_on_computed_columns(self):
        """Multiple filters on different computed columns - nested subquery pattern."""
        data = {
            "id": list(range(1, 21)),
            "x": [float(i) for i in range(20)],
            "y": [float(i * 2) for i in range(20)],
        }

        pd_df = pd.DataFrame(data)
        pd_df["sum_xy"] = pd_df["x"] + pd_df["y"]
        pd_df["diff_xy"] = pd_df["x"] - pd_df["y"]
        pd_filtered = pd_df[pd_df["sum_xy"] > 20]
        pd_result = pd_filtered[pd_filtered["diff_xy"] < -5]

        ds_df = DataStore(data)
        ds_df["sum_xy"] = ds_df["x"] + ds_df["y"]
        ds_df["diff_xy"] = ds_df["x"] - ds_df["y"]
        ds_filtered = ds_df[ds_df["sum_xy"] > 20]
        ds_result = ds_filtered[ds_filtered["diff_xy"] < -5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_computed_column_then_sort(self):
        """Assign computed column -> filter -> sort: subquery chain determinism."""
        data = {
            "name": ["alpha", "beta", "gamma", "delta", "epsilon",
                     "zeta", "eta", "theta", "iota", "kappa"],
            "score": [80, 45, 92, 67, 55, 88, 33, 71, 99, 60],
            "weight": [1.2, 0.8, 1.5, 1.0, 0.9, 1.3, 0.7, 1.1, 1.6, 0.85],
        }

        pd_df = pd.DataFrame(data)
        pd_df["weighted_score"] = pd_df["score"] * pd_df["weight"]
        pd_filtered = pd_df[pd_df["weighted_score"] > 70]
        pd_result = pd_filtered.sort_values("weighted_score", ascending=False)

        ds_df = DataStore(data)
        ds_df["weighted_score"] = ds_df["score"] * ds_df["weight"]
        ds_filtered = ds_df[ds_df["weighted_score"] > 70]
        ds_result = ds_filtered.sort_values("weighted_score", ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 3: Concat Determinism (Multiple DataStores)
# =============================================================================


class TestConcatDeterminism:
    """Verify concat of multiple DataStores matches pandas order and values."""

    def test_concat_two_datastores_order_matches_pandas(self):
        """Concat two DataStores: result order must match pd.concat."""
        data1 = {"id": [1, 2, 3], "val": ["a", "b", "c"]}
        data2 = {"id": [4, 5, 6], "val": ["d", "e", "f"]}

        pd_result = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )

        ds_result = ds.concat(
            [DataStore(data1), DataStore(data2)], ignore_index=True
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_three_datastores_order_matches_pandas(self):
        """Concat three DataStores: result order must match pd.concat."""
        data1 = {"id": [1, 2], "val": [10, 20]}
        data2 = {"id": [3, 4], "val": [30, 40]}
        data3 = {"id": [5, 6], "val": [50, 60]}

        pd_result = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2), pd.DataFrame(data3)],
            ignore_index=True,
        )

        ds_result = ds.concat(
            [DataStore(data1), DataStore(data2), DataStore(data3)],
            ignore_index=True,
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_then_filter_deterministic(self):
        """Concat -> filter: result must match pandas exactly."""
        data1 = {"id": [1, 2, 3], "score": [80, 45, 92]}
        data2 = {"id": [4, 5, 6], "score": [67, 55, 88]}

        pd_df = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )
        pd_result = pd_df[pd_df["score"] > 60]

        ds_df = ds.concat(
            [DataStore(data1), DataStore(data2)], ignore_index=True
        )
        ds_result = ds_df[ds_df["score"] > 60]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_then_sort_deterministic(self):
        """Concat -> sort: result order must match pandas."""
        data1 = {"name": ["Charlie", "Alice"], "age": [30, 25]}
        data2 = {"name": ["Bob", "Diana"], "age": [28, 35]}

        pd_df = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )
        pd_result = pd_df.sort_values("age")

        ds_df = ds.concat(
            [DataStore(data1), DataStore(data2)], ignore_index=True
        )
        ds_result = ds_df.sort_values("age")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_then_assign_filter_sort(self):
        """Concat -> assign -> filter -> sort: full chain determinism."""
        data1 = {"x": [1, 2, 3], "y": [10, 20, 30]}
        data2 = {"x": [4, 5, 6], "y": [40, 50, 60]}
        data3 = {"x": [7, 8, 9], "y": [70, 80, 90]}

        pd_df = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2), pd.DataFrame(data3)],
            ignore_index=True,
        )
        pd_df["product"] = pd_df["x"] * pd_df["y"]
        pd_filtered = pd_df[pd_df["product"] > 100]
        pd_result = pd_filtered.sort_values("product")

        ds_df = ds.concat(
            [DataStore(data1), DataStore(data2), DataStore(data3)],
            ignore_index=True,
        )
        ds_df["product"] = ds_df["x"] * ds_df["y"]
        ds_filtered = ds_df[ds_df["product"] > 100]
        ds_result = ds_filtered.sort_values("product")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_repeated_5x(self):
        """Concat executed 5 times must produce identical results each time."""
        data1 = {"a": [10, 20, 30], "b": ["x", "y", "z"]}
        data2 = {"a": [40, 50, 60], "b": ["p", "q", "r"]}

        pd_result = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )

        for i in range(5):
            ds_result = ds.concat(
                [DataStore(data1), DataStore(data2)], ignore_index=True
            )
            assert_datastore_equals_pandas(
                ds_result, pd_result,
                msg=f"Iteration {i}: concat result differs"
            )

    def test_concat_overlapping_data(self):
        """Concat DataStores with overlapping/duplicate rows matches pandas."""
        data1 = {"id": [1, 2, 3], "val": [100, 200, 300]}
        data2 = {"id": [2, 3, 4], "val": [200, 300, 400]}

        pd_result = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )

        ds_result = ds.concat(
            [DataStore(data1), DataStore(data2)], ignore_index=True
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_different_column_order(self):
        """Concat DataStores with same columns in different order."""
        data1 = {"a": [1, 2], "b": [3, 4]}
        data2 = {"b": [5, 6], "a": [7, 8]}

        pd_result = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )

        ds_result = ds.concat(
            [DataStore(data1), DataStore(data2)], ignore_index=True
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 4: Larger Scale Concat Determinism
# =============================================================================


class TestLargeScaleConcatDeterminism:
    """Test concat determinism with larger datasets to catch parallel execution issues."""

    def test_concat_large_datasets_order(self):
        """Concat two 1000-row DataStores: row order must match pandas."""
        np.random.seed(42)
        data1 = {
            "id": list(range(1000)),
            "value": np.random.randn(1000).tolist(),
            "category": np.random.choice(["A", "B", "C"], size=1000).tolist(),
        }
        data2 = {
            "id": list(range(1000, 2000)),
            "value": np.random.randn(1000).tolist(),
            "category": np.random.choice(["A", "B", "C"], size=1000).tolist(),
        }

        pd_result = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )

        ds_result = ds.concat(
            [DataStore(data1), DataStore(data2)], ignore_index=True
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_large_then_filter_sort(self):
        """Concat large DataStores -> filter -> sort: full chain determinism."""
        np.random.seed(123)
        data1 = {
            "id": list(range(500)),
            "score": np.random.randint(0, 100, size=500).tolist(),
        }
        data2 = {
            "id": list(range(500, 1000)),
            "score": np.random.randint(0, 100, size=500).tolist(),
        }

        pd_df = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )
        pd_filtered = pd_df[pd_df["score"] > 80]
        pd_result = pd_filtered.sort_values("score", ascending=False, kind="stable")

        ds_df = ds.concat(
            [DataStore(data1), DataStore(data2)], ignore_index=True
        )
        ds_filtered = ds_df[ds_df["score"] > 80]
        ds_result = ds_filtered.sort_values("score", ascending=False, kind="stable")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_large_repeated_3x(self):
        """Large concat repeated 3 times must produce identical results."""
        np.random.seed(77)
        data1 = {
            "x": np.random.randn(500).tolist(),
            "y": np.random.choice(["p", "q", "r", "s"], size=500).tolist(),
        }
        data2 = {
            "x": np.random.randn(500).tolist(),
            "y": np.random.choice(["p", "q", "r", "s"], size=500).tolist(),
        }

        pd_result = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )

        for i in range(3):
            ds_result = ds.concat(
                [DataStore(data1), DataStore(data2)], ignore_index=True
            )
            assert_datastore_equals_pandas(
                ds_result, pd_result,
                msg=f"Iteration {i}: large concat result differs"
            )


# =============================================================================
# Section 5: UNION ALL Source Data Determinism
# =============================================================================


class TestUnionAllSourceDeterminism:
    """Verify DataStore operations on data sourced from UNION ALL queries."""

    def test_union_all_source_with_order_by(self):
        """UNION ALL with ORDER BY: subsequent operations must be deterministic."""
        df = chdb.query(
            """
            SELECT * FROM (
                SELECT 1 as id, 'alpha' as name, 100 as score
                UNION ALL
                SELECT 2 as id, 'beta' as name, 200 as score
                UNION ALL
                SELECT 3 as id, 'gamma' as name, 150 as score
                UNION ALL
                SELECT 4 as id, 'delta' as name, 300 as score
                UNION ALL
                SELECT 5 as id, 'epsilon' as name, 250 as score
            ) ORDER BY id
            """,
            "DataFrame",
        )

        # Derive expected pandas DataFrame from the same chdb output to match dtypes
        pd_df = df.copy()
        ds_df = DataStore.from_df(df)

        # Filter
        pd_result = pd_df[pd_df["score"] > 150]
        ds_result = ds_df[ds_df["score"] > 150]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_union_all_source_repeated_query_5x(self):
        """Same UNION ALL query executed 5 times must produce identical DataStore results."""
        query = """
            SELECT * FROM (
                SELECT 1 as id, 10 as val
                UNION ALL
                SELECT 2 as id, 20 as val
                UNION ALL
                SELECT 3 as id, 30 as val
                UNION ALL
                SELECT 4 as id, 40 as val
            ) ORDER BY id
        """

        expected_df = None
        for i in range(5):
            df = chdb.query(query, "DataFrame")
            ds_df = DataStore.from_df(df)
            result = ds_df[ds_df["val"] > 15]
            result_df = result.to_df()

            if expected_df is None:
                expected_df = result_df
            else:
                pd.testing.assert_frame_equal(
                    result_df.reset_index(drop=True),
                    expected_df.reset_index(drop=True),
                    obj=f"Run {i} vs Run 0",
                )

    def test_union_all_with_computed_column_filter(self):
        """UNION ALL source -> compute column -> filter: deterministic."""
        df = chdb.query(
            """
            SELECT * FROM (
                SELECT 1 as id, 10.0 as price, 5 as qty
                UNION ALL
                SELECT 2 as id, 20.0 as price, 3 as qty
                UNION ALL
                SELECT 3 as id, 30.0 as price, 2 as qty
                UNION ALL
                SELECT 4 as id, 5.0 as price, 10 as qty
            ) ORDER BY id
            """,
            "DataFrame",
        )

        # Derive expected from same chdb output to match dtypes
        pd_df = df.copy()
        ds_df = DataStore.from_df(df)

        pd_df["total"] = pd_df["price"] * pd_df["qty"]
        ds_df["total"] = ds_df["price"] * ds_df["qty"]

        pd_result = pd_df[pd_df["total"] > 55]
        ds_result = ds_df[ds_df["total"] > 55]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_union_all_filter_then_sort_deterministic(self):
        """UNION ALL source -> filter -> sort: order must be deterministic."""
        df = chdb.query(
            """
            SELECT * FROM (
                SELECT 1 as id, 'alice' as name, 85 as score
                UNION ALL
                SELECT 2 as id, 'bob' as name, 45 as score
                UNION ALL
                SELECT 3 as id, 'charlie' as name, 95 as score
                UNION ALL
                SELECT 4 as id, 'diana' as name, 70 as score
                UNION ALL
                SELECT 5 as id, 'eve' as name, 60 as score
                UNION ALL
                SELECT 6 as id, 'frank' as name, 88 as score
            ) ORDER BY id
            """,
            "DataFrame",
        )

        pd_df = df.copy()
        ds_df = DataStore.from_df(df)

        pd_filtered = pd_df[pd_df["score"] > 50]
        pd_result = pd_filtered.sort_values("score", ascending=False)

        ds_filtered = ds_df[ds_df["score"] > 50]
        ds_result = ds_filtered.sort_values("score", ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_union_all_multiple_operations_chain(self):
        """UNION ALL source -> filter -> sort -> head: multi-step chain."""
        df = chdb.query(
            """
            SELECT * FROM (
                SELECT 1 as id, 10 as a, 5 as b
                UNION ALL
                SELECT 2 as id, 20 as a, 15 as b
                UNION ALL
                SELECT 3 as id, 30 as a, 25 as b
                UNION ALL
                SELECT 4 as id, 40 as a, 35 as b
                UNION ALL
                SELECT 5 as id, 50 as a, 45 as b
                UNION ALL
                SELECT 6 as id, 60 as a, 55 as b
                UNION ALL
                SELECT 7 as id, 70 as a, 65 as b
                UNION ALL
                SELECT 8 as id, 80 as a, 75 as b
            ) ORDER BY id
            """,
            "DataFrame",
        )

        pd_df = df.copy()
        ds_df = DataStore.from_df(df)

        # Filter on source column, then sort and take head
        pd_filtered = pd_df[pd_df["a"] > 30]
        ds_filtered = ds_df[ds_df["a"] > 30]

        pd_result = pd_filtered.sort_values("a", ascending=False).head(3)
        ds_result = ds_filtered.sort_values("a", ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 6: Mixed Operations Determinism
# =============================================================================


class TestMixedOperationsDeterminism:
    """Test combinations of concat, compute, filter, sort for determinism."""

    def test_concat_assign_groupby_deterministic(self):
        """Concat -> assign -> groupby: aggregated results must match pandas."""
        data1 = {"grp": ["A", "B", "A"], "val": [10, 20, 30]}
        data2 = {"grp": ["B", "A", "B"], "val": [40, 50, 60]}

        pd_df = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        )
        pd_df["doubled"] = pd_df["val"] * 2
        pd_result = pd_df.groupby("grp")["doubled"].sum().reset_index()

        ds_df = ds.concat(
            [DataStore(data1), DataStore(data2)], ignore_index=True
        )
        ds_df["doubled"] = ds_df["val"] * 2
        ds_result = ds_df.groupby("grp")["doubled"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multi_step_chain_deterministic_5x(self):
        """Complex chain executed 5x must always produce identical results."""
        data = {
            "id": list(range(50)),
            "category": ["X", "Y", "Z"] * 16 + ["X", "Y"],
            "amount": [float(i * 7 % 100) for i in range(50)],
        }

        pd_df = pd.DataFrame(data)
        pd_df["adjusted"] = pd_df["amount"] * 1.1
        pd_filtered = pd_df[pd_df["adjusted"] > 50]
        pd_result = pd_filtered.sort_values("adjusted", ascending=False, kind="stable")

        for i in range(5):
            ds_df = DataStore(data)
            ds_df["adjusted"] = ds_df["amount"] * 1.1
            ds_filtered = ds_df[ds_df["adjusted"] > 50]
            ds_result = ds_filtered.sort_values("adjusted", ascending=False, kind="stable")

            assert_datastore_equals_pandas(
                ds_result, pd_result,
                msg=f"Iteration {i}: multi-step chain differs"
            )

    def test_concat_with_mixed_pandas_datastore(self):
        """Concat mixing pandas DataFrame and DataStore objects."""
        pd_df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df2 = DataStore({"a": [5, 6], "b": [7, 8]})

        pd_expected = pd.concat(
            [pd_df1, pd.DataFrame({"a": [5, 6], "b": [7, 8]})], ignore_index=True
        )

        ds_result = ds.concat([pd_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_expected)

    def test_concat_empty_and_nonempty(self):
        """Concat empty + non-empty DataStore: result order matches pandas."""
        data_empty = {"id": pd.array([], dtype="int64"), "val": pd.array([], dtype="float64")}
        data_full = {"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]}

        pd_result = pd.concat(
            [pd.DataFrame(data_empty), pd.DataFrame(data_full)], ignore_index=True
        )

        ds_result = ds.concat(
            [DataStore(data_empty), DataStore(data_full)], ignore_index=True
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_many_small_datastores(self):
        """Concat 10 small DataStores: order must match pandas."""
        dfs_pd = []
        dfs_ds = []
        for i in range(10):
            data = {"id": [i * 3 + j for j in range(3)], "val": [float(i * 3 + j) * 10 for j in range(3)]}
            dfs_pd.append(pd.DataFrame(data))
            dfs_ds.append(DataStore(data))

        pd_result = pd.concat(dfs_pd, ignore_index=True)
        ds_result = ds.concat(dfs_ds, ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_then_head(self):
        """Concat -> head: first N rows must match pandas."""
        data1 = {"x": [1, 2, 3, 4, 5]}
        data2 = {"x": [6, 7, 8, 9, 10]}

        pd_result = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2)], ignore_index=True
        ).head(7)

        ds_result = ds.concat(
            [DataStore(data1), DataStore(data2)], ignore_index=True
        ).head(7)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_axis1_deterministic(self):
        """Concat along axis=1: column order and values must match pandas."""
        data1 = {"a": [1, 2, 3]}
        data2 = {"b": [4, 5, 6]}
        data3 = {"c": [7, 8, 9]}

        pd_result = pd.concat(
            [pd.DataFrame(data1), pd.DataFrame(data2), pd.DataFrame(data3)], axis=1
        )

        ds_result = ds.concat(
            [DataStore(data1), DataStore(data2), DataStore(data3)], axis=1
        )

        assert_datastore_equals_pandas(ds_result, pd_result)
