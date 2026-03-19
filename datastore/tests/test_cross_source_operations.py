"""
CH-9: Cross-Source Operations (SQL Table + Local DataFrame)

Verifies correctness of mixed operations between SQL-backed DataStores
(simulating remote ClickHouse tables) and DataFrame-backed DataStores (local data).

Test scenarios:
1. merge between SQL-backed and local DataStore
2. Explicit materialization then concat
3. isin with local list on SQL-backed DataStore
4. Chained operations on cross-source results (subquery-like)
5. assign() on SQL result then filter

All tests use Mirror Code Pattern: compare DataStore results with pandas results.
"""

import pandas as pd
import numpy as np
import pytest

from datastore import DataStore
import datastore as ds_module
from tests.test_utils import assert_datastore_equals_pandas


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sql_connection():
    """Shared chDB connection for creating SQL tables."""
    ds = DataStore(database=":memory:")
    ds.connect()
    return ds


@pytest.fixture
def sql_orders(sql_connection):
    """SQL-backed DataStore simulating a remote orders table."""
    sql_connection._connection.execute("""
        CREATE TABLE IF NOT EXISTS cross_orders (
            id UInt32, customer_id UInt32, product String,
            amount Float64, region String
        ) ENGINE = Memory
    """)
    sql_connection._connection.execute("TRUNCATE TABLE cross_orders")
    sql_connection._connection.execute("""
        INSERT INTO cross_orders VALUES
        (1, 101, 'Widget', 29.99, 'East'),
        (2, 102, 'Gadget', 49.99, 'West'),
        (3, 101, 'Doohickey', 19.99, 'East'),
        (4, 103, 'Thingamajig', 99.99, 'North'),
        (5, 102, 'Whatchamacallit', 39.99, 'West'),
        (6, 104, 'Widget', 29.99, 'South')
    """)
    return DataStore(table="cross_orders", database=":memory:")


@pytest.fixture
def pd_orders():
    """Pandas mirror of the orders table."""
    return pd.DataFrame({
        "id": pd.array([1, 2, 3, 4, 5, 6], dtype="uint32"),
        "customer_id": pd.array([101, 102, 101, 103, 102, 104], dtype="uint32"),
        "product": ["Widget", "Gadget", "Doohickey", "Thingamajig",
                     "Whatchamacallit", "Widget"],
        "amount": [29.99, 49.99, 19.99, 99.99, 39.99, 29.99],
        "region": ["East", "West", "East", "North", "West", "South"],
    })


@pytest.fixture
def local_customers():
    """Local DataFrame DataStore with customer info."""
    return DataStore({
        "customer_id": pd.array([101, 102, 103, 105], dtype="uint32"),
        "name": ["Alice", "Bob", "Charlie", "Eve"],
        "tier": ["Gold", "Silver", "Bronze", "Gold"],
    })


@pytest.fixture
def pd_customers():
    """Pandas mirror of local customers."""
    return pd.DataFrame({
        "customer_id": pd.array([101, 102, 103, 105], dtype="uint32"),
        "name": ["Alice", "Bob", "Charlie", "Eve"],
        "tier": ["Gold", "Silver", "Bronze", "Gold"],
    })


# ===========================================================================
# 1. Merge between SQL-backed and local DataStore
# ===========================================================================

class TestCrossSourceMerge:
    """Test merge() between SQL table DataStore and DataFrame DataStore."""

    def test_merge_sql_with_local_inner(self, sql_orders, local_customers,
                                        pd_orders, pd_customers):
        ds_result = sql_orders.merge(local_customers, on="customer_id",
                                     how="inner")
        pd_result = pd_orders.merge(pd_customers, on="customer_id",
                                    how="inner")
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_local_with_sql_inner(self, sql_orders, local_customers,
                                        pd_orders, pd_customers):
        ds_result = local_customers.merge(sql_orders, on="customer_id",
                                          how="inner")
        pd_result = pd_customers.merge(pd_orders, on="customer_id",
                                       how="inner")
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_sql_with_local_left(self, sql_orders, local_customers,
                                       pd_orders, pd_customers):
        ds_result = sql_orders.merge(local_customers, on="customer_id",
                                     how="left")
        pd_result = pd_orders.merge(pd_customers, on="customer_id",
                                    how="left")
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_local_with_sql_left(self, sql_orders, local_customers,
                                       pd_orders, pd_customers):
        ds_result = local_customers.merge(sql_orders, on="customer_id",
                                          how="left")
        pd_result = pd_customers.merge(pd_orders, on="customer_id",
                                       how="left")
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_sql_with_local_right(self, sql_orders, local_customers,
                                        pd_orders, pd_customers):
        ds_result = sql_orders.merge(local_customers, on="customer_id",
                                     how="right")
        pd_result = pd_orders.merge(pd_customers, on="customer_id",
                                    how="right")
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_sql_with_local_outer(self, sql_orders, local_customers,
                                        pd_orders, pd_customers):
        ds_result = sql_orders.merge(local_customers, on="customer_id",
                                     how="outer")
        pd_result = pd_orders.merge(pd_customers, on="customer_id",
                                    how="outer")
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)


class TestCrossSourceMergeDataFrameOnly:
    """Test merge() between two DataFrame-backed DataStores."""

    def test_merge_inner(self):
        ds1 = DataStore({"key": [1, 2, 3, 4], "val": [10, 20, 30, 40]})
        ds2 = DataStore({"key": [2, 3, 5], "info": ["x", "y", "z"]})
        ds_result = ds1.merge(ds2, on="key", how="inner")

        pd1 = pd.DataFrame({"key": [1, 2, 3, 4], "val": [10, 20, 30, 40]})
        pd2 = pd.DataFrame({"key": [2, 3, 5], "info": ["x", "y", "z"]})
        pd_result = pd1.merge(pd2, on="key", how="inner")

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_no_matching_keys(self):
        ds1 = DataStore({"key": [1, 2, 3], "val": ["a", "b", "c"]})
        ds2 = DataStore({"key": [4, 5, 6], "info": ["x", "y", "z"]})
        ds_result = ds1.merge(ds2, on="key", how="inner")

        pd1 = pd.DataFrame({"key": [1, 2, 3], "val": ["a", "b", "c"]})
        pd2 = pd.DataFrame({"key": [4, 5, 6], "info": ["x", "y", "z"]})
        pd_result = pd1.merge(pd2, on="key", how="inner")

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)
        assert len(ds_result) == 0

    def test_merge_many_to_many(self):
        ds1 = DataStore({"key": [1, 1, 2, 2], "a": ["x", "y", "p", "q"]})
        ds2 = DataStore({"key": [1, 1, 2], "b": ["m", "n", "o"]})
        ds_result = ds1.merge(ds2, on="key", how="inner")

        pd1 = pd.DataFrame({"key": [1, 1, 2, 2], "a": ["x", "y", "p", "q"]})
        pd2 = pd.DataFrame({"key": [1, 1, 2], "b": ["m", "n", "o"]})
        pd_result = pd1.merge(pd2, on="key", how="inner")

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_left_on_right_on(self):
        ds1 = DataStore({"id": [1, 2, 3], "x": [10, 20, 30]})
        ds2 = DataStore({"ref_id": [1, 2, 4], "y": [100, 200, 400]})
        ds_result = ds1.merge(ds2, left_on="id", right_on="ref_id",
                              how="inner")

        pd1 = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
        pd2 = pd.DataFrame({"ref_id": [1, 2, 4], "y": [100, 200, 400]})
        pd_result = pd1.merge(pd2, left_on="id", right_on="ref_id",
                              how="inner")

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_with_nan_keys(self):
        ds1 = DataStore({"key": [1.0, 2.0, np.nan, 4.0],
                         "val": [10, 20, 30, 40]})
        ds2 = DataStore({"key": [1.0, 2.0, 5.0],
                         "info": ["x", "y", "z"]})
        ds_result = ds1.merge(ds2, on="key", how="left")

        pd1 = pd.DataFrame({"key": [1.0, 2.0, np.nan, 4.0],
                             "val": [10, 20, 30, 40]})
        pd2 = pd.DataFrame({"key": [1.0, 2.0, 5.0],
                             "info": ["x", "y", "z"]})
        pd_result = pd1.merge(pd2, on="key", how="left")

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)


# ===========================================================================
# 2. Explicit materialization then concat
# ===========================================================================

class TestCrossSourceConcat:
    """Test concat of DataStores from different sources."""

    def test_concat_two_datastores(self):
        ds1 = DataStore({"key": [1, 2, 3], "val": ["a", "b", "c"]})
        ds2 = DataStore({"key": [4, 5], "val": ["d", "e"]})
        ds_result = ds_module.concat([ds1, ds2], ignore_index=True)

        pd1 = pd.DataFrame({"key": [1, 2, 3], "val": ["a", "b", "c"]})
        pd2 = pd.DataFrame({"key": [4, 5], "val": ["d", "e"]})
        pd_result = pd.concat([pd1, pd2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_explicit_materialization(self, sql_orders, local_customers,
                                             pd_orders, pd_customers):
        """Concat after explicit to_df() (simulating remote + local)."""
        ds_result = pd.concat(
            [sql_orders[["customer_id"]].to_df(),
             local_customers[["customer_id"]].to_df()],
            ignore_index=True,
        )
        pd_result = pd.concat(
            [pd_orders[["customer_id"]],
             pd_customers[["customer_id"]]],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(ds_result, pd_result)

    def test_concat_different_columns(self):
        ds1 = DataStore({"a": [1, 2], "b": [3, 4]})
        ds2 = DataStore({"b": [5, 6], "c": [7, 8]})
        ds_result = ds_module.concat([ds1, ds2], ignore_index=True)

        pd1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        pd2 = pd.DataFrame({"b": [5, 6], "c": [7, 8]})
        pd_result = pd.concat([pd1, pd2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_three_datastores(self):
        ds1 = DataStore({"x": [1, 2]})
        ds2 = DataStore({"x": [3, 4]})
        ds3 = DataStore({"x": [5, 6]})
        ds_result = ds_module.concat([ds1, ds2, ds3], ignore_index=True)

        pd1 = pd.DataFrame({"x": [1, 2]})
        pd2 = pd.DataFrame({"x": [3, 4]})
        pd3 = pd.DataFrame({"x": [5, 6]})
        pd_result = pd.concat([pd1, pd2, pd3], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_mixed_datastore_and_dataframe(self):
        ds1 = DataStore({"x": [1, 2], "y": [10, 20]})
        pd2 = pd.DataFrame({"x": [3, 4], "y": [30, 40]})
        ds_result = ds_module.concat([ds1, pd2], ignore_index=True)

        pd1 = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
        pd_result = pd.concat([pd1, pd2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ===========================================================================
# 3. isin with local list on SQL-backed DataStore
# ===========================================================================

class TestCrossSourceIsin:
    """Test isin() with local Python lists on DataStore."""

    def test_isin_with_local_list_on_sql(self, sql_orders, pd_orders):
        local_ids = [101, 103]
        ds_result = sql_orders[sql_orders["customer_id"].isin(local_ids)]
        pd_result = pd_orders[pd_orders["customer_id"].isin(local_ids)]
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_isin_with_local_list_on_dataframe(self):
        ds = DataStore({"id": [1, 2, 3, 4, 5],
                        "val": ["a", "b", "c", "d", "e"]})
        local_ids = [2, 4]
        ds_result = ds[ds["id"].isin(local_ids)]

        pd_df = pd.DataFrame({"id": [1, 2, 3, 4, 5],
                               "val": ["a", "b", "c", "d", "e"]})
        pd_result = pd_df[pd_df["id"].isin(local_ids)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_empty_list(self, sql_orders, pd_orders):
        ds_result = sql_orders[sql_orders["customer_id"].isin([])]
        pd_result = pd_orders[pd_orders["customer_id"].isin([])]
        assert len(ds_result) == 0
        assert len(pd_result) == 0
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_isin_all_match(self):
        ds = DataStore({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        ds_result = ds[ds["id"].isin([1, 2, 3])]

        pd_df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        pd_result = pd_df[pd_df["id"].isin([1, 2, 3])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_with_strings(self, sql_orders, pd_orders):
        ds_result = sql_orders[
            sql_orders["region"].isin(["East", "West"])
        ]
        pd_result = pd_orders[pd_orders["region"].isin(["East", "West"])]
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_isin_no_match(self):
        ds = DataStore({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        ds_result = ds[ds["id"].isin([10, 20])]

        pd_df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        pd_result = pd_df[pd_df["id"].isin([10, 20])]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0


# ===========================================================================
# 4. Chained operations on cross-source results (subquery-like)
# ===========================================================================

class TestCrossSourceChainedOps:
    """Test chained operations after cross-source merge/join."""

    def test_merge_then_filter(self):
        ds1 = DataStore({"key": [1, 2, 3, 4], "val": [10, 20, 30, 40]})
        ds2 = DataStore({"key": [2, 3, 5], "info": ["x", "y", "z"]})
        ds_result = ds1.merge(ds2, on="key", how="inner")
        ds_result = ds_result[ds_result["val"] > 15]

        pd1 = pd.DataFrame({"key": [1, 2, 3, 4], "val": [10, 20, 30, 40]})
        pd2 = pd.DataFrame({"key": [2, 3, 5], "info": ["x", "y", "z"]})
        pd_result = pd1.merge(pd2, on="key", how="inner")
        pd_result = pd_result[pd_result["val"] > 15]

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_then_assign_then_filter(self):
        ds1 = DataStore({"key": [1, 2, 3], "x": [10, 20, 30]})
        ds2 = DataStore({"key": [1, 2, 3], "y": [100, 200, 300]})
        merged = ds1.merge(ds2, on="key")
        result = merged.assign(z=lambda df: df["x"] + df["y"])
        ds_result = result[result["z"] > 200]

        pd1 = pd.DataFrame({"key": [1, 2, 3], "x": [10, 20, 30]})
        pd2 = pd.DataFrame({"key": [1, 2, 3], "y": [100, 200, 300]})
        pd_merged = pd1.merge(pd2, on="key")
        pd_result = pd_merged.assign(z=lambda df: df["x"] + df["y"])
        pd_result = pd_result[pd_result["z"] > 200]

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_then_groupby_sum(self):
        ds1 = DataStore({"key": [1, 1, 2, 2, 3],
                         "val": [10, 20, 30, 40, 50]})
        ds2 = DataStore({"key": [1, 2, 3], "group": ["A", "B", "A"]})
        merged = ds1.merge(ds2, on="key")
        ds_result = merged.groupby("group")["val"].sum()

        pd1 = pd.DataFrame({"key": [1, 1, 2, 2, 3],
                             "val": [10, 20, 30, 40, 50]})
        pd2 = pd.DataFrame({"key": [1, 2, 3], "group": ["A", "B", "A"]})
        pd_merged = pd1.merge(pd2, on="key")
        pd_result = pd_merged.groupby("group")["val"].sum()

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_then_sort(self):
        ds1 = DataStore({"key": [3, 1, 2], "val": [30, 10, 20]})
        ds2 = DataStore({"key": [1, 2, 3], "info": ["a", "b", "c"]})
        ds_result = ds1.merge(ds2, on="key").sort_values("val")

        pd1 = pd.DataFrame({"key": [3, 1, 2], "val": [30, 10, 20]})
        pd2 = pd.DataFrame({"key": [1, 2, 3], "info": ["a", "b", "c"]})
        pd_result = pd1.merge(pd2, on="key").sort_values("val")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_double_merge_chain(self):
        ds1 = DataStore({"id": [1, 2, 3], "a": [10, 20, 30]})
        ds2 = DataStore({"id": [1, 2, 3], "b": [100, 200, 300]})
        ds3 = DataStore({"id": [1, 2, 3], "c": ["x", "y", "z"]})

        ds_result = ds1.merge(ds2, on="id").merge(ds3, on="id")

        pd1 = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        pd2 = pd.DataFrame({"id": [1, 2, 3], "b": [100, 200, 300]})
        pd3 = pd.DataFrame({"id": [1, 2, 3], "c": ["x", "y", "z"]})
        pd_result = pd1.merge(pd2, on="id").merge(pd3, on="id")

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_filter_then_merge_then_filter(self):
        ds1 = DataStore({"key": [1, 2, 3, 4, 5],
                         "val": [10, 20, 30, 40, 50]})
        ds2 = DataStore({"key": [2, 3, 4], "label": ["a", "b", "c"]})

        ds_filtered = ds1[ds1["val"] >= 20]
        merged = ds_filtered.merge(ds2, on="key", how="inner")
        ds_result = merged[merged["val"] <= 40]

        pd1 = pd.DataFrame({"key": [1, 2, 3, 4, 5],
                             "val": [10, 20, 30, 40, 50]})
        pd2 = pd.DataFrame({"key": [2, 3, 4], "label": ["a", "b", "c"]})

        pd_filtered = pd1[pd1["val"] >= 20]
        pd_merged = pd_filtered.merge(pd2, on="key", how="inner")
        pd_result = pd_merged[pd_merged["val"] <= 40]

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_isin_then_merge(self, sql_orders, local_customers,
                             pd_orders, pd_customers):
        """Filter SQL DataStore with isin, then merge with local."""
        target_ids = [101, 102]
        ds_filtered = sql_orders[
            sql_orders["customer_id"].isin(target_ids)
        ]
        ds_result = ds_filtered.merge(local_customers, on="customer_id",
                                      how="inner")

        pd_filtered = pd_orders[pd_orders["customer_id"].isin(target_ids)]
        pd_result = pd_filtered.merge(pd_customers, on="customer_id",
                                      how="inner")

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)


# ===========================================================================
# 5. assign() on remote/SQL result then filter
# ===========================================================================

class TestCrossSourceAssignFilter:
    """Test assign() adding computed columns, then filter."""

    def test_assign_sql_column_expr_then_filter(self, sql_orders, pd_orders):
        ds_result = sql_orders.assign(
            discounted=sql_orders["amount"] * 0.9
        )
        ds_result = ds_result[ds_result["discounted"] > 30]

        pd_result = pd_orders.assign(
            discounted=pd_orders["amount"] * 0.9
        )
        pd_result = pd_result[pd_result["discounted"] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_assign_lambda_then_filter(self):
        ds = DataStore({"id": [1, 2, 3], "x": [10, 20, 30],
                        "y": [1, 2, 3]})
        ds_result = ds.assign(z=lambda df: df["x"] + df["y"])
        ds_result = ds_result[ds_result["z"] > 15]

        pd_df = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30],
                               "y": [1, 2, 3]})
        pd_result = pd_df.assign(z=lambda df: df["x"] + df["y"])
        pd_result = pd_result[pd_result["z"] > 15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_on_merge_result(self):
        ds1 = DataStore({"id": [1, 2, 3], "price": [100.0, 200.0, 300.0]})
        ds2 = DataStore({"id": [1, 2, 3], "qty": [5, 3, 1]})
        merged = ds1.merge(ds2, on="id")
        ds_result = merged.assign(
            total=lambda df: df["price"] * df["qty"]
        )

        pd1 = pd.DataFrame({"id": [1, 2, 3],
                             "price": [100.0, 200.0, 300.0]})
        pd2 = pd.DataFrame({"id": [1, 2, 3], "qty": [5, 3, 1]})
        pd_merged = pd1.merge(pd2, on="id")
        pd_result = pd_merged.assign(
            total=lambda df: df["price"] * df["qty"]
        )

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_assign_scalar_then_filter(self, sql_orders, pd_orders):
        ds_result = sql_orders.assign(flag=1)
        ds_result = ds_result[ds_result["amount"] > 30]

        pd_result = pd_orders.assign(flag=1)
        pd_result = pd_result[pd_result["amount"] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_multiple_assigns_chained(self):
        ds = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds_result = (
            ds.assign(c=ds["a"] + ds["b"])
              .assign(d=lambda df: df["c"] * 2)
        )

        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        pd_result = (
            pd_df.assign(c=pd_df["a"] + pd_df["b"])
                 .assign(d=lambda df: df["c"] * 2)
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


# ===========================================================================
# 6. SQL table + local DataFrame join (using join method)
# ===========================================================================

class TestCrossSourceJoin:
    """Test join() between SQL-backed and DataFrame-backed DataStores."""

    def test_sql_join_local_inner(self, sql_orders, local_customers,
                                  pd_orders, pd_customers):
        ds_result = sql_orders.join(local_customers, on="customer_id",
                                    how="inner")
        pd_result = pd_orders.merge(pd_customers, on="customer_id",
                                    how="inner")
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_sql_join_local_left(self, sql_orders, local_customers,
                                 pd_orders, pd_customers):
        ds_result = sql_orders.join(local_customers, on="customer_id",
                                    how="left")
        pd_result = pd_orders.merge(pd_customers, on="customer_id",
                                    how="left")
        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)


# ===========================================================================
# 7. Complex cross-source workflow
# ===========================================================================

class TestCrossSourceWorkflow:
    """End-to-end workflow combining multiple cross-source operations."""

    def test_filter_merge_assign_sort(self):
        """Full workflow: filter -> merge -> assign -> sort."""
        ds_products = DataStore({
            "pid": [1, 2, 3, 4, 5],
            "name": ["A", "B", "C", "D", "E"],
            "price": [10.0, 25.0, 15.0, 50.0, 30.0],
        })
        ds_categories = DataStore({
            "pid": [1, 2, 3, 4, 5],
            "cat": ["X", "Y", "X", "Z", "Y"],
        })

        filtered = ds_products[ds_products["price"] > 12]
        merged = filtered.merge(ds_categories, on="pid")
        assigned = merged.assign(
            tax=lambda df: df["price"] * 0.1
        )
        ds_result = assigned.sort_values("price")

        pd_products = pd.DataFrame({
            "pid": [1, 2, 3, 4, 5],
            "name": ["A", "B", "C", "D", "E"],
            "price": [10.0, 25.0, 15.0, 50.0, 30.0],
        })
        pd_categories = pd.DataFrame({
            "pid": [1, 2, 3, 4, 5],
            "cat": ["X", "Y", "X", "Z", "Y"],
        })

        pd_filtered = pd_products[pd_products["price"] > 12]
        pd_merged = pd_filtered.merge(pd_categories, on="pid")
        pd_assigned = pd_merged.assign(
            tax=lambda df: df["price"] * 0.1
        )
        pd_result = pd_assigned.sort_values("price")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_merge_groupby(self, sql_orders, local_customers,
                                pd_orders, pd_customers):
        """Filter with isin, merge, then groupby."""
        target_regions = ["East", "West"]
        ds_filtered = sql_orders[
            sql_orders["region"].isin(target_regions)
        ]
        ds_merged = ds_filtered.merge(local_customers, on="customer_id",
                                      how="inner")
        ds_result = ds_merged.groupby("tier")["amount"].sum()

        pd_filtered = pd_orders[pd_orders["region"].isin(target_regions)]
        pd_merged = pd_filtered.merge(pd_customers, on="customer_id",
                                      how="inner")
        pd_result = pd_merged.groupby("tier")["amount"].sum()

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_concat_then_merge(self):
        """Concat two DataStores, then merge result with another."""
        ds1 = DataStore({"key": [1, 2], "val": [10, 20]})
        ds2 = DataStore({"key": [3, 4], "val": [30, 40]})
        ds_combined = ds_module.concat([ds1, ds2], ignore_index=True)

        ds_info = DataStore({"key": [1, 2, 3, 4],
                             "label": ["a", "b", "c", "d"]})
        ds_result = ds_combined.merge(ds_info, on="key")

        pd1 = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
        pd2 = pd.DataFrame({"key": [3, 4], "val": [30, 40]})
        pd_combined = pd.concat([pd1, pd2], ignore_index=True)
        pd_info = pd.DataFrame({"key": [1, 2, 3, 4],
                                "label": ["a", "b", "c", "d"]})
        pd_result = pd_combined.merge(pd_info, on="key")

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_result_column_selection(self):
        """Select specific columns from merge result."""
        ds1 = DataStore({"key": [1, 2, 3], "a": [10, 20, 30],
                         "b": [100, 200, 300]})
        ds2 = DataStore({"key": [1, 2, 3], "c": ["x", "y", "z"],
                         "d": [1.1, 2.2, 3.3]})
        merged = ds1.merge(ds2, on="key")
        ds_result = merged[["key", "a", "c"]]

        pd1 = pd.DataFrame({"key": [1, 2, 3], "a": [10, 20, 30],
                             "b": [100, 200, 300]})
        pd2 = pd.DataFrame({"key": [1, 2, 3], "c": ["x", "y", "z"],
                             "d": [1.1, 2.2, 3.3]})
        pd_merged = pd1.merge(pd2, on="key")
        pd_result = pd_merged[["key", "a", "c"]]

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_merge_with_overlapping_columns(self):
        """Merge DataStores with overlapping non-key columns."""
        ds1 = DataStore({"key": [1, 2, 3], "val": [10, 20, 30]})
        ds2 = DataStore({"key": [1, 2, 3], "val": [100, 200, 300]})
        ds_result = ds1.merge(ds2, on="key", suffixes=("_left", "_right"))

        pd1 = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
        pd2 = pd.DataFrame({"key": [1, 2, 3], "val": [100, 200, 300]})
        pd_result = pd1.merge(pd2, on="key", suffixes=("_left", "_right"))

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)


# ===========================================================================
# 8. Cross-source with SQL table operations
# ===========================================================================

class TestSQLTableCrossOps:
    """Test cross-source operations when one DataStore is backed by a SQL table."""

    def test_sql_isin_then_assign(self, sql_orders, pd_orders):
        """Filter SQL table with isin, then assign computed column."""
        target_ids = [101, 102]
        ds_filtered = sql_orders[
            sql_orders["customer_id"].isin(target_ids)
        ]
        ds_result = ds_filtered.assign(
            tax=ds_filtered["amount"] * 0.1
        )

        pd_filtered = pd_orders[pd_orders["customer_id"].isin(target_ids)]
        pd_result = pd_filtered.assign(
            tax=pd_filtered["amount"] * 0.1
        )

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)

    def test_sql_filter_then_concat_with_local(self, sql_orders, pd_orders):
        """Filter SQL table, concat result with local DataFrame."""
        ds_filtered = sql_orders[sql_orders["amount"] > 30]
        ds_local = DataStore({
            "id": pd.array([7], dtype="uint32"),
            "customer_id": pd.array([105], dtype="uint32"),
            "product": ["NewProduct"],
            "amount": [59.99],
            "region": ["Central"],
        })
        ds_result = ds_module.concat(
            [ds_filtered, ds_local], ignore_index=True
        )

        pd_filtered = pd_orders[pd_orders["amount"] > 30]
        pd_local = pd.DataFrame({
            "id": pd.array([7], dtype="uint32"),
            "customer_id": pd.array([105], dtype="uint32"),
            "product": ["NewProduct"],
            "amount": [59.99],
            "region": ["Central"],
        })
        pd_result = pd.concat([pd_filtered, pd_local], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sql_head_then_merge(self, sql_orders, local_customers,
                                 pd_orders, pd_customers):
        """Take head of SQL result, then merge with local."""
        ds_head = sql_orders.head(3)
        ds_result = ds_head.merge(local_customers, on="customer_id",
                                  how="inner")

        pd_head = pd_orders.head(3)
        pd_result = pd_head.merge(pd_customers, on="customer_id",
                                  how="inner")

        assert_datastore_equals_pandas(ds_result, pd_result,
                                       check_row_order=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
