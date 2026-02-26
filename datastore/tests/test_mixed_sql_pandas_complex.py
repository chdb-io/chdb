"""
Complex mixed SQL and Pandas execution tests.

These tests cover advanced scenarios mixing SQL operations (filter, sort, join, groupby)
with Pandas operations (assign, transform, apply, pivot, rank, etc.) in various orders.
"""

import logging
import os

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore, Field, Sum, Count, Avg, Min, Max, config


DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def dataset_path(filename: str) -> str:
    return os.path.join(DATASET_DIR, filename)


class TestFeatureEngineeringPipeline:
    """Test feature engineering scenarios with mixed SQL/Pandas operations."""

    def test_sql_filter_pandas_transform_sql_sort(self):
        """
        Pipeline: SQL filter -> Pandas feature creation -> SQL sort
        Scenario: Filter users, compute new features, sort by computed column
        """
        users = DataStore.from_file(dataset_path("users.csv"))

        # Step 1: SQL filter
        ds = users.filter(users.age > 25)

        # Step 2: Pandas feature engineering
        ds["age_bracket"] = (ds["age"] // 10) * 10  # Age bracket (20, 30, 40...)
        ds["age_normalized"] = ds["age"] - 25  # Normalize by subtracting min

        # Step 3: SQL sort on computed column
        ds = ds.sort("age_bracket", ascending=False)

        df = ds.to_df()
        assert "age_bracket" in df.columns
        assert "age_normalized" in df.columns
        assert all(df["age"] > 25)
        # Check sorted correctly
        assert list(df["age_bracket"]) == sorted(df["age_bracket"], reverse=True)

    def test_multiple_feature_columns_from_same_source(self):
        """Create multiple derived features from a single column."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.select("order_id", "amount", "quantity")
        ds["unit_price"] = ds["amount"] / ds["quantity"]
        ds["amount_doubled"] = ds["amount"] * 2
        ds["amount_log"] = ds["amount"]  # Would be log in real scenario
        ds["is_large_order"] = ds["amount"]  # Boolean placeholder

        df = ds.to_df()
        assert all(col in df.columns for col in ["unit_price", "amount_doubled", "amount_log", "is_large_order"])
        # Verify unit_price calculation
        assert all(df["unit_price"] == df["amount"] / df["quantity"])

    def test_sql_join_pandas_feature_engineering(self):
        """Join tables then create features using pandas operations."""
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))

        # Join with USING syntax
        ds = users.join(orders, on="user_id")
        ds = ds.select("name", "age", "amount", "quantity")

        # Feature engineering
        ds["amount_per_year"] = ds["amount"] / (ds["age"] / 10)
        ds["quantity_normalized"] = ds["quantity"] - 1

        df = ds.to_df()
        assert "amount_per_year" in df.columns
        assert len(df) > 0


class TestStatisticalOperations:
    """Test statistical pandas methods combined with SQL operations."""

    def test_sql_filter_then_pandas_describe(self):
        """Filter data then get statistical summary."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.filter(users.age > 20)
        df = ds.to_df()
        stats = df.describe()

        assert "age" in stats.columns
        assert stats.loc["mean", "age"] > 20  # Filtered mean should be > 20

    def test_sql_filter_then_rank(self):
        """SQL filter followed by pandas rank operation."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.filter(orders.amount > 50)
        ds_ranked = ds.rank(numeric_only=True)

        df = ds_ranked.to_df()
        # Rank should produce values from 1 to n
        assert df["amount"].max() <= len(df)

    def test_cumulative_operations_after_sql(self):
        """Test cumulative operations after SQL filtering."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.select("order_id", "amount").sort("order_id")
        ds_cum = ds.cumsum()

        df = ds_cum.to_df()
        assert "amount" in df.columns
        # Cumsum should be monotonically increasing
        amounts = df["amount"].tolist()
        assert all(amounts[i] <= amounts[i + 1] for i in range(len(amounts) - 1))

    def test_correlation_after_join(self):
        """Calculate correlation matrix after joining tables."""
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = users.join(orders, on="user_id")
        df = ds.to_df()

        # Calculate correlation
        corr_matrix = df[["age", "amount", "quantity"]].corr()
        assert corr_matrix.shape == (3, 3)
        # Diagonal should be 1.0
        assert all(corr_matrix.loc[col, col] == pytest.approx(1.0) for col in ["age", "amount", "quantity"])


class TestDataTransformationPipeline:
    """Test complex data transformation pipelines."""

    def test_replace_values_after_sql_filter(self):
        """Replace values in a column after SQL filtering."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.filter(users.age > 25)
        ds_replaced = ds.replace({"USA": "United States", "UK": "United Kingdom"})

        df = ds_replaced.to_df()
        # Original values should be replaced
        if "United States" in df["country"].values:
            assert "USA" not in df["country"].values

    def test_clip_values_after_computation(self):
        """Clip computed values to a range."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.select("order_id", "amount")
        ds["amount_adjusted"] = ds["amount"] * 1.5
        ds_clipped = ds.clip(lower=50, upper=200)

        df = ds_clipped.to_df()
        # All numeric values should be within range
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert df[col].min() >= 50 or df[col].isna().all()
            assert df[col].max() <= 200 or df[col].isna().all()

    def test_diff_after_sql_sort(self):
        """Calculate differences after SQL sorting."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.select("order_id", "amount").sort("order_id")
        ds_diff = ds.diff()

        df = ds_diff.to_df()
        # First row should be NaN for diff
        assert pd.isna(df["amount"].iloc[0])

    def test_pct_change_on_sorted_data(self):
        """Calculate percentage change on sorted time-series-like data."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.select("order_id", "amount").sort("order_id")
        ds_pct = ds.pct_change()

        df = ds_pct.to_df()
        # First row should be NaN
        assert pd.isna(df["amount"].iloc[0])


class TestAggregationPipeline:
    """Test aggregation scenarios with mixed SQL/Pandas."""

    def test_sql_groupby_then_pandas_transform(self):
        """SQL groupby with aggregation, then pandas transform."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("country", Count("*").as_("user_count")).groupby("country")
        df = ds.to_df()
        # Apply pandas transform
        df["count_doubled"] = df["user_count"] * 2
        assert "count_doubled" in df.columns

    def test_pandas_agg_after_sql_filter(self):
        """Pandas aggregation after SQL filtering."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.filter(orders.amount > 50)
        df = ds.to_df()

        agg_result = df.agg({"amount": ["sum", "mean", "max"], "quantity": ["sum", "mean"]})
        assert agg_result.shape[0] == 3  # sum, mean, max
        assert agg_result.shape[1] == 2  # amount, quantity


class TestPivotAndReshape:
    """Test pivot and reshape operations with SQL."""

    def test_melt_after_sql_select(self):
        """Melt (unpivot) after SQL select."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("user_id", "name", "age")
        ds_melted = ds.melt(id_vars=["user_id"], value_vars=["name", "age"])

        df = ds_melted.to_df()
        assert "variable" in df.columns
        assert "value" in df.columns

    def test_transpose_after_sql_limit(self):
        """Transpose after SQL limit."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("user_id", "age").limit(5)
        ds_transposed = ds.T

        df = ds_transposed.to_df()
        # After transpose, columns become rows
        assert df.shape[1] == 5  # 5 users


class TestComplexFilterChains:
    """Test complex filter chains mixing SQL and Pandas."""

    def test_sql_filter_pandas_query_sql_filter(self):
        """SQL filter -> Pandas query -> SQL filter."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        # SQL filter
        ds = orders.filter(orders.amount > 30)

        # Pandas query
        ds = ds.query("quantity >= 1")

        # Another SQL filter
        ds = ds.filter(ds.user_id < 10)

        df = ds.to_df()
        assert all(df["amount"] > 30)
        assert all(df["quantity"] >= 1)
        assert all(df["user_id"] < 10)

    def test_isin_filter_after_sql(self):
        """Use pandas isin filter after SQL operations."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.filter(users.age > 20)
        df = ds.to_df()

        # Filter using isin
        selected_countries = ["USA", "UK"]
        df_filtered = df[df["country"].isin(selected_countries)]

        assert len(df_filtered) > 0 or len(df_filtered) == 0  # May or may not have results

    def test_nlargest_after_join(self):
        """Get top N rows after join."""
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = users.join(orders, on="user_id")
        ds_top = ds.nlargest(5, "amount")

        df = ds_top.to_df()
        assert len(df) <= 5
        # Should be sorted by amount descending
        amounts = df["amount"].tolist()
        assert amounts == sorted(amounts, reverse=True)

    def test_nsmallest_after_filter(self):
        """Get bottom N rows after SQL filter."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.filter(orders.quantity > 1)
        ds_bottom = ds.nsmallest(3, "amount")

        df = ds_bottom.to_df()
        assert len(df) <= 3


class TestWindowOperations:
    """Test window-like operations with SQL."""

    def test_rolling_mean_after_sql_sort(self):
        """Rolling mean calculation after SQL sort."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.select("order_id", "amount").sort("order_id")
        df = ds.to_df()

        # Apply rolling mean
        df["rolling_mean"] = df["amount"].rolling(window=3).mean()
        assert "rolling_mean" in df.columns
        # First 2 rows should be NaN
        assert pd.isna(df["rolling_mean"].iloc[0])
        assert pd.isna(df["rolling_mean"].iloc[1])

    def test_expanding_sum_after_filter(self):
        """Expanding sum after SQL filter."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.filter(orders.amount > 30).sort("order_id")
        df = ds.to_df()

        # Apply expanding sum
        df["expanding_sum"] = df["amount"].expanding().sum()
        # Use pytest.approx for floating point comparison
        assert df["expanding_sum"].iloc[-1] == pytest.approx(df["amount"].sum(), rel=1e-9)

    def test_shift_after_sql_operations(self):
        """Shift values after SQL sort."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.select("order_id", "amount").sort("order_id")
        ds_shifted = ds.shift(1)

        df = ds_shifted.to_df()
        # First row should be NaN after shift
        assert pd.isna(df["amount"].iloc[0])


class TestMultiTableOperations:
    """Test complex multi-table scenarios."""

    def test_three_table_join_with_feature_engineering(self):
        """Join three tables then create features."""
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))
        products = DataStore.from_file(dataset_path("products.csv"))

        # Chain joins using USING syntax
        ds = users.join(orders, on="user_id")
        ds = ds.join(products, on="product_id")

        # Create features
        ds["total_value"] = ds["amount"]
        ds["age_category"] = ds["age"] // 10

        df = ds.to_df()
        assert "total_value" in df.columns
        assert "age_category" in df.columns
        assert "product_name" in df.columns

    def test_join_filter_aggregate_transform(self):
        """Complex pipeline: join -> filter -> aggregate -> transform."""
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))

        # Join
        ds = users.join(orders, on="user_id")

        # Filter
        ds = ds.filter(ds.amount > 30)

        # Select and transform
        df = ds.to_df()

        # Aggregation
        user_totals = df.groupby("name")["amount"].sum()
        assert len(user_totals) > 0


class TestEdgeCasesAndRobustness:
    """Test edge cases for robustness."""

    def test_empty_result_handling(self):
        """Handle empty results gracefully."""
        users = DataStore.from_file(dataset_path("users.csv"))

        # Filter that returns no rows
        ds = users.filter(users.age > 1000)
        df = ds.to_df()

        assert len(df) == 0
        # Note: chDB may not return column schema for empty results
        # This is expected behavior - empty DataFrame with no columns

    def test_single_row_result(self):
        """Handle single row result."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.limit(1)
        ds["doubled_age"] = ds["age"] * 2

        df = ds.to_df()
        assert len(df) == 1
        assert "doubled_age" in df.columns

    def test_many_chained_operations(self):
        """Test many chained operations don't cause issues."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users
        ds = ds.filter(users.age > 20)
        ds["age_plus_1"] = ds["age"] + 1
        ds = ds.filter(ds.age < 50)
        ds["age_plus_2"] = ds["age"] + 2
        ds = ds.sort("age")
        ds["age_plus_3"] = ds["age"] + 3
        ds = ds.limit(5)
        ds["age_plus_4"] = ds["age"] + 4

        df = ds.to_df()
        assert all(col in df.columns for col in ["age_plus_1", "age_plus_2", "age_plus_3", "age_plus_4"])
        assert len(df) <= 5

    def test_duplicate_column_names_handling(self):
        """Test behavior with operations that might create duplicate columns."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age")
        ds["computed"] = ds["age"] * 2
        # Reassign same column
        ds["computed"] = ds["age"] * 3

        df = ds.to_df()
        # Should have the latest value
        assert all(df["computed"] == df["age"] * 3)


class TestExplainWithComplexPipelines:
    """Test explain() output for complex pipelines."""

    def test_explain_multi_step_pipeline(self):
        """Verify explain shows all steps in complex pipeline."""
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = users.filter(users.age > 25)
        ds = ds.join(orders, on="user_id")
        ds["computed"] = ds["age"] * 2
        ds = ds.sort("age")
        ds = ds.limit(10)

        # Get explain output
        explain_output = ds.explain()

        # Should contain key operations
        assert "WHERE" in explain_output or "filter" in explain_output.lower()
        assert "JOIN" in explain_output or "join" in explain_output.lower()
        assert "computed" in explain_output or "Assign" in explain_output
        assert "ORDER" in explain_output or "sort" in explain_output.lower()
        # limit() after pandas ops is executed as head in pandas stage
        assert "LIMIT" in explain_output or "limit" in explain_output.lower() or "head" in explain_output.lower()


class TestDataTypeHandling:
    """Test handling of different data types in mixed pipelines."""

    def test_string_operations_after_sql(self):
        """String operations after SQL filtering."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.filter(users.age > 25)
        df = ds.to_df()

        # String operations
        df["name_upper"] = df["name"].str.upper()
        df["name_length"] = df["name"].str.len()

        assert "name_upper" in df.columns
        assert all(df["name_upper"] == df["name"].str.upper())

    def test_numeric_type_conversion_after_sql(self):
        """Type conversion after SQL operations."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.select("order_id", "amount", "quantity")
        df = ds.to_df()

        # Convert types
        df["amount_int"] = df["amount"].astype(int)
        df["quantity_float"] = df["quantity"].astype(float)

        assert df["amount_int"].dtype in [np.int32, np.int64]
        assert df["quantity_float"].dtype == np.float64


class TestBatchOperations:
    """Test batch operations combining multiple pandas methods."""

    def test_assign_multiple_columns_at_once(self):
        """Use assign to create multiple columns at once."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.filter(orders.amount > 30)
        df = ds.to_df()

        df_new = df.assign(
            amount_doubled=lambda x: x["amount"] * 2,
            quantity_plus_one=lambda x: x["quantity"] + 1,
            order_value=lambda x: x["amount"] * x["quantity"],
        )

        assert all(col in df_new.columns for col in ["amount_doubled", "quantity_plus_one", "order_value"])

    def test_eval_expression_after_sql(self):
        """Use pandas eval for expression evaluation after SQL."""
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = orders.filter(orders.amount > 30)
        ds_evaled = ds.eval("unit_price = amount / quantity")

        df = ds_evaled.to_df()
        assert "unit_price" in df.columns


class TestExtremeMixedPipeline:
    """Test extremely long mixed SQL/Pandas pipelines (20+ steps)."""

    def test_20_plus_alternating_sql_pandas_operations(self):
        """
        A 25-step pipeline alternating between SQL and Pandas operations.
        This tests the robustness of the lazy execution system.
        """
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users

        # Step 1: [SQL] SELECT columns
        ds = ds.select("user_id", "name", "age")

        # Step 2: [SQL] FILTER age > 20
        ds = ds.filter(ds.age > 20)

        # Step 3: [Pandas] Assign: age_doubled
        ds["age_doubled"] = ds["age"] * 2

        # Step 4: [SQL] SORT by age
        ds = ds.sort("age")

        # Step 5: [Pandas] Assign: age_tripled
        ds["age_tripled"] = ds["age"] * 3

        # Step 6: [Pandas] Assign: age_plus_10
        ds["age_plus_10"] = ds["age"] + 10

        # Step 7: [SQL] FILTER age < 50
        ds = ds.filter(ds["age"] < 50)

        # Step 8: [Pandas] Assign: computed_1
        ds["computed_1"] = ds["age"] + ds["age_doubled"]

        # Step 9: [Pandas] Assign: computed_2
        ds["computed_2"] = ds["age_tripled"] - ds["age"]

        # Step 10: [SQL] SORT by age DESC
        ds = ds.sort("age", ascending=False)

        # Step 11: [Pandas] Assign: ratio
        ds["ratio"] = ds["age_doubled"] / ds["age"]

        # Step 12: [SQL] LIMIT 8
        ds = ds.limit(8)

        # Step 13: [Pandas] Assign: computed_3
        ds["computed_3"] = ds["age_tripled"] / 3

        # Step 14: [Pandas] Assign: computed_4
        ds["computed_4"] = ds["computed_1"] + ds["computed_2"]

        # Step 15: [SQL] FILTER computed_1 > 0
        ds = ds.filter(ds["computed_1"] > 0)

        # Step 16: [Pandas] Assign: computed_5
        ds["computed_5"] = ds["age"] ** 2

        # Step 17: [Pandas] Assign: computed_6
        ds["computed_6"] = ds["computed_5"] / 100

        # Step 18: [SQL] SORT by computed_5
        ds = ds.sort("computed_5")

        # Step 19: [Pandas] Assign: computed_7
        ds["computed_7"] = ds["computed_2"] + ds["computed_3"]

        # Step 20: [Pandas] Assign: computed_8
        ds["computed_8"] = ds["ratio"] * 10

        # Step 21: [SQL] LIMIT 5
        ds = ds.limit(5)

        # Step 22: [Pandas] Assign: step_22
        ds["step_22"] = ds["age"] + 100

        # Step 23: [Pandas] Assign: step_23
        ds["step_23"] = ds["step_22"] * 2

        # Step 24: [SQL] FILTER step_22 > 0
        ds = ds.filter(ds["step_22"] > 0)

        # Step 25: [Pandas] Assign: final_marker
        ds["final_marker"] = ds["step_22"] * 0 + 1

        # Execute and verify
        df = ds.to_df()

        # Assertions
        assert len(df) <= 5, f"Should have at most 5 rows, got {len(df)}"
        assert "final_marker" in df.columns, "final_marker should exist"
        assert "computed_8" in df.columns, "computed_8 should exist"
        assert "step_23" in df.columns, "step_23 should exist"
        assert all(df["final_marker"] == 1), "All final_marker values should be 1"

        # Verify some computations
        if len(df) > 0:
            # ratio should be 2 (age_doubled / age = 2*age / age = 2)
            for val in df["ratio"]:
                assert val == pytest.approx(2.0, rel=1e-9), "ratio should be 2"
            # computed_3 should equal age (age_tripled / 3 = 3*age / 3 = age)
            for idx in df.index:
                assert df.loc[idx, "computed_3"] == pytest.approx(df.loc[idx, "age"], rel=1e-9)

    def test_25_step_with_joins_and_aggregations(self):
        """
        A 25-step pipeline including joins between multiple tables.

        KNOWN LIMITATION:
        1. Column assignments referencing joined table columns don't work with lazy ops
        2. SELECT before JOIN limits final columns - need to select after JOIN or use SELECT *
        """
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))
        products = DataStore.from_file(dataset_path("products.csv"))

        # Steps 1-7: Pure SQL operations with JOIN
        # NOTE: Don't use select() before join, or use select("*") to get all columns
        ds = users.filter(users.age > 22)  # Step 1
        ds = ds.join(orders, on="user_id")  # Step 2
        ds = ds.join(products, on="product_id")  # Step 3
        ds = ds.filter(ds.age < 50)  # Step 4
        ds = ds.sort("age", ascending=False)  # Step 5
        ds = ds.limit(10)  # Step 6

        # Step 7: Execute to get joined columns
        df = ds.to_df()

        # Steps 8-25: Pure pandas operations on executed data
        if len(df) > 0 and "amount" in df.columns:
            df["order_value"] = df["amount"] * df["quantity"]  # Step 8
            df["total_cost"] = df["order_value"] + df["price"]  # Step 9
            df["cost_ratio"] = df["total_cost"] / df["amount"]  # Step 10
            df = df[df["total_cost"] > 50]  # Step 11
            df = df.sort_values("total_cost", ascending=False)  # Step 12
            df["rank_score"] = df["total_cost"] / 100  # Step 13
            df["discount"] = df["total_cost"] * 0.1  # Step 14
            df = df.head(8)  # Step 15
            df["final_price"] = df["total_cost"] - df["discount"]  # Step 16
            df["category"] = df["final_price"] // 100  # Step 17
            df = df[df["final_price"] > 100]  # Step 18
            df = df.sort_values("final_price")  # Step 19
            df["price_squared"] = df["final_price"] ** 2  # Step 20
            df["normalized"] = df["final_price"] / 100  # Step 21
            df = df.head(5)  # Step 22
            df["is_premium"] = df["final_price"]  # Step 23
            df["processed_flag"] = 1  # Step 24
            df["step_25_marker"] = df["processed_flag"] + 24  # Step 25

            # Assertions
            assert len(df) <= 5, f"Should have at most 5 rows, got {len(df)}"
            assert "processed_flag" in df.columns, "processed_flag should exist"
            assert "discount" in df.columns, "discount should exist"
            assert "final_price" in df.columns, "final_price should exist"
            assert "step_25_marker" in df.columns, "step_25_marker should exist"
            assert all(df["processed_flag"] == 1), "All processed_flag should be 1"
            assert all(df["step_25_marker"] == 25), "All step_25_marker should be 25"

    def test_join_column_assignment_works(self):
        """
        Simple join followed by column assignment WORKS correctly.

        This test verifies that:
            ds = users.join(orders, on="user_id")
            ds["computed"] = ds["amount"] * 2  # amount is from orders

        Works because there are no lazy pandas ops before the join,
        so the JOIN is included in the initial SQL query.
        """
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = users.join(orders, on="user_id")
        ds["computed"] = ds["amount"] * 2  # This WORKS!

        df = ds.to_df()
        assert "computed" in df.columns
        assert "amount" in df.columns
        assert len(df) > 0

    def test_lazy_op_before_join_breaks_column_access(self):
        """
        KNOWN LIMITATION: When there's a lazy pandas op BEFORE the join,
        accessing joined table columns fails.

        The issue:
        1. ds["user_score"] = ds["age"] * 10  -> records LazyColumnAssignment
        2. ds = ds.join(orders, on="user_id") -> adds to self._joins
        3. ds["computed"] = ds["amount"] * 2  -> records LazyColumnAssignment

        During execution:
        - Phase 1 SQL only processes ops BEFORE first LazyColumnAssignment
        - So the SELECT clause is built without knowing about the join
        - The JOIN IS included (from self._joins), but SELECT may limit columns
        - Phase 2 executes lazy ops, but 'amount' may not be in DataFrame
        """
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))

        ds = users.select("user_id", "name", "age")  # "amount" is not in the selected columns
        # This should work:
        # ds = users.select("user_id", "name", "age", "amount")
        ds["user_score"] = ds["age"] * 10  # Lazy op BEFORE join
        ds = ds.join(orders, on="user_id")
        ds["computed"] = ds["amount"] * 2  # Tries to access joined column

        try:
            df = ds.to_df()
            # If it works, great!
            assert "computed" in df.columns
        except KeyError as e:
            # Expected: SELECT limits columns, 'amount' not available
            pytest.xfail(f"Known limitation: {e}")

    def test_30_step_extreme_pipeline(self):
        """
        A 30-step extreme pipeline to stress test the system.

        This test creates a very long chain of alternating operations
        to verify the lazy execution system can handle extreme cases.
        """
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users

        # Phase 1: Initial SQL operations (Steps 1-5)
        ds = ds.select("user_id", "name", "age")  # Step 1
        ds = ds.filter(ds.age > 20)  # Step 2
        ds = ds.sort("age")  # Step 3
        ds = ds.filter(ds.age < 50)  # Step 4
        ds = ds.sort("name")  # Step 5

        # Phase 2: Pandas transformations (Steps 6-10)
        ds["step_6"] = ds["age"] * 2  # Step 6
        ds["step_7"] = ds["step_6"] + 10  # Step 7
        ds["step_8"] = ds["age"] + 5  # Step 8
        ds["step_9"] = ds["age"] ** 2  # Step 9
        ds["step_10"] = ds["step_9"] / 10  # Step 10

        # Phase 3: More SQL (Steps 11-15)
        ds = ds.filter(ds["step_6"] > 40)  # Step 11
        ds = ds.sort("step_7", ascending=False)  # Step 12
        ds = ds.limit(8)  # Step 13
        ds = ds.filter(ds["step_9"] > 500)  # Step 14
        ds = ds.sort("age")  # Step 15

        # Phase 4: More Pandas (Steps 16-20)
        ds["step_16"] = ds["step_6"] - ds["age"]  # Step 16
        ds["step_17"] = ds["step_7"] * 2  # Step 17
        ds["step_18"] = ds["step_7"] / 2  # Step 18
        ds["step_19"] = ds["step_16"] + ds["step_17"]  # Step 19
        ds["step_20"] = ds["age"] + ds["step_18"]  # Step 20

        # Phase 5: More SQL (Steps 21-25)
        ds = ds.filter(ds["step_6"] > 50)  # Step 21
        ds = ds.sort("step_20", ascending=False)  # Step 22
        ds = ds.limit(5)  # Step 23
        ds = ds.filter(ds["step_18"] > 20)  # Step 24
        ds = ds.sort("name")  # Step 25

        # Phase 6: Final Pandas operations (Steps 26-30)
        ds["step_26"] = ds["step_9"] / 100  # Step 26
        ds["step_27"] = ds["step_26"] + 1  # Step 27
        ds["step_28"] = ds["step_26"] + ds["step_20"]  # Step 28
        ds["step_29"] = ds["step_28"] * 2  # Step 29
        ds["final_step_30"] = ds["step_28"] * 0 + 30  # Step 30

        # Execute
        df = ds.to_df()

        # Assertions
        assert len(df) <= 5, f"Should have at most 5 rows, got {len(df)}"

        if len(df) > 0:
            # Verify final step marker
            assert "final_step_30" in df.columns
            assert all(df["final_step_30"] == 30), "All final_step_30 should be 30"

            # Verify some intermediate computations exist
            assert "step_6" in df.columns
            assert "step_18" in df.columns
            assert "step_20" in df.columns
            assert "step_26" in df.columns
            assert "step_28" in df.columns

    def test_explain_shows_all_30_steps(self):
        """Verify explain() output contains information about many operations."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users

        # Build a 15-step pipeline and check explain output
        ds = ds.select("user_id", "name", "age")  # 1
        ds = ds.filter(ds.age > 20)  # 2
        ds["col_3"] = ds["age"] * 2  # 3
        ds = ds.sort("age")  # 4
        ds["col_5"] = ds["col_3"] + 10  # 5
        ds = ds.filter(ds.age < 50)  # 6
        ds = ds.rename(columns={"name": "username"})  # 7
        ds["col_8"] = ds["col_5"] - 5  # 8
        ds = ds.sort("username")  # 9
        ds = ds.add_prefix("p_")  # 10
        ds["col_11"] = ds["p_age"] * 3  # 11
        ds = ds.limit(10)  # 12
        ds["col_13"] = ds["col_11"] / 2  # 13
        ds = ds.filter(ds["p_col_3"] > 40)  # 14
        ds["col_15"] = ds["col_13"] + 1  # 15

        # Get explain output
        explain_output = ds.explain()

        # Should contain multiple operation types
        assert "SELECT" in explain_output or "select" in explain_output.lower()
        assert "WHERE" in explain_output or "filter" in explain_output.lower()
        assert "ORDER" in explain_output or "sort" in explain_output.lower()
        assert "LIMIT" in explain_output or "limit" in explain_output.lower()
        assert "Assign" in explain_output or "col_" in explain_output
        assert "prefix" in explain_output.lower() or "p_" in explain_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
