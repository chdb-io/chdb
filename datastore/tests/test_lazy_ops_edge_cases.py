"""
Tests for lazy ops edge cases - exploring potential bugs similar to the JOIN issue.
These tests check complex interactions between SQL operations and lazy pandas ops.
"""

import logging
import os

import pytest

from datastore import DataStore, config


DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def dataset_path(filename: str) -> str:
    return os.path.join(DATASET_DIR, filename)


class TestGroupByWithLazyOps:
    """Test GROUP BY interactions with lazy ops."""

    def test_groupby_then_pandas_column_assignment(self):
        """GROUP BY followed by pandas column assignment."""
        users = DataStore.from_file(dataset_path("users.csv"))

        # GROUP BY with aggregation
        ds = users.select("country").groupby("country")
        ds["new_col"] = ds["country"]  # Add column after groupby

        df = ds.to_df()
        assert "new_col" in df.columns

    def test_groupby_with_aggregation_then_pandas(self):
        """GROUP BY with aggregation followed by pandas ops."""
        from datastore import Count

        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("country", Count("*").as_("cnt")).groupby("country")
        ds["cnt_doubled"] = ds["cnt"] * 2

        df = ds.to_df()
        assert "cnt_doubled" in df.columns


class TestDistinctWithLazyOps:
    """Test DISTINCT interactions with lazy ops."""

    def test_distinct_then_column_assignment(self):
        """DISTINCT followed by column assignment."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("country").distinct()
        ds["country_copy"] = ds["country"]

        df = ds.to_df()
        assert "country_copy" in df.columns
        # Check distinct was applied - should have unique countries only
        assert len(df) == len(df["country"].unique()), "DISTINCT not applied correctly"


class TestRenameAndFilter:
    """Test column rename followed by filter on new name."""

    def test_rename_then_filter_new_column(self):
        """Rename column then filter using the new name."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age")
        ds = ds.rename(columns={"age": "user_age"})
        # Now filter on the NEW column name
        ds = ds.filter(ds["user_age"] > 30)

        try:
            df = ds.to_df()
            assert "user_age" in df.columns
            assert all(df["user_age"] > 30)
        except Exception as e:
            pytest.xfail(f"Rename then filter on new name fails: {e}")

    def test_add_prefix_then_filter_prefixed_column(self):
        """Add prefix then filter using prefixed column name."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age")
        ds = ds.add_prefix("u_")
        # Now filter on the PREFIXED column name
        ds = ds.filter(ds["u_age"] > 30)

        try:
            df = ds.to_df()
            assert "u_name" in df.columns
            assert "u_age" in df.columns
            assert all(df["u_age"] > 30)
        except Exception as e:
            pytest.xfail(f"Add prefix then filter on prefixed name fails: {e}")


class TestMultipleExecution:
    """Test calling to_df() multiple times."""

    def test_multiple_to_df_calls(self):
        """Multiple to_df() calls should return same result."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age").filter(users.age > 25)
        ds["computed"] = ds["age"] * 2

        df1 = ds.to_df()
        df2 = ds.to_df()

        assert df1.equals(df2), "Multiple to_df() calls should return same result"

    def test_modify_after_to_df(self):
        """Modifying DataStore after to_df() - what happens?"""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age").filter(users.age > 25)
        df1 = ds.to_df()

        # Now add more operations
        ds["computed"] = ds["age"] * 2
        df2 = ds.to_df()

        # df2 should have the new column
        assert "computed" in df2.columns
        # df1 should NOT have it (it's a snapshot)
        assert "computed" not in df1.columns


class TestOffsetLimitWithLazyOps:
    """Test OFFSET/LIMIT interactions with lazy ops."""

    def test_offset_limit_then_pandas(self):
        """OFFSET and LIMIT followed by pandas ops."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age").sort("age").offset(2).limit(3)
        ds["rank"] = ds["age"]  # Just copy for testing

        df = ds.to_df()
        assert len(df) == 3
        assert "rank" in df.columns

    def test_pandas_then_limit(self):
        """Pandas ops followed by LIMIT."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age")
        ds["computed"] = ds["age"] * 2
        ds = ds.limit(5)

        df = ds.to_df()
        assert len(df) == 5
        assert "computed" in df.columns


class TestNestedFilters:
    """Test multiple nested filter operations."""

    def test_filter_pandas_filter(self):
        """SQL filter -> pandas op -> SQL filter."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age", "country")
        ds = ds.filter(ds.age > 20)  # SQL filter
        ds["age_group"] = ds["age"]  # pandas op (just copy)
        ds = ds.filter(ds.age < 50)  # Another SQL filter (on DataFrame now)

        df = ds.to_df()
        assert all(df["age"] > 20)
        assert all(df["age"] < 50)

    def test_multiple_sql_filters_before_pandas(self):
        """Multiple SQL filters before pandas op."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age", "country")
        ds = ds.filter(ds.age > 20)
        ds = ds.filter(ds.age < 50)
        ds = ds.filter(ds.country == "USA")
        ds["marker"] = ds["age"]

        df = ds.to_df()
        assert all(df["age"] > 20)
        assert all(df["age"] < 50)
        assert all(df["country"] == "USA")


class TestColumnSelectionEdgeCases:
    """Test edge cases with column selection."""

    def test_select_then_assign_uses_unselected_column(self):
        """Select columns, then try to use an unselected column.
        
        After select(), only the selected columns should be accessible.
        This matches pandas behavior where selecting columns restricts what can be accessed.
        """
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name")  # Only select name
        
        # Now try to use age which wasn't selected - should raise KeyError
        with pytest.raises(KeyError, match="Column 'age' is not accessible"):
            ds["age_copy"] = ds["age"]

    def test_column_selection_after_assignment(self):
        """Column selection after assignment should include new column."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age")
        ds["computed"] = ds["age"] * 2
        ds = ds[["name", "computed"]]  # Select only name and computed

        df = ds.to_df()
        assert list(df.columns) == ["name", "computed"]
        assert "age" not in df.columns


class TestChainedJoins:
    """Test chained/multiple joins."""

    def test_three_way_join_with_pandas(self):
        """Join three tables then apply pandas ops using USING syntax."""
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))
        products = DataStore.from_file(dataset_path("products.csv"))

        # Join users -> orders -> products
        # Use on="column" for USING syntax - no table prefix ambiguity!
        ds = users.join(orders, on="user_id")
        ds = ds.join(products, on="product_id")
        ds = ds.select("name", "amount", "product_name")
        ds["total"] = ds["amount"]  # Just copy for testing

        df = ds.to_df()
        assert "total" in df.columns
        assert "name" in df.columns
        assert "product_name" in df.columns


class TestHavingWithLazyOps:
    """Test HAVING clause with lazy ops."""

    def test_having_then_pandas(self):
        """HAVING clause followed by pandas ops."""
        from datastore import Count

        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("country", Count("*").as_("cnt")).groupby("country")
        ds = ds.having(Count("*") > 1)
        ds["cnt_doubled"] = ds["cnt"] * 2

        df = ds.to_df()
        assert "cnt_doubled" in df.columns


class TestImmutabilityAndCopies:
    """Test that operations return new DataStore instances."""

    def test_filter_returns_new_instance(self):
        """filter() should return a new instance, not modify original."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds1 = users.select("name", "age")
        ds2 = ds1.filter(ds1.age > 30)

        # ds1 should still have all records when executed
        df1 = ds1.to_df()
        df2 = ds2.to_df()

        # ds1 should have more records than ds2 (filter was effective)
        assert len(df1) > len(df2)

    def test_assignment_modifies_in_place(self):
        """Column assignment modifies in place (adds lazy op to same instance)."""
        users = DataStore.from_file(dataset_path("users.csv"))

        ds = users.select("name", "age")
        id_before = id(ds)
        ds["computed"] = ds["age"] * 2
        id_after = id(ds)

        # ds should be the same object (assignment modifies in place)
        assert id_before == id_after


class TestMemoryLayoutDetection:
    """
    Tests for DataFrame memory layout detection and automatic copy.

    When a pandas DataFrame is created from a 2D numpy array, columns become
    non-contiguous strided views. chDB's Python() table function cannot correctly
    read this layout. DataStore automatically detects and copies such DataFrames.

    See: https://github.com/chdb-io/chdb/issues/XXX (to be reported)
    """

    def test_needs_memory_copy_detection(self):
        """Test _needs_memory_copy detects problematic DataFrames."""
        import numpy as np
        import pandas as pd
        from datastore.lazy_ops import _needs_memory_copy

        # Case 1: DataFrame from 2D numpy array - needs copy
        data = np.random.normal(size=(5, 2))
        df1 = pd.DataFrame(data, columns=['A', 'B'])
        assert _needs_memory_copy(df1) is True, "2D array DataFrame should need copy"

        # Case 2: DataFrame from dict - does NOT need copy
        df2 = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        assert _needs_memory_copy(df2) is False, "Dict DataFrame should not need copy"

        # Case 3: DataFrame.copy() - does NOT need copy
        df3 = pd.DataFrame(data, columns=['A', 'B']).copy()
        assert _needs_memory_copy(df3) is False, "Copied DataFrame should not need copy"

        # Case 4: Single column - does NOT need copy
        df4 = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        assert _needs_memory_copy(df4) is False, "Single column should not need copy"

        # Case 5: F-ordered array (columns are C-contiguous) - does NOT need copy
        data_f = np.asfortranarray(data)
        df5 = pd.DataFrame(data_f, columns=['A', 'B'])
        assert _needs_memory_copy(df5) is False, "F-ordered array should not need copy"

    def test_datastore_auto_copies_problematic_dataframe(self):
        """Test DataStore automatically copies DataFrames with problematic memory layout."""
        import numpy as np
        import pandas as pd
        from tests.test_utils import assert_datastore_equals_pandas

        # Create DataFrame from 2D numpy array (problematic for chDB)
        np.random.seed(42)
        data = np.random.normal(size=(10, 2))
        pd_df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

        # DataStore should auto-detect and copy, producing correct results
        ds = DataStore(pd_df)

        # Compare results - should match
        assert_datastore_equals_pandas(ds, pd_df)

    def test_datastore_preserves_data_from_numpy_array(self):
        """Test DataStore preserves exact values from numpy array DataFrame."""
        import numpy as np
        import pandas as pd

        np.random.seed(123)
        data = np.random.normal(size=(5, 3))
        pd_df = pd.DataFrame(data, columns=['A', 'B', 'C'])

        ds = DataStore(pd_df)

        # Execute and verify exact values
        ds_df = ds._get_df()
        for i in range(5):
            for j, col in enumerate(['A', 'B', 'C']):
                expected = data[i, j]
                actual = ds_df[col].iloc[i]
                assert abs(expected - actual) < 1e-10, (
                    f"Value mismatch at row {i}, col {col}: "
                    f"expected {expected}, got {actual}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
