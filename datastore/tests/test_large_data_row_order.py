"""
Tests for row order preservation with large data volumes.

These tests are specifically designed to expose non-deterministic row ordering issues
that may occur due to chDB's parallel execution. Small data tests may pass intermittently,
but large data volumes (100k+ rows) reliably expose these issues.

The key scenarios tested:
1. dropna + assign: Non-contiguous index after dropna, then SQL expression evaluation
2. filter + assign: Similar pattern with boolean filtering
3. Multiple chained operations: Complex pipelines that may lose row order
"""

import unittest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestLargeDataRowOrder(unittest.TestCase):
    """Test row order preservation with large data volumes (100k rows)."""

    @classmethod
    def setUpClass(cls):
        """Create large test data once for all tests."""
        np.random.seed(42)
        cls.n = 100000  # 100k rows to reliably expose parallel execution issues
        
        # Data with ~17% NaN in column 'a' - use float type for proper NaN handling
        a_vals = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], size=cls.n)
        b_vals = np.random.choice([10.0, 20.0, 30.0, 40.0, 50.0, np.nan], size=cls.n)
        c_vals = np.random.randint(100, 600, size=cls.n).astype(float)
        
        cls.data_with_nulls = {
            "a": a_vals,
            "b": b_vals,
            "c": c_vals,
        }
        
        # Data without nulls for filter tests
        cls.data_no_nulls = {
            "x": np.random.randint(1, 100, size=cls.n).astype(float),
            "y": np.random.randint(1, 100, size=cls.n).astype(float),
            "z": np.random.randint(1, 100, size=cls.n).astype(float),
        }

    def test_dropna_then_assign_arithmetic_expression(self):
        """Test row order: dropna creates non-contiguous index, then assign with SQL expression.
        
        This is the exact pattern that caused CI failures on macOS py3.12.
        dropna() creates a DataFrame with non-contiguous index (e.g., [0, 1, 2, 4, 5, ...]),
        then assign() with a SQL expression (b + c) must preserve this row order.
        """
        # pandas operations
        pd_df = pd.DataFrame(self.data_with_nulls)
        pd_result = pd_df.dropna(subset=["a"])
        pd_result = pd_result.assign(d=pd_result["b"] + pd_result["c"])
        
        # DataStore operations (mirror of pandas)
        ds_df = DataStore(self.data_with_nulls)
        ds_result = ds_df.dropna(subset=["a"])
        ds_result = ds_result.assign(d=ds_result["b"] + ds_result["c"])
        
        # Verify exact match including row order
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_then_assign_multiple_expressions(self):
        """Test row order with multiple column assignments after dropna."""
        # pandas operations
        pd_df = pd.DataFrame(self.data_with_nulls)
        pd_result = pd_df.dropna(subset=["a"])
        pd_result = pd_result.assign(
            sum_bc=pd_result["b"] + pd_result["c"],
            diff_bc=pd_result["b"] - pd_result["c"],
            prod_bc=pd_result["b"] * pd_result["c"],
        )
        
        # DataStore operations
        ds_df = DataStore(self.data_with_nulls)
        ds_result = ds_df.dropna(subset=["a"])
        ds_result = ds_result.assign(
            sum_bc=ds_result["b"] + ds_result["c"],
            diff_bc=ds_result["b"] - ds_result["c"],
            prod_bc=ds_result["b"] * ds_result["c"],
        )
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_assign(self):
        """Test row order: filter creates non-contiguous index, then assign."""
        # pandas operations
        pd_df = pd.DataFrame(self.data_no_nulls)
        pd_result = pd_df[pd_df["x"] > 50]  # Filter ~50% of rows
        pd_result = pd_result.assign(sum_xy=pd_result["x"] + pd_result["y"])
        
        # DataStore operations
        ds_df = DataStore(self.data_no_nulls)
        ds_result = ds_df[ds_df["x"] > 50]
        ds_result = ds_result.assign(sum_xy=ds_result["x"] + ds_result["y"])
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters_then_assign(self):
        """Test row order with multiple chained filters then assign."""
        # pandas operations
        pd_df = pd.DataFrame(self.data_no_nulls)
        pd_result = pd_df[pd_df["x"] > 30]
        pd_result = pd_result[pd_result["y"] > 30]
        pd_result = pd_result[pd_result["z"] > 30]
        pd_result = pd_result.assign(total=pd_result["x"] + pd_result["y"] + pd_result["z"])
        
        # DataStore operations
        ds_df = DataStore(self.data_no_nulls)
        ds_result = ds_df[ds_df["x"] > 30]
        ds_result = ds_result[ds_result["y"] > 30]
        ds_result = ds_result[ds_result["z"] > 30]
        ds_result = ds_result.assign(total=ds_result["x"] + ds_result["y"] + ds_result["z"])
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_filter_assign_chain(self):
        """Test complex chain: dropna -> filter -> assign."""
        # pandas operations
        pd_df = pd.DataFrame(self.data_with_nulls)
        pd_result = pd_df.dropna(subset=["a"])
        pd_result = pd_result[pd_result["a"] > 2]
        pd_result = pd_result.assign(computed=pd_result["b"] * 2 + pd_result["c"])
        
        # DataStore operations
        ds_df = DataStore(self.data_with_nulls)
        ds_result = ds_df.dropna(subset=["a"])
        ds_result = ds_result[ds_result["a"] > 2]
        ds_result = ds_result.assign(computed=ds_result["b"] * 2 + ds_result["c"])
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_then_assign(self):
        """Test row order with head() then assign."""
        # pandas operations
        pd_df = pd.DataFrame(self.data_no_nulls)
        pd_result = pd_df.head(50000)
        pd_result = pd_result.assign(doubled=pd_result["x"] * 2)
        
        # DataStore operations
        ds_df = DataStore(self.data_no_nulls)
        ds_result = ds_df.head(50000)
        ds_result = ds_result.assign(doubled=ds_result["x"] * 2)
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_slice_then_assign(self):
        """Test row order with iloc slicing then assign."""
        # pandas operations
        pd_df = pd.DataFrame(self.data_no_nulls)
        pd_result = pd_df.iloc[10000:60000]
        pd_result = pd_result.assign(sum_val=pd_result["x"] + pd_result["y"])
        
        # DataStore operations
        ds_df = DataStore(self.data_no_nulls)
        ds_result = ds_df.iloc[10000:60000]
        ds_result = ds_result.assign(sum_val=ds_result["x"] + ds_result["y"])
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_all_columns_then_assign(self):
        """Test dropna on all columns then assign."""
        # pandas operations
        pd_df = pd.DataFrame(self.data_with_nulls)
        pd_result = pd_df.dropna()  # Drop rows with any NaN
        pd_result = pd_result.assign(total=pd_result["a"] + pd_result["b"] + pd_result["c"])
        
        # DataStore operations
        ds_df = DataStore(self.data_with_nulls)
        ds_result = ds_df.dropna()
        ds_result = ds_result.assign(total=ds_result["a"] + ds_result["b"] + ds_result["c"])
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestLargeDataRowOrderRepeated(unittest.TestCase):
    """Run key tests multiple times to catch intermittent failures."""

    def test_dropna_assign_repeated_10x(self):
        """Run dropna + assign test 10 times to catch intermittent failures."""
        np.random.seed(42)
        n = 100000
        
        for iteration in range(10):
            # Use float arrays with np.nan for proper null handling
            data = {
                "a": np.where(
                    np.random.random(n) < 0.17,
                    np.nan,
                    np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n)
                ),
                "b": np.where(
                    np.random.random(n) < 0.17,
                    np.nan,
                    np.random.choice([10.0, 20.0, 30.0, 40.0, 50.0], size=n)
                ),
                "c": np.random.randint(100, 600, size=n).astype(float),
            }
            
            pd_df = pd.DataFrame(data)
            pd_result = pd_df.dropna(subset=["a"])
            pd_result = pd_result.assign(d=pd_result["b"] + pd_result["c"])
            
            ds_df = DataStore(data)
            ds_result = ds_df.dropna(subset=["a"])
            ds_result = ds_result.assign(d=ds_result["b"] + ds_result["c"])
            
            assert_datastore_equals_pandas(
                ds_result, pd_result,
                msg=f"Failed on iteration {iteration}"
            )

    def test_filter_assign_repeated_10x(self):
        """Run filter + assign test 10 times to catch intermittent failures."""
        np.random.seed(42)
        n = 100000
        
        for iteration in range(10):
            data = {
                "x": np.random.randint(1, 100, size=n).astype(float),
                "y": np.random.randint(1, 100, size=n).astype(float),
            }
            
            pd_df = pd.DataFrame(data)
            pd_result = pd_df[pd_df["x"] > 50]
            pd_result = pd_result.assign(sum_xy=pd_result["x"] + pd_result["y"])
            
            ds_df = DataStore(data)
            ds_result = ds_df[ds_df["x"] > 50]
            ds_result = ds_result.assign(sum_xy=ds_result["x"] + ds_result["y"])
            
            assert_datastore_equals_pandas(
                ds_result, pd_result,
                msg=f"Failed on iteration {iteration}"
            )


class TestLargeDataComplexChains(unittest.TestCase):
    """Test complex operation chains with large data."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n = 100000
        cls.data = {
            "a": np.random.randint(1, 10, size=cls.n).astype(float),
            "b": np.random.randint(1, 100, size=cls.n).astype(float),
            "c": np.random.randint(1, 1000, size=cls.n).astype(float),
        }

    def test_filter_filter_assign(self):
        """Test double filter then assign."""
        pd_df = pd.DataFrame(self.data)
        pd_result = pd_df[pd_df["a"] > 5]
        pd_result = pd_result[pd_result["b"] > 50]
        pd_result = pd_result.assign(sum_abc=pd_result["a"] + pd_result["b"] + pd_result["c"])
        
        ds_df = DataStore(self.data)
        ds_result = ds_df[ds_df["a"] > 5]
        ds_result = ds_result[ds_result["b"] > 50]
        ds_result = ds_result.assign(sum_abc=ds_result["a"] + ds_result["b"] + ds_result["c"])
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_filter_assign(self):
        """Test select columns, filter, then assign."""
        pd_df = pd.DataFrame(self.data)
        pd_result = pd_df[["a", "b"]]
        pd_result = pd_result[pd_result["a"] > 5]
        pd_result = pd_result.assign(doubled_b=pd_result["b"] * 2)
        
        ds_df = DataStore(self.data)
        ds_result = ds_df[["a", "b"]]
        ds_result = ds_result[ds_result["a"] > 5]
        ds_result = ds_result.assign(doubled_b=ds_result["b"] * 2)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == "__main__":
    unittest.main()
