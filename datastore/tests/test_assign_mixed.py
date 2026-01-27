"""
Test cases for assign() method with mixed SQL expressions and lambda/pandas expressions.

These tests verify that assign() works correctly when:
1. Only SQL expressions are provided
2. Only lambda/pandas expressions are provided  
3. Mixed SQL expressions and lambda expressions are provided together
"""

import pandas as pd
import pytest
from datastore import DataStore
from tests.test_utils import assert_frame_equal


class TestAssignMixed:
    """Test assign() with mixed expression types."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame and DataStore for testing."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })
        ds = DataStore(df.copy())
        return df, ds

    def test_assign_only_sql_expression(self, sample_data):
        """Test assign with only SQL expressions (ColumnExpr)."""
        df, ds = sample_data
        
        # DataStore with SQL expression
        ds_result = ds.assign(D=ds['A'] + ds['B'])
        
        # pandas equivalent
        pd_result = df.assign(D=df['A'] + df['B'])
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_only_lambda(self, sample_data):
        """Test assign with only lambda expressions."""
        df, ds = sample_data
        
        # DataStore with lambda
        ds_result = ds.assign(D=lambda x: x['A'] * 2)
        
        # pandas equivalent
        pd_result = df.assign(D=lambda x: x['A'] * 2)
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_mixed_expr_and_lambda(self, sample_data):
        """Test assign with both SQL expression and lambda."""
        df, ds = sample_data
        
        # DataStore with mixed expressions
        ds_result = ds.assign(
            D=ds['A'] + ds['B'],        # SQL expression
            E=lambda x: x['C'] * 2      # Lambda
        )
        
        # pandas equivalent
        pd_result = df.assign(
            D=df['A'] + df['B'],
            E=lambda x: x['C'] * 2
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_mixed_multiple_of_each(self, sample_data):
        """Test assign with multiple SQL expressions and multiple lambdas."""
        df, ds = sample_data
        
        # DataStore with multiple mixed expressions
        ds_result = ds.assign(
            D=ds['A'] + ds['B'],        # SQL expression 1
            E=ds['A'] * ds['B'],        # SQL expression 2
            F=lambda x: x['C'] - 50,    # Lambda 1
            G=lambda x: x['A'] ** 2     # Lambda 2
        )
        
        # pandas equivalent
        pd_result = df.assign(
            D=df['A'] + df['B'],
            E=df['A'] * df['B'],
            F=lambda x: x['C'] - 50,
            G=lambda x: x['A'] ** 2
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_with_scalar_value(self, sample_data):
        """Test assign with scalar values (treated as pandas expression)."""
        df, ds = sample_data
        
        # DataStore with scalar
        ds_result = ds.assign(constant=42)
        
        # pandas equivalent
        pd_result = df.assign(constant=42)
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_mixed_expr_lambda_scalar(self, sample_data):
        """Test assign with SQL expression, lambda, and scalar mixed together."""
        df, ds = sample_data
        
        # DataStore with all types mixed
        ds_result = ds.assign(
            D=ds['A'] + ds['B'],        # SQL expression
            E=lambda x: x['C'] * 2,     # Lambda
            F=999                       # Scalar
        )
        
        # pandas equivalent
        pd_result = df.assign(
            D=df['A'] + df['B'],
            E=lambda x: x['C'] * 2,
            F=999
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_preserves_original_columns(self, sample_data):
        """Test that assign preserves all original columns."""
        df, ds = sample_data
        
        original_cols = ds.columns.tolist()
        
        # Assign new columns
        ds_result = ds.assign(
            D=ds['A'] + 1,
            E=lambda x: x['B'] + 1
        )
        
        # Check all original columns are present
        for col in original_cols:
            assert col in ds_result.columns.tolist(), f"Original column {col} missing"
        
        # Check new columns are present
        assert 'D' in ds_result.columns.tolist()
        assert 'E' in ds_result.columns.tolist()

    def test_assign_chain_operations(self, sample_data):
        """Test chained assign operations."""
        df, ds = sample_data
        
        # Chain assigns
        ds_result = ds.assign(D=ds['A'] + ds['B']).assign(E=lambda x: x['D'] * 2)
        
        # pandas equivalent
        pd_result = df.assign(D=df['A'] + df['B']).assign(E=lambda x: x['D'] * 2)
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_lambda_references_new_sql_column(self, sample_data):
        """Test that lambda can reference columns created by SQL expressions in same assign."""
        df, ds = sample_data
        
        # In mixed assign, SQL expressions are processed first
        # Then lambda can reference the new columns
        ds_result = ds.assign(
            D=ds['A'] + ds['B'],
            E=lambda x: x['D'] * 2  # References D created by SQL expr
        )
        
        # pandas equivalent
        pd_result = df.assign(
            D=df['A'] + df['B'],
            E=lambda x: x['D'] * 2
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestAssignWithColumnExpr:
    """Test assign() with ColumnExpr that lacks _expr (executor/method mode)."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame and DataStore for testing."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds = DataStore(df.copy())
        return df, ds

    def test_assign_with_transform_result(self, sample_data):
        """Test assign with groupby transform result (executor mode ColumnExpr).
        
        This is a common pandas pattern for feature engineering:
        df.assign(group_mean=df.groupby('category')['value'].transform('mean'))
        """
        df, ds = sample_data
        
        # pandas: assign with transform result
        pd_result = df.assign(
            group_mean=df.groupby('category')['value'].transform('mean')
        )
        
        # DataStore: transform returns ColumnExpr in executor mode (_expr is None)
        ds_transform = ds.groupby('category')['value'].transform('mean')
        ds_result = ds.assign(group_mean=ds_transform)
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_with_transform_mixed_with_sql_expr(self, sample_data):
        """Test assign mixing transform result with SQL expression.
        
        Note: Column order may differ from pandas when mixing SQL expressions
        (processed first) with transform results (processed later).
        This is a known behavior difference.
        """
        df, ds = sample_data
        
        # pandas
        pd_result = df.assign(
            group_mean=df.groupby('category')['value'].transform('mean'),
            doubled=df['value'] * 2
        )
        
        # DataStore: mixed transform (executor mode) + SQL expression
        ds_result = ds.assign(
            group_mean=ds.groupby('category')['value'].transform('mean'),
            doubled=ds['value'] * 2
        )
        
        # Compare (ignoring column order - known difference)
        ds_df = ds_result._get_df().reset_index(drop=True)
        pd_df = pd_result.reset_index(drop=True)
        
        # Reorder ds columns to match pandas for comparison
        ds_df = ds_df[pd_df.columns]
        assert_frame_equal(ds_df, pd_df)

    def test_assign_with_transform_mixed_with_lambda(self, sample_data):
        """Test assign mixing transform result with lambda."""
        df, ds = sample_data
        
        # pandas
        pd_result = df.assign(
            group_mean=df.groupby('category')['value'].transform('mean'),
            tripled=lambda x: x['value'] * 3
        )
        
        # DataStore: mixed transform (executor mode) + lambda
        ds_result = ds.assign(
            group_mean=ds.groupby('category')['value'].transform('mean'),
            tripled=lambda x: x['value'] * 3
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_with_multiple_transforms(self, sample_data):
        """Test assign with multiple transform results."""
        df, ds = sample_data
        
        # pandas
        pd_result = df.assign(
            group_mean=df.groupby('category')['value'].transform('mean'),
            group_sum=df.groupby('category')['value'].transform('sum'),
            group_std=df.groupby('category')['value'].transform('std')
        )
        
        # DataStore
        ds_result = ds.assign(
            group_mean=ds.groupby('category')['value'].transform('mean'),
            group_sum=ds.groupby('category')['value'].transform('sum'),
            group_std=ds.groupby('category')['value'].transform('std')
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_assign_transform_chain_operations(self, sample_data):
        """Test chained assign with transforms."""
        df, ds = sample_data
        
        # Chain: first assign transform, then use it in another assign
        pd_result = (
            df
            .assign(group_mean=df.groupby('category')['value'].transform('mean'))
            .assign(diff_from_mean=lambda x: x['value'] - x['group_mean'])
        )
        
        ds_result = (
            ds
            .assign(group_mean=ds.groupby('category')['value'].transform('mean'))
            .assign(diff_from_mean=lambda x: x['value'] - x['group_mean'])
        )
        
        # Compare
        assert ds_result.columns.tolist() == pd_result.columns.tolist()
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))


class TestAssignThenFilter:
    """Test assign() followed by filter using the newly assigned column.
    
    This tests the scenario described in issue:
    - assign() creates a new column C
    - filter using C (e.g., ds[ds['C'] > 30])
    
    Previously this could fail with 'Unknown identifier C' due to lazy execution.
    """

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame and DataStore for testing."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds = DataStore(df.copy())
        return df, ds

    def test_assign_sql_expr_then_filter_on_new_column(self, sample_data):
        """Test: assign new column with SQL expr, then filter on that column.
        
        Mirror Pattern:
        - pandas: df.assign(C=...) then df_new[df_new['C'] > 30]
        - DataStore: ds.assign(C=...) then ds_new[ds_new['C'] > 30]
        """
        df, ds = sample_data
        
        # pandas
        pd_with_c = df.assign(C=df['A'] + df['B'])
        pd_filtered = pd_with_c[pd_with_c['C'] > 30]
        
        # DataStore (this was the failing case)
        ds_with_c = ds.assign(C=ds['A'] + ds['B'])
        ds_filtered = ds_with_c[ds_with_c['C'] > 30]
        
        # Compare
        assert_frame_equal(
            ds_filtered._get_df().reset_index(drop=True),
            pd_filtered.reset_index(drop=True))

    def test_assign_lambda_then_filter_on_new_column(self, sample_data):
        """Test: assign new column with lambda, then filter on that column."""
        df, ds = sample_data
        
        # pandas
        pd_with_c = df.assign(C=lambda x: x['A'] * x['B'])
        pd_filtered = pd_with_c[pd_with_c['C'] > 50]
        
        # DataStore
        ds_with_c = ds.assign(C=lambda x: x['A'] * x['B'])
        ds_filtered = ds_with_c[ds_with_c['C'] > 50]
        
        # Compare
        assert_frame_equal(
            ds_filtered._get_df().reset_index(drop=True),
            pd_filtered.reset_index(drop=True))

    def test_assign_multiple_then_filter_on_last_column(self, sample_data):
        """Test: assign multiple columns, then filter on the last one."""
        df, ds = sample_data
        
        # pandas
        pd_result = df.assign(C=df['A'] + df['B'], D=lambda x: x['C'] * 2)
        pd_filtered = pd_result[pd_result['D'] > 100]
        
        # DataStore
        ds_result = ds.assign(C=ds['A'] + ds['B'], D=lambda x: x['C'] * 2)
        ds_filtered = ds_result[ds_result['D'] > 100]
        
        # Compare
        assert_frame_equal(
            ds_filtered._get_df().reset_index(drop=True),
            pd_filtered.reset_index(drop=True))

    def test_chained_assign_then_filter(self, sample_data):
        """Test: chained assign().assign() then filter on first assigned column."""
        df, ds = sample_data
        
        # pandas
        pd_result = df.assign(C=df['A'] * 2).assign(D=df['B'] * 3)
        pd_filtered = pd_result[pd_result['C'] > 5]
        
        # DataStore
        ds_result = ds.assign(C=ds['A'] * 2).assign(D=ds['B'] * 3)
        ds_filtered = ds_result[ds_result['C'] > 5]
        
        # Compare
        assert_frame_equal(
            ds_filtered._get_df().reset_index(drop=True),
            pd_filtered.reset_index(drop=True))

    def test_assign_then_filter_with_compound_condition(self, sample_data):
        """Test: assign then filter with AND/OR conditions using new column."""
        df, ds = sample_data
        
        # pandas
        pd_with_c = df.assign(C=df['A'] + df['B'])
        pd_filtered = pd_with_c[(pd_with_c['C'] > 20) & (pd_with_c['A'] > 1)]
        
        # DataStore
        ds_with_c = ds.assign(C=ds['A'] + ds['B'])
        ds_filtered = ds_with_c[(ds_with_c['C'] > 20) & (ds_with_c['A'] > 1)]
        
        # Compare
        assert_frame_equal(
            ds_filtered._get_df().reset_index(drop=True),
            pd_filtered.reset_index(drop=True))

    def test_assign_then_filter_then_assign(self, sample_data):
        """Test: assign -> filter -> assign chain."""
        df, ds = sample_data
        
        # pandas
        pd_step1 = df.assign(C=df['A'] + df['B'])
        pd_step2 = pd_step1[pd_step1['C'] > 20]
        pd_result = pd_step2.assign(D=pd_step2['C'] * 2)
        
        # DataStore
        ds_step1 = ds.assign(C=ds['A'] + ds['B'])
        ds_step2 = ds_step1[ds_step1['C'] > 20]
        ds_result = ds_step2.assign(D=ds_step2['C'] * 2)
        
        # Compare
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))

    def test_filter_then_assign_then_filter_on_new_column(self, sample_data):
        """Test: filter -> assign -> filter (on new column) chain."""
        df, ds = sample_data
        
        # pandas
        pd_step1 = df[df['A'] > 1]
        pd_step2 = pd_step1.assign(C=pd_step1['A'] * pd_step1['B'])
        pd_result = pd_step2[pd_step2['C'] > 100]
        
        # DataStore
        ds_step1 = ds[ds['A'] > 1]
        ds_step2 = ds_step1.assign(C=ds_step1['A'] * ds_step1['B'])
        ds_result = ds_step2[ds_step2['C'] > 100]
        
        # Compare
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))
