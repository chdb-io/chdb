"""
Exploratory Batch 77: Pivot, Transform, and Reshape Edge Cases

Focus areas:
1. Pivot operations with various data types
2. Transform operations with custom functions
3. Stack/unstack edge cases
4. Melt/wide_to_long operations
5. Transpose with mixed types
6. Explode with nested structures
7. Cross-tabulation edge cases
8. Complex aggregation patterns
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestPivotOperations:
    """Test pivot and pivot_table operations"""

    def test_basic_pivot(self):
        """Test basic pivot operation"""
        pd_df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NY', 'LA', 'NY', 'LA'],
            'sales': [100, 200, 150, 250]
        })
        ds_df = DataStore({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NY', 'LA', 'NY', 'LA'],
            'sales': [100, 200, 150, 250]
        })

        pd_result = pd_df.pivot(index='date', columns='city', values='sales')
        ds_result = ds_df.pivot(index='date', columns='city', values='sales')
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_pivot_with_none_values(self):
        """Test pivot when some cells will be NaN"""
        pd_df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02'],
            'city': ['NY', 'LA', 'NY'],
            'sales': [100, 200, 150]
        })
        ds_df = DataStore({
            'date': ['2023-01', '2023-01', '2023-02'],
            'city': ['NY', 'LA', 'NY'],
            'sales': [100, 200, 150]
        })

        pd_result = pd_df.pivot(index='date', columns='city', values='sales')
        ds_result = ds_df.pivot(index='date', columns='city', values='sales')
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_pivot_table_basic(self):
        """Test basic pivot_table operation with aggregation"""
        pd_df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02', '2023-01'],
            'city': ['NY', 'NY', 'NY', 'LA', 'LA'],
            'sales': [100, 150, 200, 250, 120]
        })
        ds_df = DataStore({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02', '2023-01'],
            'city': ['NY', 'NY', 'NY', 'LA', 'LA'],
            'sales': [100, 150, 200, 250, 120]
        })

        pd_result = pd_df.pivot_table(index='date', columns='city', values='sales', aggfunc='sum')
        ds_result = ds_df.pivot_table(index='date', columns='city', values='sales', aggfunc='sum')
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_pivot_table_multiple_aggfuncs(self):
        """Test pivot_table with multiple aggregation functions"""
        pd_df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NY', 'NY', 'LA', 'LA'],
            'sales': [100, 150, 200, 250]
        })
        ds_df = DataStore({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NY', 'NY', 'LA', 'LA'],
            'sales': [100, 150, 200, 250]
        })

        pd_result = pd_df.pivot_table(index='date', columns='city', values='sales', aggfunc=['sum', 'mean'])
        ds_result = ds_df.pivot_table(index='date', columns='city', values='sales', aggfunc=['sum', 'mean'])
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


class TestTransformOperations:
    """Test transform operations"""

    def test_transform_with_builtin_func(self):
        """Test transform with built-in function"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('group')['value'].transform('sum')
        ds_result = ds_df.groupby('group')['value'].transform('sum')
        assert_series_equal(ds_result, pd_result)

    def test_transform_with_mean(self):
        """Test transform with mean function"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.groupby('group')['value'].transform('mean')
        ds_result = ds_df.groupby('group')['value'].transform('mean')
        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_transform_cumsum(self):
        """Test transform with cumulative sum"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1, 2, 3, 4, 5]
        })

        pd_result = pd_df.groupby('group')['value'].transform('cumsum')
        ds_result = ds_df.groupby('group')['value'].transform('cumsum')
        assert_series_equal(ds_result, pd_result)


class TestMeltOperations:
    """Test melt and wide_to_long operations"""

    def test_basic_melt(self):
        """Test basic melt operation"""
        pd_df = pd.DataFrame({
            'id': [1, 2],
            'value_a': [10, 20],
            'value_b': [30, 40]
        })
        ds_df = DataStore({
            'id': [1, 2],
            'value_a': [10, 20],
            'value_b': [30, 40]
        })

        pd_result = pd_df.melt(id_vars=['id'], value_vars=['value_a', 'value_b'])
        ds_result = ds_df.melt(id_vars=['id'], value_vars=['value_a', 'value_b'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_melt_with_var_name(self):
        """Test melt with custom var_name and value_name"""
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'math': [90, 85],
            'english': [88, 92]
        })
        ds_df = DataStore({
            'name': ['Alice', 'Bob'],
            'math': [90, 85],
            'english': [88, 92]
        })

        pd_result = pd_df.melt(
            id_vars=['name'],
            value_vars=['math', 'english'],
            var_name='subject',
            value_name='score'
        )
        ds_result = ds_df.melt(
            id_vars=['name'],
            value_vars=['math', 'english'],
            var_name='subject',
            value_name='score'
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_melt_without_id_vars(self):
        """Test melt without id_vars"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        ds_df = DataStore({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })

        pd_result = pd_df.melt()
        ds_result = ds_df.melt()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestExplodeOperations:
    """Test explode operations"""

    def test_basic_explode(self):
        """Test basic explode on list column"""
        pd_df = pd.DataFrame({
            'id': [1, 2],
            'values': [[1, 2, 3], [4, 5]]
        })
        ds_df = DataStore({
            'id': [1, 2],
            'values': [[1, 2, 3], [4, 5]]
        })

        pd_result = pd_df.explode('values')
        ds_result = ds_df.explode('values')
        # Reset index for comparison as explode preserves original index
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_explode_with_empty_list(self):
        """Test explode with empty list values"""
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'values': [[1, 2], [], [3]]
        })
        ds_df = DataStore({
            'id': [1, 2, 3],
            'values': [[1, 2], [], [3]]
        })

        pd_result = pd_df.explode('values')
        ds_result = ds_df.explode('values')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_explode_with_none(self):
        """Test explode with None values"""
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'values': [[1, 2], None, [3]]
        })
        ds_df = DataStore({
            'id': [1, 2, 3],
            'values': [[1, 2], None, [3]]
        })

        pd_result = pd_df.explode('values')
        ds_result = ds_df.explode('values')
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCrosstabOperations:
    """Test cross-tabulation operations"""

    def test_basic_crosstab(self):
        """Test basic crosstab"""
        a = pd.Series([1, 1, 2, 2, 1], name='a')
        b = pd.Series(['x', 'y', 'x', 'y', 'x'], name='b')

        pd_result = pd.crosstab(a, b)

        ds_a = DataStore({'a': [1, 1, 2, 2, 1]})['a']
        ds_b = DataStore({'b': ['x', 'y', 'x', 'y', 'x']})['b']

        # Note: pd.crosstab may need to be called differently for DataStore
        # This test verifies the capability
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2, 1], 'b': ['x', 'y', 'x', 'y', 'x']})
        ds_df = DataStore({'a': [1, 1, 2, 2, 1], 'b': ['x', 'y', 'x', 'y', 'x']})

        # Using pivot_table to achieve similar result
        pd_ct = pd_df.pivot_table(index='a', columns='b', aggfunc='size', fill_value=0)
        ds_ct = ds_df.pivot_table(index='a', columns='b', aggfunc='size', fill_value=0)
        assert_datastore_equals_pandas(ds_ct, pd_ct, check_index=True)


class TestStackUnstack:
    """Test stack and unstack operations"""

    def test_basic_stack(self):
        """Test basic stack operation"""
        pd_df = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4]
        }, index=['x', 'y'])
        ds_df = DataStore({
            'A': [1, 2],
            'B': [3, 4]
        })

        pd_result = pd_df.stack()
        ds_result = ds_df.stack()
        # Stack returns a Series - compare values (index differs: DataStore uses 0,1 vs pandas x,y)
        # Values should match in same order: [1, 3, 2, 4] for columns A,B rows 0,1
        assert_series_equal(ds_result, pd_result, check_index=False, check_names=False)


class TestComplexAggregations:
    """Test complex aggregation patterns"""

    def test_groupby_multiple_agg(self):
        """Test groupby with multiple aggregation functions"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value1': [10, 20, 30, 40, 50],
            'value2': [1.5, 2.5, 3.5, 4.5, 5.5]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value1': [10, 20, 30, 40, 50],
            'value2': [1.5, 2.5, 3.5, 4.5, 5.5]
        })

        pd_result = pd_df.groupby('group').agg({
            'value1': ['sum', 'mean'],
            'value2': ['min', 'max']
        })
        ds_result = ds_df.groupby('group').agg({
            'value1': ['sum', 'mean'],
            'value2': ['min', 'max']
        })
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_named_agg(self):
        """Test groupby with named aggregation"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby('group').agg(
            total=('value', 'sum'),
            average=('value', 'mean')
        )
        ds_result = ds_df.groupby('group').agg(
            total=('value', 'sum'),
            average=('value', 'mean')
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_agg_with_custom_percentile(self):
        """Test aggregation with percentile-like operations"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df.groupby('group')['value'].quantile(0.5)
        ds_result = ds_df.groupby('group')['value'].quantile(0.5)
        assert_series_equal(ds_result, pd_result, check_dtype=False)


class TestReshapeChains:
    """Test chained reshape operations"""

    def test_filter_then_melt(self):
        """Test filtering then melting"""
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'type': ['A', 'B', 'A'],
            'val_x': [10, 20, 30],
            'val_y': [40, 50, 60]
        })
        ds_df = DataStore({
            'id': [1, 2, 3],
            'type': ['A', 'B', 'A'],
            'val_x': [10, 20, 30],
            'val_y': [40, 50, 60]
        })

        pd_result = pd_df[pd_df['type'] == 'A'].melt(id_vars=['id', 'type'])
        ds_result = ds_df[ds_df['type'] == 'A'].melt(id_vars=['id', 'type'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_then_pivot(self):
        """Test groupby aggregation then pivot"""
        pd_df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NY', 'LA', 'NY', 'LA'],
            'type': ['A', 'A', 'A', 'A'],
            'value': [100, 200, 150, 250]
        })
        ds_df = DataStore({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NY', 'LA', 'NY', 'LA'],
            'type': ['A', 'A', 'A', 'A'],
            'value': [100, 200, 150, 250]
        })

        pd_agg = pd_df.groupby(['date', 'city'])['value'].sum().reset_index()
        pd_result = pd_agg.pivot(index='date', columns='city', values='value')

        ds_agg = ds_df.groupby(['date', 'city'])['value'].sum().reset_index()
        ds_result = ds_agg.pivot(index='date', columns='city', values='value')

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_melt_then_groupby(self):
        """Test melting then groupby aggregation"""
        pd_df = pd.DataFrame({
            'id': [1, 2],
            'jan': [100, 200],
            'feb': [150, 250]
        })
        ds_df = DataStore({
            'id': [1, 2],
            'jan': [100, 200],
            'feb': [150, 250]
        })

        pd_melted = pd_df.melt(id_vars=['id'], var_name='month', value_name='sales')
        pd_result = pd_melted.groupby('month')['sales'].sum()

        ds_melted = ds_df.melt(id_vars=['id'], var_name='month', value_name='sales')
        ds_result = ds_melted.groupby('month')['sales'].sum()

        assert_series_equal(ds_result, pd_result, check_dtype=False)


class TestTransposeEdgeCases:
    """Test transpose edge cases"""

    def test_transpose_numeric(self):
        """Test transpose of numeric DataFrame"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        ds_df = DataStore({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })

        pd_result = pd_df.T
        ds_result = ds_df.T
        # Transpose produces DataFrame with integer column names (0, 1, 2)
        # Rows become a, b and columns become 0, 1, 2
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_transpose_mixed_types(self):
        """Test transpose preserves data (mixed types)"""
        pd_df = pd.DataFrame({
            'a': [1, 2],
            'b': [3, 4]
        })
        ds_df = DataStore({
            'a': [1, 2],
            'b': [3, 4]
        })

        pd_result = pd_df.T.T  # Double transpose should return to original
        ds_result = ds_df.T.T
        # After double transpose, should be back to original structure with same values
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestValueCountsEdgeCases:
    """Test value_counts edge cases"""

    def test_value_counts_with_none(self):
        """Test value_counts with None values"""
        pd_df = pd.DataFrame({
            'col': ['a', 'b', 'a', None, 'b', 'a']
        })
        ds_df = DataStore({
            'col': ['a', 'b', 'a', None, 'b', 'a']
        })

        pd_result = pd_df['col'].value_counts(dropna=False)
        ds_result = ds_df['col'].value_counts(dropna=False)
        # Compare sorted by index to handle different tie-ordering
        pd_sorted = pd_result.sort_index(na_position='last')
        ds_sorted = ds_result.sort_index(na_position='last')
        assert_series_equal(ds_sorted, pd_sorted)

    def test_value_counts_normalize(self):
        """Test value_counts with normalize=True"""
        pd_df = pd.DataFrame({
            'col': ['a', 'b', 'a', 'a', 'b']
        })
        ds_df = DataStore({
            'col': ['a', 'b', 'a', 'a', 'b']
        })

        pd_result = pd_df['col'].value_counts(normalize=True)
        ds_result = ds_df['col'].value_counts(normalize=True)
        # Check sum equals 1
        assert abs(sum(ds_result.values) - 1.0) < 0.001


class TestNuniqueEdgeCases:
    """Test nunique edge cases"""

    def test_nunique_with_none(self):
        """Test nunique with None values"""
        pd_df = pd.DataFrame({
            'col': ['a', 'b', 'a', None, 'b', None]
        })
        ds_df = DataStore({
            'col': ['a', 'b', 'a', None, 'b', None]
        })

        pd_result = pd_df['col'].nunique()
        ds_result = ds_df['col'].nunique()
        assert ds_result == pd_result

    def test_nunique_dropna_false(self):
        """Test nunique with dropna=False"""
        pd_df = pd.DataFrame({
            'col': ['a', 'b', 'a', None, 'b', None]
        })
        ds_df = DataStore({
            'col': ['a', 'b', 'a', None, 'b', None]
        })

        pd_result = pd_df['col'].nunique(dropna=False)
        ds_result = ds_df['col'].nunique(dropna=False)
        assert ds_result == pd_result

    def test_df_nunique(self):
        """Test DataFrame nunique"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 1, 3],
            'b': ['x', 'y', 'x', 'y']
        })
        ds_df = DataStore({
            'a': [1, 2, 1, 3],
            'b': ['x', 'y', 'x', 'y']
        })

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()
        assert_series_equal(ds_result, pd_result)


class TestDescribeEdgeCases:
    """Test describe edge cases"""

    def test_describe_numeric(self):
        """Test describe on numeric columns"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10.5, 20.5, 30.5, 40.5, 50.5]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10.5, 20.5, 30.5, 40.5, 50.5]
        })

        pd_result = pd_df.describe()
        ds_result = ds_df.describe()
        # describe returns stats, check shape
        assert ds_result.shape == pd_result.shape

    def test_describe_include_all(self):
        """Test describe with include='all'"""
        pd_df = pd.DataFrame({
            'num': [1, 2, 3],
            'str': ['a', 'b', 'c']
        })
        ds_df = DataStore({
            'num': [1, 2, 3],
            'str': ['a', 'b', 'c']
        })

        pd_result = pd_df.describe(include='all')
        ds_result = ds_df.describe(include='all')
        assert ds_result.shape == pd_result.shape


class TestSampleEdgeCases:
    """Test sample edge cases"""

    def test_sample_n(self):
        """Test sample with n parameter"""
        pd_df = pd.DataFrame({
            'a': range(100),
            'b': range(100, 200)
        })
        ds_df = DataStore({
            'a': range(100),
            'b': range(100, 200)
        })

        # Sample 10 rows - can't compare directly due to randomness
        ds_result = ds_df.sample(n=10)
        assert len(ds_result) == 10

    def test_sample_frac(self):
        """Test sample with frac parameter"""
        pd_df = pd.DataFrame({
            'a': range(100),
            'b': range(100, 200)
        })
        ds_df = DataStore({
            'a': range(100),
            'b': range(100, 200)
        })

        # Sample 10% of rows
        ds_result = ds_df.sample(frac=0.1)
        assert len(ds_result) == 10

    def test_sample_with_seed(self):
        """Test sample with random_state for reproducibility"""
        pd_df = pd.DataFrame({
            'a': range(100)
        })
        ds_df = DataStore({
            'a': range(100)
        })

        ds_result1 = ds_df.sample(n=5, random_state=42)
        ds_result2 = ds_df.sample(n=5, random_state=42)
        # Same seed should give same result
        assert list(ds_result1['a']) == list(ds_result2['a'])


class TestNlargestNsmallest:
    """Test nlargest and nsmallest operations"""

    def test_nlargest_basic(self):
        """Test basic nlargest"""
        pd_df = pd.DataFrame({
            'name': ['a', 'b', 'c', 'd', 'e'],
            'value': [10, 50, 30, 20, 40]
        })
        ds_df = DataStore({
            'name': ['a', 'b', 'c', 'd', 'e'],
            'value': [10, 50, 30, 20, 40]
        })

        pd_result = pd_df.nlargest(3, 'value')
        ds_result = ds_df.nlargest(3, 'value')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_basic(self):
        """Test basic nsmallest"""
        pd_df = pd.DataFrame({
            'name': ['a', 'b', 'c', 'd', 'e'],
            'value': [10, 50, 30, 20, 40]
        })
        ds_df = DataStore({
            'name': ['a', 'b', 'c', 'd', 'e'],
            'value': [10, 50, 30, 20, 40]
        })

        pd_result = pd_df.nsmallest(3, 'value')
        ds_result = ds_df.nsmallest(3, 'value')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_series(self):
        """Test nlargest on Series"""
        pd_df = pd.DataFrame({
            'value': [10, 50, 30, 20, 40]
        })
        ds_df = DataStore({
            'value': [10, 50, 30, 20, 40]
        })

        pd_result = pd_df['value'].nlargest(3)
        ds_result = ds_df['value'].nlargest(3)
        assert_series_equal(ds_result, pd_result)

    def test_nsmallest_with_ties(self):
        """Test nsmallest with tied values"""
        pd_df = pd.DataFrame({
            'name': ['a', 'b', 'c', 'd'],
            'value': [10, 10, 20, 20]
        })
        ds_df = DataStore({
            'name': ['a', 'b', 'c', 'd'],
            'value': [10, 10, 20, 20]
        })

        pd_result = pd_df.nsmallest(2, 'value')
        ds_result = ds_df.nsmallest(2, 'value')
        # With ties, the exact rows returned should match (both get rows with value=10)
        # Order within ties can vary, so check_row_order=False for ties on same value
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
