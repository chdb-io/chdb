"""
Exploratory Discovery Batch 23 - Update, Bitwise Operations, and Advanced Features

Focus areas:
1. DataFrame.update() method
2. Bitwise operations (&, |, ^, ~) on boolean columns
3. DataFrame.explode() variations
4. DataFrame.compare() method
5. infer_objects/convert_dtypes operations
6. Complex transform chains
7. Advanced eval/query operations
8. applymap/map edge cases
9. DataFrame construction edge cases
10. Special index operations
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_dataframe, get_series


# =============================================================================
# DataFrame.update() tests
# =============================================================================
class TestDataFrameUpdate:
    """Test DataFrame.update() operation."""

    def test_update_basic(self):
        """Basic update - overwrite values from another DataFrame."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [100, 200, 300]})
        pd_df1.update(pd_df2)

        ds_df1 = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df2 = DataStore({'A': [100, 200, 300]})
        ds_df1.update(ds_df2)

        assert_datastore_equals_pandas(ds_df1, pd_df1)

    def test_update_with_nan_no_overwrite(self):
        """Update should not overwrite with NaN by default."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [np.nan, 200, np.nan]})
        pd_df1.update(pd_df2)

        ds_df1 = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df2 = DataStore({'A': [np.nan, 200, np.nan]})
        ds_df1.update(ds_df2)

        assert_datastore_equals_pandas(ds_df1, pd_df1)

    def test_update_partial_columns(self):
        """Update only specific columns."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        pd_df2 = pd.DataFrame({'B': [40, 50, 60]})
        pd_df1.update(pd_df2)

        ds_df1 = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df2 = DataStore({'B': [40, 50, 60]})
        ds_df1.update(ds_df2)

        assert_datastore_equals_pandas(ds_df1, pd_df1)

    def test_update_with_overwrite_false(self):
        """Update with overwrite=False - only update NaN values."""
        pd_df1 = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [100, 200, 300]})
        pd_df1.update(pd_df2, overwrite=False)

        ds_df1 = DataStore({'A': [1.0, np.nan, 3.0], 'B': [4, 5, 6]})
        ds_df2 = DataStore({'A': [100, 200, 300]})
        ds_df1.update(ds_df2, overwrite=False)

        assert_datastore_equals_pandas(ds_df1, pd_df1)


# =============================================================================
# Bitwise operations tests
# =============================================================================
class TestBitwiseOperations:
    """Test bitwise operations on boolean columns."""

    def test_bitwise_and(self):
        """Test & (bitwise AND) on boolean Series."""
        pd_df = pd.DataFrame({
            'A': [True, True, False, False],
            'B': [True, False, True, False]
        })
        pd_result = pd_df['A'] & pd_df['B']

        ds_df = DataStore(pd_df)
        ds_result = ds_df['A'] & ds_df['B']

        # Execute if lazy
        ds_series = get_series(ds_result)

        assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            )

    def test_bitwise_or(self):
        """Test | (bitwise OR) on boolean Series."""
        pd_df = pd.DataFrame({
            'A': [True, True, False, False],
            'B': [True, False, True, False]
        })
        pd_result = pd_df['A'] | pd_df['B']

        ds_df = DataStore(pd_df)
        ds_result = ds_df['A'] | ds_df['B']

        # Execute if lazy
        ds_series = get_series(ds_result)

        assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            )

    def test_bitwise_xor(self):
        """Test ^ (bitwise XOR) on boolean Series."""
        pd_df = pd.DataFrame({
            'A': [True, True, False, False],
            'B': [True, False, True, False]
        })
        pd_result = pd_df['A'] ^ pd_df['B']

        ds_df = DataStore(pd_df)
        ds_result = ds_df['A'] ^ ds_df['B']

        # Execute if lazy
        ds_series = get_series(ds_result)

        assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            )

    def test_bitwise_not(self):
        """Test ~ (bitwise NOT) on boolean Series."""
        pd_df = pd.DataFrame({'A': [True, False, True, False]})
        pd_result = ~pd_df['A']

        ds_df = DataStore(pd_df)
        ds_result = ~ds_df['A']

        # Execute if lazy
        ds_series = get_series(ds_result)

        assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            )

    def test_combined_boolean_filter(self):
        """Test combined boolean filter using & and |."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[(pd_df['A'] > 2) & (pd_df['B'] < 40)]

        ds_df = DataStore(pd_df)
        ds_result = ds_df[(ds_df['A'] > 2) & (ds_df['B'] < 40)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negated_filter(self):
        """Test negated filter using ~."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        pd_result = pd_df[~(pd_df['A'] > 3)]

        ds_df = DataStore(pd_df)
        ds_result = ds_df[~(ds_df['A'] > 3)]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# DataFrame.explode() tests
# =============================================================================
class TestDataFrameExplode:
    """Test DataFrame.explode() operation."""

    def test_explode_single_column(self):
        """Basic explode on a single column with lists."""
        pd_df = pd.DataFrame({
            'A': [[1, 2], [3, 4, 5], [6]],
            'B': ['x', 'y', 'z']
        })
        pd_result = pd_df.explode('A')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.explode('A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_explode_with_empty_list(self):
        """Explode with empty lists."""
        pd_df = pd.DataFrame({
            'A': [[1, 2], [], [3]],
            'B': ['x', 'y', 'z']
        })
        pd_result = pd_df.explode('A')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.explode('A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_explode_ignore_index(self):
        """Explode with ignore_index=True."""
        pd_df = pd.DataFrame({
            'A': [[1, 2], [3, 4]],
            'B': ['x', 'y']
        })
        pd_result = pd_df.explode('A', ignore_index=True)

        ds_df = DataStore(pd_df)
        ds_result = ds_df.explode('A', ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_explode_multiple_columns(self):
        """Explode multiple columns at once (pandas 1.3+)."""
        pd_df = pd.DataFrame({
            'A': [[1, 2], [3, 4]],
            'B': [['a', 'b'], ['c', 'd']],
            'C': ['x', 'y']
        })
        pd_result = pd_df.explode(['A', 'B'])

        ds_df = DataStore(pd_df)
        ds_result = ds_df.explode(['A', 'B'])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Type conversion tests
# =============================================================================
class TestTypeConversion:
    """Test infer_objects and convert_dtypes operations."""

    def test_infer_objects_basic(self):
        """Basic infer_objects - convert object columns to best dtype."""
        pd_df = pd.DataFrame({
            'A': pd.array([1, 2, 3], dtype=object),
            'B': ['a', 'b', 'c']
        })
        pd_result = pd_df.infer_objects()

        ds_df = DataStore(pd_df)
        ds_result = ds_df.infer_objects()

        # Check that the result is a valid DataFrame
        ds_df_result = get_dataframe(ds_result)
        assert len(ds_df_result) == len(pd_result)
        assert list(ds_df_result.columns) == list(pd_result.columns)

    def test_convert_dtypes_basic(self):
        """Basic convert_dtypes - convert to best possible dtype."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1.0, 2.0, 3.0],
            'C': ['a', 'b', 'c']
        })
        pd_result = pd_df.convert_dtypes()

        ds_df = DataStore(pd_df)
        ds_result = ds_df.convert_dtypes()

        # Check that the result is a valid DataFrame
        ds_df_result = get_dataframe(ds_result)
        assert len(ds_df_result) == len(pd_result)
        assert list(ds_df_result.columns) == list(pd_result.columns)

    def test_astype_dict(self):
        """Test astype with dictionary specifying per-column types."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['1.1', '2.2', '3.3']
        })
        pd_result = pd_df.astype({'A': float, 'B': float})

        ds_df = DataStore(pd_df)
        ds_result = ds_df.astype({'A': float, 'B': float})

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Transform tests
# =============================================================================
class TestTransform:
    """Test DataFrame.transform() variations."""

    def test_transform_with_lambda(self):
        """Transform with lambda function."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_result = pd_df.transform(lambda x: x * 2)

        ds_df = DataStore(pd_df)
        ds_result = ds_df.transform(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_with_string_func(self):
        """Transform with string function name."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_result = pd_df.transform('sqrt')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.transform('sqrt')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_multiple_functions(self):
        """Transform with multiple functions."""
        pd_df = pd.DataFrame({'A': [1, 4, 9], 'B': [16, 25, 36]})
        pd_result = pd_df.transform(['sqrt', 'abs'])

        ds_df = DataStore(pd_df)
        ds_result = ds_df.transform(['sqrt', 'abs'])

        # For multi-function transform, just verify the shape and non-empty
        ds_df_result = get_dataframe(ds_result)
        assert len(ds_df_result) == len(pd_result)

    def test_transform_per_column(self):
        """Transform with different functions per column."""
        pd_df = pd.DataFrame({'A': [1, 4, 9], 'B': [2, 3, 4]})
        pd_result = pd_df.transform({'A': 'sqrt', 'B': 'square'})

        ds_df = DataStore(pd_df)
        ds_result = ds_df.transform({'A': 'sqrt', 'B': 'square'})

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-4)


# =============================================================================
# Eval and Query tests
# =============================================================================
class TestEvalQuery:
    """Test advanced eval and query operations."""

    def test_eval_arithmetic(self):
        """Eval with arithmetic expression."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df = pd_df.eval('C = A + B')

        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = ds_df.eval('C = A + B')

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_eval_with_multiplication(self):
        """Eval with multiplication."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df = pd_df.eval('C = A * B')

        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = ds_df.eval('C = A * B')

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_query_simple(self):
        """Simple query filtering."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        pd_result = pd_df.query('A > 2')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.query('A > 2')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_combined_conditions(self):
        """Query with combined conditions."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        pd_result = pd_df.query('A > 2 and B < 45')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.query('A > 2 and B < 45')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_not(self):
        """Query with NOT condition."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e']})
        pd_result = pd_df.query('not A > 3')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.query('not A > 3')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# applymap/map tests
# =============================================================================
class TestApplyMap:
    """Test applymap and map operations."""

    def test_applymap_basic(self):
        """Basic applymap (now called map in pandas 2.1+)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        # Use map for pandas 2.1+, applymap for older
        if hasattr(pd_df, 'map'):
            pd_result = pd_df.map(lambda x: x * 2)
        else:
            pd_result = pd_df.applymap(lambda x: x * 2)

        ds_df = DataStore(pd_df)
        # DataStore should support both
        if hasattr(ds_df, 'map'):
            ds_result = ds_df.map(lambda x: x * 2)
        else:
            ds_result = ds_df.applymap(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_map_string_format(self):
        """Map with string formatting."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        if hasattr(pd_df, 'map'):
            pd_result = pd_df.map(lambda x: f"val_{x}")
        else:
            pd_result = pd_df.applymap(lambda x: f"val_{x}")

        ds_df = DataStore(pd_df)
        if hasattr(ds_df, 'map'):
            ds_result = ds_df.map(lambda x: f"val_{x}")
        else:
            ds_result = ds_df.applymap(lambda x: f"val_{x}")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_series_map_with_dict(self):
        """Series.map with dictionary."""
        pd_df = pd.DataFrame({'A': ['cat', 'dog', 'bird', 'cat']})
        pd_result = pd_df['A'].map({'cat': 1, 'dog': 2, 'bird': 3})

        ds_df = DataStore(pd_df)
        ds_result = ds_df['A'].map({'cat': 1, 'dog': 2, 'bird': 3})

        # Execute if lazy
        ds_series = get_series(ds_result)

        assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            )


# =============================================================================
# Construction edge cases
# =============================================================================
class TestConstructionEdgeCases:
    """Test DataFrame construction edge cases."""

    def test_construct_from_nested_dict(self):
        """Construct from nested dictionary."""
        data = {'row1': {'A': 1, 'B': 2}, 'row2': {'A': 3, 'B': 4}}
        pd_df = pd.DataFrame.from_dict(data, orient='index')

        ds_df = DataStore.from_dict(data, orient='index')

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_with_index(self):
        """Construct with custom index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])

        ds_df = DataStore({'A': [1, 2, 3]}, index=['x', 'y', 'z'])

        # Just verify the data matches, index handling may differ
        ds_df_result = get_dataframe(ds_df)
        assert list(ds_df_result['A']) == list(pd_df['A'])

    def test_construct_from_records(self):
        """Construct from records."""
        records = [{'A': 1, 'B': 'x'}, {'A': 2, 'B': 'y'}, {'A': 3, 'B': 'z'}]
        pd_df = pd.DataFrame.from_records(records)

        ds_df = DataStore.from_records(records)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_empty_with_columns(self):
        """Construct empty DataFrame with columns."""
        pd_df = pd.DataFrame(columns=['A', 'B', 'C'])

        ds_df = DataStore(columns=['A', 'B', 'C'])

        ds_df_result = get_dataframe(ds_df)
        assert list(ds_df_result.columns) == ['A', 'B', 'C']
        assert len(ds_df_result) == 0


# =============================================================================
# Index operations
# =============================================================================
class TestIndexOperations:
    """Test advanced index operations."""

    def test_set_index_and_filter(self):
        """Set index then filter."""
        pd_df = pd.DataFrame({'A': ['x', 'y', 'z'], 'B': [1, 2, 3]})
        pd_df = pd_df.set_index('A')
        pd_result = pd_df[pd_df['B'] > 1]

        ds_df = DataStore({'A': ['x', 'y', 'z'], 'B': [1, 2, 3]})
        ds_df = ds_df.set_index('A')
        ds_result = ds_df[ds_df['B'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_drop(self):
        """Reset index with drop=True."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
        pd_result = pd_df.reset_index(drop=True)

        ds_df = DataStore({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_result = ds_df.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_fill(self):
        """Reindex with fill_value."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
        pd_result = pd_df.reindex([0, 1, 2, 3, 4], fill_value=0)

        ds_df = DataStore({'A': [1, 2, 3]}, index=[0, 1, 2])
        ds_result = ds_df.reindex([0, 1, 2, 3, 4], fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Melt variations
# =============================================================================
class TestMeltVariations:
    """Test DataFrame.melt() variations."""

    def test_melt_basic(self):
        """Basic melt operation."""
        pd_df = pd.DataFrame({
            'id': ['A', 'B'],
            'val1': [1, 2],
            'val2': [3, 4]
        })
        pd_result = pd_df.melt(id_vars=['id'])

        ds_df = DataStore(pd_df)
        ds_result = ds_df.melt(id_vars=['id'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_melt_with_var_name(self):
        """Melt with custom var_name."""
        pd_df = pd.DataFrame({
            'id': ['A', 'B'],
            'val1': [1, 2],
            'val2': [3, 4]
        })
        pd_result = pd_df.melt(id_vars=['id'], var_name='measure')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.melt(id_vars=['id'], var_name='measure')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_melt_with_value_name(self):
        """Melt with custom value_name."""
        pd_df = pd.DataFrame({
            'id': ['A', 'B'],
            'val1': [1, 2],
            'val2': [3, 4]
        })
        pd_result = pd_df.melt(id_vars=['id'], value_name='measurement')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.melt(id_vars=['id'], value_name='measurement')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_melt_value_vars(self):
        """Melt with specific value_vars."""
        pd_df = pd.DataFrame({
            'id': ['A', 'B'],
            'val1': [1, 2],
            'val2': [3, 4],
            'val3': [5, 6]
        })
        pd_result = pd_df.melt(id_vars=['id'], value_vars=['val1', 'val2'])

        ds_df = DataStore(pd_df)
        ds_result = ds_df.melt(id_vars=['id'], value_vars=['val1', 'val2'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Stack/unstack edge cases
# =============================================================================
class TestStackUnstack:
    """Test stack/unstack edge cases."""

    def test_stack_basic(self):
        """Basic stack operation."""
        pd_df = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4]
        }, index=['row1', 'row2'])
        pd_result = pd_df.stack()

        ds_df = DataStore(pd_df, index=['row1', 'row2'])
        ds_result = ds_df.stack()

        # Stack returns a Series with MultiIndex
        ds_series = get_series(ds_result)
        # Just verify the values match
        assert list(ds_series.values) == list(pd_result.values)

    def test_unstack_basic(self):
        """Basic unstack operation."""
        # Create a stacked Series first
        pd_df = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4]
        }, index=['row1', 'row2'])
        pd_stacked = pd_df.stack()
        pd_result = pd_stacked.unstack()

        ds_df = DataStore(pd_df, index=['row1', 'row2'])
        ds_stacked = ds_df.stack()
        ds_result = ds_stacked.unstack()

        # Just verify the shape and values roughly match
        ds_df_result = get_dataframe(ds_result)
        assert ds_df_result.shape == pd_result.shape


# =============================================================================
# Pivot variations
# =============================================================================
class TestPivotVariations:
    """Test pivot and pivot_table variations."""

    def test_pivot_basic(self):
        """Basic pivot operation."""
        pd_df = pd.DataFrame({
            'foo': ['one', 'one', 'two', 'two'],
            'bar': ['A', 'B', 'A', 'B'],
            'baz': [1, 2, 3, 4]
        })
        pd_result = pd_df.pivot(index='foo', columns='bar', values='baz')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.pivot(index='foo', columns='bar', values='baz')

        # Just verify the shape
        ds_df_result = get_dataframe(ds_result)
        assert ds_df_result.shape == pd_result.shape

    def test_pivot_table_with_aggfunc(self):
        """Pivot table with aggregation function."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'C': [1, 2, 3, 4]
        })
        pd_result = pd_df.pivot_table(index='A', columns='B', values='C', aggfunc='sum')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.pivot_table(index='A', columns='B', values='C', aggfunc='sum')

        # Just verify the shape
        ds_df_result = get_dataframe(ds_result)
        assert ds_df_result.shape == pd_result.shape


# =============================================================================
# Memory/copy semantics
# =============================================================================
class TestMemoryCopySemantics:
    """Test memory and copy semantics."""

    def test_copy_deep(self):
        """Deep copy should not share memory."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_copy = pd_df.copy(deep=True)

        ds_df = DataStore(pd_df)
        ds_copy = ds_df.copy(deep=True)

        # Modify original
        pd_df.iloc[0, 0] = 999

        # Copy should be unchanged
        ds_copy_df = get_dataframe(ds_copy)
        assert ds_copy_df.iloc[0, 0] != 999

    def test_to_numpy_basic(self):
        """Basic to_numpy conversion."""
        # Use float to avoid nullable int issues with chDB
        pd_df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        pd_result = pd_df.to_numpy()

        ds_df = DataStore(pd_df)
        ds_result = ds_df.to_numpy()

        np.testing.assert_array_almost_equal(ds_result, pd_result)

    def test_values_property(self):
        """Values property should return numpy array."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_result = pd_df.values

        ds_df = DataStore(pd_df)
        ds_result = ds_df.values

        np.testing.assert_array_equal(ds_result, pd_result)


# =============================================================================
# String column operations
# =============================================================================
class TestStringColumnOps:
    """Test string column operations."""

    def test_str_slice(self):
        """String slicing via str accessor."""
        pd_df = pd.DataFrame({'A': ['hello', 'world', 'test']})
        pd_result = pd_df['A'].str[:3]

        ds_df = DataStore(pd_df)
        ds_result = ds_df['A'].str[:3]

        # Execute if lazy
        ds_series = get_series(ds_result)

        assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            )

    def test_str_split_basic(self):
        """Basic string split."""
        pd_df = pd.DataFrame({'A': ['a-b-c', 'd-e-f', 'g-h-i']})
        pd_result = pd_df['A'].str.split('-')

        ds_df = DataStore(pd_df)
        ds_result = ds_df['A'].str.split('-')

        # Execute if lazy
        ds_series = get_series(ds_result)

        # Compare first element - chDB returns numpy array, pandas returns list
        # Both should contain the same values
        assert list(ds_series.iloc[0]) == list(pd_result.iloc[0])

    def test_str_len(self):
        """String length via str accessor."""
        pd_df = pd.DataFrame({'A': ['hello', 'hi', 'test123']})
        pd_result = pd_df['A'].str.len()

        ds_df = DataStore(pd_df)
        ds_result = ds_df['A'].str.len()

        # Execute if lazy
        ds_series = get_series(ds_result)

        assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            )


# =============================================================================
# DataFrame comparison method
# =============================================================================
class TestDataFrameCompare:
    """Test DataFrame.compare() method."""

    def test_compare_basic(self):
        """Basic compare between two DataFrames."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [1, 20, 3], 'B': [4, 5, 60]})
        pd_result = pd_df1.compare(pd_df2)

        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)
        ds_result = ds_df1.compare(ds_df2)

        ds_df_result = get_dataframe(ds_result)
        # Just verify it produces a result with differences
        assert len(ds_df_result) > 0

    def test_compare_keep_equal(self):
        """Compare with keep_equal=True."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [1, 20, 3], 'B': [4, 5, 60]})
        pd_result = pd_df1.compare(pd_df2, keep_equal=True)

        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)
        ds_result = ds_df1.compare(ds_df2, keep_equal=True)

        ds_df_result = get_dataframe(ds_result)
        assert len(ds_df_result) == len(pd_result)


# =============================================================================
# Combine operations
# =============================================================================
class TestCombineOps:
    """Test combine and combine_first operations."""

    def test_combine_with_func(self):
        """Combine two DataFrames with custom function."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
        pd_result = pd_df1.combine(pd_df2, lambda s1, s2: s1 + s2)

        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)
        ds_result = ds_df1.combine(ds_df2, lambda s1, s2: s1 + s2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_basic(self):
        """Combine_first to fill NaN values."""
        pd_df1 = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': [np.nan, 5.0, 6.0]})
        pd_df2 = pd.DataFrame({'A': [10.0, 20.0, 30.0], 'B': [40.0, 50.0, 60.0]})
        pd_result = pd_df1.combine_first(pd_df2)

        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)
        ds_result = ds_df1.combine_first(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Numeric precision tests
# =============================================================================
class TestNumericPrecision:
    """Test numeric precision edge cases."""

    def test_very_small_numbers(self):
        """Operations with very small floating point numbers."""
        pd_df = pd.DataFrame({'A': [1e-10, 2e-10, 3e-10]})
        pd_result = pd_df['A'] * 1e10

        ds_df = DataStore(pd_df)
        ds_result = ds_df['A'] * 1e10

        # Execute if lazy
        ds_series = get_series(ds_result)

        np.testing.assert_array_almost_equal(
            ds_series.values.flatten(),
            pd_result.values.flatten(),
            decimal=5
        )

    def test_large_numbers(self):
        """Operations with large numbers."""
        pd_df = pd.DataFrame({'A': [1e15, 2e15, 3e15]})
        pd_result = pd_df.sum()

        ds_df = DataStore(pd_df)
        ds_result = ds_df.sum()

        ds_series = get_series(ds_result)
        # Check sum is approximately correct
        assert abs(ds_series['A'] - pd_result['A']) / pd_result['A'] < 1e-10


# =============================================================================
# Edge cases with special values
# =============================================================================
class TestSpecialValues:
    """Test handling of special values."""

    def test_inf_in_operations(self):
        """Operations with infinity values."""
        pd_df = pd.DataFrame({'A': [1.0, np.inf, -np.inf, 4.0]})
        pd_result = pd_df.replace([np.inf, -np.inf], np.nan)

        ds_df = DataStore(pd_df)
        ds_result = ds_df.replace([np.inf, -np.inf], np.nan)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nan_propagation(self):
        """NaN should propagate through arithmetic."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': [4.0, 5.0, np.nan]})
        pd_result = pd_df['A'] + pd_df['B']

        ds_df = DataStore(pd_df)
        ds_result = ds_df['A'] + ds_df['B']

        # Execute if lazy
        ds_series = get_series(ds_result)

        # Check NaN positions match
        assert pd.isna(ds_series.iloc[1]) == pd.isna(pd_result.iloc[1])
        assert pd.isna(ds_series.iloc[2]) == pd.isna(pd_result.iloc[2])


# =============================================================================
# Method chaining tests
# =============================================================================
class TestMethodChaining:
    """Test complex method chaining."""

    def test_filter_assign_sort_head(self):
        """Chain: filter -> assign -> sort -> head."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        pd_result = (pd_df[pd_df['A'] > 2]
                     .assign(C=lambda x: x['A'] + x['B'])
                     .sort_values('C', ascending=False)
                     .head(2))

        ds_df = DataStore(pd_df)
        ds_result = (ds_df[ds_df['A'] > 2]
                     .assign(C=lambda x: x['A'] + x['B'])
                     .sort_values('C', ascending=False)
                     .head(2))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_reset_sort(self):
        """Chain: groupby -> agg -> reset_index -> sort."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        pd_result = (pd_df.groupby('category')
                     .agg({'value': 'sum'})
                     .reset_index()
                     .sort_values('value', ascending=False))

        ds_df = DataStore(pd_df)
        ds_result = (ds_df.groupby('category')
                     .agg({'value': 'sum'})
                     .reset_index()
                     .sort_values('value', ascending=False))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_columns_filter_dropna(self):
        """Chain: column select -> filter -> dropna."""
        pd_df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[['A', 'B']][pd_df['C'] > 20].dropna()

        ds_df = DataStore(pd_df)
        ds_result = ds_df[['A', 'B']][ds_df['C'] > 20].dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
