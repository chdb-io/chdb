"""
Exploratory Batch 85: Deep Chain Operations and Boundary Conditions

Focus areas:
1. Deep chain operations (5+ layers of lazy operations)
2. Setter/getter chains with complex expressions
3. SQL aggregation + window function combinations
4. Boundary values: empty DataFrame, single row, many columns
5. Complex filter chains with various dtypes
6. Nested subquery patterns
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestDeepChainOperations:
    """Test deep chains of lazy operations (5+ operations)"""

    def test_deep_filter_chain(self):
        """5+ filter operations in sequence"""
        pd_df = pd.DataFrame({
            'a': range(100),
            'b': range(100, 200),
            'c': [i % 10 for i in range(100)]
        })
        ds_df = DataStore({
            'a': range(100),
            'b': range(100, 200),
            'c': [i % 10 for i in range(100)]
        })

        # 5 filters in sequence
        pd_result = pd_df[pd_df['a'] > 10]
        pd_result = pd_result[pd_result['a'] < 90]
        pd_result = pd_result[pd_result['b'] > 120]
        pd_result = pd_result[pd_result['b'] < 180]
        pd_result = pd_result[pd_result['c'] > 2]

        ds_result = ds_df[ds_df['a'] > 10]
        ds_result = ds_result[ds_result['a'] < 90]
        ds_result = ds_result[ds_result['b'] > 120]
        ds_result = ds_result[ds_result['b'] < 180]
        ds_result = ds_result[ds_result['c'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_deep_select_filter_sort_chain(self):
        """Mixed operations: select + filter + sort in deep chain"""
        pd_df = pd.DataFrame({
            'a': [5, 2, 8, 1, 9, 3, 7, 4, 6, 10],
            'b': [50, 20, 80, 10, 90, 30, 70, 40, 60, 100],
            'c': ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y'],
            'd': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
        })
        ds_df = DataStore({
            'a': [5, 2, 8, 1, 9, 3, 7, 4, 6, 10],
            'b': [50, 20, 80, 10, 90, 30, 70, 40, 60, 100],
            'c': ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y'],
            'd': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
        })

        # select -> filter -> sort -> filter -> select -> sort
        pd_result = pd_df[['a', 'b', 'd']]
        pd_result = pd_result[pd_result['a'] > 2]
        pd_result = pd_result.sort_values('a')
        pd_result = pd_result[pd_result['b'] < 90]
        pd_result = pd_result[['a', 'b']]
        pd_result = pd_result.sort_values('b', ascending=False)

        ds_result = ds_df[['a', 'b', 'd']]
        ds_result = ds_result[ds_result['a'] > 2]
        ds_result = ds_result.sort_values('a')
        ds_result = ds_result[ds_result['b'] < 90]
        ds_result = ds_result[['a', 'b']]
        ds_result = ds_result.sort_values('b', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_deep_assign_filter_chain(self):
        """Deep chain with column assignments and filters"""
        pd_df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        # assign -> filter -> assign -> filter -> assign
        pd_result = pd_df.assign(z=lambda d: d['x'] + d['y'])
        pd_result = pd_result[pd_result['z'] > 15]
        pd_result = pd_result.assign(w=lambda d: d['z'] * 2)
        pd_result = pd_result[pd_result['w'] < 150]
        pd_result = pd_result.assign(v=lambda d: d['w'] - d['x'])

        ds_result = ds_df.assign(z=lambda d: d['x'] + d['y'])
        ds_result = ds_result[ds_result['z'] > 15]
        ds_result = ds_result.assign(w=lambda d: d['z'] * 2)
        ds_result = ds_result[ds_result['w'] < 150]
        ds_result = ds_result.assign(v=lambda d: d['w'] - d['x'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_deep_groupby_filter_sort_chain(self):
        """Deep chain with groupby, filter, and sort"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })

        # filter -> groupby -> agg -> sort -> head
        pd_result = pd_df[pd_df['value'] > 20]
        pd_result = pd_result.groupby('group')['value'].sum().reset_index()
        pd_result = pd_result.sort_values('value', ascending=False)
        pd_result = pd_result.head(2)

        ds_result = ds_df[ds_df['value'] > 20]
        ds_result = ds_result.groupby('group')['value'].sum().reset_index()
        ds_result = ds_result.sort_values('value', ascending=False)
        ds_result = ds_result.head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_deep_head_filter_head_chain(self):
        """LIMIT -> WHERE -> LIMIT pattern requiring nested subqueries"""
        pd_df = pd.DataFrame({
            'id': range(100),
            'val': [i % 20 for i in range(100)]
        })
        ds_df = DataStore({
            'id': range(100),
            'val': [i % 20 for i in range(100)]
        })

        # head -> filter -> head -> filter
        pd_result = pd_df.head(50)
        pd_result = pd_result[pd_result['val'] > 5]
        pd_result = pd_result.head(20)
        pd_result = pd_result[pd_result['val'] < 15]

        ds_result = ds_df.head(50)
        ds_result = ds_result[ds_result['val'] > 5]
        ds_result = ds_result.head(20)
        ds_result = ds_result[ds_result['val'] < 15]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBoundaryConditions:
    """Test boundary conditions: empty, single row, many columns"""

    def test_empty_dataframe_filter(self):
        """Operations on empty DataFrame"""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore({'a': [], 'b': []})

        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_empty_dataframe_groupby(self):
        """GroupBy on empty DataFrame"""
        pd_df = pd.DataFrame({'group': pd.Series([], dtype=str), 'value': pd.Series([], dtype=float)})
        ds_df = DataStore({'group': pd.Series([], dtype=str), 'value': pd.Series([], dtype=float)})

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_single_row_operations(self):
        """All operations on single row DataFrame"""
        pd_df = pd.DataFrame({'a': [42], 'b': [100], 'c': ['x']})
        ds_df = DataStore({'a': [42], 'b': [100], 'c': ['x']})

        # filter
        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

        # sort
        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')
        assert_datastore_equals_pandas(ds_result, pd_result)

        # head
        pd_result = pd_df.head(1)
        ds_result = ds_df.head(1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_to_empty(self):
        """Filter single row DataFrame to empty"""
        pd_df = pd.DataFrame({'a': [42], 'b': [100]})
        ds_df = DataStore({'a': [42], 'b': [100]})

        pd_result = pd_df[pd_df['a'] > 100]  # No match
        ds_result = ds_df[ds_df['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_many_columns_select(self):
        """Select from DataFrame with many columns"""
        n_cols = 50
        data = {f'col_{i}': range(10) for i in range(n_cols)}

        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        # Select subset
        selected_cols = [f'col_{i}' for i in range(0, n_cols, 5)]  # Every 5th column
        pd_result = pd_df[selected_cols]
        ds_result = ds_df[selected_cols]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_many_columns_filter_sort(self):
        """Filter and sort on DataFrame with many columns"""
        n_cols = 30
        data = {f'col_{i}': range(20) for i in range(n_cols)}

        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[pd_df['col_0'] > 5].sort_values('col_1', ascending=False).head(10)
        ds_result = ds_df[ds_df['col_0'] > 5].sort_values('col_1', ascending=False).head(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_null_column(self):
        """Operations with all-NULL column"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [None, None, None, None, None]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [None, None, None, None, None]
        })

        # Filter on non-null column
        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexSetterGetter:
    """Test complex setter and getter chains"""

    def test_setitem_then_filter_on_new_column(self):
        """Set new column then filter on it"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_df['b'] = pd_df['a'] * 2
        ds_df['b'] = ds_df['a'] * 2

        pd_result = pd_df[pd_df['b'] > 4]
        ds_result = ds_df[ds_df['b'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_setitem_chain(self):
        """Multiple setitem operations in sequence"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_df['b'] = pd_df['a'] * 2
        pd_df['c'] = pd_df['b'] + 10
        pd_df['d'] = pd_df['c'] - pd_df['a']

        ds_df['b'] = ds_df['a'] * 2
        ds_df['c'] = ds_df['b'] + 10
        ds_df['d'] = ds_df['c'] - ds_df['a']

        pd_result = pd_df[['a', 'b', 'c', 'd']]
        ds_result = ds_df[['a', 'b', 'c', 'd']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_setitem_with_boolean_mask(self):
        """Set values using boolean mask"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        # Create boolean column
        pd_df['is_high'] = pd_df['a'] > 2
        ds_df['is_high'] = ds_df['a'] > 2

        # Filter using that column
        pd_result = pd_df[pd_df['is_high']]
        ds_result = ds_df[ds_df['is_high']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_getitem_nested_expression(self):
        """Complex nested expression in getitem"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500]
        })

        # Complex condition
        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] < 40) | (pd_df['c'] > 400)]
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] < 40) | (ds_df['c'] > 400)]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAggregationCombinations:
    """Test aggregation with various operation combinations"""

    def test_groupby_multiple_aggs_then_filter(self):
        """GroupBy with multiple aggregations then filter result"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        # Multiple aggregations
        pd_result = pd_df.groupby('group').agg({'value': ['sum', 'mean', 'count']}).reset_index()
        pd_result.columns = ['group', 'sum', 'mean', 'count']

        ds_result = ds_df.groupby('group').agg({'value': ['sum', 'mean', 'count']}).reset_index()
        ds_result.columns = ['group', 'sum', 'mean', 'count']

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_then_groupby_then_sort(self):
        """Filter -> GroupBy -> Sort chain"""
        pd_df = pd.DataFrame({
            'category': ['X', 'X', 'Y', 'Y', 'Z', 'Z', 'X', 'Y', 'Z'],
            'amount': [100, 200, 150, 250, 300, 100, 50, 175, 225]
        })
        ds_df = DataStore({
            'category': ['X', 'X', 'Y', 'Y', 'Z', 'Z', 'X', 'Y', 'Z'],
            'amount': [100, 200, 150, 250, 300, 100, 50, 175, 225]
        })

        pd_result = pd_df[pd_df['amount'] > 100]
        pd_result = pd_result.groupby('category')['amount'].sum().reset_index()
        pd_result = pd_result.sort_values('amount', ascending=False)

        ds_result = ds_df[ds_df['amount'] > 100]
        ds_result = ds_result.groupby('category')['amount'].sum().reset_index()
        ds_result = ds_result.sort_values('amount', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumulative_after_filter(self):
        """Cumulative operations after filter"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })

        pd_result = pd_df[pd_df['a'] > 3].copy()
        pd_result['cumsum_b'] = pd_result['b'].cumsum()

        ds_result = ds_df[ds_df['a'] > 3]
        ds_result = ds_result.assign(cumsum_b=lambda d: d['b'].cumsum())

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDtypePreservation:
    """Test dtype preservation through operations"""

    def test_int_dtype_after_filter(self):
        """Integer dtype preserved after filter"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_float_dtype_after_operations(self):
        """Float dtype preserved through operations"""
        pd_df = pd.DataFrame({
            'a': [1.1, 2.2, 3.3, 4.4, 5.5],
            'b': [10.5, 20.5, 30.5, 40.5, 50.5]
        })
        ds_df = DataStore({
            'a': [1.1, 2.2, 3.3, 4.4, 5.5],
            'b': [10.5, 20.5, 30.5, 40.5, 50.5]
        })

        pd_result = pd_df[pd_df['a'] > 2.0].sort_values('b', ascending=False)
        ds_result = ds_df[ds_df['a'] > 2.0].sort_values('b', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_dtype_after_filter(self):
        """String dtype preserved after filter"""
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'score': [85, 90, 78, 92]
        })
        ds_df = DataStore({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'score': [85, 90, 78, 92]
        })

        pd_result = pd_df[pd_df['score'] > 80]
        ds_result = ds_df[ds_df['score'] > 80]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_dtype_operations(self):
        """Mixed dtype columns through various operations"""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True]
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True]
        })

        pd_result = pd_df[pd_df['int_col'] > 2].sort_values('float_col')
        ds_result = ds_df[ds_df['int_col'] > 2].sort_values('float_col')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSpecialValues:
    """Test handling of special values"""

    def test_nan_in_filter(self):
        """NaN values in filter conditions"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nan_preservation_in_sort(self):
        """NaN values preserved in sorted result"""
        pd_df = pd.DataFrame({
            'a': [3.0, np.nan, 1.0, np.nan, 2.0],
            'b': [30, 20, 10, 40, 50]
        })
        ds_df = DataStore({
            'a': [3.0, np.nan, 1.0, np.nan, 2.0],
            'b': [30, 20, 10, 40, 50]
        })

        pd_result = pd_df.sort_values('a', na_position='last')
        ds_result = ds_df.sort_values('a', na_position='last')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_inf_values(self):
        """Infinity values in operations"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.inf, 3.0, -np.inf, 5.0],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1.0, np.inf, 3.0, -np.inf, 5.0],
            'b': [10, 20, 30, 40, 50]
        })

        # Filter finite values
        pd_result = pd_df[pd_df['a'] < np.inf]
        pd_result = pd_result[pd_result['a'] > -np.inf]

        ds_result = ds_df[ds_df['a'] < np.inf]
        ds_result = ds_result[ds_result['a'] > -np.inf]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_zero_and_negative_values(self):
        """Zero and negative values in operations"""
        pd_df = pd.DataFrame({
            'a': [-5, -2, 0, 2, 5],
            'b': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'a': [-5, -2, 0, 2, 5],
            'b': [1, 2, 3, 4, 5]
        })

        # Filter negative
        pd_result = pd_df[pd_df['a'] < 0]
        ds_result = ds_df[ds_df['a'] < 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

        # Filter zero
        pd_result = pd_df[pd_df['a'] == 0]
        ds_result = ds_df[ds_df['a'] == 0]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexFilterExpressions:
    """Test complex filter expressions"""

    def test_between_filter(self):
        """Filter using between condition"""
        pd_df = pd.DataFrame({
            'value': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        })
        ds_df = DataStore({
            'value': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        })

        pd_result = pd_df[(pd_df['value'] >= 20) & (pd_df['value'] <= 60)]
        ds_result = ds_df[(ds_df['value'] >= 20) & (ds_df['value'] <= 60)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_filter(self):
        """Filter using isin condition"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D', 'E', 'F'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'C', 'D', 'E', 'F'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df[pd_df['category'].isin(['A', 'C', 'E'])]
        ds_result = ds_df[ds_df['category'].isin(['A', 'C', 'E'])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_not_filter(self):
        """Filter using NOT condition"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [True, False, True, False, True]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [True, False, True, False, True]
        })

        pd_result = pd_df[~pd_df['b']]
        ds_result = ds_df[~ds_df['b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_column_comparison(self):
        """Filter comparing multiple columns"""
        pd_df = pd.DataFrame({
            'a': [1, 5, 3, 7, 2],
            'b': [2, 4, 6, 3, 8]
        })
        ds_df = DataStore({
            'a': [1, 5, 3, 7, 2],
            'b': [2, 4, 6, 3, 8]
        })

        pd_result = pd_df[pd_df['a'] > pd_df['b']]
        ds_result = ds_df[ds_df['a'] > ds_df['b']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRenameDropOperations:
    """Test rename and drop operations in chains"""

    def test_rename_then_filter(self):
        """Rename columns then filter"""
        pd_df = pd.DataFrame({'old_a': [1, 2, 3], 'old_b': [10, 20, 30]})
        ds_df = DataStore({'old_a': [1, 2, 3], 'old_b': [10, 20, 30]})

        pd_result = pd_df.rename(columns={'old_a': 'new_a', 'old_b': 'new_b'})
        pd_result = pd_result[pd_result['new_a'] > 1]

        ds_result = ds_df.rename(columns={'old_a': 'new_a', 'old_b': 'new_b'})
        ds_result = ds_result[ds_result['new_a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_then_filter(self):
        """Drop columns then filter"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30], 'c': [100, 200, 300]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30], 'c': [100, 200, 300]})

        pd_result = pd_df.drop(columns=['c'])
        pd_result = pd_result[pd_result['a'] > 1]

        ds_result = ds_df.drop(columns=['c'])
        ds_result = ds_result[ds_result['a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_drop_rename_chain(self):
        """Filter -> Drop -> Rename chain"""
        pd_df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
            'z': [100, 200, 300, 400, 500]
        })
        ds_df = DataStore({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
            'z': [100, 200, 300, 400, 500]
        })

        pd_result = pd_df[pd_df['x'] > 2]
        pd_result = pd_result.drop(columns=['z'])
        pd_result = pd_result.rename(columns={'x': 'a', 'y': 'b'})

        ds_result = ds_df[ds_df['x'] > 2]
        ds_result = ds_result.drop(columns=['z'])
        ds_result = ds_result.rename(columns={'x': 'a', 'y': 'b'})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEdgeCaseCombinations:
    """Test edge case combinations"""

    def test_empty_result_from_filter_chain(self):
        """Chain that results in empty DataFrame"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        # Filter to nothing
        pd_result = pd_df[pd_df['a'] > 10]
        ds_result = ds_df[ds_df['a'] > 10]

        assert len(pd_result) == 0
        assert len(ds_result) == 0
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_head_zero(self):
        """head(0) returns empty DataFrame"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30]})

        pd_result = pd_df.head(0)
        ds_result = ds_df.head(0)

        assert len(pd_result) == 0
        assert len(ds_result) == 0
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_tail_larger_than_rows(self):
        """tail(n) where n > number of rows"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30]})

        pd_result = pd_df.tail(100)
        ds_result = ds_df.tail(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_all_rows(self):
        """Filter that matches all rows"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30]})

        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_duplicate_column_select(self):
        """Select same column multiple times (should not duplicate)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30], 'c': [100, 200, 300]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30], 'c': [100, 200, 300]})

        pd_result = pd_df[['a', 'b']]
        ds_result = ds_df[['a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)
