"""
Exploratory Batch 70: Complex Type Combinations and Long Operation Chains

Focus areas:
1. Mixed nullable types in groupby aggregations
2. Long operation chains (5+ ops) with mixed SQL/Pandas segments
3. Type combinations: int64 + float64 + string in complex operations
4. Chain patterns: filter -> assign -> groupby -> filter -> assign -> sort
5. Operations mixing SQL-able and Pandas-only ops

Discovery method: Architecture-based exploration based on query_planner.py and lazy_ops.py
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestMixedNullableTypesGroupby:
    """Test groupby aggregations with mixed nullable types"""

    def test_groupby_agg_int_float_mixed(self):
        """Groupby with int and float columns, aggregate both"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.5, 2.5, 3.5, 4.5, 5.5]
        })
        pd_result = pd_df.groupby('group').agg({
            'int_col': 'sum',
            'float_col': 'mean'
        }).reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.groupby('group').agg({
            'int_col': 'sum',
            'float_col': 'mean'
        }).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_multiple_agg_same_col(self):
        """Multiple aggregations on same column"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        pd_result = pd_df.groupby('group').agg({
            'value': ['sum', 'mean', 'min', 'max']
        })
        pd_result.columns = ['value_sum', 'value_mean', 'value_min', 'value_max']
        pd_result = pd_result.reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.groupby('group').agg({
            'value': ['sum', 'mean', 'min', 'max']
        })
        ds_result.columns = ['value_sum', 'value_mean', 'value_min', 'value_max']
        ds_result = ds_result.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_with_nan_in_value_col(self):
        """Groupby with NaN values in aggregated column"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        pd_result = pd_df.groupby('group')['value'].sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_count_with_nan(self):
        """Groupby count with NaN values - count should exclude NaN"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [1.0, np.nan, 3.0, np.nan, np.nan]
        })
        pd_result = pd_df.groupby('group')['value'].count().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.groupby('group')['value'].count().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_multiple_columns_mixed_types(self):
        """Groupby by multiple columns with mixed types"""
        pd_df = pd.DataFrame({
            'cat': ['X', 'X', 'Y', 'Y', 'X'],
            'num': [1, 1, 2, 2, 1],
            'value': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        pd_result = pd_df.groupby(['cat', 'num'])['value'].sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.groupby(['cat', 'num'])['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestLongOperationChains:
    """Test long operation chains (5+ operations)"""

    def test_filter_assign_filter_sort(self):
        """Chain: filter -> assign -> filter -> sort_values"""
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'score': [85.0, 90.0, 75.0, 88.0, 92.0]
        })
        pd_result = pd_df[pd_df['age'] > 28].copy()
        pd_result['age_score'] = pd_result['age'] * pd_result['score']
        pd_result = pd_result[pd_result['age_score'] > 2500]
        pd_result = pd_result.sort_values('age_score', ascending=False)
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['age'] > 28]
        ds_result['age_score'] = ds_result['age'] * ds_result['score']
        ds_result = ds_result[ds_result['age_score'] > 2500]
        ds_result = ds_result.sort_values('age_score', ascending=False)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_filter_assign_groupby_sort(self):
        """Chain: select -> filter -> assign -> groupby -> sort"""
        pd_df = pd.DataFrame({
            'dept': ['Sales', 'Sales', 'IT', 'IT', 'HR', 'HR'],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
            'salary': [50000, 60000, 70000, 80000, 45000, 55000],
            'bonus': [5000, 6000, 7000, 8000, 4500, 5500]
        })
        pd_result = pd_df[['dept', 'salary', 'bonus']].copy()
        pd_result = pd_result[pd_result['salary'] > 45000]
        pd_result['total'] = pd_result['salary'] + pd_result['bonus']
        pd_result = pd_result.groupby('dept')['total'].mean().reset_index()
        pd_result = pd_result.sort_values('total', ascending=False)
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[['dept', 'salary', 'bonus']]
        ds_result = ds_result[ds_result['salary'] > 45000]
        ds_result['total'] = ds_result['salary'] + ds_result['bonus']
        ds_result = ds_result.groupby('dept')['total'].mean().reset_index()
        ds_result = ds_result.sort_values('total', ascending=False)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_with_multiple_filters(self):
        """Chain with multiple consecutive filters"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'c': ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y']
        })
        pd_result = pd_df[pd_df['a'] > 2]
        pd_result = pd_result[pd_result['b'] < 90]
        pd_result = pd_result[pd_result['c'] == 'x']
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 2]
        ds_result = ds_result[ds_result['b'] < 90]
        ds_result = ds_result[ds_result['c'] == 'x']
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_filter_groupby_filter_sort_limit(self):
        """Chain: filter -> groupby -> agg -> filter -> sort -> head"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D'],
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        pd_result = pd_df[pd_df['value'] > 15]
        pd_result = pd_result.groupby('category')['value'].sum().reset_index()
        pd_result = pd_result[pd_result['value'] > 50]
        pd_result = pd_result.sort_values('value', ascending=False)
        pd_result = pd_result.head(2)
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['value'] > 15]
        ds_result = ds_result.groupby('category')['value'].sum().reset_index()
        ds_result = ds_result[ds_result['value'] > 50]
        ds_result = ds_result.sort_values('value', ascending=False)
        ds_result = ds_result.head(2)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_assign_multiple_columns(self):
        """Chain with multiple column assignments"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.copy()
        pd_result['c'] = pd_result['a'] + pd_result['b']
        pd_result['d'] = pd_result['c'] * 2
        pd_result['e'] = pd_result['d'] - pd_result['a']
        pd_result['f'] = pd_result['e'] / pd_result['b']

        ds_df = DataStore(pd_df.copy())
        ds_df['c'] = ds_df['a'] + ds_df['b']
        ds_df['d'] = ds_df['c'] * 2
        ds_df['e'] = ds_df['d'] - ds_df['a']
        ds_df['f'] = ds_df['e'] / ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_result)


class TestChainWithStringOperations:
    """Test chains that mix string operations with numeric operations"""

    def test_filter_str_assign_groupby(self):
        """Filter by string prefix, then groupby numeric"""
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Anna', 'Bob', 'Barbara', 'Charlie'],
            'dept': ['Sales', 'IT', 'Sales', 'IT', 'HR'],
            'salary': [50000, 60000, 55000, 65000, 45000]
        })
        pd_result = pd_df[pd_df['name'].str.startswith('A')].copy()
        pd_result['bonus'] = pd_result['salary'] * 0.1
        pd_result = pd_result.groupby('dept').agg({
            'salary': 'sum',
            'bonus': 'sum'
        }).reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['name'].str.startswith('A')]
        ds_result['bonus'] = ds_result['salary'] * 0.1
        ds_result = ds_result.groupby('dept').agg({
            'salary': 'sum',
            'bonus': 'sum'
        }).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_str_upper_filter_assign(self):
        """String upper transformation followed by filter and assign"""
        pd_df = pd.DataFrame({
            'text': ['hello', 'world', 'foo', 'bar'],
            'value': [1, 2, 3, 4]
        })
        pd_result = pd_df.copy()
        pd_result['upper_text'] = pd_result['text'].str.upper()
        pd_result = pd_result[pd_result['value'] > 1]
        pd_result['doubled'] = pd_result['value'] * 2
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_df['upper_text'] = ds_df['text'].str.upper()
        ds_result = ds_df[ds_df['value'] > 1]
        ds_result['doubled'] = ds_result['value'] * 2
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_len_filter_groupby(self):
        """String length filter followed by groupby"""
        pd_df = pd.DataFrame({
            'name': ['Al', 'Bob', 'Charlie', 'Dan', 'Elizabeth'],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[pd_df['name'].str.len() > 3].copy()
        pd_result = pd_result.groupby('category')['value'].sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['name'].str.len() > 3]
        ds_result = ds_result.groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestChainWithDateTimeOperations:
    """Test chains with datetime operations"""

    def test_dt_year_filter_groupby(self):
        """Extract year, filter, then groupby"""
        pd_df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10', '2021-08-25', '2022-01-01']),
            'value': [100, 200, 300, 400, 500]
        })
        pd_result = pd_df.copy()
        pd_result['year'] = pd_result['date'].dt.year
        pd_result = pd_result[pd_result['year'] >= 2021]
        pd_result = pd_result.groupby('year')['value'].sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_df['year'] = ds_df['date'].dt.year
        ds_result = ds_df[ds_df['year'] >= 2021]
        ds_result = ds_result.groupby('year')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_month_quarter_filter(self):
        """Extract month and quarter, filter by both"""
        pd_df = pd.DataFrame({
            'date': pd.to_datetime([
                '2021-01-15', '2021-04-20', '2021-07-10', '2021-10-25',
                '2022-02-15', '2022-05-20', '2022-08-10', '2022-11-25'
            ]),
            'sales': [100, 200, 300, 400, 150, 250, 350, 450]
        })
        pd_result = pd_df.copy()
        pd_result['month'] = pd_result['date'].dt.month
        pd_result['quarter'] = pd_result['date'].dt.quarter
        pd_result = pd_result[pd_result['quarter'].isin([1, 4])]
        pd_result = pd_result[pd_result['sales'] > 100]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_df['month'] = ds_df['date'].dt.month
        ds_df['quarter'] = ds_df['date'].dt.quarter
        ds_result = ds_df[ds_df['quarter'].isin([1, 4])]
        ds_result = ds_result[ds_result['sales'] > 100]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestChainWithNullHandling:
    """Test chains that involve NULL/NaN handling"""

    def test_filter_dropna_groupby(self):
        """Filter, drop NaN, then groupby"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [1.0, np.nan, 3.0, 4.0, np.nan]
        })
        pd_result = pd_df[pd_df['group'] != 'C'].copy()
        pd_result = pd_result.dropna(subset=['value'])
        pd_result = pd_result.groupby('group')['value'].sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['group'] != 'C']
        ds_result = ds_result.dropna(subset=['value'])
        ds_result = ds_result.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_fillna_filter_assign(self):
        """Fill NaN, filter, then assign"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.copy()
        pd_result['a'] = pd_result['a'].fillna(0)
        pd_result = pd_result[pd_result['a'] > 0]
        pd_result['c'] = pd_result['a'] * pd_result['b']
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_df['a'] = ds_df['a'].fillna(0)
        ds_result = ds_df[ds_df['a'] > 0]
        ds_result['c'] = ds_result['a'] * ds_result['b']
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_with_nan_filter_groupby(self):
        """Assign column that produces NaN, filter it out, groupby"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'x': [10.0, 0.0, 20.0, 0.0, 30.0],
            'y': [2.0, 5.0, 4.0, 0.0, 6.0]
        })
        pd_result = pd_df.copy()
        pd_result['ratio'] = pd_result['x'] / pd_result['y']
        # Filter out NaN and inf values
        pd_result = pd_result[pd_result['ratio'].notna() & np.isfinite(pd_result['ratio'])]
        pd_result = pd_result.groupby('group')['ratio'].mean().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_df['ratio'] = ds_df['x'] / ds_df['y']
        # DataStore doesn't support np.isfinite directly on ColumnExpr
        # Use dropna to filter out NaN, then filter out inf using comparison
        ds_result = ds_df.dropna(subset=['ratio'])
        ds_result = ds_result[(ds_result['ratio'] != np.inf) & (ds_result['ratio'] != -np.inf)]
        ds_result = ds_result.groupby('group')['ratio'].mean().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestComplexAggregations:
    """Test complex aggregation scenarios"""

    def test_groupby_agg_named_functions(self):
        """Groupby with named aggregation functions"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.groupby('group').agg(
            x_sum=('x', 'sum'),
            x_mean=('x', 'mean'),
            y_max=('y', 'max'),
            y_min=('y', 'min')
        ).reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.groupby('group').agg(
            x_sum=('x', 'sum'),
            x_mean=('x', 'mean'),
            y_max=('y', 'max'),
            y_min=('y', 'min')
        ).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_size_and_count(self):
        """Groupby with size() and count() - they differ with NaN"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        # size counts all rows, count excludes NaN
        pd_size = pd_df.groupby('group').size().reset_index(name='size')
        pd_count = pd_df.groupby('group')['value'].count().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_size = ds_df.groupby('group').size().reset_index(name='size')
        ds_count = ds_df.groupby('group')['value'].count().reset_index()

        assert_datastore_equals_pandas(ds_size, pd_size, check_row_order=False)
        assert_datastore_equals_pandas(ds_count, pd_count, check_row_order=False)

    @pytest.mark.xfail(reason="chDB uses population variance (ddof=0), pandas uses sample variance (ddof=1)")
    def test_groupby_std_var(self):
        """Groupby with std and var - note: chDB uses population variance, pandas uses sample"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
        })
        pd_result = pd_df.groupby('group').agg({
            'value': ['std', 'var']
        })
        pd_result.columns = ['std', 'var']
        pd_result = pd_result.reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.groupby('group').agg({
            'value': ['std', 'var']
        })
        ds_result.columns = ['std', 'var']
        ds_result = ds_result.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestEdgeCasesChains:
    """Test edge cases in operation chains"""

    def test_empty_after_filter_chain(self):
        """Chain that results in empty DataFrame"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [10, 20, 30]
        })
        pd_result = pd_df[pd_df['a'] > 100]
        pd_result = pd_result[pd_result['b'] < 0]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 100]
        ds_result = ds_result[ds_result['b'] < 0]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_chain(self):
        """Chain operating on single row"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[pd_df['a'] == 3].copy()
        pd_result['c'] = pd_result['a'] * pd_result['b']
        pd_result['d'] = pd_result['c'] + 100
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] == 3]
        ds_result['c'] = ds_result['a'] * ds_result['b']
        ds_result['d'] = ds_result['c'] + 100
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_rows_filtered(self):
        """Filter that keeps all rows, then further operations"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [10, 20, 30]
        })
        pd_result = pd_df[pd_df['a'] > 0].copy()  # All rows
        pd_result['c'] = pd_result['a'] + pd_result['b']
        pd_result = pd_result.sort_values('c', ascending=False)
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 0]
        ds_result['c'] = ds_result['a'] + ds_result['b']
        ds_result = ds_result.sort_values('c', ascending=False)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_with_head_in_middle(self):
        """Chain with head() limiting rows, then further ops"""
        pd_df = pd.DataFrame({
            'a': [5, 3, 1, 4, 2],
            'b': [50, 30, 10, 40, 20]
        })
        pd_result = pd_df.sort_values('a').head(3).copy()
        pd_result['c'] = pd_result['a'] * 10
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.sort_values('a').head(3)
        ds_result['c'] = ds_result['a'] * 10
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBooleanMaskChains:
    """Test chains with boolean mask operations"""

    def test_combined_and_filters(self):
        """Filter with combined AND conditions"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'x', 'y', 'x']
        })
        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] < 50) & (pd_df['c'] == 'x')]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] < 50) & (ds_df['c'] == 'x')]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combined_or_filters(self):
        """Filter with combined OR conditions"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'x', 'y', 'x']
        })
        pd_result = pd_df[(pd_df['a'] == 1) | (pd_df['a'] == 5) | (pd_df['c'] == 'y')]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[(ds_df['a'] == 1) | (ds_df['a'] == 5) | (ds_df['c'] == 'y')]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negation_filter(self):
        """Filter with negation"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'x', 'y', 'x']
        })
        pd_result = pd_df[~(pd_df['b'] == 'y')]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[~(ds_df['b'] == 'y')]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_filter_chain(self):
        """Filter with isin() followed by more operations"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D', 'E'],
            'value': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[pd_df['category'].isin(['A', 'C', 'E'])].copy()
        pd_result['doubled'] = pd_result['value'] * 2
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['category'].isin(['A', 'C', 'E'])]
        ds_result['doubled'] = ds_result['value'] * 2
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnReferencesInChain:
    """Test column references in long chains"""

    def test_reference_newly_created_column(self):
        """Reference column created earlier in chain"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.copy()
        pd_result['c'] = pd_result['a'] + pd_result['b']
        pd_result['d'] = pd_result['c'] * 2  # Reference c
        pd_result['e'] = pd_result['d'] + pd_result['a']  # Reference d and a

        ds_df = DataStore(pd_df.copy())
        ds_df['c'] = ds_df['a'] + ds_df['b']
        ds_df['d'] = ds_df['c'] * 2
        ds_df['e'] = ds_df['d'] + ds_df['a']

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_overwrite_column_in_chain(self):
        """Overwrite existing column multiple times"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.copy()
        pd_result['a'] = pd_result['a'] * 2
        pd_result['a'] = pd_result['a'] + pd_result['b']
        pd_result['a'] = pd_result['a'] / 10

        ds_df = DataStore(pd_df.copy())
        ds_df['a'] = ds_df['a'] * 2
        ds_df['a'] = ds_df['a'] + ds_df['b']
        ds_df['a'] = ds_df['a'] / 10

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_filter_on_newly_created_column(self):
        """Filter based on a column created in the same chain"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.copy()
        pd_result['c'] = pd_result['a'] + pd_result['b']
        pd_result = pd_result[pd_result['c'] > 30]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_df['c'] = ds_df['a'] + ds_df['b']
        ds_result = ds_df[ds_df['c'] > 30]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
