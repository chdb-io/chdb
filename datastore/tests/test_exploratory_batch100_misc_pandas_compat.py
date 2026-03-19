"""
Exploratory Batch 100: Miscellaneous Pandas Compatibility Tests

Targets under-tested areas:
- select_dtypes edge cases
- where/mask with other parameter
- query() string evaluation
- eval() string expressions
- compare() method
- update() method
- nunique with parameters
- DataFrame-level cumsum/diff/pct_change chains
- Complex multi-step chains (filter -> assign -> groupby -> agg -> merge -> sort)
- merge with indicator parameter
- pivot_table with margins
- apply axis=1 with complex functions
"""

import unittest
import numpy as np
import pandas as pd

from datastore import DataStore
from datastore.tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
)


class TestSelectDtypes(unittest.TestCase):
    """Test select_dtypes with various include/exclude combinations."""

    def setUp(self):
        self.data = {
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_select_dtypes_include_number(self):
        pd_result = self.pd_df.select_dtypes(include='number')
        ds_result = self.ds_df.select_dtypes(include='number')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_include_object(self):
        pd_result = self.pd_df.select_dtypes(include='object')
        ds_result = self.ds_df.select_dtypes(include='object')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_include_bool(self):
        pd_result = self.pd_df.select_dtypes(include='bool')
        ds_result = self.ds_df.select_dtypes(include='bool')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_exclude_object(self):
        pd_result = self.pd_df.select_dtypes(exclude='object')
        ds_result = self.ds_df.select_dtypes(exclude='object')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_include_list(self):
        pd_result = self.pd_df.select_dtypes(include=['int64', 'float64'])
        ds_result = self.ds_df.select_dtypes(include=['int64', 'float64'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_exclude_list(self):
        pd_result = self.pd_df.select_dtypes(exclude=['object', 'bool'])
        ds_result = self.ds_df.select_dtypes(exclude=['object', 'bool'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_on_filtered_df(self):
        pd_result = self.pd_df[self.pd_df['int_col'] > 1].select_dtypes(include='number')
        ds_result = self.ds_df[self.ds_df['int_col'] > 1].select_dtypes(include='number')
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestWhereWithOther(unittest.TestCase):
    """Test where() and mask() with the other parameter."""

    def setUp(self):
        self.data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_where_scalar_other(self):
        pd_result = self.pd_df.where(self.pd_df['A'] > 2, -1)
        ds_result = self.ds_df.where(self.ds_df['A'] > 2, -1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_where_zero_other(self):
        pd_result = self.pd_df.where(self.pd_df['A'] > 3, 0)
        ds_result = self.ds_df.where(self.ds_df['A'] > 3, 0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_where_no_other(self):
        """where() without other should fill NaN."""
        pd_result = self.pd_df.where(self.pd_df['A'] > 2)
        ds_result = self.ds_df.where(self.ds_df['A'] > 2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mask_scalar_other(self):
        pd_result = self.pd_df.mask(self.pd_df['A'] > 3, -99)
        ds_result = self.ds_df.mask(self.ds_df['A'] > 3, -99)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mask_no_other(self):
        """mask() without other should fill NaN where True."""
        pd_result = self.pd_df.mask(self.pd_df['A'] > 3)
        ds_result = self.ds_df.mask(self.ds_df['A'] > 3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_where_with_multiple_conditions(self):
        cond_pd = (self.pd_df['A'] > 1) & (self.pd_df['B'] < 50)
        cond_ds = (self.ds_df['A'] > 1) & (self.ds_df['B'] < 50)
        pd_result = self.pd_df.where(cond_pd, 0)
        ds_result = self.ds_df.where(cond_ds, 0)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestQueryMethod(unittest.TestCase):
    """Test query() string evaluation."""

    def setUp(self):
        self.data = {'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
                     'age': [25, 30, 35, 28],
                     'salary': [50000, 60000, 70000, 55000]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_query_simple_comparison(self):
        pd_result = self.pd_df.query('age > 28')
        ds_result = self.ds_df.query('age > 28')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_compound_condition(self):
        pd_result = self.pd_df.query('age > 25 and salary < 65000')
        ds_result = self.ds_df.query('age > 25 and salary < 65000')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_or_condition(self):
        pd_result = self.pd_df.query('age < 26 or salary > 65000')
        ds_result = self.ds_df.query('age < 26 or salary > 65000')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_variable(self):
        min_age = 30
        pd_result = self.pd_df.query('age >= @min_age')
        ds_result = self.ds_df.query('age >= @min_age')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_column_comparison(self):
        data = {'A': [10, 20, 30], 'B': [15, 15, 15]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.query('A > B')
        ds_result = ds_df.query('A > B')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_string_method(self):
        pd_result = self.pd_df.query('name == "Alice"')
        ds_result = self.ds_df.query('name == "Alice"')
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEvalMethod(unittest.TestCase):
    """Test eval() string expressions."""

    def setUp(self):
        self.data = {'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_eval_arithmetic(self):
        pd_result = self.pd_df.eval('C = A + B')
        ds_result = self.ds_df.eval('C = A + B')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_complex_expression(self):
        pd_result = self.pd_df.eval('C = A * 2 + B / 10')
        ds_result = self.ds_df.eval('C = A * 2 + B / 10')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_boolean_expression(self):
        pd_result = self.pd_df.eval('A > 2')
        ds_result = self.ds_df.eval('A > 2')
        # eval returning a Series
        if isinstance(pd_result, pd.Series):
            assert_series_equal(ds_result, pd_result)
        else:
            assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_multiple_columns(self):
        pd_result = self.pd_df.eval('C = A + B\nD = A * B')
        ds_result = self.ds_df.eval('C = A + B\nD = A * B')
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCompareMethod(unittest.TestCase):
    """Test compare() method for finding differences between DataFrames."""

    def test_compare_basic(self):
        data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        data2 = {'A': [1, 2, 99], 'B': [4, 55, 6]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)

        pd_result = pd_df1.compare(pd_df2)
        ds_result = ds_df1.compare(pd_df2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_keep_shape(self):
        data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        data2 = {'A': [1, 2, 99], 'B': [4, 55, 6]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)

        pd_result = pd_df1.compare(pd_df2, keep_shape=True)
        ds_result = ds_df1.compare(pd_df2, keep_shape=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_keep_equal(self):
        data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        data2 = {'A': [1, 2, 99], 'B': [4, 55, 6]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)

        pd_result = pd_df1.compare(pd_df2, keep_equal=True)
        ds_result = ds_df1.compare(pd_df2, keep_equal=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_with_datastore_other(self):
        data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        data2 = {'A': [1, 2, 99], 'B': [4, 55, 6]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)
        ds_df2 = DataStore(data2)

        pd_result = pd_df1.compare(pd_df2)
        ds_result = ds_df1.compare(ds_df2)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestUpdateMethod(unittest.TestCase):
    """Test update() method for in-place modification."""

    def test_update_basic(self):
        data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        data2 = {'A': [10, np.nan, 30]}
        pd_df = pd.DataFrame(data1)
        ds_df = DataStore(data1)
        other = pd.DataFrame(data2)

        pd_df.update(other)
        ds_df.update(other)
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_update_with_datastore_other(self):
        data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        data2 = {'A': [10, np.nan, 30]}
        pd_df = pd.DataFrame(data1)
        ds_df = DataStore(data1)
        other_ds = DataStore(data2)
        other_pd = pd.DataFrame(data2)

        pd_df.update(other_pd)
        ds_df.update(other_ds)
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_update_overwrite_false(self):
        data1 = {'A': [1, np.nan, 3], 'B': [4, 5, 6]}
        data2 = {'A': [10, 20, 30]}
        pd_df = pd.DataFrame(data1)
        ds_df = DataStore(data1)
        other = pd.DataFrame(data2)

        pd_df.update(other, overwrite=False)
        ds_df.update(other, overwrite=False)
        assert_datastore_equals_pandas(ds_df, pd_df)


class TestNunique(unittest.TestCase):
    """Test nunique with various parameters."""

    def setUp(self):
        self.data = {
            'A': [1, 2, 2, 3, np.nan],
            'B': ['x', 'y', 'x', 'z', 'y'],
            'C': [1.0, 1.0, 2.0, np.nan, np.nan],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_nunique_default(self):
        pd_result = self.pd_df.nunique()
        ds_result = self.ds_df.nunique()
        assert_series_equal(ds_result, pd_result)

    def test_nunique_dropna_false(self):
        pd_result = self.pd_df.nunique(dropna=False)
        ds_result = self.ds_df.nunique(dropna=False)
        assert_series_equal(ds_result, pd_result)

    def test_series_nunique(self):
        pd_result = self.pd_df['A'].nunique()
        ds_result = self.ds_df['A'].nunique()
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_series_nunique_dropna_false(self):
        pd_result = self.pd_df['A'].nunique(dropna=False)
        ds_result = self.ds_df['A'].nunique(dropna=False)
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"


class TestDataFrameCumulativeOps(unittest.TestCase):
    """Test DataFrame-level cumulative operations and chains."""

    def setUp(self):
        self.data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_cumsum(self):
        pd_result = self.pd_df.cumsum()
        ds_result = self.ds_df.cumsum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff(self):
        pd_result = self.pd_df.diff()
        ds_result = self.ds_df.diff()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_periods_2(self):
        pd_result = self.pd_df.diff(periods=2)
        ds_result = self.ds_df.diff(periods=2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pct_change(self):
        pd_result = self.pd_df.pct_change()
        ds_result = self.ds_df.pct_change()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_then_diff(self):
        pd_result = self.pd_df.cumsum().diff()
        ds_result = self.ds_df.cumsum().diff()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_abs_on_negative_data(self):
        data = {'A': [-1, -2, 3, -4], 'B': [5, -6, -7, 8]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.abs()
        ds_result = ds_df.abs()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAnyAll(unittest.TestCase):
    """Test any() and all() on DataFrames."""

    def setUp(self):
        self.data = {'A': [True, True, False], 'B': [True, False, False], 'C': [True, True, True]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_any_default(self):
        pd_result = self.pd_df.any()
        ds_result = self.ds_df.any()
        assert_series_equal(ds_result, pd_result)

    def test_all_default(self):
        pd_result = self.pd_df.all()
        ds_result = self.ds_df.all()
        assert_series_equal(ds_result, pd_result)

    def test_any_numeric(self):
        data = {'A': [0, 0, 0], 'B': [0, 1, 0], 'C': [1, 1, 1]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.any()
        ds_result = ds_df.any()
        assert_series_equal(ds_result, pd_result)

    def test_all_numeric(self):
        data = {'A': [1, 1, 1], 'B': [1, 0, 1], 'C': [0, 0, 0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.all()
        ds_result = ds_df.all()
        assert_series_equal(ds_result, pd_result)

    def test_any_with_nan(self):
        data = {'A': [np.nan, np.nan, np.nan], 'B': [np.nan, 1.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.any()
        ds_result = ds_df.any()
        assert_series_equal(ds_result, pd_result)


class TestMergeWithIndicator(unittest.TestCase):
    """Test merge with indicator parameter."""

    def setUp(self):
        self.left = {'key': ['A', 'B', 'C', 'D'], 'val_left': [1, 2, 3, 4]}
        self.right = {'key': ['B', 'C', 'E', 'F'], 'val_right': [20, 30, 50, 60]}

    def test_merge_outer_with_indicator(self):
        pd_left = pd.DataFrame(self.left)
        pd_right = pd.DataFrame(self.right)
        ds_left = DataStore(self.left)
        ds_right = DataStore(self.right)

        pd_result = pd_left.merge(pd_right, on='key', how='outer', indicator=True)
        ds_result = ds_left.merge(ds_right, on='key', how='outer', indicator=True)
        # Merge order may differ, compare without row order
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_left_with_indicator(self):
        pd_left = pd.DataFrame(self.left)
        pd_right = pd.DataFrame(self.right)
        ds_left = DataStore(self.left)

        pd_result = pd_left.merge(pd_right, on='key', how='left', indicator=True)
        ds_result = ds_left.merge(pd_right, on='key', how='left', indicator=True)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_with_different_suffixes(self):
        left = {'key': [1, 2], 'value': [10, 20]}
        right = {'key': [1, 2], 'value': [100, 200]}
        pd_left = pd.DataFrame(left)
        pd_right = pd.DataFrame(right)
        ds_left = DataStore(left)

        pd_result = pd_left.merge(pd_right, on='key', suffixes=('_left', '_right'))
        ds_result = ds_left.merge(pd_right, on='key', suffixes=('_left', '_right'))
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestPivotTableWithMargins(unittest.TestCase):
    """Test pivot_table with margins and various aggfuncs."""

    def setUp(self):
        self.data = {
            'category': ['A', 'A', 'B', 'B', 'A', 'B'],
            'subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'value': [10, 20, 30, 40, 50, 60],
            'count': [1, 2, 3, 4, 5, 6],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_pivot_table_basic(self):
        pd_result = self.pd_df.pivot_table(values='value', index='category', columns='subcategory', aggfunc='mean')
        ds_result = self.ds_df.pivot_table(values='value', index='category', columns='subcategory', aggfunc='mean')
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_table_sum(self):
        pd_result = self.pd_df.pivot_table(values='value', index='category', columns='subcategory', aggfunc='sum')
        ds_result = self.ds_df.pivot_table(values='value', index='category', columns='subcategory', aggfunc='sum')
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_table_with_margins(self):
        pd_result = self.pd_df.pivot_table(
            values='value', index='category', columns='subcategory',
            aggfunc='sum', margins=True
        )
        ds_result = self.ds_df.pivot_table(
            values='value', index='category', columns='subcategory',
            aggfunc='sum', margins=True
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_table_with_fill_value(self):
        data = {
            'cat': ['A', 'A', 'B'],
            'sub': ['X', 'Y', 'X'],
            'val': [10, 20, 30],
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.pivot_table(values='val', index='cat', columns='sub', aggfunc='sum', fill_value=0)
        ds_result = ds_df.pivot_table(values='val', index='cat', columns='sub', aggfunc='sum', fill_value=0)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_table_multiple_values(self):
        pd_result = self.pd_df.pivot_table(
            values=['value', 'count'], index='category', aggfunc='sum'
        )
        ds_result = self.ds_df.pivot_table(
            values=['value', 'count'], index='category', aggfunc='sum'
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestComplexChains(unittest.TestCase):
    """Test complex multi-step operation chains."""

    def setUp(self):
        self.data = {
            'department': ['Engineering', 'Engineering', 'Sales', 'Sales', 'Marketing', 'Marketing',
                          'Engineering', 'Sales', 'Marketing', 'Engineering'],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank',
                    'Grace', 'Hank', 'Ivy', 'Jack'],
            'salary': [80000, 90000, 60000, 65000, 55000, 58000,
                      85000, 62000, 57000, 95000],
            'experience': [5, 8, 3, 6, 2, 4, 7, 5, 3, 10],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_filter_then_groupby_agg(self):
        """filter -> groupby -> agg"""
        pd_result = (self.pd_df[self.pd_df['salary'] > 60000]
                     .groupby('department')['salary']
                     .agg(['mean', 'count'])
                     .reset_index())
        ds_result = (self.ds_df[self.ds_df['salary'] > 60000]
                     .groupby('department')['salary']
                     .agg(['mean', 'count'])
                     .reset_index())
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_assign_sort(self):
        """filter -> assign -> sort_values"""
        pd_result = (self.pd_df[self.pd_df['experience'] > 3]
                     .assign(salary_per_year=lambda df: df['salary'] / df['experience'])
                     .sort_values('salary_per_year', ascending=False))
        ds_result = (self.ds_df[self.ds_df['experience'] > 3]
                     .assign(salary_per_year=lambda df: df['salary'] / df['experience'])
                     .sort_values('salary_per_year', ascending=False))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_then_filter(self):
        """groupby -> agg -> filter on aggregated result"""
        pd_agg = (self.pd_df.groupby('department')
                  .agg(avg_salary=('salary', 'mean'), count=('name', 'count'))
                  .reset_index())
        pd_result = pd_agg[pd_agg['count'] >= 3]

        ds_agg = (self.ds_df.groupby('department')
                  .agg(avg_salary=('salary', 'mean'), count=('name', 'count'))
                  .reset_index())
        ds_result = ds_agg[ds_agg['count'] >= 3]
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_then_groupby(self):
        """merge -> groupby -> agg"""
        dept_data = {
            'department': ['Engineering', 'Sales', 'Marketing'],
            'budget': [500000, 300000, 200000],
        }
        pd_dept = pd.DataFrame(dept_data)
        ds_dept = DataStore(dept_data)

        pd_merged = self.pd_df.merge(pd_dept, on='department')
        pd_result = (pd_merged.groupby('department')
                     .agg(total_salary=('salary', 'sum'), budget=('budget', 'first'))
                     .reset_index())

        ds_merged = self.ds_df.merge(ds_dept, on='department')
        ds_result = (ds_merged.groupby('department')
                     .agg(total_salary=('salary', 'sum'), budget=('budget', 'first'))
                     .reset_index())
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multiple_filter_chain(self):
        """multiple sequential filters"""
        pd_result = self.pd_df[self.pd_df['salary'] > 55000][
            self.pd_df['experience'] > 3
        ].sort_values('salary')

        ds_result = self.ds_df[self.ds_df['salary'] > 55000][
            self.ds_df['experience'] > 3
        ].sort_values('salary')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_select_filter_sort(self):
        """column selection -> filter -> sort"""
        pd_result = (self.pd_df[['name', 'salary', 'department']]
                     [self.pd_df['salary'] > 70000]
                     .sort_values('salary'))
        ds_result = (self.ds_df[['name', 'salary', 'department']]
                     [self.ds_df['salary'] > 70000]
                     .sort_values('salary'))
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestApplyAxis1Complex(unittest.TestCase):
    """Test apply with axis=1 (row-wise) for complex functions."""

    def setUp(self):
        self.data = {'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40], 'C': [100, 200, 300, 400]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_apply_row_sum(self):
        pd_result = self.pd_df.apply(sum, axis=1)
        ds_result = self.ds_df.apply(sum, axis=1)
        assert_series_equal(ds_result, pd_result)

    def test_apply_row_max(self):
        pd_result = self.pd_df.apply(max, axis=1)
        ds_result = self.ds_df.apply(max, axis=1)
        assert_series_equal(ds_result, pd_result)

    def test_apply_row_lambda(self):
        pd_result = self.pd_df.apply(lambda row: row['A'] + row['B'] * 2, axis=1)
        ds_result = self.ds_df.apply(lambda row: row['A'] + row['B'] * 2, axis=1)
        assert_series_equal(ds_result, pd_result)

    def test_apply_column_wise_mean(self):
        pd_result = self.pd_df.apply(np.mean, axis=0)
        ds_result = self.ds_df.apply(np.mean, axis=0)
        assert_series_equal(ds_result, pd_result)


class TestRollingExpandingChains(unittest.TestCase):
    """Test rolling and expanding window operations."""

    def setUp(self):
        self.data = {'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_rolling_mean(self):
        pd_result = self.pd_df.rolling(3).mean()
        ds_result = self.ds_df.rolling(3).mean()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_sum(self):
        pd_result = self.pd_df.rolling(3).sum()
        ds_result = self.ds_df.rolling(3).sum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_std(self):
        pd_result = self.pd_df.rolling(3).std()
        ds_result = self.ds_df.rolling(3).std()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_min_periods(self):
        pd_result = self.pd_df.rolling(3, min_periods=1).mean()
        ds_result = self.ds_df.rolling(3, min_periods=1).mean()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_mean(self):
        pd_result = self.pd_df.expanding().mean()
        ds_result = self.ds_df.expanding().mean()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_sum(self):
        pd_result = self.pd_df.expanding().sum()
        ds_result = self.ds_df.expanding().sum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ewm_mean(self):
        pd_result = self.pd_df.ewm(span=3).mean()
        ds_result = self.ds_df.ewm(span=3).mean()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMeltEdgeCases(unittest.TestCase):
    """Test melt with edge cases."""

    def test_melt_single_id_var(self):
        data = {'id': [1, 2], 'A': [10, 20], 'B': [30, 40], 'C': [50, 60]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.melt(id_vars='id')
        ds_result = ds_df.melt(id_vars='id')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_melt_specific_value_vars(self):
        data = {'id': [1, 2], 'A': [10, 20], 'B': [30, 40], 'C': [50, 60]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.melt(id_vars='id', value_vars=['A', 'B'])
        ds_result = ds_df.melt(id_vars='id', value_vars=['A', 'B'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_melt_custom_names(self):
        data = {'id': [1, 2], 'A': [10, 20], 'B': [30, 40]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.melt(id_vars='id', var_name='metric', value_name='measurement')
        ds_result = ds_df.melt(id_vars='id', var_name='metric', value_name='measurement')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_melt_no_id_vars(self):
        data = {'A': [1, 2], 'B': [3, 4]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.melt()
        ds_result = ds_df.melt()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestExplode(unittest.TestCase):
    """Test explode() method."""

    def test_explode_list_column(self):
        data = {'A': [[1, 2], [3, 4, 5], [6]], 'B': ['x', 'y', 'z']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.explode('A').reset_index(drop=True)
        ds_result = ds_df.explode('A').reset_index(drop=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_explode_with_ignore_index(self):
        data = {'A': [[1, 2], [3]], 'B': ['x', 'y']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.explode('A', ignore_index=True)
        ds_result = ds_df.explode('A', ignore_index=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_explode_empty_list(self):
        data = {'A': [[1, 2], [], [3]], 'B': ['x', 'y', 'z']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.explode('A', ignore_index=True)
        ds_result = ds_df.explode('A', ignore_index=True)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTranspose(unittest.TestCase):
    """Test transpose operations."""

    def test_transpose_numeric(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.T
        ds_result = ds_df.transpose()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transpose_mixed_types(self):
        data = {'A': [1, 2], 'B': [3.0, 4.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.T
        ds_result = ds_df.transpose()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIsinEdgeCases(unittest.TestCase):
    """Test isin() with various input types."""

    def test_isin_with_list(self):
        data = {'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df[pd_df['A'].isin([2, 4])]
        ds_result = ds_df[ds_df['A'].isin([2, 4])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_with_empty_list(self):
        data = {'A': [1, 2, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df[pd_df['A'].isin([])]
        ds_result = ds_df[ds_df['A'].isin([])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_dataframe_level(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.isin([1, 4, 5])
        ds_result = ds_df.isin([1, 4, 5])
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullHandlingInChains(unittest.TestCase):
    """Test null handling across chained operations."""

    def setUp(self):
        self.data = {
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1.0, np.nan, 3.0, np.nan, 5.0],
            'count': [10, 20, np.nan, 40, 50],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_fillna_then_groupby_sum(self):
        pd_result = (self.pd_df.fillna(0)
                     .groupby('group')['value']
                     .sum()
                     .reset_index())
        ds_result = (self.ds_df.fillna(0)
                     .groupby('group')['value']
                     .sum()
                     .reset_index())
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_dropna_then_groupby(self):
        pd_result = (self.pd_df.dropna()
                     .groupby('group')['value']
                     .mean()
                     .reset_index())
        ds_result = (self.ds_df.dropna()
                     .groupby('group')['value']
                     .mean()
                     .reset_index())
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_fillna_different_values_per_column(self):
        pd_result = self.pd_df.fillna({'value': 0, 'count': -1})
        ds_result = self.ds_df.fillna({'value': 0, 'count': -1})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_sum(self):
        pd_result = self.pd_df.isna().sum()
        ds_result = self.ds_df.isna().sum()
        assert_series_equal(ds_result, pd_result)


class TestValueCounts(unittest.TestCase):
    """Test value_counts on Series."""

    def setUp(self):
        self.data = {'A': ['cat', 'dog', 'cat', 'bird', 'dog', 'cat', np.nan]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_value_counts_default(self):
        pd_result = self.pd_df['A'].value_counts()
        ds_result = self.ds_df['A'].value_counts()
        assert_series_equal(ds_result, pd_result, check_index=False)

    def test_value_counts_normalize(self):
        pd_result = self.pd_df['A'].value_counts(normalize=True)
        ds_result = self.ds_df['A'].value_counts(normalize=True)
        assert_series_equal(ds_result, pd_result, check_index=False)

    def test_value_counts_dropna_false(self):
        pd_result = self.pd_df['A'].value_counts(dropna=False)
        ds_result = self.ds_df['A'].value_counts(dropna=False)
        assert_series_equal(ds_result, pd_result, check_index=False)

    def test_value_counts_ascending(self):
        pd_result = self.pd_df['A'].value_counts(ascending=True)
        ds_result = self.ds_df['A'].value_counts(ascending=True)
        assert_series_equal(ds_result, pd_result, check_index=False)


class TestDescribeEdgeCases(unittest.TestCase):
    """Test describe() with various parameters."""

    def test_describe_numeric_only(self):
        data = {'A': [1, 2, 3, 4, 5], 'B': [1.5, 2.5, 3.5, 4.5, 5.5], 'C': ['a', 'b', 'c', 'd', 'e']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.describe()
        ds_result = ds_df.describe()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_include_all(self):
        data = {'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'C': [1.0, 2.0, 3.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.describe(include='all')
        ds_result = ds_df.describe(include='all')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_custom_percentiles(self):
        data = {'A': list(range(100))}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.describe(percentiles=[0.1, 0.5, 0.9])
        ds_result = ds_df.describe(percentiles=[0.1, 0.5, 0.9])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_single_column(self):
        data = {'A': [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df['A'].describe()
        ds_result = ds_df['A'].describe()
        assert_series_equal(ds_result, pd_result)


class TestDuplicatesAdvanced(unittest.TestCase):
    """Test drop_duplicates and duplicated with various parameters."""

    def setUp(self):
        self.data = {
            'A': [1, 1, 2, 2, 3],
            'B': ['x', 'x', 'y', 'y', 'z'],
            'C': [10, 10, 20, 30, 40],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_drop_duplicates_subset(self):
        pd_result = self.pd_df.drop_duplicates(subset=['A'])
        ds_result = self.ds_df.drop_duplicates(subset=['A'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self):
        pd_result = self.pd_df.drop_duplicates(subset=['A'], keep='last')
        ds_result = self.ds_df.drop_duplicates(subset=['A'], keep='last')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_false(self):
        pd_result = self.pd_df.drop_duplicates(subset=['A'], keep=False)
        ds_result = self.ds_df.drop_duplicates(subset=['A'], keep=False)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_duplicated_default(self):
        pd_result = self.pd_df.duplicated()
        ds_result = self.ds_df.duplicated()
        assert_series_equal(ds_result, pd_result)

    def test_duplicated_subset(self):
        pd_result = self.pd_df.duplicated(subset=['A'])
        ds_result = self.ds_df.duplicated(subset=['A'])
        assert_series_equal(ds_result, pd_result)


class TestEmptyDataFrameOps(unittest.TestCase):
    """Test operations on empty DataFrames."""

    def test_empty_after_filter(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df[pd_df['A'] > 100]
        ds_result = ds_df[ds_df['A'] > 100]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_shape(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_empty = pd_df[pd_df['A'] > 100]
        ds_empty = ds_df[ds_df['A'] > 100]
        assert pd_empty.shape == ds_empty.shape

    def test_empty_groupby(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df[pd_df['A'] > 100].groupby('A')['B'].sum().reset_index()
        ds_result = ds_df[ds_df['A'] > 100].groupby('A')['B'].sum().reset_index()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_len_of_empty(self):
        data = {'A': [1, 2, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        assert len(pd_df[pd_df['A'] > 100]) == len(ds_df[ds_df['A'] > 100])


class TestSingleRowOps(unittest.TestCase):
    """Test operations on single-row DataFrames."""

    def setUp(self):
        self.data = {'A': [42], 'B': [3.14], 'C': ['hello']}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_single_row_describe(self):
        pd_result = self.pd_df.describe()
        ds_result = self.ds_df.describe()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_mean(self):
        pd_result = self.pd_df[['A', 'B']].mean()
        ds_result = self.ds_df[['A', 'B']].mean()
        assert_series_equal(ds_result, pd_result)

    def test_single_row_filter_match(self):
        pd_result = self.pd_df[self.pd_df['A'] == 42]
        ds_result = self.ds_df[self.ds_df['A'] == 42]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_no_match(self):
        pd_result = self.pd_df[self.pd_df['A'] == 999]
        ds_result = self.ds_df[self.ds_df['A'] == 999]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupbyNamedAgg(unittest.TestCase):
    """Test groupby with named aggregation (pd.NamedAgg style)."""

    def setUp(self):
        self.data = {
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [10, 20, 30, 40, 50],
            'weight': [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_named_agg_basic(self):
        pd_result = (self.pd_df.groupby('group')
                     .agg(mean_value=('value', 'mean'), sum_weight=('weight', 'sum'))
                     .reset_index())
        ds_result = (self.ds_df.groupby('group')
                     .agg(mean_value=('value', 'mean'), sum_weight=('weight', 'sum'))
                     .reset_index())
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_named_agg_multiple_funcs_same_column(self):
        pd_result = (self.pd_df.groupby('group')
                     .agg(val_min=('value', 'min'), val_max=('value', 'max'), val_mean=('value', 'mean'))
                     .reset_index())
        ds_result = (self.ds_df.groupby('group')
                     .agg(val_min=('value', 'min'), val_max=('value', 'max'), val_mean=('value', 'mean'))
                     .reset_index())
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_size(self):
        pd_result = self.pd_df.groupby('group').size().reset_index(name='count')
        ds_result = self.ds_df.groupby('group').size().reset_index(name='count')
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestConcatEdgeCases(unittest.TestCase):
    """Test concat with various parameters."""

    def test_concat_two_datastores(self):
        data1 = {'A': [1, 2], 'B': [3, 4]}
        data2 = {'A': [5, 6], 'B': [7, 8]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)
        ds_df2 = DataStore(data2)

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = ds_df1.concat([ds_df1, ds_df2], ignore_index=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_with_different_columns(self):
        data1 = {'A': [1, 2], 'B': [3, 4]}
        data2 = {'A': [5, 6], 'C': [7, 8]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)
        ds_df2 = DataStore(data2)

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = ds_df1.concat([ds_df1, ds_df2], ignore_index=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_axis1(self):
        data1 = {'A': [1, 2, 3]}
        data2 = {'B': [4, 5, 6]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)
        ds_df2 = DataStore(data2)

        pd_result = pd.concat([pd_df1, pd_df2], axis=1)
        ds_result = ds_df1.concat([ds_df1, ds_df2], axis=1)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestHeadTailSample(unittest.TestCase):
    """Test head, tail, sample operations."""

    def setUp(self):
        self.data = {'A': list(range(20)), 'B': list(range(100, 120))}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_head_default(self):
        pd_result = self.pd_df.head()
        ds_result = self.ds_df.head()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_n(self):
        pd_result = self.pd_df.head(3)
        ds_result = self.ds_df.head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_default(self):
        pd_result = self.pd_df.tail()
        ds_result = self.ds_df.tail()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_n(self):
        pd_result = self.pd_df.tail(3)
        ds_result = self.ds_df.tail(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sample_n(self):
        ds_result = self.ds_df.sample(5, random_state=42)
        pd_result = self.pd_df.sample(5, random_state=42)
        # sample results should have same shape
        assert ds_result.shape == pd_result.shape

    def test_head_after_filter(self):
        pd_result = self.pd_df[self.pd_df['A'] > 5].head(3)
        ds_result = self.ds_df[self.ds_df['A'] > 5].head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_after_sort(self):
        pd_result = self.pd_df.sort_values('A', ascending=False).tail(3)
        ds_result = self.ds_df.sort_values('A', ascending=False).tail(3)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRenameDrop(unittest.TestCase):
    """Test rename and drop operations."""

    def setUp(self):
        self.data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_rename_columns(self):
        pd_result = self.pd_df.rename(columns={'A': 'alpha', 'B': 'beta'})
        ds_result = self.ds_df.rename(columns={'A': 'alpha', 'B': 'beta'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_columns(self):
        pd_result = self.pd_df.drop(columns=['C'])
        ds_result = self.ds_df.drop(columns=['C'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_multiple_columns(self):
        pd_result = self.pd_df.drop(columns=['A', 'C'])
        ds_result = self.ds_df.drop(columns=['A', 'C'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_filter(self):
        pd_result = self.pd_df.rename(columns={'A': 'x'})
        pd_result = pd_result[pd_result['x'] > 1]
        ds_result = self.ds_df.rename(columns={'A': 'x'})
        ds_result = ds_result[ds_result['x'] > 1]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_then_groupby(self):
        data = {'grp': ['A', 'A', 'B'], 'val': [1, 2, 3], 'extra': [10, 20, 30]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.drop(columns=['extra']).groupby('grp')['val'].sum().reset_index()
        ds_result = ds_df.drop(columns=['extra']).groupby('grp')['val'].sum().reset_index()
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


if __name__ == '__main__':
    unittest.main()
