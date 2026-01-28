"""
Exploratory Batch 87: Advanced Edge Cases and Less-Tested Operations

Focus areas:
1. eval() with various expressions and special characters
2. query() with complex boolean expressions
3. Chained operations with assign and filter
4. DataFrame comparison edge cases (equals, compare)
5. Series/DataFrame arithmetic with different types
6. squeeze() behavior with different shapes
7. xs() cross-section access
8. isin() with edge cases (empty, None values)
9. truncate() with various index types
10. first_valid_index() and last_valid_index()
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestEvalEdgeCases:
    """Test eval() with various expression patterns"""

    def test_eval_simple_arithmetic(self):
        """Basic arithmetic in eval"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.eval('c = a + b')
        ds_result = ds_df.eval('c = a + b')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_multiple_operations(self):
        """Multiple operations in single eval"""
        pd_df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0]
        })
        ds_df = DataStore({
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0]
        })

        pd_result = pd_df.eval('z = x * 2 + y / 2')
        ds_result = ds_df.eval('z = x * 2 + y / 2')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_with_negative_values(self):
        """Eval with negative numbers"""
        pd_df = pd.DataFrame({
            'a': [-5, -3, 0, 3, 5],
            'b': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'a': [-5, -3, 0, 3, 5],
            'b': [1, 2, 3, 4, 5]
        })

        pd_result = pd_df.eval('c = a * b')
        ds_result = ds_df.eval('c = a * b')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestQueryComplexExpressions:
    """Test query() with complex boolean expressions"""

    def test_query_and_or_combination(self):
        """Query with AND and OR"""
        pd_df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6],
            'y': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'x': [1, 2, 3, 4, 5, 6],
            'y': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df.query('x > 2 and y < 50')
        ds_result = ds_df.query('x > 2 and y < 50')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_parentheses(self):
        """Query with explicit parentheses"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.query('(a > 2) and (b < 4 or c > 30)')
        ds_result = ds_df.query('(a > 2) and (b < 4 or c > 30)')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_in_list(self):
        """Query with 'in' operator"""
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'score': [85, 90, 78, 92]
        })
        ds_df = DataStore({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'score': [85, 90, 78, 92]
        })

        pd_result = pd_df.query('name in ["Alice", "Charlie"]')
        ds_result = ds_df.query('name in ["Alice", "Charlie"]')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_not_in(self):
        """Query with 'not in' operator"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.query('category not in ["C"]')
        ds_result = ds_df.query('category not in ["C"]')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestChainedAssignFilter:
    """Test chained operations with assign and filter"""

    def test_assign_then_filter(self):
        """Assign new column then filter"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.assign(c=pd_df['a'] + pd_df['b'])
        pd_result = pd_result[pd_result['c'] > 30]

        ds_result = ds_df.assign(c=ds_df['a'] + ds_df['b'])
        ds_result = ds_result[ds_result['c'] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_assign(self):
        """Filter then assign new column"""
        pd_df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        pd_filtered = pd_df[pd_df['x'] > 2]
        pd_result = pd_filtered.assign(z=pd_filtered['x'] * 2)

        ds_filtered = ds_df[ds_df['x'] > 2]
        ds_result = ds_filtered.assign(z=ds_filtered['x'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_assigns_chain(self):
        """Multiple assigns in sequence"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        ds_df = DataStore({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })

        pd_result = pd_df.assign(c=pd_df['a'] * 2).assign(d=lambda df: df['b'] + df['c'])
        ds_result = ds_df.assign(c=ds_df['a'] * 2).assign(d=lambda df: df['b'] + df['c'])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSqueezeEdgeCases:
    """Test squeeze() with different DataFrame shapes"""

    def test_squeeze_single_column(self):
        """Squeeze DataFrame with single column"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        assert_series_equal(ds_result, pd_result)

    def test_squeeze_single_row(self):
        """Squeeze DataFrame with single row"""
        pd_df = pd.DataFrame({'a': [10], 'b': [20], 'c': [30]})
        ds_df = DataStore({'a': [10], 'b': [20], 'c': [30]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        assert_series_equal(ds_result, pd_result)

    def test_squeeze_single_cell(self):
        """Squeeze DataFrame with single cell"""
        pd_df = pd.DataFrame({'a': [42]})
        ds_df = DataStore({'a': [42]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        # Single cell squeezes to scalar
        assert ds_result == pd_result

    def test_squeeze_multi_row_multi_col_no_effect(self):
        """Squeeze has no effect on multi-row multi-col DataFrame"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIsinEdgeCases:
    """Test isin() with various edge cases"""

    def test_isin_empty_list(self):
        """isin with empty list should return all False"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].isin([])
        ds_result = ds_df['a'].isin([])

        assert_series_equal(ds_result, pd_result)

    def test_isin_no_matches(self):
        """isin where nothing matches"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df['a'].isin([10, 20, 30])
        ds_result = ds_df['a'].isin([10, 20, 30])

        assert_series_equal(ds_result, pd_result)

    def test_isin_all_match(self):
        """isin where everything matches"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df['a'].isin([1, 2, 3, 4, 5])
        ds_result = ds_df['a'].isin([1, 2, 3, 4, 5])

        assert_series_equal(ds_result, pd_result)

    def test_isin_with_strings(self):
        """isin with string values"""
        pd_df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'David']})
        ds_df = DataStore({'name': ['Alice', 'Bob', 'Charlie', 'David']})

        pd_result = pd_df['name'].isin(['Alice', 'David'])
        ds_result = ds_df['name'].isin(['Alice', 'David'])

        assert_series_equal(ds_result, pd_result)

    def test_isin_filter_dataframe(self):
        """Use isin to filter DataFrame"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df[pd_df['category'].isin(['A', 'C'])]
        ds_result = ds_df[ds_df['category'].isin(['A', 'C'])]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArithmeticEdgeCases:
    """Test arithmetic operations with edge cases"""

    def test_divide_by_zero(self):
        """Division by zero should produce inf"""
        pd_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0],
            'b': [2.0, 0.0, 5.0]
        })
        ds_df = DataStore({
            'a': [10.0, 20.0, 30.0],
            'b': [2.0, 0.0, 5.0]
        })

        pd_result = pd_df['a'] / pd_df['b']
        ds_result = ds_df['a'] / ds_df['b']

        assert_series_equal(ds_result, pd_result)

    def test_modulo_with_zero(self):
        """Modulo with zero in divisor"""
        pd_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0],
            'b': [3.0, 0.0, 7.0]
        })
        ds_df = DataStore({
            'a': [10.0, 20.0, 30.0],
            'b': [3.0, 0.0, 7.0]
        })

        pd_result = pd_df['a'] % pd_df['b']
        ds_result = ds_df['a'] % ds_df['b']

        assert_series_equal(ds_result, pd_result)

    def test_negative_power(self):
        """Negative exponent"""
        pd_df = pd.DataFrame({
            'base': [2.0, 3.0, 4.0],
            'exp': [-1.0, -2.0, 0.0]
        })
        ds_df = DataStore({
            'base': [2.0, 3.0, 4.0],
            'exp': [-1.0, -2.0, 0.0]
        })

        pd_result = pd_df['base'] ** pd_df['exp']
        ds_result = ds_df['base'] ** ds_df['exp']

        assert_series_equal(ds_result, pd_result)

    def test_scalar_operations(self):
        """Arithmetic with scalar values"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        # Addition
        pd_add = pd_df['a'] + 10
        ds_add = ds_df['a'] + 10
        assert_series_equal(ds_add, pd_add)

        # Subtraction
        pd_sub = pd_df['a'] - 2
        ds_sub = ds_df['a'] - 2
        assert_series_equal(ds_sub, pd_sub)

        # Multiplication
        pd_mul = pd_df['a'] * 3
        ds_mul = ds_df['a'] * 3
        assert_series_equal(ds_mul, pd_mul)

        # Division
        pd_div = pd_df['a'] / 2
        ds_div = ds_df['a'] / 2
        assert_series_equal(ds_div, pd_div)


class TestFirstLastValidIndex:
    """Test first_valid_index() and last_valid_index()"""

    def test_first_valid_index_with_leading_nan(self):
        """First valid index when leading values are NaN"""
        pd_df = pd.DataFrame({'a': [np.nan, np.nan, 1.0, 2.0, 3.0]})
        ds_df = DataStore({'a': [np.nan, np.nan, 1.0, 2.0, 3.0]})

        pd_result = pd_df['a'].first_valid_index()
        ds_result = ds_df['a'].first_valid_index()

        assert ds_result == pd_result

    def test_last_valid_index_with_trailing_nan(self):
        """Last valid index when trailing values are NaN"""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, np.nan, np.nan]})
        ds_df = DataStore({'a': [1.0, 2.0, 3.0, np.nan, np.nan]})

        pd_result = pd_df['a'].last_valid_index()
        ds_result = ds_df['a'].last_valid_index()

        assert ds_result == pd_result

    def test_first_valid_index_no_nan(self):
        """First valid index with no NaN values"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].first_valid_index()
        ds_result = ds_df['a'].first_valid_index()

        assert ds_result == pd_result

    def test_last_valid_index_no_nan(self):
        """Last valid index with no NaN values"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].last_valid_index()
        ds_result = ds_df['a'].last_valid_index()

        assert ds_result == pd_result


class TestTruncateEdgeCases:
    """Test truncate() with various parameters"""

    def test_truncate_with_integer_index(self):
        """Truncate with integer index"""
        pd_df = pd.DataFrame({'a': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [10, 20, 30, 40, 50]})

        pd_result = pd_df.truncate(before=1, after=3)
        ds_result = ds_df.truncate(before=1, after=3)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_truncate_only_before(self):
        """Truncate with only before parameter"""
        pd_df = pd.DataFrame({'a': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [10, 20, 30, 40, 50]})

        pd_result = pd_df.truncate(before=2)
        ds_result = ds_df.truncate(before=2)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_truncate_only_after(self):
        """Truncate with only after parameter"""
        pd_df = pd.DataFrame({'a': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [10, 20, 30, 40, 50]})

        pd_result = pd_df.truncate(after=2)
        ds_result = ds_df.truncate(after=2)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


class TestXsCrossSection:
    """Test xs() cross-section access"""

    def test_xs_with_single_index(self):
        """xs() with regular index"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        }, index=['x', 'y', 'z', 'w', 'v'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.xs('y')
        ds_result = ds_df.xs('y')

        assert_series_equal(ds_result, pd_result)


class TestShiftEdgeCases:
    """Test shift() with various parameters"""

    def test_shift_positive(self):
        """Shift with positive periods"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].shift(2)
        ds_result = ds_df['a'].shift(2)

        assert_series_equal(ds_result, pd_result)

    def test_shift_negative(self):
        """Shift with negative periods (shift backward)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].shift(-2)
        ds_result = ds_df['a'].shift(-2)

        assert_series_equal(ds_result, pd_result)

    def test_shift_with_fill_value(self):
        """Shift with fill_value parameter"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].shift(2, fill_value=0)
        ds_result = ds_df['a'].shift(2, fill_value=0)

        assert_series_equal(ds_result, pd_result)

    def test_shift_dataframe(self):
        """Shift entire DataFrame"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4],
            'b': [10, 20, 30, 40]
        })

        pd_result = pd_df.shift(1)
        ds_result = ds_df.shift(1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDiffEdgeCases:
    """Test diff() with various parameters"""

    def test_diff_default(self):
        """diff() with default parameters"""
        pd_df = pd.DataFrame({'a': [1, 3, 6, 10, 15]})
        ds_df = DataStore({'a': [1, 3, 6, 10, 15]})

        pd_result = pd_df['a'].diff()
        ds_result = ds_df['a'].diff()

        assert_series_equal(ds_result, pd_result)

    def test_diff_periods_2(self):
        """diff() with periods=2"""
        pd_df = pd.DataFrame({'a': [1, 2, 4, 7, 11]})
        ds_df = DataStore({'a': [1, 2, 4, 7, 11]})

        pd_result = pd_df['a'].diff(periods=2)
        ds_result = ds_df['a'].diff(periods=2)

        assert_series_equal(ds_result, pd_result)

    def test_diff_negative_periods(self):
        """diff() with negative periods"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].diff(periods=-1)
        ds_result = ds_df['a'].diff(periods=-1)

        assert_series_equal(ds_result, pd_result)


class TestPctChangeEdgeCases:
    """Test pct_change() percentage change calculation"""

    def test_pct_change_default(self):
        """pct_change() with default parameters"""
        pd_df = pd.DataFrame({'a': [10.0, 12.0, 15.0, 18.0, 20.0]})
        ds_df = DataStore({'a': [10.0, 12.0, 15.0, 18.0, 20.0]})

        pd_result = pd_df['a'].pct_change()
        ds_result = ds_df['a'].pct_change()

        assert_series_equal(ds_result, pd_result)

    def test_pct_change_periods_2(self):
        """pct_change() with periods=2"""
        pd_df = pd.DataFrame({'a': [100.0, 110.0, 121.0, 133.1, 146.41]})
        ds_df = DataStore({'a': [100.0, 110.0, 121.0, 133.1, 146.41]})

        pd_result = pd_df['a'].pct_change(periods=2)
        ds_result = ds_df['a'].pct_change(periods=2)

        assert_series_equal(ds_result, pd_result)

    def test_pct_change_with_zero(self):
        """pct_change() with zero values (division by zero)"""
        pd_df = pd.DataFrame({'a': [0.0, 10.0, 20.0, 0.0, 30.0]})
        ds_df = DataStore({'a': [0.0, 10.0, 20.0, 0.0, 30.0]})

        pd_result = pd_df['a'].pct_change()
        ds_result = ds_df['a'].pct_change()

        assert_series_equal(ds_result, pd_result)


class TestCumsumCumprodEdgeCases:
    """Test cumsum() and cumprod() edge cases"""

    def test_cumsum_with_nan(self):
        """cumsum() with NaN values"""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0, 5.0]})
        ds_df = DataStore({'a': [1.0, 2.0, np.nan, 4.0, 5.0]})

        pd_result = pd_df['a'].cumsum()
        ds_result = ds_df['a'].cumsum()

        assert_series_equal(ds_result, pd_result)

    def test_cumprod_with_nan(self):
        """cumprod() with NaN values"""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 3.0, 4.0]})
        ds_df = DataStore({'a': [1.0, 2.0, np.nan, 3.0, 4.0]})

        pd_result = pd_df['a'].cumprod()
        ds_result = ds_df['a'].cumprod()

        assert_series_equal(ds_result, pd_result)

    def test_cummax_cummin(self):
        """cummax() and cummin()"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5, 9, 2, 6]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5, 9, 2, 6]})

        pd_max = pd_df['a'].cummax()
        ds_max = ds_df['a'].cummax()
        assert_series_equal(ds_max, pd_max)

        pd_min = pd_df['a'].cummin()
        ds_min = ds_df['a'].cummin()
        assert_series_equal(ds_min, pd_min)


class TestClipEdgeCases:
    """Test clip() with various parameters"""

    def test_clip_both_bounds(self):
        """clip() with both lower and upper bounds"""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15, 20]})
        ds_df = DataStore({'a': [1, 5, 10, 15, 20]})

        pd_result = pd_df['a'].clip(lower=5, upper=15)
        ds_result = ds_df['a'].clip(lower=5, upper=15)

        assert_series_equal(ds_result, pd_result)

    def test_clip_only_lower(self):
        """clip() with only lower bound"""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15, 20]})
        ds_df = DataStore({'a': [1, 5, 10, 15, 20]})

        pd_result = pd_df['a'].clip(lower=10)
        ds_result = ds_df['a'].clip(lower=10)

        assert_series_equal(ds_result, pd_result)

    def test_clip_only_upper(self):
        """clip() with only upper bound"""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15, 20]})
        ds_df = DataStore({'a': [1, 5, 10, 15, 20]})

        pd_result = pd_df['a'].clip(upper=10)
        ds_result = ds_df['a'].clip(upper=10)

        assert_series_equal(ds_result, pd_result)

    def test_clip_dataframe(self):
        """clip() on entire DataFrame"""
        pd_df = pd.DataFrame({
            'a': [1, 5, 10, 15, 20],
            'b': [2, 6, 11, 16, 21]
        })
        ds_df = DataStore({
            'a': [1, 5, 10, 15, 20],
            'b': [2, 6, 11, 16, 21]
        })

        pd_result = pd_df.clip(lower=5, upper=15)
        ds_result = ds_df.clip(lower=5, upper=15)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRankEdgeCases:
    """Test rank() with various parameters"""

    def test_rank_default(self):
        """rank() with default parameters"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df['a'].rank()
        ds_result = ds_df['a'].rank()

        assert_series_equal(ds_result, pd_result)

    def test_rank_method_min(self):
        """rank() with method='min'"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df['a'].rank(method='min')
        ds_result = ds_df['a'].rank(method='min')

        assert_series_equal(ds_result, pd_result)

    def test_rank_method_max(self):
        """rank() with method='max'"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df['a'].rank(method='max')
        ds_result = ds_df['a'].rank(method='max')

        assert_series_equal(ds_result, pd_result)

    def test_rank_ascending_false(self):
        """rank() with ascending=False"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df['a'].rank(ascending=False)
        ds_result = ds_df['a'].rank(ascending=False)

        assert_series_equal(ds_result, pd_result)

    def test_rank_with_nan(self):
        """rank() with NaN values"""
        pd_df = pd.DataFrame({'a': [3.0, np.nan, 4.0, 1.0, 5.0]})
        ds_df = DataStore({'a': [3.0, np.nan, 4.0, 1.0, 5.0]})

        pd_result = pd_df['a'].rank()
        ds_result = ds_df['a'].rank()

        assert_series_equal(ds_result, pd_result)


class TestAbsEdgeCases:
    """Test abs() with various data types"""

    def test_abs_mixed_signs(self):
        """abs() with mixed positive and negative values"""
        pd_df = pd.DataFrame({'a': [-5, -3, 0, 3, 5]})
        ds_df = DataStore({'a': [-5, -3, 0, 3, 5]})

        pd_result = pd_df['a'].abs()
        ds_result = ds_df['a'].abs()

        assert_series_equal(ds_result, pd_result)

    def test_abs_floats(self):
        """abs() with float values"""
        pd_df = pd.DataFrame({'a': [-5.5, -3.3, 0.0, 3.3, 5.5]})
        ds_df = DataStore({'a': [-5.5, -3.3, 0.0, 3.3, 5.5]})

        pd_result = pd_df['a'].abs()
        ds_result = ds_df['a'].abs()

        assert_series_equal(ds_result, pd_result)

    def test_abs_dataframe(self):
        """abs() on entire DataFrame"""
        pd_df = pd.DataFrame({
            'a': [-1, -2, 3],
            'b': [4, -5, -6]
        })
        ds_df = DataStore({
            'a': [-1, -2, 3],
            'b': [4, -5, -6]
        })

        pd_result = pd_df.abs()
        ds_result = ds_df.abs()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRoundEdgeCases:
    """Test round() with various decimal places"""

    def test_round_default(self):
        """round() with default decimals=0"""
        pd_df = pd.DataFrame({'a': [1.4, 2.5, 3.6, 4.5, 5.4]})
        ds_df = DataStore({'a': [1.4, 2.5, 3.6, 4.5, 5.4]})

        pd_result = pd_df['a'].round()
        ds_result = ds_df['a'].round()

        assert_series_equal(ds_result, pd_result)

    def test_round_2_decimals(self):
        """round() with decimals=2"""
        pd_df = pd.DataFrame({'a': [1.234, 2.567, 3.891]})
        ds_df = DataStore({'a': [1.234, 2.567, 3.891]})

        pd_result = pd_df['a'].round(2)
        ds_result = ds_df['a'].round(2)

        assert_series_equal(ds_result, pd_result)

    def test_round_negative_decimals(self):
        """round() with negative decimals (round to tens, hundreds, etc)"""
        pd_df = pd.DataFrame({'a': [1234, 5678, 9012]})
        ds_df = DataStore({'a': [1234, 5678, 9012]})

        pd_result = pd_df['a'].round(-2)
        ds_result = ds_df['a'].round(-2)

        assert_series_equal(ds_result, pd_result)


class TestComparisonChainEdgeCases:
    """Test comparison operations chained together"""

    def test_between_equivalent(self):
        """Test equivalent of between using comparisons"""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15, 20]})
        ds_df = DataStore({'a': [1, 5, 10, 15, 20]})

        # Using between
        pd_result_between = pd_df[pd_df['a'].between(5, 15)]
        ds_result_between = ds_df[ds_df['a'].between(5, 15)]
        assert_datastore_equals_pandas(ds_result_between, pd_result_between)

        # Using chained comparisons
        pd_result_chain = pd_df[(pd_df['a'] >= 5) & (pd_df['a'] <= 15)]
        ds_result_chain = ds_df[(ds_df['a'] >= 5) & (ds_df['a'] <= 15)]
        assert_datastore_equals_pandas(ds_result_chain, pd_result_chain)

    def test_multiple_or_conditions(self):
        """Multiple OR conditions"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df[(pd_df['a'] == 1) | (pd_df['a'] == 3) | (pd_df['a'] == 5)]
        ds_result = ds_df[(ds_df['a'] == 1) | (ds_df['a'] == 3) | (ds_df['a'] == 5)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negation(self):
        """Test negation (~) operator"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df[~(pd_df['a'] > 3)]
        ds_result = ds_df[~(ds_df['a'] > 3)]

        assert_datastore_equals_pandas(ds_result, pd_result)
