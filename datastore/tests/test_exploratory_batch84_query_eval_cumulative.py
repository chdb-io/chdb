"""
Exploratory Batch 84: Query, Eval, Cumulative Operations Edge Cases

Focus areas:
1. query() with various expressions and @variable references
2. eval() for column expressions
3. Cumulative operations (cumsum, cummax, cummin, cumprod)
4. diff() and pct_change() operations
5. clip() with various bounds
6. idxmax/idxmin operations
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestQueryEdgeCases:
    """Test query() method edge cases"""

    def test_query_simple(self):
        """query() with simple boolean expression"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.query('A > 2')
        ds_result = ds_df.query('A > 2')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_compound_condition(self):
        """query() with compound boolean expression"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.query('A > 2 and B < 50')
        ds_result = ds_df.query('A > 2 and B < 50')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_or_condition(self):
        """query() with OR condition"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.query('A == 1 or A == 5')
        ds_result = ds_df.query('A == 1 or A == 5')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_variable_reference(self):
        """query() with @variable reference"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        threshold = 3
        pd_result = pd_df.query('A > @threshold')
        ds_result = ds_df.query('A > @threshold')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_in_operator(self):
        """query() with 'in' operator"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })

        pd_result = pd_df.query('A in [1, 3, 5]')
        ds_result = ds_df.query('A in [1, 3, 5]')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_not_in(self):
        """query() with 'not in' operator"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.query('A not in [2, 4]')
        ds_result = ds_df.query('A not in [2, 4]')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_column_comparison(self):
        """query() comparing two columns"""
        pd_df = pd.DataFrame({
            'A': [1, 5, 3, 4, 2],
            'B': [2, 3, 4, 3, 5]
        })
        ds_df = DataStore({
            'A': [1, 5, 3, 4, 2],
            'B': [2, 3, 4, 3, 5]
        })

        pd_result = pd_df.query('A > B')
        ds_result = ds_df.query('A > B')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEvalEdgeCases:
    """Test eval() method edge cases"""

    def test_eval_simple_expression(self):
        """eval() with simple arithmetic expression"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.eval('C = A + B')
        ds_result = ds_df.eval('C = A + B')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_complex_expression(self):
        """eval() with complex arithmetic expression"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })

        pd_result = pd_df.eval('D = A + B * C')
        ds_result = ds_df.eval('D = A + B * C')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_division(self):
        """eval() with division operation"""
        pd_df = pd.DataFrame({
            'A': [10.0, 20.0, 30.0],
            'B': [2.0, 4.0, 5.0]
        })
        ds_df = DataStore({
            'A': [10.0, 20.0, 30.0],
            'B': [2.0, 4.0, 5.0]
        })

        pd_result = pd_df.eval('C = A / B')
        ds_result = ds_df.eval('C = A / B')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_without_assignment(self):
        """eval() returning computed Series without assignment"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.eval('A + B')
        ds_result = ds_df.eval('A + B')

        assert_series_equal(ds_result, pd_result)


class TestCumulativeOperations:
    """Test cumulative operations (cumsum, cummax, cummin, cumprod)"""

    def test_cumsum_simple(self):
        """cumsum() basic operation"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })

        pd_result = pd_df.cumsum()
        ds_result = ds_df.cumsum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_with_nulls(self):
        """cumsum() with NULL values"""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, 4.0],
            'B': [10.0, 20.0, np.nan, 40.0]
        })
        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, 4.0],
            'B': [10.0, 20.0, np.nan, 40.0]
        })

        pd_result = pd_df.cumsum()
        ds_result = ds_df.cumsum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cummax_simple(self):
        """cummax() basic operation"""
        pd_df = pd.DataFrame({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        ds_df = DataStore({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })

        pd_result = pd_df.cummax()
        ds_result = ds_df.cummax()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cummin_simple(self):
        """cummin() basic operation"""
        pd_df = pd.DataFrame({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        ds_df = DataStore({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })

        pd_result = pd_df.cummin()
        ds_result = ds_df.cummin()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumprod_simple(self):
        """cumprod() basic operation"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [2, 2, 2, 2]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [2, 2, 2, 2]
        })

        pd_result = pd_df.cumprod()
        ds_result = ds_df.cumprod()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_axis1(self):
        """cumsum() with axis=1 (row-wise)"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })

        pd_result = pd_df.cumsum(axis=1)
        ds_result = ds_df.cumsum(axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDiffOperations:
    """Test diff() and related operations"""

    def test_diff_simple(self):
        """diff() basic operation"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 4, 7, 11],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 4, 7, 11],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.diff()
        ds_result = ds_df.diff()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_periods_2(self):
        """diff() with periods=2"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.diff(periods=2)
        ds_result = ds_df.diff(periods=2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_negative_periods(self):
        """diff() with negative periods (backward diff)"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.diff(periods=-1)
        ds_result = ds_df.diff(periods=-1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_axis1(self):
        """diff() with axis=1"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [10, 11, 12]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [10, 11, 12]
        })

        pd_result = pd_df.diff(axis=1)
        ds_result = ds_df.diff(axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestClipOperations:
    """Test clip() operations"""

    def test_clip_both_bounds(self):
        """clip() with both lower and upper bounds"""
        pd_df = pd.DataFrame({
            'A': [1, 5, 10, 15, 20],
            'B': [2, 4, 6, 8, 10]
        })
        ds_df = DataStore({
            'A': [1, 5, 10, 15, 20],
            'B': [2, 4, 6, 8, 10]
        })

        pd_result = pd_df.clip(lower=5, upper=15)
        ds_result = ds_df.clip(lower=5, upper=15)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_lower_only(self):
        """clip() with only lower bound"""
        pd_df = pd.DataFrame({
            'A': [1, 5, 10, 15, 20],
            'B': [2, 4, 6, 8, 10]
        })
        ds_df = DataStore({
            'A': [1, 5, 10, 15, 20],
            'B': [2, 4, 6, 8, 10]
        })

        pd_result = pd_df.clip(lower=5)
        ds_result = ds_df.clip(lower=5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_upper_only(self):
        """clip() with only upper bound"""
        pd_df = pd.DataFrame({
            'A': [1, 5, 10, 15, 20],
            'B': [2, 4, 6, 8, 10]
        })
        ds_df = DataStore({
            'A': [1, 5, 10, 15, 20],
            'B': [2, 4, 6, 8, 10]
        })

        pd_result = pd_df.clip(upper=10)
        ds_result = ds_df.clip(upper=10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_with_series_bounds(self):
        """clip() with Series as bounds"""
        pd_df = pd.DataFrame({
            'A': [1, 5, 10, 15, 20],
            'B': [2, 4, 6, 8, 10]
        })
        ds_df = DataStore({
            'A': [1, 5, 10, 15, 20],
            'B': [2, 4, 6, 8, 10]
        })

        lower_bound = pd.Series([2, 3, 4, 5, 6])
        upper_bound = pd.Series([10, 11, 12, 13, 14])

        pd_result = pd_df.clip(lower=lower_bound, upper=upper_bound, axis=0)
        ds_result = ds_df.clip(lower=lower_bound, upper=upper_bound, axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIdxMaxMinOperations:
    """Test idxmax() and idxmin() operations"""

    def test_idxmax_simple(self):
        """idxmax() basic operation"""
        pd_df = pd.DataFrame({
            'A': [1, 5, 3],
            'B': [4, 2, 6]
        })
        ds_df = DataStore({
            'A': [1, 5, 3],
            'B': [4, 2, 6]
        })

        pd_result = pd_df.idxmax()
        ds_result = ds_df.idxmax()

        assert_series_equal(ds_result, pd_result)

    def test_idxmin_simple(self):
        """idxmin() basic operation"""
        pd_df = pd.DataFrame({
            'A': [1, 5, 3],
            'B': [4, 2, 6]
        })
        ds_df = DataStore({
            'A': [1, 5, 3],
            'B': [4, 2, 6]
        })

        pd_result = pd_df.idxmin()
        ds_result = ds_df.idxmin()

        assert_series_equal(ds_result, pd_result)

    def test_idxmax_axis1(self):
        """idxmax() with axis=1"""
        pd_df = pd.DataFrame({
            'A': [1, 5, 3],
            'B': [4, 2, 6],
            'C': [2, 3, 1]
        })
        ds_df = DataStore({
            'A': [1, 5, 3],
            'B': [4, 2, 6],
            'C': [2, 3, 1]
        })

        pd_result = pd_df.idxmax(axis=1)
        ds_result = ds_df.idxmax(axis=1)

        assert_series_equal(ds_result, pd_result)

    def test_idxmin_with_nulls(self):
        """idxmin() with NULL values"""
        pd_df = pd.DataFrame({
            'A': [np.nan, 5, 3],
            'B': [4, np.nan, 6]
        })
        ds_df = DataStore({
            'A': [np.nan, 5, 3],
            'B': [4, np.nan, 6]
        })

        pd_result = pd_df.idxmin()
        ds_result = ds_df.idxmin()

        assert_series_equal(ds_result, pd_result)


class TestCorrCovOperations:
    """Test correlation and covariance operations"""

    def test_corr_simple(self):
        """corr() basic operation"""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [1.0, 2.0, 4.0, 8.0, 16.0]
        })
        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [1.0, 2.0, 4.0, 8.0, 16.0]
        })

        pd_result = pd_df.corr()
        ds_result = ds_df.corr()

        # Compare values with tolerance for floating point
        np.testing.assert_array_almost_equal(pd_result.values, ds_result.values, decimal=10)

    def test_cov_simple(self):
        """cov() basic operation"""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0]
        })
        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0]
        })

        pd_result = pd_df.cov()
        ds_result = ds_df.cov()

        np.testing.assert_array_almost_equal(pd_result.values, ds_result.values, decimal=10)


class TestAllAnyOperations:
    """Test all() and any() operations"""

    def test_all_simple(self):
        """all() basic operation"""
        pd_df = pd.DataFrame({
            'A': [True, True, True],
            'B': [True, False, True]
        })
        ds_df = DataStore({
            'A': [True, True, True],
            'B': [True, False, True]
        })

        pd_result = pd_df.all()
        ds_result = ds_df.all()

        assert_series_equal(ds_result, pd_result)

    def test_any_simple(self):
        """any() basic operation"""
        pd_df = pd.DataFrame({
            'A': [False, False, False],
            'B': [False, True, False]
        })
        ds_df = DataStore({
            'A': [False, False, False],
            'B': [False, True, False]
        })

        pd_result = pd_df.any()
        ds_result = ds_df.any()

        assert_series_equal(ds_result, pd_result)

    def test_all_axis1(self):
        """all() with axis=1"""
        pd_df = pd.DataFrame({
            'A': [True, True, False],
            'B': [True, False, False]
        })
        ds_df = DataStore({
            'A': [True, True, False],
            'B': [True, False, False]
        })

        pd_result = pd_df.all(axis=1)
        ds_result = ds_df.all(axis=1)

        assert_series_equal(ds_result, pd_result)

    def test_any_axis1(self):
        """any() with axis=1"""
        pd_df = pd.DataFrame({
            'A': [True, False, False],
            'B': [False, False, False]
        })
        ds_df = DataStore({
            'A': [True, False, False],
            'B': [False, False, False]
        })

        pd_result = pd_df.any(axis=1)
        ds_result = ds_df.any(axis=1)

        assert_series_equal(ds_result, pd_result)


class TestQueryAfterOperations:
    """Test query() after other operations"""

    def test_query_after_assign(self):
        """query() after assign operation"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.assign(C=lambda x: x['A'] * 2).query('C > 4')
        ds_result = ds_df.assign(C=lambda x: x['A'] * 2).query('C > 4')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_after_filter(self):
        """cumsum() after filter operation"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df[pd_df['A'] > 2].cumsum()
        ds_result = ds_df[ds_df['A'] > 2].cumsum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_after_sort(self):
        """diff() after sort operation"""
        pd_df = pd.DataFrame({
            'A': [3, 1, 4, 1, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [3, 1, 4, 1, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.sort_values('A').diff()
        ds_result = ds_df.sort_values('A').diff()

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
