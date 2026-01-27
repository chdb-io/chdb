"""
Exploratory Batch 31: Iteration Protocols and Rare Methods

Focus areas:
1. Iteration protocols (iterrows, itertuples, items, __iter__)
2. Rare methods (squeeze, asfreq, from_dict, from_records, to_xml, isetitem)
3. Cumulative operations (cumprod, cummin, cummax)
4. mode/prod methods
5. __array__ protocol
6. Additional edge cases

Note: DataStore is immutable, so methods like isetitem and insert return new
DataStore objects instead of modifying in place. Tests are designed accordingly.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import get_dataframe, assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal, get_series, get_value


class TestIterationProtocols:
    """Test iteration methods for DataFrame."""
    
    def test_iterrows_basic(self):
        """Test basic iterrows functionality."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())
        
        pd_rows = list(pd_df.iterrows())
        ds_rows = list(ds_df.iterrows())
        
        assert len(pd_rows) == len(ds_rows)
        for (pd_idx, pd_row), (ds_idx, ds_row) in zip(pd_rows, ds_rows):
            assert pd_idx == ds_idx
            assert_series_equal(pd_row, ds_row)
    
    def test_iterrows_empty(self):
        """Test iterrows on empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore(pd_df.copy())
        
        pd_rows = list(pd_df.iterrows())
        ds_rows = list(ds_df.iterrows())
        
        assert len(pd_rows) == 0
        assert len(ds_rows) == 0
    
    def test_itertuples_basic(self):
        """Test basic itertuples functionality."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_tuples = list(pd_df.itertuples())
        ds_tuples = list(ds_df.itertuples())
        
        assert len(pd_tuples) == len(ds_tuples)
        for pd_t, ds_t in zip(pd_tuples, ds_tuples):
            assert pd_t == ds_t
    
    def test_itertuples_with_index_false(self):
        """Test itertuples with index=False."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_tuples = list(pd_df.itertuples(index=False))
        ds_tuples = list(ds_df.itertuples(index=False))
        
        assert pd_tuples == ds_tuples
    
    def test_itertuples_with_name(self):
        """Test itertuples with custom name."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2]})
        ds_df = DataStore(pd_df.copy())
        
        pd_tuple = next(pd_df.itertuples(name='MyRow'))
        ds_tuple = next(ds_df.itertuples(name='MyRow'))
        
        assert type(pd_tuple).__name__ == type(ds_tuple).__name__
        assert pd_tuple == ds_tuple
    
    def test_items_basic(self):
        """Test items() method."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_items = list(pd_df.items())
        ds_items = list(ds_df.items())
        
        assert len(pd_items) == len(ds_items)
        for (pd_col, pd_series), (ds_col, ds_series) in zip(pd_items, ds_items):
            assert pd_col == ds_col
            assert_series_equal(pd_series, ds_series)
    
    def test_iter_columns(self):
        """Test __iter__ iterates over column names."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_cols = list(iter(pd_df))
        ds_cols = list(iter(ds_df))
        
        assert pd_cols == ds_cols


class TestRareMethods:
    """Test uncommon DataFrame methods."""
    
    def test_squeeze_single_column(self):
        """Test squeeze on single-column DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()
        
        # Should return Series
        assert isinstance(pd_result, pd.Series)
        assert_series_equal(pd_result, ds_result)
    
    def test_squeeze_single_row(self):
        """Test squeeze on single-row DataFrame."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()
        
        assert isinstance(pd_result, pd.Series)
        assert_series_equal(pd_result, ds_result)
    
    def test_squeeze_single_value(self):
        """Test squeeze on 1x1 DataFrame."""
        pd_df = pd.DataFrame({'A': [42]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()
        
        # Should return scalar
        assert pd_result == ds_result == 42
    
    def test_squeeze_axis_columns(self):
        """Test squeeze with axis='columns'."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.squeeze(axis='columns')
        ds_result = ds_df.squeeze(axis='columns')
        
        assert_series_equal(pd_result, ds_result)
    
    def test_squeeze_no_change(self):
        """Test squeeze on multi-row multi-column DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()
        
        # Should return DataFrame unchanged
        assert isinstance(pd_result, pd.DataFrame)
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_from_dict_columns(self):
        """Test from_dict class method with columns orient."""
        data = {'A': [1, 2], 'B': [3, 4]}
        
        pd_df = pd.DataFrame.from_dict(data)
        ds_df = DataStore.from_dict(data)
        
        assert_datastore_equals_pandas(ds_df, pd_df)
    
    def test_from_dict_index(self):
        """Test from_dict with orient='index'."""
        data = {'row1': {'A': 1, 'B': 2}, 'row2': {'A': 3, 'B': 4}}
        
        pd_df = pd.DataFrame.from_dict(data, orient='index')
        ds_df = DataStore.from_dict(data, orient='index')
        
        assert_datastore_equals_pandas(ds_df, pd_df)
    
    def test_from_records_basic(self):
        """Test from_records class method."""
        data = [{'A': 1, 'B': 2}, {'A': 3, 'B': 4}]
        
        pd_df = pd.DataFrame.from_records(data)
        ds_df = DataStore.from_records(data)
        
        assert_datastore_equals_pandas(ds_df, pd_df)
    
    def test_from_records_with_columns(self):
        """Test from_records with explicit columns."""
        data = [(1, 2, 3), (4, 5, 6)]
        columns = ['A', 'B', 'C']
        
        pd_df = pd.DataFrame.from_records(data, columns=columns)
        ds_df = DataStore.from_records(data, columns=columns)
        
        assert_datastore_equals_pandas(ds_df, pd_df)
    
    def test_isetitem_returns_new_datastore(self):
        """Test isetitem returns new DataStore (immutable design).
        
        DataStore is immutable, so isetitem returns a new DataStore
        with the modified column instead of modifying in place.
        """
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        # isetitem returns new DataStore
        ds_result = ds_df.isetitem(0, [10, 20])
        
        # Original should be unchanged
        assert_datastore_equals_pandas(ds_df, pd_df)
        
        # Result should have the new values
        expected = pd.DataFrame({'A': [10, 20], 'B': [3, 4]})
        assert_datastore_equals_pandas(ds_result, expected)
    
    def test_isetitem_last_column(self):
        """Test isetitem on last column returns new DataStore."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
        ds_df = DataStore(pd_df.copy())
        
        ds_result = ds_df.isetitem(2, [100])
        
        expected = pd.DataFrame({'A': [1], 'B': [2], 'C': [100]})
        assert_datastore_equals_pandas(ds_result, expected)


class TestCumulativeOperations:
    """Test cumulative operations."""
    
    def test_cumprod_basic(self):
        """Test cumulative product."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 2, 2, 2]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.cumprod()
        ds_result = ds_df.cumprod()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_cumprod_with_nan(self):
        """Test cumulative product with NaN."""
        pd_df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.cumprod()
        ds_result = ds_df.cumprod()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_cummin_basic(self):
        """Test cumulative minimum."""
        pd_df = pd.DataFrame({'A': [3, 1, 4, 1, 5], 'B': [9, 2, 6, 5, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.cummin()
        ds_result = ds_df.cummin()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_cummin_with_nan(self):
        """Test cumulative minimum with NaN."""
        pd_df = pd.DataFrame({'A': [3.0, np.nan, 1.0, 2.0]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.cummin()
        ds_result = ds_df.cummin()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_cummax_basic(self):
        """Test cumulative maximum."""
        pd_df = pd.DataFrame({'A': [1, 4, 2, 5, 3], 'B': [5, 2, 8, 1, 9]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.cummax()
        ds_result = ds_df.cummax()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_cummax_with_nan(self):
        """Test cumulative maximum with NaN."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, 3.0, 2.0]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.cummax()
        ds_result = ds_df.cummax()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestModeAndProd:
    """Test mode and prod methods."""
    
    def test_mode_single_mode(self):
        """Test mode with single mode value."""
        pd_df = pd.DataFrame({'A': [1, 2, 2, 3, 2]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.mode()
        ds_result = ds_df.mode()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_mode_multiple_modes(self):
        """Test mode with multiple mode values."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.mode()
        ds_result = ds_df.mode()
        
        # Mode returns all modes sorted
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_mode_numeric_only(self):
        """Test mode with numeric_only parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 2], 'B': ['x', 'x', 'y']})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.mode(numeric_only=True)
        ds_result = ds_df.mode(numeric_only=True)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_prod_basic(self):
        """Test product of values."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.prod()
        ds_result = ds_df.prod()
        
        assert_series_equal(ds_result, pd_result)
    
    def test_prod_with_nan_skipna(self):
        """Test product with NaN and skipna."""
        pd_df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.prod(skipna=True)
        ds_result = ds_df.prod(skipna=True)
        
        assert_series_equal(ds_result, pd_result)
    
    def test_prod_axis_columns(self):
        """Test product along columns axis."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.prod(axis=1)
        ds_result = ds_df.prod(axis=1)
        
        assert_series_equal(ds_result, pd_result)
    
    def test_product_alias(self):
        """Test product() as alias for prod()."""
        pd_df = pd.DataFrame({'A': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.product()
        ds_result = ds_df.product()
        
        assert_series_equal(ds_result, pd_result)


class TestArrayProtocol:
    """Test numpy array protocol."""
    
    def test_array_basic(self):
        """Test __array__ conversion."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_arr = np.array(pd_df)
        ds_arr = np.array(ds_df)
        
        np.testing.assert_array_equal(pd_arr, ds_arr)
    
    def test_array_with_dtype(self):
        """Test __array__ with explicit dtype."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_arr = np.array(pd_df, dtype=float)
        ds_arr = np.array(ds_df, dtype=float)
        
        np.testing.assert_array_equal(pd_arr, ds_arr)
        assert pd_arr.dtype == ds_arr.dtype
    
    def test_array_in_numpy_function(self):
        """Test DataStore works with numpy functions expecting arrays."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        # Use axis=0 to get sum per column
        pd_sum = np.sum(pd_df, axis=0)
        ds_sum = np.sum(ds_df, axis=0)
        
        assert_series_equal(pd_sum, ds_sum)
    
    def test_array_column_expr(self):
        """Test __array__ on ColumnExpr."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_arr = np.array(pd_df['A'])
        ds_arr = np.array(ds_df['A'])
        
        np.testing.assert_array_equal(pd_arr, ds_arr)


class TestXSMethod:
    """Test cross-section selection."""
    
    def test_xs_row_selection(self):
        """Test xs for row selection."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.xs('y')
        ds_result = ds_df.xs('y')
        
        assert_series_equal(pd_result, ds_result)
    
    def test_xs_column_selection(self):
        """Test xs for column selection."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.xs('A', axis=1)
        ds_result = ds_df.xs('A', axis=1)
        
        assert_series_equal(pd_result, ds_result)


class TestSwapMethods:
    """Test swaplevel and swapaxes methods."""
    
    def test_swapaxes_basic(self):
        """Test swapaxes (transpose equivalent)."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.swapaxes(0, 1)
        ds_result = ds_df.swapaxes(0, 1)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_swaplevel_multiindex(self):
        """Test swaplevel on MultiIndex."""
        arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.swaplevel()
        ds_result = ds_df.swaplevel()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestImmutableDesign:
    """Test that DataStore follows immutable design principles.
    
    DataStore is designed to be immutable for functional programming style
    and to prevent side effects. Methods that would mutate in pandas
    return new DataStore objects instead.
    """
    
    def test_drop_inplace_not_supported(self):
        """Test that inplace=True raises ValueError."""
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]})
        
        with pytest.raises(ValueError, match="inplace=True is not supported"):
            ds_df.drop(columns=['B'], inplace=True)
    
    def test_fillna_inplace_not_supported(self):
        """Test that fillna inplace=True raises ValueError."""
        ds_df = DataStore({'A': [1.0, np.nan, 3.0]})
        
        with pytest.raises(ValueError, match="inplace=True is not supported"):
            ds_df.fillna(0, inplace=True)
    
    def test_rename_inplace_not_supported(self):
        """Test that rename inplace=True raises ValueError."""
        ds_df = DataStore({'A': [1, 2]})
        
        with pytest.raises(ValueError, match="inplace=True is not supported"):
            ds_df.rename(columns={'A': 'B'}, inplace=True)
    
    def test_reset_index_inplace_not_supported(self):
        """Test that reset_index inplace=True raises ValueError."""
        ds_df = DataStore({'A': [1, 2]})
        
        with pytest.raises(ValueError, match="inplace=True is not supported"):
            ds_df.reset_index(inplace=True, drop=True)
    
    def test_drop_returns_new_datastore(self):
        """Test that drop returns new DataStore without modifying original."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        ds_result = ds_df.drop(columns=['B'])
        
        # Original unchanged
        assert list(ds_df.columns) == ['A', 'B']
        # Result has dropped column
        assert list(ds_result.columns) == ['A']
    
    def test_fillna_returns_new_datastore(self):
        """Test that fillna returns new DataStore without modifying original."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, 3.0]})
        ds_df = DataStore(pd_df.copy())
        
        ds_result = ds_df.fillna(0)
        
        # Original has NaN
        assert np.isnan(get_dataframe(ds_df)['A'].iloc[1])
        # Result has 0
        assert get_dataframe(ds_result)['A'].iloc[1] == 0


class TestInsertMethod:
    """Test insert method behavior (inplace like pandas)."""
    
    def test_insert_inplace(self):
        """Test that insert modifies DataStore in place (like pandas)."""
        pd_df = pd.DataFrame({'A': [1, 2], 'C': [5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        # pandas insert is inplace
        pd_df.insert(1, 'B', [3, 4])
        ds_df.insert(1, 'B', [3, 4])
        
        # DataStore has inserted column (inplace)
        assert list(ds_df.columns) == ['A', 'B', 'C']
        
        expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        assert_datastore_equals_pandas(ds_df, expected)
    
    def test_insert_returns_none(self):
        """Test that insert returns None (like pandas)."""
        ds_df = DataStore(pd.DataFrame({'A': [1, 2], 'C': [5, 6]}))
        result = ds_df.insert(1, 'B', [3, 4])
        
        assert result is None
        assert list(ds_df.columns) == ['A', 'B', 'C']
    
    def test_insert_at_beginning(self):
        """Test insert at position 0."""
        pd_df = pd.DataFrame({'B': [2], 'C': [3]})
        ds_df = DataStore(pd_df.copy())
        
        ds_df.insert(0, 'A', [1])
        
        assert list(ds_df.columns) == ['A', 'B', 'C']
    
    def test_insert_at_end(self):
        """Test insert at end position."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2]})
        ds_df = DataStore(pd_df.copy())
        
        ds_df.insert(2, 'C', [3])
        
        assert list(ds_df.columns) == ['A', 'B', 'C']


class TestEdgeCases:
    """Additional edge cases."""
    
    def test_pop_column(self):
        """Test pop method to remove and return column."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_popped = pd_df.pop('A')
        ds_popped = ds_df.pop('A')
        
        assert_series_equal(pd_popped, ds_popped)
        assert_datastore_equals_pandas(ds_df, pd_df)
    
    def test_get_with_default(self):
        """Test get method with default value."""
        pd_df = pd.DataFrame({'A': [1, 2]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.get('B', default='missing')
        ds_result = ds_df.get('B', default='missing')
        
        assert pd_result == ds_result == 'missing'
    
    def test_keys_method(self):
        """Test keys method (column names)."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_keys = list(pd_df.keys())
        ds_keys = list(ds_df.keys())
        
        assert pd_keys == ds_keys
    
    def test_bool_single_element_true(self):
        """Test bool() behavior on single True value."""
        pd_df = pd.DataFrame({'A': [True]})
        ds_df = DataStore(pd_df.copy())
        
        # Access single element via iloc
        assert bool(pd_df.iloc[0, 0]) == bool(ds_df.iloc[0, 0]) == True
    
    def test_bool_single_element_false(self):
        """Test bool() behavior on single False value."""
        pd_df = pd.DataFrame({'A': [0]})
        ds_df = DataStore(pd_df.copy())
        
        # Access single element via iloc
        assert bool(pd_df.iloc[0, 0]) == bool(ds_df.iloc[0, 0]) == False
    
    def test_abs_method(self):
        """Test abs method."""
        pd_df = pd.DataFrame({'A': [-1, 2, -3], 'B': [-4, -5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.abs()
        ds_result = ds_df.abs()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_round_method(self):
        """Test round method."""
        pd_df = pd.DataFrame({'A': [1.234, 2.567], 'B': [3.891, 4.123]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.round(2)
        ds_result = ds_df.round(2)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_clip_with_axis(self):
        """Test clip with different axis."""
        pd_df = pd.DataFrame({'A': [1, 5, 10], 'B': [2, 6, 11]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.clip(lower=3, upper=8)
        ds_result = ds_df.clip(lower=3, upper=8)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_idxmax_method(self):
        """Test idxmax method."""
        pd_df = pd.DataFrame({'A': [1, 5, 3], 'B': [4, 2, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.idxmax()
        ds_result = ds_df.idxmax()
        
        assert_series_equal(pd_result, ds_result)
    
    def test_idxmin_method(self):
        """Test idxmin method."""
        pd_df = pd.DataFrame({'A': [1, 5, 3], 'B': [4, 2, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.idxmin()
        ds_result = ds_df.idxmin()
        
        assert_series_equal(pd_result, ds_result)


class TestColumnExprCumulative:
    """Test cumulative operations on ColumnExpr."""
    
    def test_column_expr_cumprod(self):
        """Test cumprod on ColumnExpr."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['A'].cumprod()
        ds_result = ds_df['A'].cumprod()
        
        # Execute and compare
        ds_series = get_series(ds_result)
        assert_series_equal(pd_result, ds_series)
    
    def test_column_expr_cummin(self):
        """Test cummin on ColumnExpr."""
        pd_df = pd.DataFrame({'A': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['A'].cummin()
        ds_result = ds_df['A'].cummin()
        
        ds_series = get_series(ds_result)
        assert_series_equal(pd_result, ds_series)
    
    def test_column_expr_cummax(self):
        """Test cummax on ColumnExpr."""
        pd_df = pd.DataFrame({'A': [1, 4, 2, 5, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['A'].cummax()
        ds_result = ds_df['A'].cummax()
        
        ds_series = get_series(ds_result)
        assert_series_equal(pd_result, ds_series)
    
    def test_column_expr_prod(self):
        """Test prod on ColumnExpr."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['A'].prod()
        ds_result = ds_df['A'].prod()
        
        # Execute if lazy
        ds_value = get_value(ds_result)
        assert pd_result == ds_value


class TestMiscellaneousEdgeCases:
    """Miscellaneous edge cases."""
    
    def test_empty_df_items_iteration(self):
        """Test items iteration on empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore(pd_df.copy())
        
        pd_items = list(pd_df.items())
        ds_items = list(ds_df.items())
        
        assert len(pd_items) == len(ds_items)
        for (pd_col, pd_series), (ds_col, ds_series) in zip(pd_items, ds_items):
            assert pd_col == ds_col
            assert_series_equal(pd_series, ds_series)
    
    def test_empty_df_itertuples(self):
        """Test itertuples on empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore(pd_df.copy())
        
        assert list(pd_df.itertuples()) == list(ds_df.itertuples())
    
    def test_single_row_squeeze(self):
        """Test squeeze on single row."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.squeeze(axis=0)
        ds_result = ds_df.squeeze(axis=0)
        
        assert_series_equal(pd_result, ds_result)
    
    def test_filter_then_cumulative(self):
        """Test filter followed by cumulative operation."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[pd_df['A'] > 2].cumsum()
        ds_result = ds_df[ds_df['A'] > 2].cumsum()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_xs_after_groupby(self):
        """Test xs after aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())
        
        pd_agg = pd_df.groupby('group').sum()
        ds_agg = ds_df.groupby('group').sum()
        
        # xs on aggregated result
        pd_result = pd_agg.xs('A')
        ds_result = ds_agg.xs('A')
        
        assert_series_equal(pd_result, ds_result)
    
    def test_large_cumulative(self):
        """Test cumulative on larger dataset."""
        n = 1000
        pd_df = pd.DataFrame({'A': np.random.randn(n)})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.cumsum()
        ds_result = ds_df.cumsum()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
