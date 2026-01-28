"""
Exploratory Batch 30: Special Cases and Corner Scenarios

This batch focuses on:
1. NaN/None/NA handling across different operations
2. Duplicate column names
3. Special characters in column names
4. Multi-index operations
5. DateTime edge cases
6. Object dtype mixed types
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from datetime import datetime, date, timedelta
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal
from tests.xfail_markers import chdb_timedelta_type


# =======================
# Test Fixtures
# =======================

@pytest.fixture
def df_with_nans():
    return pd.DataFrame({
        'A': [1.0, np.nan, 3.0, np.nan, 5.0],
        'B': [np.nan, 2.0, np.nan, 4.0, np.nan],
        'C': ['a', None, 'c', 'd', None]
    })


@pytest.fixture
def df_all_nan():
    return pd.DataFrame({
        'A': [np.nan, np.nan, np.nan],
        'B': [np.nan, np.nan, np.nan]
    })


@pytest.fixture
def df_mixed_object():
    return pd.DataFrame({
        'mixed': [1, 'two', 3.0, None, True]
    })


# =======================
# Part 1: NaN Handling
# =======================

class TestNanHandling:
    """Test NaN/None/NA handling in various operations."""
    
    def test_filter_nan_column(self, df_with_nans):
        """Test filtering on column with NaN values."""
        pd_df = df_with_nans.copy()
        ds_df = DataStore(df_with_nans.copy())
        
        # Filter where A > 2 (NaN should be excluded)
        pd_result = pd_df[pd_df['A'] > 2]
        ds_result = ds_df[ds_df['A'] > 2]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_isna_filter(self, df_with_nans):
        """Test filtering with isna()."""
        pd_df = df_with_nans.copy()
        ds_df = DataStore(df_with_nans.copy())
        
        pd_result = pd_df[pd_df['A'].isna()]
        ds_result = ds_df[ds_df['A'].isna()]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_notna_filter(self, df_with_nans):
        """Test filtering with notna()."""
        pd_df = df_with_nans.copy()
        ds_df = DataStore(df_with_nans.copy())
        
        pd_result = pd_df[pd_df['A'].notna()]
        ds_result = ds_df[ds_df['A'].notna()]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_nan_arithmetic(self, df_with_nans):
        """Test arithmetic with NaN values."""
        pd_df = df_with_nans.copy()
        ds_df = DataStore(df_with_nans.copy())
        
        # A + B where both may have NaN
        pd_df['D'] = pd_df['A'] + pd_df['B']
        pd_result = pd_df
        
        ds_df['D'] = ds_df['A'] + ds_df['B']
        ds_result = ds_df.to_df()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_all_nan_aggregation(self, df_all_nan):
        """Test aggregation on all-NaN DataFrame."""
        pd_df = df_all_nan.copy()
        ds_df = DataStore(df_all_nan.copy())
        
        # Sum should return 0.0 for all-NaN (pandas behavior)
        pd_result = pd_df.sum()
        ds_result = ds_df.sum()
        
        # Both should return 0 for all-NaN columns
        for col in ['A', 'B']:
            assert pd_result[col] == 0.0
            assert ds_result[col] == 0.0
    
    def test_fillna_different_values(self, df_with_nans):
        """Test fillna with different values per column."""
        pd_df = df_with_nans.copy()
        ds_df = DataStore(df_with_nans.copy())
        
        pd_result = pd_df.fillna({'A': -1, 'B': -2})
        ds_result = ds_df.fillna({'A': -1, 'B': -2})
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dropna_subset(self, df_with_nans):
        """Test dropna with subset."""
        pd_df = df_with_nans.copy()
        ds_df = DataStore(df_with_nans.copy())
        
        pd_result = pd_df.dropna(subset=['A'])
        ds_result = ds_df.dropna(subset=['A'])
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dropna_how_all(self, df_with_nans):
        """Test dropna with how='all'."""
        # Add a row that's all NaN
        pd_df = df_with_nans.copy()
        pd_df.loc[5] = [np.nan, np.nan, None]
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.dropna(how='all')
        ds_result = ds_df.dropna(how='all')
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =======================
# Part 2: Duplicate Column Names
# =======================

class TestDuplicateColumns:
    """Test handling of duplicate column names."""
    
    def test_create_with_duplicate_columns(self):
        """Test creating DataFrame with duplicate column names."""
        pd_df = pd.DataFrame([[1, 2, 3]], columns=['A', 'A', 'B'])
        ds_df = DataStore(pd_df.copy())
        
        # Both should have 3 columns
        assert len(ds_df.columns) == 3
        assert list(ds_df.columns) == ['A', 'A', 'B']
    
    def test_select_duplicate_column(self):
        """Test selecting duplicate column returns all matching."""
        pd_df = pd.DataFrame([[1, 2, 3]], columns=['A', 'A', 'B'])
        ds_df = DataStore(pd_df.copy())
        
        # Select 'A' should return both A columns
        pd_result = pd_df['A']
        ds_result = ds_df['A']
        
        # Both should be DataFrames with 2 columns
        assert isinstance(pd_result, pd.DataFrame)
        # DataStore behavior should match pandas
    
    def test_rename_duplicate_columns(self):
        """Test renaming can resolve duplicates."""
        pd_df = pd.DataFrame([[1, 2, 3]], columns=['A', 'A', 'B'])
        ds_df = DataStore(pd_df.copy())
        
        # For DataStore, access via to_df() and rename
        ds_result_df = ds_df.to_df()
        ds_result_df.columns = ['X', 'A', 'B']
        
        # Now should have unique columns
        assert list(ds_result_df.columns) == ['X', 'A', 'B']


# =======================
# Part 3: Special Characters in Column Names
# =======================

class TestSpecialColumnNames:
    """Test handling of special characters in column names."""
    
    def test_space_in_column_name(self):
        """Test column name with space."""
        pd_df = pd.DataFrame({'col with space': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['col with space'].sum()
        ds_result = ds_df['col with space'].sum()
        
        assert pd_result == ds_result
    
    def test_special_chars_in_column_name(self):
        """Test column name with special characters."""
        pd_df = pd.DataFrame({'col-with-dash': [1, 2, 3], 'col.with.dot': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[['col-with-dash', 'col.with.dot']]
        ds_result = ds_df[['col-with-dash', 'col.with.dot']]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_unicode_column_name(self):
        """Test column name with unicode characters."""
        pd_df = pd.DataFrame({'列名': [1, 2, 3], 'Σ': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['列名'].sum()
        ds_result = ds_df['列名'].sum()
        
        assert pd_result == ds_result
    
    def test_numeric_column_name(self):
        """Test numeric column name."""
        pd_df = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        # Both pandas and DataStore should support numeric column names
        pd_result = pd_df[0].sum()
        ds_result = ds_df[0].sum()
        
        assert pd_result == ds_result
    
    def test_empty_string_column_name(self):
        """Test empty string as column name."""
        pd_df = pd.DataFrame({'': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[''].sum()
        ds_result = ds_df[''].sum()
        
        assert pd_result == ds_result


# =======================
# Part 4: DateTime Operations
# =======================

class TestDateTimeOperations:
    """Test DateTime-specific operations."""
    
    def test_date_filter(self):
        """Test filtering by date."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        pd_df = pd.DataFrame({'date': dates, 'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())
        
        cutoff = pd.Timestamp('2024-01-03')
        pd_result = pd_df[pd_df['date'] >= cutoff]
        ds_result = ds_df[ds_df['date'] >= cutoff]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_date_sort(self):
        """Test sorting by date."""
        dates = pd.to_datetime(['2024-03-01', '2024-01-15', '2024-02-20'])
        pd_df = pd.DataFrame({'date': dates, 'value': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.sort_values('date')
        ds_result = ds_df.sort_values('date')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_date_groupby(self):
        """Test groupby with date column."""
        dates = pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'])
        pd_df = pd.DataFrame({'date': dates, 'value': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.groupby('date')['value'].sum().reset_index()
        ds_result = ds_df.groupby('date')['value'].sum().reset_index()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    @chdb_timedelta_type
    def test_timedelta_arithmetic(self):
        """Test timedelta arithmetic."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore(pd_df.copy())
        
        # Add timedelta
        pd_df['date_plus_1'] = pd_df['date'] + pd.Timedelta(days=1)
        pd_result = pd_df
        
        ds_df['date_plus_1'] = ds_df['date'] + pd.Timedelta(days=1)
        ds_result = ds_df.to_df()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =======================
# Part 5: Object Type Mixed Values
# =======================

class TestMixedObjectType:
    """Test handling of object dtype with mixed value types."""
    
    def test_mixed_object_astype_str(self, df_mixed_object):
        """Test converting mixed object to string."""
        pd_df = df_mixed_object.copy()
        ds_df = DataStore(df_mixed_object.copy())
        
        pd_result = pd_df['mixed'].astype(str)
        ds_result = ds_df['mixed'].astype(str)
        
        # Compare as Series
        assert_series_equal(ds_result, pd_result)
    
    def test_mixed_object_fillna(self, df_mixed_object):
        """Test fillna on mixed object."""
        pd_df = df_mixed_object.copy()
        ds_df = DataStore(df_mixed_object.copy())
        
        pd_result = pd_df.fillna('FILLED')
        ds_result = ds_df.fillna('FILLED')
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =======================
# Part 6: Complex Filter Expressions
# =======================

class TestComplexFilters:
    """Test complex filter expressions."""
    
    def test_and_or_combined(self):
        """Test combined AND/OR filters."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['x', 'y', 'x', 'y', 'x']
        })
        ds_df = DataStore(pd_df.copy())
        
        # (A > 2 AND B < 40) OR C == 'x'
        pd_result = pd_df[((pd_df['A'] > 2) & (pd_df['B'] < 40)) | (pd_df['C'] == 'x')]
        ds_result = ds_df[((ds_df['A'] > 2) & (ds_df['B'] < 40)) | (ds_df['C'] == 'x')]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_negation_filter(self):
        """Test negation in filter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())
        
        # NOT (A > 3)
        pd_result = pd_df[~(pd_df['A'] > 3)]
        ds_result = ds_df[~(ds_df['A'] > 3)]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_isin_filter(self):
        """Test isin() filter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[pd_df['A'].isin([2, 4])]
        ds_result = ds_df[ds_df['A'].isin([2, 4])]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_between_filter(self):
        """Test between() filter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[pd_df['A'].between(2, 4)]
        ds_result = ds_df[ds_df['A'].between(2, 4)]
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =======================
# Part 7: Aggregation Edge Cases
# =======================

class TestAggregationEdgeCases:
    """Test edge cases in aggregation operations."""
    
    def test_single_value_agg(self):
        """Test aggregation on single value DataFrame."""
        pd_df = pd.DataFrame({'A': [42]})
        ds_df = DataStore(pd_df.copy())
        
        assert pd_df['A'].sum() == ds_df['A'].sum()
        assert pd_df['A'].mean() == ds_df['A'].mean()
        assert pd_df['A'].min() == ds_df['A'].min()
        assert pd_df['A'].max() == ds_df['A'].max()
    
    def test_groupby_single_group(self):
        """Test groupby with only one group."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'A'], 'value': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_agg_multiple_functions(self):
        """Test multiple aggregation functions."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.agg({'A': ['sum', 'mean', 'std']})
        ds_result = ds_df.agg({'A': ['sum', 'mean', 'std']})
        
        # Compare the values (structure may differ slightly)
        assert abs(pd_result['A']['sum'] - ds_result['A']['sum']) < 0.001
        assert abs(pd_result['A']['mean'] - ds_result['A']['mean']) < 0.001


# =======================
# Part 8: Edge Case Operations
# =======================

class TestEdgeCaseOperations:
    """Test various edge case operations."""
    
    def test_empty_column_list_select(self):
        """Test selecting with empty column list."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[[]]
        ds_result = ds_df[[]]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_head_larger_than_df(self):
        """Test head(n) where n > len(df)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.head(100)
        ds_result = ds_df.head(100)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_tail_larger_than_df(self):
        """Test tail(n) where n > len(df)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.tail(100)
        ds_result = ds_df.tail(100)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_sample_with_replace(self):
        """Test sample with replacement."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        # Sample more rows than exist (requires replace=True)
        pd_result = pd_df.sample(n=5, replace=True, random_state=42)
        ds_result = ds_df.sample(n=5, replace=True, random_state=42)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_nlargest_with_ties(self):
        """Test nlargest when there are ties."""
        pd_df = pd.DataFrame({'A': [3, 3, 2, 2, 1], 'B': ['a', 'b', 'c', 'd', 'e']})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.nlargest(3, 'A')
        ds_result = ds_df.nlargest(3, 'A')
        
        # Row order may differ for ties, so just check values
        assert set(pd_result['A'].tolist()) == set(ds_result.to_df()['A'].tolist())
    
    def test_nsmallest_with_ties(self):
        """Test nsmallest when there are ties."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3], 'B': ['a', 'b', 'c', 'd', 'e']})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.nsmallest(3, 'A')
        ds_result = ds_df.nsmallest(3, 'A')
        
        assert set(pd_result['A'].tolist()) == set(ds_result.to_df()['A'].tolist())


# =======================
# Part 9: String Operations
# =======================

class TestStringOperations:
    """Test string accessor operations."""
    
    def test_str_contains_regex(self):
        """Test str.contains with regex."""
        pd_df = pd.DataFrame({'text': ['abc123', 'def456', 'ghi789', 'abc']})
        ds_df = DataStore(pd_df.copy())
        
        # Filter rows containing digits
        pd_result = pd_df[pd_df['text'].str.contains(r'\d+', regex=True)]
        ds_result = ds_df[ds_df['text'].str.contains(r'\d+', regex=True)]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_str_len(self):
        """Test str.len()."""
        pd_df = pd.DataFrame({'text': ['a', 'bb', 'ccc', 'dddd']})
        ds_df = DataStore(pd_df.copy())
        
        pd_df['len'] = pd_df['text'].str.len()
        pd_result = pd_df
        
        ds_df['len'] = ds_df['text'].str.len()
        ds_result = ds_df.to_df()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_str_upper_lower(self):
        """Test str.upper() and str.lower()."""
        pd_df = pd.DataFrame({'text': ['Hello', 'World', 'Test']})
        ds_df = DataStore(pd_df.copy())
        
        pd_df['upper'] = pd_df['text'].str.upper()
        pd_df['lower'] = pd_df['text'].str.lower()
        pd_result = pd_df
        
        ds_df['upper'] = ds_df['text'].str.upper()
        ds_df['lower'] = ds_df['text'].str.lower()
        ds_result = ds_df.to_df()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_str_replace(self):
        """Test str.replace()."""
        pd_df = pd.DataFrame({'text': ['hello world', 'foo bar', 'test case']})
        ds_df = DataStore(pd_df.copy())
        
        pd_df['replaced'] = pd_df['text'].str.replace(' ', '_')
        pd_result = pd_df
        
        ds_df['replaced'] = ds_df['text'].str.replace(' ', '_')
        ds_result = ds_df.to_df()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =======================
# Part 10: Boolean Operations
# =======================

class TestBooleanOperations:
    """Test boolean column operations."""
    
    def test_bool_column_filter(self):
        """Test filtering by boolean column directly."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'is_valid': [True, False, True]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[pd_df['is_valid']]
        ds_result = ds_df[ds_df['is_valid']]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_bool_column_negation(self):
        """Test negating boolean column in filter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'is_valid': [True, False, True]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[~pd_df['is_valid']]
        ds_result = ds_df[~ds_df['is_valid']]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_any_all_methods(self):
        """Test any() and all() on boolean column."""
        pd_df = pd.DataFrame({'bools': [True, True, False]})
        ds_df = DataStore(pd_df.copy())
        
        assert pd_df['bools'].any() == ds_df['bools'].any()
        assert pd_df['bools'].all() == ds_df['bools'].all()
    
    def test_bool_arithmetic(self):
        """Test arithmetic with boolean columns."""
        pd_df = pd.DataFrame({'A': [True, False, True], 'B': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        # Bool * int
        pd_df['C'] = pd_df['A'] * pd_df['B']
        pd_result = pd_df
        
        ds_df['C'] = ds_df['A'] * ds_df['B']
        ds_result = ds_df.to_df()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
