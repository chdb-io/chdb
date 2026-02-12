"""
Exploratory Batch 53: Module-Level Functions, CaseWhen Advanced, Deep Chains

This batch tests:
1. Module-level functions (crosstab, wide_to_long, factorize, json_normalize)
2. CaseWhen advanced scenarios (nested, NULL handling, expressions as values)
3. Deep operation chains (10+ operations, various combinations)
4. Unicode and special character handling
5. ColumnExpr advanced datetime methods (isocalendar)

Discovery approach: Architecture-based testing targeting gaps in test coverage.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from datastore import DataStore, concat
import datastore as ds_module
from tests.test_utils import assert_datastore_equals_pandas
from tests.xfail_markers import (
    chdb_no_day_month_name,
    chdb_strftime_format_difference,
    chdb_pad_no_side_param,
    chdb_center_implementation,
    chdb_startswith_no_tuple,
    limit_datastore_index_setter,
    chdb_python_table_rownumber_nondeterministic,
)


# ============================================================================
# Module-level Functions
# ============================================================================


class TestCrosstab:
    """Test crosstab function."""

    def test_crosstab_basic(self):
        """Test basic crosstab functionality."""
        # pandas
        pd_a = pd.Series([1, 1, 2, 2, 3])
        pd_b = pd.Series(['a', 'b', 'a', 'b', 'a'])
        pd_result = pd.crosstab(pd_a, pd_b)

        # DataStore
        ds_a = DataStore({'a': [1, 1, 2, 2, 3]})['a']
        ds_b = DataStore({'b': ['a', 'b', 'a', 'b', 'a']})['b']
        ds_result = ds_module.crosstab(ds_a, ds_b)

        # Both should produce a frequency table with same shape and values
        assert ds_result.shape == pd_result.shape
        # Compare values (column order may differ)
        for col in pd_result.columns:
            if col in ds_result.columns:
                # Get values from DataStore column (may be ColumnExpr)
                ds_col = ds_result[col]
                ds_values = list(ds_col) if hasattr(ds_col, '__iter__') else [ds_col]
                pd_values = pd_result[col].sort_index().reset_index(drop=True).tolist()
                assert ds_values == pd_values or sorted(ds_values) == sorted(pd_values)

    def test_crosstab_with_values(self):
        """Test crosstab with values and aggfunc."""
        pd_a = pd.Series(['A', 'A', 'B', 'B'])
        pd_b = pd.Series(['x', 'y', 'x', 'y'])
        pd_v = pd.Series([10, 20, 30, 40])
        pd_result = pd.crosstab(pd_a, pd_b, values=pd_v, aggfunc='sum')

        ds_df = DataStore({'a': ['A', 'A', 'B', 'B'], 'b': ['x', 'y', 'x', 'y'], 'v': [10, 20, 30, 40]})
        ds_result = ds_module.crosstab(ds_df['a'], ds_df['b'], values=ds_df['v'], aggfunc='sum')

        # Compare shape and values
        assert ds_result.shape == pd_result.shape

    def test_crosstab_margins(self):
        """Test crosstab with margins (totals)."""
        pd_a = pd.Series(['A', 'A', 'B', 'B'])
        pd_b = pd.Series(['x', 'y', 'x', 'y'])
        pd_result = pd.crosstab(pd_a, pd_b, margins=True)

        ds_df = DataStore({'a': ['A', 'A', 'B', 'B'], 'b': ['x', 'y', 'x', 'y']})
        ds_result = ds_module.crosstab(ds_df['a'], ds_df['b'], margins=True)

        assert ds_result.shape == pd_result.shape


class TestWideToLong:
    """Test wide_to_long function."""

    def test_wide_to_long_basic(self):
        """Test basic wide_to_long transformation."""
        # Create wide format data
        pd_df = pd.DataFrame(
            {'id': [1, 2], 'A1999': [10, 20], 'A2000': [11, 21], 'B1999': [100, 200], 'B2000': [110, 210]}
        )
        pd_result = pd.wide_to_long(pd_df, stubnames=['A', 'B'], i='id', j='year')

        ds_df = DataStore(
            {'id': [1, 2], 'A1999': [10, 20], 'A2000': [11, 21], 'B1999': [100, 200], 'B2000': [110, 210]}
        )
        ds_result = ds_module.wide_to_long(ds_df, stubnames=['A', 'B'], i='id', j='year')

        # Compare values (index structure may differ)
        # Compare DataFrames after converting DataStore to pandas
        ds_df_result = ds_result.reset_index() if hasattr(ds_result, 'reset_index') else ds_result
        if hasattr(ds_df_result, '_execute'):
            ds_df_result = ds_df_result._execute()
        pd_df_result = pd_result.reset_index()

        # Compare sorted by id and year
        ds_sorted = ds_df_result.sort_values(by=['id', 'year']).reset_index(drop=True)
        pd_sorted = pd_df_result.sort_values(by=['id', 'year']).reset_index(drop=True)

        pd.testing.assert_frame_equal(ds_sorted, pd_sorted, check_names=False)


class TestFactorize:
    """Test factorize function."""

    def test_factorize_basic(self):
        """Test basic factorize functionality."""
        pd_data = pd.Series(['a', 'b', 'a', 'c', 'b'])
        pd_codes, pd_uniques = pd.factorize(pd_data)

        ds_data = DataStore({'col': ['a', 'b', 'a', 'c', 'b']})['col']
        ds_codes, ds_uniques = ds_module.factorize(ds_data)

        # Codes should map same values to same integers
        assert len(ds_codes) == len(pd_codes)
        assert len(ds_uniques) == len(pd_uniques)

    def test_factorize_with_na(self):
        """Test factorize with NA values."""
        pd_data = pd.Series(['a', None, 'a', 'b', None])
        pd_codes, pd_uniques = pd.factorize(pd_data)

        ds_data = DataStore({'col': ['a', None, 'a', 'b', None]})['col']
        ds_codes, ds_uniques = ds_module.factorize(ds_data)

        # NA values should get -1 code
        assert -1 in ds_codes

    def test_factorize_sorted(self):
        """Test factorize with sort=True."""
        pd_data = pd.Series(['c', 'a', 'b', 'a'])
        pd_codes, pd_uniques = pd.factorize(pd_data, sort=True)

        ds_data = DataStore({'col': ['c', 'a', 'b', 'a']})['col']
        ds_codes, ds_uniques = ds_module.factorize(ds_data, sort=True)

        # Uniques should be sorted
        assert list(ds_uniques) == list(pd_uniques)


class TestJsonNormalize:
    """Test json_normalize function."""

    def test_json_normalize_basic(self):
        """Test basic json_normalize functionality."""
        data = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
        pd_result = pd.json_normalize(data)
        ds_result = ds_module.json_normalize(data)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_json_normalize_nested(self):
        """Test json_normalize with nested data."""
        data = [{'id': 1, 'info': {'name': 'Alice', 'age': 30}}, {'id': 2, 'info': {'name': 'Bob', 'age': 25}}]
        pd_result = pd.json_normalize(data)
        ds_result = ds_module.json_normalize(data)

        assert set(ds_result.columns) == set(pd_result.columns)

    def test_json_normalize_with_record_path(self):
        """Test json_normalize with record_path."""
        data = [{'id': 1, 'items': [{'name': 'a'}, {'name': 'b'}]}, {'id': 2, 'items': [{'name': 'c'}]}]
        pd_result = pd.json_normalize(data, record_path='items', meta='id')
        ds_result = ds_module.json_normalize(data, record_path='items', meta='id')

        assert len(ds_result) == len(pd_result)


class TestUnique:
    """Test unique function."""

    def test_unique_basic(self):
        """Test basic unique functionality."""
        pd_data = pd.Series([1, 2, 2, 3, 1])
        pd_result = pd.unique(pd_data)

        ds_data = DataStore({'col': [1, 2, 2, 3, 1]})['col']
        ds_result = ds_module.unique(ds_data)

        assert set(ds_result) == set(pd_result)

    def test_unique_with_na(self):
        """Test unique with NA values."""
        pd_data = pd.Series([1, None, 2, None, 1])
        pd_result = pd.unique(pd_data)

        ds_data = DataStore({'col': [1, None, 2, None, 1]})['col']
        ds_result = ds_module.unique(ds_data)

        # NA should be included once
        assert len([x for x in ds_result if pd.isna(x)]) <= 1


class TestValueCounts:
    """Test module-level value_counts function."""

    def test_value_counts_basic(self):
        """Test basic value_counts functionality."""
        pd_data = pd.Series(['a', 'b', 'a', 'c', 'a', 'b'])
        # pandas 3.0 removed pd.value_counts(), use Series.value_counts() instead
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        if pandas_version >= (3, 0):
            pd_result = pd_data.value_counts()
        else:
            pd_result = pd.value_counts(pd_data)

        ds_data = DataStore({'col': ['a', 'b', 'a', 'c', 'a', 'b']})['col']
        ds_result = ds_module.value_counts(ds_data)

        # Sort both by index for comparison (ignore index name difference)
        pd_sorted = pd_result.sort_index()
        ds_sorted = ds_result.sort_index()
        pd.testing.assert_series_equal(ds_sorted, pd_sorted, check_names=False)

    def test_value_counts_normalize(self):
        """Test value_counts with normalize=True."""
        pd_data = pd.Series(['a', 'b', 'a', 'a'])
        # pandas 3.0 removed pd.value_counts(), use Series.value_counts() instead
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        if pandas_version >= (3, 0):
            pd_result = pd_data.value_counts(normalize=True)
        else:
            pd_result = pd.value_counts(pd_data, normalize=True)

        ds_data = DataStore({'col': ['a', 'b', 'a', 'a']})['col']
        ds_result = ds_module.value_counts(ds_data, normalize=True)

        pd_sorted = pd_result.sort_index()
        ds_sorted = ds_result.sort_index()
        pd.testing.assert_series_equal(ds_sorted, pd_sorted, rtol=1e-5, check_names=False)


# ============================================================================
# CaseWhen Advanced
# ============================================================================


class TestCaseWhenAdvanced:
    """Test advanced CaseWhen scenarios."""

    def test_case_when_basic(self):
        """Test basic CaseWhen functionality."""
        pd_df = pd.DataFrame({'score': [95, 85, 75, 65, 55]})
        pd_df['grade'] = np.select(
            [pd_df['score'] >= 90, pd_df['score'] >= 80, pd_df['score'] >= 70, pd_df['score'] >= 60],
            ['A', 'B', 'C', 'D'],
            default='F',
        )

        ds_df = DataStore({'score': [95, 85, 75, 65, 55]})
        ds_df['grade'] = (
            ds_df.when(ds_df['score'] >= 90, 'A')
            .when(ds_df['score'] >= 80, 'B')
            .when(ds_df['score'] >= 70, 'C')
            .when(ds_df['score'] >= 60, 'D')
            .otherwise('F')
        )

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_case_when_with_null(self):
        """Test CaseWhen with NULL values in condition column."""
        pd_df = pd.DataFrame({'value': [10, None, 30, None, 50]})
        pd_df['category'] = np.select([pd_df['value'].isna(), pd_df['value'] >= 30], ['missing', 'high'], default='low')

        ds_df = DataStore({'value': [10, None, 30, None, 50]})
        ds_df['category'] = (
            ds_df.when(ds_df['value'].isna(), 'missing').when(ds_df['value'] >= 30, 'high').otherwise('low')
        )

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_case_when_with_expression_values(self):
        """Test CaseWhen with column expressions as result values."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [1, 2, 3]})
        pd_df['result'] = np.where(pd_df['a'] > 15, pd_df['a'] + pd_df['b'], pd_df['a'] - pd_df['b'])

        ds_df = DataStore({'a': [10, 20, 30], 'b': [1, 2, 3]})
        ds_df['result'] = ds_df.when(ds_df['a'] > 15, ds_df['a'] + ds_df['b']).otherwise(ds_df['a'] - ds_df['b'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_case_when_boolean_result(self):
        """Test CaseWhen with boolean result values."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        pd_df['is_even'] = np.select([pd_df['value'] % 2 == 0], [True], default=False)

        ds_df = DataStore({'value': [1, 2, 3, 4, 5]})
        ds_df['is_even'] = ds_df.when(ds_df['value'] % 2 == 0, True).otherwise(False)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_case_when_numeric_result(self):
        """Test CaseWhen with numeric result values."""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'C', 'A', 'B']})
        pd_df['weight'] = np.select([pd_df['category'] == 'A', pd_df['category'] == 'B'], [1.0, 2.0], default=3.0)

        ds_df = DataStore({'category': ['A', 'B', 'C', 'A', 'B']})
        ds_df['weight'] = ds_df.when(ds_df['category'] == 'A', 1.0).when(ds_df['category'] == 'B', 2.0).otherwise(3.0)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_case_when_multiple_conditions(self):
        """Test CaseWhen with compound conditions using AND/OR."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        pd_df['result'] = np.select(
            [(pd_df['a'] > 2) & (pd_df['b'] > 2), (pd_df['a'] <= 2) | (pd_df['b'] <= 2)],
            ['both_high', 'one_low'],
            default='other',
        )

        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df['result'] = (
            ds_df.when((ds_df['a'] > 2) & (ds_df['b'] > 2), 'both_high')
            .when((ds_df['a'] <= 2) | (ds_df['b'] <= 2), 'one_low')
            .otherwise('other')
        )

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_case_when_chain_with_filter(self):
        """Test CaseWhen result used in subsequent filter."""
        pd_df = pd.DataFrame({'score': [95, 85, 75, 65, 55]})
        pd_df['grade'] = np.select([pd_df['score'] >= 90, pd_df['score'] >= 80], ['A', 'B'], default='C')
        pd_result = pd_df[pd_df['grade'].isin(['A', 'B'])]

        ds_df = DataStore({'score': [95, 85, 75, 65, 55]})
        ds_df['grade'] = ds_df.when(ds_df['score'] >= 90, 'A').when(ds_df['score'] >= 80, 'B').otherwise('C')
        ds_result = ds_df[ds_df['grade'].isin(['A', 'B'])]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Deep Operation Chains
# ============================================================================


class TestDeepOperationChains:
    """Test deep operation chains (10+ operations)."""

    def test_deep_filter_chain(self):
        """Test 10+ filter operations in chain."""
        data = {'a': list(range(100)), 'b': list(range(100, 200))}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        # Apply 10 sequential filters
        pd_result = pd_df
        ds_result = ds_df

        for i in range(10):
            threshold = i * 5
            pd_result = pd_result[pd_result['a'] >= threshold]
            ds_result = ds_result[ds_result['a'] >= threshold]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_deep_assign_chain(self):
        """Test 10+ assign operations in chain."""
        data = {'a': [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        # Apply 10 sequential assigns
        for i in range(10):
            col_name = f'col_{i}'
            pd_df = pd_df.assign(**{col_name: pd_df['a'] + i})
            ds_df = ds_df.assign(**{col_name: ds_df['a'] + i})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_mixed_deep_chain(self):
        """Test mixed operations: filter -> assign -> sort -> head -> filter -> assign -> ..."""
        data = {'a': list(range(100)), 'b': ['x', 'y'] * 50}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        # Complex chain
        pd_result = (
            pd_df[pd_df['a'] > 10]
            .assign(c=lambda x: x['a'] * 2)
            .sort_values('c', ascending=False)
            .head(50)
            .assign(d=lambda x: x['c'] + 10)[lambda x: x['d'] > 50]
            .assign(e=lambda x: x['a'] + x['d'])
            .sort_values('e')
            .head(20)
            .assign(f=lambda x: x['e'] - x['a'])[lambda x: x['f'] > 40]
        )

        ds_result = ds_df[ds_df['a'] > 10].assign(c=ds_df['a'] * 2).sort_values('c', ascending=False).head(50)
        ds_result = ds_result.assign(d=ds_result['c'] + 10)
        ds_result = ds_result[ds_result['d'] > 50]
        ds_result = ds_result.assign(e=ds_result['a'] + ds_result['d'])
        ds_result = ds_result.sort_values('e').head(20)
        ds_result = ds_result.assign(f=ds_result['e'] - ds_result['a'])
        ds_result = ds_result[ds_result['f'] > 40]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_chain(self):
        """Test groupby with multiple aggregations chained."""
        data = {'category': ['A', 'B', 'A', 'B', 'A', 'B'] * 10, 'value': list(range(60))}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.groupby('category').agg({'value': ['sum', 'mean', 'min', 'max']})
        pd_result.columns = ['_'.join(col).strip() for col in pd_result.columns.values]
        pd_result = pd_result.reset_index()

        ds_result = ds_df.groupby('category').agg({'value': ['sum', 'mean', 'min', 'max']})
        ds_result.columns = ['_'.join(col).strip() for col in ds_result.columns.values]
        ds_result = ds_result.reset_index()

        assert_datastore_equals_pandas(
            ds_result.sort_values('category').reset_index(drop=True),
            pd_result.sort_values('category').reset_index(drop=True),
        )


# ============================================================================
# Unicode and Special Characters
# ============================================================================


class TestUnicodeHandling:
    """Test Unicode and special character handling."""

    def test_unicode_column_values(self):
        """Test Unicode values in columns."""
        data = {'name': ['Alice', 'Bob', 'Charlie'], 'city': ['Beijing', 'Munich', 'Tokyo']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_unicode_filter(self):
        """Test filter with Unicode values."""
        data = {'city': ['Beijing', 'Munich', 'Tokyo']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[pd_df['city'] == 'Munich']
        ds_result = ds_df[ds_df['city'] == 'Munich']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_emoji_values(self):
        """Test emoji values in columns."""
        data = {'emoji': ['smile', 'heart', 'star'], 'count': [10, 20, 30]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_special_characters_in_strings(self):
        """Test special characters like quotes, backslashes."""
        data = {'text': ["it's", 'say "hello"', 'path\\to\\file', 'new\nline']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_empty_strings(self):
        """Test empty strings handling."""
        data = {'text': ['', 'a', '', 'b', '']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[pd_df['text'] == '']
        ds_result = ds_df[ds_df['text'] == '']

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# DateTime Advanced
# ============================================================================


class TestDateTimeAdvanced:
    """Test advanced datetime functionality."""

    def test_isocalendar_year(self):
        """Test dt.isocalendar().year accessor."""
        dates = pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31'])
        pd_df = pd.DataFrame({'date': dates})
        pd_df['iso_year'] = pd_df['date'].dt.isocalendar().year

        ds_df = DataStore({'date': dates})
        ds_df['iso_year'] = ds_df['date'].dt.isocalendar().year

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_isocalendar_week(self):
        """Test dt.isocalendar().week accessor."""
        dates = pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31'])
        pd_df = pd.DataFrame({'date': dates})
        pd_df['iso_week'] = pd_df['date'].dt.isocalendar().week

        ds_df = DataStore({'date': dates})
        ds_df['iso_week'] = ds_df['date'].dt.isocalendar().week

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_isocalendar_day(self):
        """Test dt.isocalendar().day accessor."""
        dates = pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31'])
        pd_df = pd.DataFrame({'date': dates})
        pd_df['iso_day'] = pd_df['date'].dt.isocalendar().day

        ds_df = DataStore({'date': dates})
        ds_df['iso_day'] = ds_df['date'].dt.isocalendar().day

        assert_datastore_equals_pandas(ds_df, pd_df)

    @chdb_no_day_month_name
    def test_dt_day_name(self):
        """Test dt.day_name() accessor."""
        dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        pd_df = pd.DataFrame({'date': dates})
        pd_df['day_name'] = pd_df['date'].dt.day_name()

        ds_df = DataStore({'date': dates})
        ds_df['day_name'] = ds_df['date'].dt.day_name()

        assert_datastore_equals_pandas(ds_df, pd_df)

    @chdb_no_day_month_name
    def test_dt_month_name(self):
        """Test dt.month_name() accessor."""
        dates = pd.to_datetime(['2023-01-15', '2023-06-15', '2023-12-15'])
        pd_df = pd.DataFrame({'date': dates})
        pd_df['month_name'] = pd_df['date'].dt.month_name()

        ds_df = DataStore({'date': dates})
        ds_df['month_name'] = ds_df['date'].dt.month_name()

        assert_datastore_equals_pandas(ds_df, pd_df)

    @chdb_strftime_format_difference
    def test_dt_strftime_complex(self):
        """Test dt.strftime with complex format."""
        dates = pd.to_datetime(['2023-01-15 10:30:45', '2023-06-20 14:15:30'])
        pd_df = pd.DataFrame({'date': dates})
        pd_df['formatted'] = pd_df['date'].dt.strftime('%Y-%m-%d %H:%M')

        ds_df = DataStore({'date': dates})
        ds_df['formatted'] = ds_df['date'].dt.strftime('%Y-%m-%d %H:%M')

        assert_datastore_equals_pandas(ds_df, pd_df)


# ============================================================================
# String Accessor Advanced
# ============================================================================


class TestStringAccessorAdvanced:
    """Test advanced string accessor functionality."""

    @chdb_pad_no_side_param
    def test_str_pad(self):
        """Test str.pad() method."""
        pd_df = pd.DataFrame({'text': ['a', 'bb', 'ccc']})
        pd_df['padded'] = pd_df['text'].str.pad(5, side='left', fillchar='0')

        ds_df = DataStore({'text': ['a', 'bb', 'ccc']})
        ds_df['padded'] = ds_df['text'].str.pad(5, side='left', fillchar='0')

        assert_datastore_equals_pandas(ds_df, pd_df)

    @chdb_center_implementation
    def test_str_center(self):
        """Test str.center() method."""
        pd_df = pd.DataFrame({'text': ['a', 'bb', 'ccc']})
        pd_df['centered'] = pd_df['text'].str.center(7, fillchar='-')

        ds_df = DataStore({'text': ['a', 'bb', 'ccc']})
        ds_df['centered'] = ds_df['text'].str.center(7, fillchar='-')

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_zfill(self):
        """Test str.zfill() method."""
        pd_df = pd.DataFrame({'num': ['1', '22', '333']})
        pd_df['padded'] = pd_df['num'].str.zfill(5)

        ds_df = DataStore({'num': ['1', '22', '333']})
        ds_df['padded'] = ds_df['num'].str.zfill(5)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_count_pattern(self):
        """Test str.count() with pattern."""
        pd_df = pd.DataFrame({'text': ['aaa', 'aab', 'abc', 'bbb']})
        pd_df['a_count'] = pd_df['text'].str.count('a')

        ds_df = DataStore({'text': ['aaa', 'aab', 'abc', 'bbb']})
        ds_df['a_count'] = ds_df['text'].str.count('a')

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_find(self):
        """Test str.find() method."""
        pd_df = pd.DataFrame({'text': ['hello world', 'world hello', 'no match']})
        pd_df['pos'] = pd_df['text'].str.find('world')

        ds_df = DataStore({'text': ['hello world', 'world hello', 'no match']})
        ds_df['pos'] = ds_df['text'].str.find('world')

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_rfind(self):
        """Test str.rfind() method."""
        pd_df = pd.DataFrame({'text': ['hello hello', 'world', 'no match']})
        pd_df['pos'] = pd_df['text'].str.rfind('hello')

        ds_df = DataStore({'text': ['hello hello', 'world', 'no match']})
        ds_df['pos'] = ds_df['text'].str.rfind('hello')

        assert_datastore_equals_pandas(ds_df, pd_df)

    @chdb_startswith_no_tuple
    def test_str_startswith_tuple(self):
        """Test str.startswith() with tuple of prefixes."""
        pd_df = pd.DataFrame({'text': ['apple', 'banana', 'cherry', 'avocado']})
        pd_df['starts_with_ab'] = pd_df['text'].str.startswith(('a', 'b'))

        ds_df = DataStore({'text': ['apple', 'banana', 'cherry', 'avocado']})
        ds_df['starts_with_ab'] = ds_df['text'].str.startswith(('a', 'b'))

        assert_datastore_equals_pandas(ds_df, pd_df)

    @chdb_startswith_no_tuple
    def test_str_endswith_tuple(self):
        """Test str.endswith() with tuple of suffixes."""
        pd_df = pd.DataFrame({'text': ['apple', 'banana', 'cherry', 'mango']})
        pd_df['ends_with'] = pd_df['text'].str.endswith(('e', 'a', 'o'))

        ds_df = DataStore({'text': ['apple', 'banana', 'cherry', 'mango']})
        ds_df['ends_with'] = ds_df['text'].str.endswith(('e', 'a', 'o'))

        assert_datastore_equals_pandas(ds_df, pd_df)


# ============================================================================
# Numeric Edge Cases
# ============================================================================


class TestNumericEdgeCases:
    """Test numeric edge cases."""

    def test_integer_overflow_boundary(self):
        """Test values near integer overflow boundaries."""
        max_int = 2**62
        data = {'value': [max_int, max_int - 1, -max_int, -max_int + 1, 0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[pd_df['value'] > 0]
        ds_result = ds_df[ds_df['value'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_very_small_floats(self):
        """Test very small float values."""
        data = {'value': [1e-300, 1e-200, 1e-100, 1e-10, 0.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[pd_df['value'] > 1e-150]
        ds_result = ds_df[ds_df['value'] > 1e-150]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_inf_values(self):
        """Test infinity values."""
        data = {'value': [float('inf'), float('-inf'), 0.0, 1.0, -1.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[~np.isinf(pd_df['value'])]
        ds_result = ds_df[~ds_df['value'].isin([float('inf'), float('-inf')])]

        # Compare non-inf values
        assert len(ds_result) == len(pd_result)

    def test_mixed_int_float_arithmetic(self):
        """Test arithmetic between int and float columns."""
        data = {'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_df['sum'] = pd_df['int_col'] + pd_df['float_col']
        pd_df['diff'] = pd_df['float_col'] - pd_df['int_col']
        pd_df['prod'] = pd_df['int_col'] * pd_df['float_col']
        pd_df['div'] = pd_df['float_col'] / pd_df['int_col']

        ds_df['sum'] = ds_df['int_col'] + ds_df['float_col']
        ds_df['diff'] = ds_df['float_col'] - ds_df['int_col']
        ds_df['prod'] = ds_df['int_col'] * ds_df['float_col']
        ds_df['div'] = ds_df['float_col'] / ds_df['int_col']

        assert_datastore_equals_pandas(ds_df, pd_df)


# ============================================================================
# Complex Filter Conditions
# ============================================================================


class TestComplexFilterConditions:
    """Test complex filter condition combinations."""

    def test_triple_and_condition(self):
        """Test three conditions combined with AND."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1], 'c': [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] > 1) & (pd_df['c'] > 1)]
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] > 1) & (ds_df['c'] > 1)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_triple_or_condition(self):
        """Test three conditions combined with OR."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1], 'c': [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[(pd_df['a'] == 1) | (pd_df['b'] == 1) | (pd_df['c'] == 3)]
        ds_result = ds_df[(ds_df['a'] == 1) | (ds_df['b'] == 1) | (ds_df['c'] == 3)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_and_or_parentheses(self):
        """Test mixed AND/OR with parentheses for precedence."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        # (a > 2 AND b > 2) OR (a < 2 AND b < 4)
        pd_result = pd_df[((pd_df['a'] > 2) & (pd_df['b'] > 2)) | ((pd_df['a'] < 2) & (pd_df['b'] < 4))]
        ds_result = ds_df[((ds_df['a'] > 2) & (ds_df['b'] > 2)) | ((ds_df['a'] < 2) & (ds_df['b'] < 4))]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negated_compound_condition(self):
        """Test negation of compound conditions."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        # NOT (a > 2 AND b > 2)
        pd_result = pd_df[~((pd_df['a'] > 2) & (pd_df['b'] > 2))]
        ds_result = ds_df[~((ds_df['a'] > 2) & (ds_df['b'] > 2))]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_with_and_or(self):
        """Test between combined with AND/OR."""
        data = {'a': list(range(10)), 'b': list(range(10, 20))}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[(pd_df['a'].between(2, 5)) & (pd_df['b'] > 13)]
        ds_result = ds_df[(ds_df['a'].between(2, 5)) & (ds_df['b'] > 13)]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Aggregation with NULL handling
# ============================================================================


class TestAggregationNullHandling:
    """Test aggregation operations with NULL handling."""

    def test_count_vs_size_with_null(self):
        """Test count() vs size() behavior with NULL values."""
        data = {'category': ['A', 'A', 'B', 'B', 'B'], 'value': [1, None, 2, None, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        # count excludes NULL, size includes all
        pd_count = pd_df.groupby('category')['value'].count()
        pd_size = pd_df.groupby('category').size()

        ds_count = ds_df.groupby('category')['value'].count()
        ds_size = ds_df.groupby('category').size()

        # Execute ColumnExpr to get Series
        ds_count_series = pd.Series(
            list(ds_count), index=ds_count.index, name=ds_count.name if hasattr(ds_count, 'name') else None
        )
        ds_size_series = pd.Series(list(ds_size), index=ds_size.index)

        pd.testing.assert_series_equal(ds_count_series.sort_index(), pd_count.sort_index(), check_names=False)
        pd.testing.assert_series_equal(ds_size_series.sort_index(), pd_size.sort_index(), check_names=False)

    @chdb_python_table_rownumber_nondeterministic
    def test_first_last_with_null(self):
        """Test first() and last() with NULL values."""
        data = {'category': ['A', 'A', 'A', 'B', 'B'], 'value': [None, 2, 3, 4, None]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_first = pd_df.groupby('category')['value'].first()
        pd_last = pd_df.groupby('category')['value'].last()

        ds_first = ds_df.groupby('category')['value'].first()
        ds_last = ds_df.groupby('category')['value'].last()

        # Execute ColumnExpr to get Series
        ds_first_series = pd.Series(list(ds_first), index=ds_first.index)
        ds_last_series = pd.Series(list(ds_last), index=ds_last.index)

        pd.testing.assert_series_equal(ds_first_series.sort_index(), pd_first.sort_index(), check_names=False)
        pd.testing.assert_series_equal(ds_last_series.sort_index(), pd_last.sort_index(), check_names=False)

    def test_min_max_all_null(self):
        """Test min/max when all values are NULL."""
        data = {'category': ['A', 'A', 'B', 'B'], 'value': [None, None, 1, 2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_min = pd_df.groupby('category')['value'].min()
        pd_max = pd_df.groupby('category')['value'].max()

        ds_min = ds_df.groupby('category')['value'].min()
        ds_max = ds_df.groupby('category')['value'].max()

        # Execute ColumnExpr to get Series
        ds_min_series = pd.Series(list(ds_min), index=ds_min.index)
        ds_max_series = pd.Series(list(ds_max), index=ds_max.index)

        # Both should return NaN for category A
        pd.testing.assert_series_equal(ds_min_series.sort_index(), pd_min.sort_index(), check_names=False)
        pd.testing.assert_series_equal(ds_max_series.sort_index(), pd_max.sort_index(), check_names=False)


# ============================================================================
# Column Selection Edge Cases
# ============================================================================


class TestColumnSelectionEdgeCases:
    """Test column selection edge cases."""

    def test_select_single_column_multiple_times(self):
        """Test selecting same column multiple times."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[['a', 'a', 'b']]
        ds_result = ds_df[['a', 'a', 'b']]

        # Both should have 3 columns
        assert len(ds_result.columns) == len(pd_result.columns)

    def test_select_reorder_columns(self):
        """Test selecting columns in different order."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[['c', 'a', 'b']]
        ds_result = ds_df[['c', 'a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_after_drop(self):
        """Test column selection after drop."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
        pd_df = pd.DataFrame(data).drop(columns=['b'])
        ds_df = DataStore(data).drop(columns=['b'])

        pd_result = pd_df[['c', 'a']]
        ds_result = ds_df[['c', 'a']]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Index Operations
# ============================================================================


class TestIndexOperations:
    """Test index-related operations."""

    @limit_datastore_index_setter
    def test_reset_index_drop_true(self):
        """Test reset_index with drop=True."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        pd_df = pd.DataFrame(data, index=['x', 'y', 'z'])
        ds_df = DataStore(data)
        ds_df.index = pd.Index(['x', 'y', 'z'])

        pd_result = pd_df.reset_index(drop=True)
        ds_result = ds_df.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_verify_integrity(self):
        """Test set_index with verify_integrity (should fail on duplicates)."""
        data = {'a': [1, 1, 2], 'b': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        # Both should raise on duplicate index values
        with pytest.raises(Exception):
            pd_df.set_index('a', verify_integrity=True)

        with pytest.raises(Exception):
            ds_df.set_index('a', verify_integrity=True)

    @limit_datastore_index_setter
    def test_reindex_new_indices(self):
        """Test reindex with new indices (adding NaN rows)."""
        data = {'a': [1, 2, 3]}
        pd_df = pd.DataFrame(data, index=['x', 'y', 'z'])
        ds_df = DataStore(data)
        ds_df.index = pd.Index(['x', 'y', 'z'])

        pd_result = pd_df.reindex(['w', 'x', 'y', 'z', 'aa'])
        ds_result = ds_df.reindex(['w', 'x', 'y', 'z', 'aa'])

        # Both should have NaN for new indices
        assert len(ds_result) == len(pd_result)


# ============================================================================
# Empty DataFrame Operations
# ============================================================================


class TestEmptyDataFrameOperations:
    """Test operations on empty DataFrames."""

    def test_empty_concat(self):
        """Test concatenating with empty DataFrame."""
        data = {'a': [1, 2, 3]}
        pd_df = pd.DataFrame(data)
        pd_empty = pd.DataFrame({'a': []})
        pd_result = pd.concat([pd_df, pd_empty], ignore_index=True)

        ds_df = DataStore(data)
        ds_empty = DataStore({'a': []})
        ds_result = concat([ds_df, ds_empty], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_merge(self):
        """Test merging with empty DataFrame."""
        pd_left = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_right = pd.DataFrame({'a': [], 'c': []})
        pd_result = pd.merge(pd_left, pd_right, on='a', how='left')

        ds_left = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_right = DataStore({'a': [], 'c': []})
        ds_result = ds_left.merge(ds_right, on='a', how='left')

        assert len(ds_result) == len(pd_result)

    def test_empty_groupby(self):
        """Test groupby on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore({'a': [], 'b': []})

        pd_result = pd_df.groupby('a')['b'].sum()
        ds_result = ds_df.groupby('a')['b'].sum()

        assert len(ds_result) == len(pd_result) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
