"""
Exploratory Batch 33: Multi-operation chains and advanced combinations

Focus areas:
1. Complex multi-table operations (chained merge/concat)
2. GroupBy advanced edge cases
3. Rolling/Expanding combined with other operations
4. Schema inference edge cases
5. Error handling and boundary conditions

Tests use Mirror Code Pattern: compare DataStore results with pandas results.
"""

import pytest
from tests.xfail_markers import lazy_index_not_preserved
import pandas as pd
import numpy as np
from datastore import DataStore
import datastore as ds
from tests.test_utils import assert_datastore_equals_pandas


# =============================================================================
# Test Group 1: Chained Merge Operations
# =============================================================================

class TestChainedMergeOperations:
    """Test chained merge operations with multiple DataFrames."""

    def test_three_way_merge_inner(self):
        """Test merging three DataFrames with inner join."""
        # pandas
        pd_df1 = pd.DataFrame({'key': [1, 2, 3, 4], 'value1': ['a', 'b', 'c', 'd']})
        pd_df2 = pd.DataFrame({'key': [2, 3, 4, 5], 'value2': ['e', 'f', 'g', 'h']})
        pd_df3 = pd.DataFrame({'key': [3, 4, 5, 6], 'value3': ['i', 'j', 'k', 'l']})
        pd_result = pd_df1.merge(pd_df2, on='key').merge(pd_df3, on='key')

        # DataStore
        ds_df1 = DataStore({'key': [1, 2, 3, 4], 'value1': ['a', 'b', 'c', 'd']})
        ds_df2 = DataStore({'key': [2, 3, 4, 5], 'value2': ['e', 'f', 'g', 'h']})
        ds_df3 = DataStore({'key': [3, 4, 5, 6], 'value3': ['i', 'j', 'k', 'l']})
        ds_result = ds_df1.merge(ds_df2, on='key').merge(ds_df3, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_three_way_merge_left(self):
        """Test merging three DataFrames with left join."""
        # pandas
        pd_df1 = pd.DataFrame({'key': [1, 2, 3], 'value1': ['a', 'b', 'c']})
        pd_df2 = pd.DataFrame({'key': [2, 3, 4], 'value2': ['d', 'e', 'f']})
        pd_df3 = pd.DataFrame({'key': [3, 4, 5], 'value3': ['g', 'h', 'i']})
        pd_result = pd_df1.merge(pd_df2, on='key', how='left').merge(pd_df3, on='key', how='left')

        # DataStore
        ds_df1 = DataStore({'key': [1, 2, 3], 'value1': ['a', 'b', 'c']})
        ds_df2 = DataStore({'key': [2, 3, 4], 'value2': ['d', 'e', 'f']})
        ds_df3 = DataStore({'key': [3, 4, 5], 'value3': ['g', 'h', 'i']})
        ds_result = ds_df1.merge(ds_df2, on='key', how='left').merge(ds_df3, on='key', how='left')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_with_different_key_columns(self):
        """Test merge with different left_on/right_on columns."""
        # pandas
        pd_df1 = pd.DataFrame({'id1': [1, 2, 3], 'value1': ['a', 'b', 'c']})
        pd_df2 = pd.DataFrame({'id2': [2, 3, 4], 'value2': ['d', 'e', 'f']})
        pd_result = pd_df1.merge(pd_df2, left_on='id1', right_on='id2')

        # DataStore
        ds_df1 = DataStore({'id1': [1, 2, 3], 'value1': ['a', 'b', 'c']})
        ds_df2 = DataStore({'id2': [2, 3, 4], 'value2': ['d', 'e', 'f']})
        ds_result = ds_df1.merge(ds_df2, left_on='id1', right_on='id2')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_after_filter(self):
        """Test merge after applying filters."""
        # pandas
        pd_df1 = pd.DataFrame({'key': [1, 2, 3, 4, 5], 'value1': [10, 20, 30, 40, 50]})
        pd_df2 = pd.DataFrame({'key': [2, 3, 4, 5, 6], 'value2': [100, 200, 300, 400, 500]})
        pd_result = pd_df1[pd_df1['value1'] > 20].merge(pd_df2[pd_df2['value2'] < 400], on='key')

        # DataStore
        ds_df1 = DataStore({'key': [1, 2, 3, 4, 5], 'value1': [10, 20, 30, 40, 50]})
        ds_df2 = DataStore({'key': [2, 3, 4, 5, 6], 'value2': [100, 200, 300, 400, 500]})
        ds_result = ds_df1[ds_df1['value1'] > 20].merge(ds_df2[ds_df2['value2'] < 400], on='key')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_then_groupby_agg(self):
        """Test merge followed by groupby aggregation."""
        # pandas
        pd_df1 = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value1': [1, 2, 3, 4]})
        pd_df2 = pd.DataFrame({'category': ['A', 'B', 'C'], 'factor': [10, 20, 30]})
        pd_result = pd_df1.merge(pd_df2, on='category').groupby('category').agg({'value1': 'sum', 'factor': 'mean'}).reset_index()

        # DataStore
        ds_df1 = DataStore({'category': ['A', 'A', 'B', 'B'], 'value1': [1, 2, 3, 4]})
        ds_df2 = DataStore({'category': ['A', 'B', 'C'], 'factor': [10, 20, 30]})
        ds_result = ds_df1.merge(ds_df2, on='category').groupby('category').agg({'value1': 'sum', 'factor': 'mean'}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 2: Concat Operations
# =============================================================================

class TestConcatOperations:
    """Test various concat scenarios."""

    def test_concat_three_dataframes(self):
        """Test concatenating three DataFrames."""
        # pandas
        pd_df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        pd_df3 = pd.DataFrame({'A': [9, 10], 'B': [11, 12]})
        pd_result = pd.concat([pd_df1, pd_df2, pd_df3], ignore_index=True)

        # DataStore
        ds_df1 = DataStore({'A': [1, 2], 'B': [3, 4]})
        ds_df2 = DataStore({'A': [5, 6], 'B': [7, 8]})
        ds_df3 = DataStore({'A': [9, 10], 'B': [11, 12]})
        ds_result = ds.concat([ds_df1, ds_df2, ds_df3], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_with_different_columns(self):
        """Test concat with partially overlapping columns."""
        # pandas
        pd_df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df2 = pd.DataFrame({'B': [5, 6], 'C': [7, 8]})
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)

        # DataStore
        ds_df1 = DataStore({'A': [1, 2], 'B': [3, 4]})
        ds_df2 = DataStore({'B': [5, 6], 'C': [7, 8]})
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_then_filter(self):
        """Test filter after concat."""
        # pandas
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
        pd_concat = pd.concat([pd_df1, pd_df2], ignore_index=True)
        pd_result = pd_concat[pd_concat['A'] > 5]

        # DataStore
        ds_df1 = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df2 = DataStore({'A': [7, 8, 9], 'B': [10, 11, 12]})
        ds_concat = ds.concat([ds_df1, ds_df2], ignore_index=True)
        ds_result = ds_concat[ds_concat['A'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_then_groupby(self):
        """Test groupby after concat."""
        # pandas
        pd_df1 = pd.DataFrame({'category': ['A', 'B'], 'value': [10, 20]})
        pd_df2 = pd.DataFrame({'category': ['A', 'B'], 'value': [30, 40]})
        pd_concat = pd.concat([pd_df1, pd_df2], ignore_index=True)
        pd_result = pd_concat.groupby('category')['value'].sum().reset_index()

        # DataStore
        ds_df1 = DataStore({'category': ['A', 'B'], 'value': [10, 20]})
        ds_df2 = DataStore({'category': ['A', 'B'], 'value': [30, 40]})
        ds_concat = ds.concat([ds_df1, ds_df2], ignore_index=True)
        ds_result = ds_concat.groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 3: GroupBy Advanced Edge Cases
# =============================================================================

class TestGroupByAdvancedEdgeCases:
    """Test advanced groupby scenarios."""

    def test_groupby_multiple_keys_filter(self):
        """Test groupby with multiple keys followed by filter."""
        # pandas
        pd_df = pd.DataFrame({
            'cat1': ['A', 'A', 'B', 'B', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        pd_result = pd_df.groupby(['cat1', 'cat2'])['value'].sum().reset_index()
        pd_result = pd_result[pd_result['value'] > 3]

        # DataStore
        ds_df = DataStore({
            'cat1': ['A', 'A', 'B', 'B', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        ds_result = ds_df.groupby(['cat1', 'cat2'])['value'].sum().reset_index()
        ds_result = ds_result[ds_result['value'] > 3]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_agg_multiple_functions_same_column(self):
        """Test groupby with multiple aggregation functions on same column."""
        # pandas
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.groupby('category')['value'].agg(['sum', 'mean', 'count']).reset_index()

        # DataStore
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_result = ds_df.groupby('category')['value'].agg(['sum', 'mean', 'count']).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_on_filtered_data(self):
        """Test groupby on pre-filtered data."""
        # pandas
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        pd_filtered = pd_df[pd_df['value'] > 20]
        pd_result = pd_filtered.groupby('category')['value'].sum().reset_index()

        # DataStore
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_filtered = ds_df[ds_df['value'] > 20]
        ds_result = ds_filtered.groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_transform_with_multiple_columns(self):
        """Test groupby transform on multiple columns."""
        # pandas
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value1': [10, 20, 30, 40],
            'value2': [100, 200, 300, 400]
        })
        pd_df['sum1'] = pd_df.groupby('category')['value1'].transform('sum')
        pd_df['sum2'] = pd_df.groupby('category')['value2'].transform('sum')
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B'],
            'value1': [10, 20, 30, 40],
            'value2': [100, 200, 300, 400]
        })
        ds_df = ds_df.assign(sum1=ds_df.groupby('category')['value1'].transform('sum'))
        ds_df = ds_df.assign(sum2=ds_df.groupby('category')['value2'].transform('sum'))
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_head_tail_different_n(self):
        """Test groupby head and tail with different n values."""
        # pandas
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        pd_head = pd_df.groupby('category').head(2)
        pd_tail = pd_df.groupby('category').tail(1)

        # DataStore
        ds_df = DataStore({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        ds_head = ds_df.groupby('category').head(2)
        ds_tail = ds_df.groupby('category').tail(1)

        assert_datastore_equals_pandas(ds_head, pd_head)
        assert_datastore_equals_pandas(ds_tail, pd_tail)


# =============================================================================
# Test Group 4: Rolling/Expanding Combined Operations
# =============================================================================

class TestRollingExpandingCombined:
    """Test rolling/expanding combined with other operations."""

    def test_rolling_after_sort(self):
        """Test rolling calculation after sorting."""
        # pandas
        pd_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02', '2023-01-04']),
            'value': [30, 10, 20, 40]
        })
        pd_df = pd_df.sort_values('date')
        pd_df['rolling_sum'] = pd_df['value'].rolling(window=2).sum()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({
            'date': pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02', '2023-01-04']),
            'value': [30, 10, 20, 40]
        })
        ds_df = ds_df.sort_values('date')
        ds_df = ds_df.assign(rolling_sum=ds_df['value'].rolling(window=2).sum())
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_mean_with_filter(self):
        """Test expanding mean followed by filter."""
        # pandas
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        pd_df['expanding_mean'] = pd_df['value'].expanding().mean()
        pd_result = pd_df[pd_df['expanding_mean'] > 3]

        # DataStore
        ds_df = DataStore({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = ds_df.assign(expanding_mean=ds_df['value'].expanding().mean())
        ds_result = ds_df[ds_df['expanding_mean'] > 3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_on_multiple_columns(self):
        """Test rolling calculations on multiple columns."""
        # pandas
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        pd_df['A_rolling'] = pd_df['A'].rolling(window=2).sum()
        pd_df['B_rolling'] = pd_df['B'].rolling(window=3).mean()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = ds_df.assign(A_rolling=ds_df['A'].rolling(window=2).sum())
        ds_df = ds_df.assign(B_rolling=ds_df['B'].rolling(window=3).mean())
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_min_periods(self):
        """Test rolling with min_periods parameter."""
        # pandas
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        pd_df['rolling_1'] = pd_df['value'].rolling(window=3, min_periods=1).sum()
        pd_df['rolling_2'] = pd_df['value'].rolling(window=3, min_periods=2).sum()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({'value': [1, 2, 3, 4, 5]})
        ds_df = ds_df.assign(rolling_1=ds_df['value'].rolling(window=3, min_periods=1).sum())
        ds_df = ds_df.assign(rolling_2=ds_df['value'].rolling(window=3, min_periods=2).sum())
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 5: Schema and Type Inference Edge Cases
# =============================================================================

class TestSchemaTypeInference:
    """Test schema and type inference edge cases."""

    def test_mixed_int_float_column(self):
        """Test column with mixed int and float values."""
        # pandas
        pd_df = pd.DataFrame({'mixed': [1, 2.5, 3, 4.5, 5]})
        pd_result = pd_df[pd_df['mixed'] > 2]

        # DataStore
        ds_df = DataStore({'mixed': [1, 2.5, 3, 4.5, 5]})
        ds_result = ds_df[ds_df['mixed'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_numeric_conversion(self):
        """Test operations after string to numeric conversion."""
        # pandas
        pd_df = pd.DataFrame({'str_col': ['1', '2', '3', '4', '5']})
        pd_df['num_col'] = pd_df['str_col'].astype(int)
        pd_result = pd_df[pd_df['num_col'] > 2]

        # DataStore
        ds_df = DataStore({'str_col': ['1', '2', '3', '4', '5']})
        ds_df = ds_df.assign(num_col=ds_df['str_col'].astype(int))
        ds_result = ds_df[ds_df['num_col'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_select_preserves_types(self):
        """Test that column selection preserves column types."""
        # pandas
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        pd_result = pd_df[['int_col', 'str_col']]

        # DataStore
        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        ds_result = ds_df[['int_col', 'str_col']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_result_preserves_schema(self):
        """Test that empty result from filter preserves column schema."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        pd_result = pd_df[pd_df['A'] > 100]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_result = ds_df[ds_df['A'] > 100]

        # Check columns are preserved
        assert list(ds_result._get_df().columns) == list(pd_result.columns)
        assert len(ds_result) == len(pd_result)


# =============================================================================
# Test Group 6: Complex Chain Operations
# =============================================================================

class TestComplexChainOperations:
    """Test complex chains of multiple operations."""

    def test_filter_sort_head_chain(self):
        """Test filter -> sort -> head chain."""
        # pandas
        pd_df = pd.DataFrame({
            'A': [5, 3, 8, 1, 9, 2, 7, 4, 6, 10],
            'B': ['j', 'c', 'h', 'a', 'i', 'b', 'g', 'd', 'f', 'e']
        })
        pd_result = pd_df[pd_df['A'] > 3].sort_values('A').head(3)

        # DataStore
        ds_df = DataStore({
            'A': [5, 3, 8, 1, 9, 2, 7, 4, 6, 10],
            'B': ['j', 'c', 'h', 'a', 'i', 'b', 'g', 'd', 'f', 'e']
        })
        ds_result = ds_df[ds_df['A'] > 3].sort_values('A').head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_filter_groupby_chain(self):
        """Test assign -> filter -> groupby chain."""
        # pandas
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        pd_df['doubled'] = pd_df['value'] * 2
        pd_result = pd_df[pd_df['doubled'] > 40].groupby('category')['doubled'].sum().reset_index()

        # DataStore
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = ds_df.assign(doubled=ds_df['value'] * 2)
        ds_result = ds_df[ds_df['doubled'] > 40].groupby('category')['doubled'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multiple_assigns_chain(self):
        """Test chain of multiple assigns."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_df['B'] = pd_df['A'] * 2
        pd_df['C'] = pd_df['B'] + 10
        pd_df['D'] = pd_df['C'] / 2
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_df = ds_df.assign(B=ds_df['A'] * 2)
        ds_df = ds_df.assign(C=ds_df['B'] + 10)
        ds_df = ds_df.assign(D=ds_df['C'] / 2)
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_filter_assign_chain(self):
        """Test column select -> filter -> assign chain."""
        # pandas
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })
        pd_result = pd_df[['A', 'B']]
        pd_result = pd_result[pd_result['A'] > 2]
        pd_result = pd_result.assign(D=pd_result['A'] + pd_result['B'])

        # DataStore
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })
        ds_result = ds_df[['A', 'B']]
        ds_result = ds_result[ds_result['A'] > 2]
        ds_result = ds_result.assign(D=ds_result['A'] + ds_result['B'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_deep_chain_with_reset_index(self):
        """Test deep chain including reset_index."""
        # pandas
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        pd_result = (pd_df
                     .groupby('category')['value']
                     .sum()
                     .reset_index()
                     .sort_values('value', ascending=False)
                     .head(1))

        # DataStore
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        ds_result = (ds_df
                     .groupby('category')['value']
                     .sum()
                     .reset_index()
                     .sort_values('value', ascending=False)
                     .head(1))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 7: Error Recovery and Edge Cases
# =============================================================================

class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_operations(self):
        """Test operations on empty DataFrame."""
        # pandas
        pd_df = pd.DataFrame({'A': [], 'B': []})
        pd_df = pd_df.astype({'A': int, 'B': int})
        pd_result = pd_df[['A']]

        # DataStore
        ds_df = DataStore({'A': [], 'B': []})
        ds_df = ds_df.astype({'A': int, 'B': int})
        ds_result = ds_df[['A']]

        assert list(ds_result._get_df().columns) == list(pd_result.columns)

    def test_single_row_operations(self):
        """Test operations on single row DataFrame."""
        # pandas
        pd_df = pd.DataFrame({'A': [1], 'B': [2]})
        pd_df['C'] = pd_df['A'] + pd_df['B']
        pd_result = pd_df.sort_values('C').head(1)

        # DataStore
        ds_df = DataStore({'A': [1], 'B': [2]})
        ds_df = ds_df.assign(C=ds_df['A'] + ds_df['B'])
        ds_result = ds_df.sort_values('C').head(1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_null_column_operations(self):
        """Test operations on column with all null values."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [None, None, None]})
        pd_df['B'] = pd_df['B'].fillna(0)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [None, None, None]})
        ds_df = ds_df.fillna({'B': 0})
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_very_large_number_operations(self):
        """Test operations with very large numbers."""
        # pandas
        pd_df = pd.DataFrame({'A': [1e15, 2e15, 3e15]})
        pd_df['B'] = pd_df['A'] * 2
        pd_result = pd_df[pd_df['B'] > 3e15]

        # DataStore
        ds_df = DataStore({'A': [1e15, 2e15, 3e15]})
        ds_df = ds_df.assign(B=ds_df['A'] * 2)
        ds_result = ds_df[ds_df['B'] > 3e15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_special_characters_in_values(self):
        """Test handling of special characters in values."""
        # pandas
        pd_df = pd.DataFrame({
            'text': ["hello'world", 'foo"bar', "test\nline", "tab\there"]
        })
        pd_result = pd_df[pd_df['text'].str.contains('hello', regex=False)]

        # DataStore
        ds_df = DataStore({
            'text': ["hello'world", 'foo"bar', "test\nline", "tab\there"]
        })
        ds_result = ds_df[ds_df['text'].str.contains('hello', regex=False)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_unicode_values(self):
        """Test handling of unicode values."""
        # pandas
        pd_df = pd.DataFrame({
            'text': ['hello', 'world', 'test', 'data']
        })
        pd_result = pd_df[pd_df['text'] == 'hello']

        # DataStore
        ds_df = DataStore({
            'text': ['hello', 'world', 'test', 'data']
        })
        ds_result = ds_df[ds_df['text'] == 'hello']

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 8: Apply/Transform Variations
# =============================================================================

class TestApplyTransformVariations:
    """Test various apply and transform scenarios."""

    def test_apply_with_lambda_single_column(self):
        """Test apply with lambda on single column."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_df['B'] = pd_df['A'].apply(lambda x: x ** 2)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_df = ds_df.assign(B=ds_df['A'].apply(lambda x: x ** 2))
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_with_numpy_function(self):
        """Test apply with numpy function."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 4, 9, 16, 25]})
        pd_df['B'] = pd_df['A'].apply(np.sqrt)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({'A': [1, 4, 9, 16, 25]})
        ds_df = ds_df.assign(B=ds_df['A'].apply(np.sqrt))
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_multiple_functions(self):
        """Test transform with multiple functions."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_df['A_shifted'] = pd_df['A'].transform(lambda x: x.shift(1))
        pd_df['A_diff'] = pd_df['A'].transform(lambda x: x.diff())
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_df = ds_df.assign(A_shifted=ds_df['A'].transform(lambda x: x.shift(1)))
        ds_df = ds_df.assign(A_diff=ds_df['A'].transform(lambda x: x.diff()))
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_on_string_column(self):
        """Test apply on string column."""
        # pandas
        pd_df = pd.DataFrame({'text': ['hello', 'WORLD', 'Test']})
        pd_df['lower'] = pd_df['text'].apply(lambda x: x.lower())
        pd_result = pd_df

        # DataStore
        ds_df = DataStore({'text': ['hello', 'WORLD', 'Test']})
        ds_df = ds_df.assign(lower=ds_df['text'].apply(lambda x: x.lower()))
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 9: Index Operations
# =============================================================================

class TestIndexOperations:
    """Test index-related operations."""

    def test_set_index_then_reset(self):
        """Test set_index followed by reset_index."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_result = pd_df.set_index('A').reset_index()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_result = ds_df.set_index('A').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    @lazy_index_not_preserved
    def test_set_index_preserve_in_operations(self):
        """Test that operations preserve custom index."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})
        pd_df = pd_df.set_index('A')
        pd_df['C'] = pd_df['B'] * 2
        pd_result = pd_df.reset_index()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [10, 20, 30]})
        ds_df = ds_df.set_index('A')
        ds_df = ds_df.assign(C=ds_df['B'] * 2)
        ds_result = ds_df.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_fill(self):
        """Test reindex with fill_value."""
        # pandas - create df with non-default index
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 2, 4])
        pd_result = pd_df.reindex([0, 1, 2, 3, 4], fill_value=0)

        # DataStore - need to create with proper index from start
        # Use pandas df as source to preserve index
        source_df = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 2, 4])
        ds_df = DataStore(source_df)
        ds_result = ds_df.reindex([0, 1, 2, 3, 4], fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_index_operations(self):
        """Test sort_index operations."""
        # pandas - create with non-default index
        pd_df = pd.DataFrame({'A': [3, 1, 2]}, index=[2, 0, 1])
        pd_result = pd_df.sort_index()

        # DataStore - use pandas df as source to preserve index
        source_df = pd.DataFrame({'A': [3, 1, 2]}, index=[2, 0, 1])
        ds_df = DataStore(source_df)
        ds_result = ds_df.sort_index()

        # Compare values after resetting index
        assert_datastore_equals_pandas(ds_result.reset_index(drop=True), pd_result.reset_index(drop=True))


# =============================================================================
# Test Group 10: Copy and Memory Semantics
# =============================================================================

class TestCopyMemorySemantics:
    """Test copy and memory semantics."""

    def test_copy_independence(self):
        """Test that copy creates independent DataFrame."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_copy = pd_df.copy()
        pd_df['A'] = [10, 20, 30]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_copy = ds_df.copy()
        ds_df = ds_df.assign(A=[10, 20, 30])

        # Original copy should be unchanged
        assert list(ds_copy._get_df()['A']) == list(pd_copy['A'])

    def test_operations_dont_modify_original(self):
        """Test that operations don't modify original."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_filtered = pd_df[pd_df['A'] > 1]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_filtered = ds_df[ds_df['A'] > 1]

        # Original should have all rows
        assert len(ds_df) == len(pd_df)
        assert len(ds_filtered) == len(pd_filtered)

    def test_assign_returns_new_datastore(self):
        """Test that assign returns new DataStore."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_assigned = pd_df.assign(B=[4, 5, 6])

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_assigned = ds_df.assign(B=[4, 5, 6])

        # Original should not have column B
        assert 'B' not in ds_df.columns
        assert 'B' in ds_assigned.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
