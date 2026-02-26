"""
Comprehensive Pandas Alignment Tests for DataStore.

These tests strictly verify that DataStore operations produce results
identical to pandas operations, following the design principle:
"Users write familiar pandas-style code, backend automatically selects optimal execution engine."

Test Categories:
1. Basic Operations: from_df, select, filter, limit
2. GroupBy & Aggregation: groupby, agg, multiple aggregations
3. Sorting & Ordering: order_by, sort_values alignment
4. Column Operations: with_column, computed columns
5. Join & Union: join, union operations
6. Distinct & Deduplication: distinct, drop_duplicates alignment
7. String Operations: str.contains, string filters
8. Null Handling: is_null, is_not_null, notna alignment
9. Conditional Logic: isin, between, case_when
10. Chained Operations: complex pipelines

DESIGN PRINCIPLES ENFORCED:
- NO reset_index() to mask alignment issues (per workspace rules)
- Index preservation is verified where applicable
- Data values AND ordering are strictly compared
- Type compatibility is checked
"""

import unittest
import numpy as np
import pandas as pd
import datastore as ds
from tests.test_utils import _normalize_chdb_dtypes, assert_frame_equal, assert_series_equal


class TestBasicOperations(unittest.TestCase):
    """Test basic DataStore operations align with pandas."""

    def test_from_pandas_roundtrip(self):
        """DataStore.from_df() should preserve all data from pandas DataFrame."""
        pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
                'value': [10, 20, 15, 30, 25, 12, 35, 22],
                'score': [1.5, 2.3, 1.8, 3.2, 2.9, 1.6, 3.5, 2.4],
            }
        )

        store = ds.DataStore.from_df(pdf)
        result = store.select('*').to_df()

        # Strict comparison: values, columns, and index
        assert_frame_equal(result, pdf)

    def test_select_columns(self):
        """select() should return exact same columns as pandas column selection."""
        pdf = pd.DataFrame(
            {
                'a': [1, 2, 3],
                'b': [4, 5, 6],
                'c': [7, 8, 9],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[['a', 'c']]
        ds_result = store.select('a', 'c').to_df()

        assert_frame_equal(ds_result, pd_result)

    def test_limit_equals_head(self):
        """limit(n) should produce same result as pandas head(n)."""
        pdf = pd.DataFrame(
            {
                'id': range(10),
                'value': range(100, 110),
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.head(5)
        ds_result = store.limit(5).to_df()

        assert_frame_equal(ds_result, pd_result)


class TestFilterOperations(unittest.TestCase):
    """Test filter operations align with pandas boolean indexing."""

    def test_simple_filter(self):
        """Simple filter should match pandas boolean indexing."""
        pdf = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [25, 30, 35, 28, 32],
                'salary': [50000, 60000, 70000, 55000, 65000],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[pdf['age'] > 28]
        ds_result = store.filter(store['age'] > 28).to_df()

        # Verify index is preserved (not reset)
        self.assertEqual(list(pd_result.index), list(ds_result.index))
        assert_frame_equal(ds_result, pd_result)

    def test_multiple_conditions_or(self):
        """OR conditions should match pandas | operator."""
        pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'C'],
                'value': [10, 25, 30, 15, 20],
                'active': [True, True, False, True, False],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[(pdf['category'] == 'A') | (pdf['value'] > 20)]
        ds_result = store.filter((store['category'] == 'A') | (store['value'] > 20)).to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        assert_frame_equal(ds_result, pd_result)

    def test_multiple_conditions_and(self):
        """AND conditions should match pandas & operator."""
        pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'C'],
                'value': [10, 25, 30, 15, 20],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[(pdf['category'] == 'A') & (pdf['value'] > 15)]
        ds_result = store.filter((store['category'] == 'A') & (store['value'] > 15)).to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        assert_frame_equal(ds_result, pd_result)

    def test_isin_filter(self):
        """isin() should match pandas isin()."""
        pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'C', 'D', 'E'],
                'value': [1, 2, 3, 4, 5],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[pdf['category'].isin(['A', 'C', 'E'])]
        ds_result = store.filter(store['category'].isin(['A', 'C', 'E'])).to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        assert_frame_equal(ds_result, pd_result)

    def test_between_filter(self):
        """between() should match pandas between()."""
        pdf = pd.DataFrame(
            {
                'value': [5, 10, 15, 20, 25, 30],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[pdf['value'].between(10, 25)]
        ds_result = store.filter(store['value'].between(10, 25)).to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        assert_frame_equal(ds_result, pd_result)


class TestGroupByOperations(unittest.TestCase):
    """Test groupby operations align with pandas groupby."""

    def test_groupby_sum(self):
        """groupby().agg({'col': 'sum'}) should match pandas groupby sum."""
        pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A'],
                'value': [10, 20, 30, 40, 50],
            }
        )

        store = ds.DataStore.from_df(pdf)

        # Pandas groupby returns Series with category as index
        pd_series = pdf.groupby('category')['value'].sum()

        # DataStore groupby
        ds_result = store.groupby('category').agg({'value': 'sum'}).to_df()

        # DataStore result should have category as index, value as column
        ds_series = ds_result['value']

        # GroupBy order is not guaranteed - sort both for comparison
        assert_series_equal(ds_series.sort_index(), pd_series.sort_index())

    def test_groupby_multiple_agg(self):
        """groupby().agg() with multiple aggregations should match pandas.

        DataStore now correctly returns MultiIndex columns matching pandas behavior
        when using agg({col: [funcs]}).
        """
        pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A'],
                'value': [10, 20, 30, 40, 50],
                'count': [1, 2, 3, 4, 5],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.groupby('category').agg({'value': ['sum', 'mean'], 'count': 'sum'})
        ds_result = store.groupby('category').agg({'value': ['sum', 'mean'], 'count': 'sum'}).to_df()

        # DataStore now returns MultiIndex columns matching pandas
        # GroupBy order is not guaranteed - sort both by index for comparison
        assert_frame_equal(ds_result.sort_index(), pd_result.sort_index())

    def test_groupby_mean(self):
        """groupby().agg({'col': 'mean'}) should match pandas groupby mean."""
        pdf = pd.DataFrame(
            {
                'group': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
                'value': [10, 20, 30, 40, 50, 60],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_series = pdf.groupby('group')['value'].mean()
        ds_result = store.groupby('group').agg({'value': 'mean'}).to_df()
        ds_series = ds_result['value']

        # GroupBy order is not guaranteed - sort both for comparison
        assert_series_equal(ds_series.sort_index(), pd_series.sort_index())


class TestSortingOperations(unittest.TestCase):
    """Test sorting operations align with pandas sort_values."""

    def test_order_by_descending(self):
        """order_by(ascending=False) should match pandas sort_values."""
        pdf = pd.DataFrame(
            {
                'name': ['Charlie', 'Alice', 'Bob'],
                'score': [85, 92, 78],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.sort_values('score', ascending=False)
        ds_result = store.order_by('score', ascending=False).to_df()

        # Verify order (index should reflect original row positions)
        self.assertEqual(list(pd_result.index), list(ds_result.index))
        assert_frame_equal(ds_result, pd_result)

    def test_order_by_ascending(self):
        """order_by(ascending=True) should match pandas sort_values."""
        pdf = pd.DataFrame(
            {
                'name': ['Charlie', 'Alice', 'Bob'],
                'score': [85, 92, 78],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.sort_values('score', ascending=True)
        ds_result = store.order_by('score', ascending=True).to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        assert_frame_equal(ds_result, pd_result)


class TestColumnOperations(unittest.TestCase):
    """Test column operations align with pandas."""

    def test_with_column_arithmetic(self):
        """with_column() with arithmetic should match pandas computed column."""
        pdf = pd.DataFrame(
            {
                'price': [100, 200, 300],
                'quantity': [2, 3, 1],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.copy()
        pd_result['total'] = pd_result['price'] * pd_result['quantity']

        ds_result = store.with_column('total', store['price'] * store['quantity']).to_df()

        assert_frame_equal(ds_result, pd_result)

    def test_assign_column(self):
        """ds['col'] = expr should match pandas df['col'] = expr."""
        pdf = pd.DataFrame(
            {
                'a': [1, 2, 3],
                'b': [4, 5, 6],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.copy()
        pd_result['c'] = pd_result['a'] + pd_result['b']

        store['c'] = store['a'] + store['b']
        ds_result = store.to_df()

        assert_frame_equal(ds_result, pd_result)


class TestJoinOperations(unittest.TestCase):
    """Test join operations align with pandas merge."""

    def test_inner_join(self):
        """inner join should match pandas merge(how='inner')."""
        users = pd.DataFrame(
            {
                'user_id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie'],
            }
        )

        orders = pd.DataFrame(
            {
                'order_id': [101, 102, 103],
                'user_id': [1, 2, 1],
                'amount': [100, 200, 150],
            }
        )

        users_store = ds.DataStore.from_df(users)
        orders_store = ds.DataStore.from_df(orders)

        pd_result = pd.merge(users, orders, on='user_id', how='inner')
        ds_result = users_store.join(orders_store, on='user_id', how='inner').to_df()

        assert_frame_equal(ds_result, pd_result)


class TestUnionOperations(unittest.TestCase):
    """Test union operations align with pandas concat."""

    def test_union(self):
        """union() should match pandas concat(ignore_index=True)."""
        pdf1 = pd.DataFrame(
            {
                'id': [1, 2, 3],
                'value': [10, 20, 30],
            }
        )
        pdf2 = pd.DataFrame(
            {
                'id': [4, 5, 6],
                'value': [40, 50, 60],
            }
        )

        store1 = ds.DataStore.from_df(pdf1)
        store2 = ds.DataStore.from_df(pdf2)

        pd_result = pd.concat([pdf1, pdf2], ignore_index=True)
        ds_result = store1.union(store2).to_df()

        assert_frame_equal(ds_result, pd_result)


class TestDistinctOperations(unittest.TestCase):
    """Test distinct operations align with pandas drop_duplicates."""

    def test_distinct(self):
        """distinct() should match pandas drop_duplicates()."""
        pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'C', 'B', 'A'],
                'value': [1, 2, 1, 3, 2, 1],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.drop_duplicates()
        ds_result = store.distinct().to_df()

        # Verify index is preserved
        self.assertEqual(list(pd_result.index), list(ds_result.index))
        assert_frame_equal(ds_result, pd_result)


class TestNullHandling(unittest.TestCase):
    """Test null handling aligns with pandas."""

    def test_filter_not_null(self):
        """is_not_null() should match pandas notna()."""
        pdf = pd.DataFrame(
            {
                'a': [1, 2, None, 4, None],
                'b': [10, None, 30, None, 50],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[pdf['a'].notna()]
        ds_result = store.filter(store['a'].is_not_null()).to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        # Normalize chDB dtypes (Float64 with <NA> → float64 with nan) before comparison
        ds_result_normalized = _normalize_chdb_dtypes(ds_result)
        assert_frame_equal(ds_result_normalized, pd_result)

    def test_filter_is_null(self):
        """is_null() should match pandas isna()."""
        pdf = pd.DataFrame(
            {
                'a': [1, 2, None, 4, None],
                'b': [10, None, 30, None, 50],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[pdf['a'].isna()]
        ds_result = store.filter(store['a'].is_null()).to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        # Normalize chDB dtypes (Float64 with <NA> → float64 with nan) before comparison
        ds_result_normalized = _normalize_chdb_dtypes(ds_result)
        assert_frame_equal(ds_result_normalized, pd_result)


class TestStringOperations(unittest.TestCase):
    """Test string operations align with pandas str accessor."""

    def test_str_contains(self):
        """str.contains() should match pandas str.contains()."""
        pdf = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David'],
                'city': ['New York', 'Boston', 'Chicago', 'New Orleans'],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[pdf['city'].str.contains('New')]
        ds_result = store.filter(store['city'].str.contains('New')).to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        assert_frame_equal(ds_result, pd_result)


class TestCaseWhen(unittest.TestCase):
    """Test CASE WHEN aligns with np.select."""

    def test_case_when_matches_np_select(self):
        """ds.when().otherwise() should match np.select()."""
        pdf = pd.DataFrame(
            {
                'score': [45, 65, 85, 95],
            }
        )

        store = ds.DataStore.from_df(pdf)

        # Pandas with np.select (reference)
        conditions = [
            pdf['score'] >= 90,
            pdf['score'] >= 80,
            pdf['score'] >= 60,
        ]
        choices = ['A', 'B', 'C']
        pd_result = pdf.copy()
        pd_result['grade'] = np.select(conditions, choices, default='F')

        # DataStore CASE WHEN
        store['grade'] = (
            store.when(store['score'] >= 90, 'A')
            .when(store['score'] >= 80, 'B')
            .when(store['score'] >= 60, 'C')
            .otherwise('F')
        )
        ds_result = store.to_df()

        assert_frame_equal(ds_result, pd_result)

    def test_case_when_binary_matches_np_where(self):
        """Simple ds.when().otherwise() should match np.where()."""
        pdf = pd.DataFrame(
            {
                'value': [10, 50, 100, 150],
            }
        )

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.copy()
        pd_result['status'] = np.where(pdf['value'] >= 100, 'high', 'low')

        store['status'] = store.when(store['value'] >= 100, 'high').otherwise('low')
        ds_result = store.to_df()

        assert_frame_equal(ds_result, pd_result)


class TestChainedOperations(unittest.TestCase):
    """Test chained operations produce same results as pandas pipelines."""

    def test_filter_groupby_orderby(self):
        """filter -> groupby -> order_by should match pandas pipeline."""
        pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'C', 'A', 'C', 'B'],
                'region': ['N', 'S', 'N', 'E', 'S', 'N', 'E', 'S'],
                'sales': [100, 200, 150, 300, 250, 180, 220, 170],
            }
        )

        store = ds.DataStore.from_df(pdf)

        # Pandas chained
        pd_series = pdf[pdf['sales'] > 150].groupby('category')['sales'].sum().sort_values(ascending=False)

        # DataStore chained
        ds_result = (
            store.filter(store['sales'] > 150)
            .groupby('category')
            .agg({'sales': 'sum'})
            .order_by('sales', ascending=False)
            .to_df()
        )

        ds_series = ds_result['sales']

        # Verify VALUES in ORDER
        self.assertEqual(list(pd_series.values), list(ds_series.values))
        # Verify INDEX (category) in ORDER
        self.assertEqual(list(pd_series.index), list(ds_series.index))


class TestAggregationFunctions(unittest.TestCase):
    """Test aggregation functions align with pandas."""

    def test_multiple_agg_functions(self):
        """agg() with multiple functions should produce correct values."""
        pdf = pd.DataFrame(
            {
                'value': [10, 20, 30, 40, 50],
            }
        )

        store = ds.DataStore.from_df(pdf)

        # Note: use 'mean' not 'avg' (pandas naming)
        pd_result = pdf['value'].agg(['sum', 'mean', 'min', 'max', 'count'])

        ds_result = store.agg({'value': ['sum', 'mean', 'min', 'max', 'count']}).to_df()

        # DataStore returns DataFrame with agg function names as index
        # Verify each aggregation value matches pandas
        self.assertEqual(pd_result['sum'], ds_result.loc['sum', 'value'])
        self.assertEqual(pd_result['mean'], ds_result.loc['mean', 'value'])
        self.assertEqual(pd_result['min'], ds_result.loc['min', 'value'])
        self.assertEqual(pd_result['max'], ds_result.loc['max', 'value'])
        self.assertEqual(pd_result['count'], ds_result.loc['count', 'value'])


class TestSQLExecution(unittest.TestCase):
    """Test SQL execution behavior (note: SQL doesn't preserve pandas index)."""

    def test_sql_direct_data_matches(self):
        """SQL execution should produce correct DATA (index may differ).

        Note: SQL execution via .sql() does NOT preserve pandas row indices.
        This is expected - SQL doesn't have row index concept.
        For index preservation, use .filter() instead.
        """
        pdf = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [25, 30, 35],
            }
        )

        store = ds.DataStore.from_df(pdf)

        # Reference data (reset index since SQL returns 0-based)
        pd_data = pdf[pdf['age'] >= 30].reset_index(drop=True)

        # DataStore SQL
        ds_result = store.sql("SELECT * FROM __df__ WHERE age >= 30").to_df()

        # Compare data only (check_index=False since SQL doesn't preserve index)
        assert_frame_equal(
            ds_result.reset_index(drop=True),
            pd_data,
            )


class TestIndexPreservation(unittest.TestCase):
    """Test that lazy operations preserve pandas index correctly."""

    def test_filter_preserves_custom_index(self):
        """filter() should preserve custom index."""
        pdf = pd.DataFrame({'value': [10, 20, 30, 40, 50]}, index=['a', 'b', 'c', 'd', 'e'])

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf[pdf['value'] > 25]
        ds_result = store.filter(store['value'] > 25).to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        self.assertEqual(['c', 'd', 'e'], list(ds_result.index))

    def test_orderby_preserves_original_index(self):
        """order_by() should preserve original row indices."""
        pdf = pd.DataFrame({'value': [30, 10, 20]}, index=['x', 'y', 'z'])

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.sort_values('value')
        ds_result = store.order_by('value').to_df()

        # Index should reflect original positions
        self.assertEqual(list(pd_result.index), list(ds_result.index))
        self.assertEqual(['y', 'z', 'x'], list(ds_result.index))

    def test_distinct_preserves_index(self):
        """distinct() should preserve first occurrence index."""
        pdf = pd.DataFrame({'value': [1, 2, 1, 3, 2]}, index=['a', 'b', 'c', 'd', 'e'])

        store = ds.DataStore.from_df(pdf)

        pd_result = pdf.drop_duplicates()
        ds_result = store.distinct().to_df()

        self.assertEqual(list(pd_result.index), list(ds_result.index))
        self.assertEqual(['a', 'b', 'd'], list(ds_result.index))


if __name__ == '__main__':
    unittest.main()
