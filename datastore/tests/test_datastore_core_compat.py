"""
DataStore Core API Compatibility Tests

Tests for DataStore's core fluent API methods (distinct, join, union, with_column, etc.)
that operate on DataStore objects created from pandas DataFrames.

These tests verify that DataStore.from_df() can be used as a drop-in replacement
for pandas operations with its fluent API.
"""

import unittest
import numpy as np
import pandas as pd

import datastore as ds


class TestDistinct(unittest.TestCase):
    """Tests for DataStore.distinct() method."""

    def test_distinct_removes_duplicates(self):
        """distinct() should remove duplicate rows."""
        pdf = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A'],
            'value': [1, 2, 1, 3, 2, 1],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.distinct().to_df()

        # Should have 3 unique rows
        self.assertEqual(len(result), 3)
        # Verify unique combinations exist
        unique_tuples = set(zip(result['category'], result['value']))
        expected = {('A', 1), ('B', 2), ('C', 3)}
        self.assertEqual(unique_tuples, expected)

    def test_distinct_with_subset(self):
        """distinct(subset=[...]) should consider only specified columns."""
        pdf = pd.DataFrame({
            'category': ['A', 'B', 'A'],
            'value': [1, 2, 3],  # Different values for same category
        })

        store = ds.DataStore.from_df(pdf)
        result = store.distinct(subset=['category']).to_df()

        # Should have 2 unique categories (A and B)
        self.assertEqual(len(result), 2)

    def test_distinct_preserves_order(self):
        """distinct() should preserve first occurrence order by default."""
        pdf = pd.DataFrame({
            'x': [3, 1, 2, 1, 3],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.distinct().to_df()

        # First occurrences: 3, 1, 2
        self.assertEqual(list(result['x']), [3, 1, 2])


class TestJoin(unittest.TestCase):
    """Tests for DataStore.join() method."""

    def test_inner_join(self):
        """Inner join should return matching rows from both tables."""
        users = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
        })

        orders = pd.DataFrame({
            'order_id': [101, 102, 103],
            'user_id': [1, 2, 1],
            'amount': [100, 200, 150],
        })

        users_store = ds.DataStore.from_df(users)
        orders_store = ds.DataStore.from_df(orders)

        result = users_store.join(orders_store, on='user_id', how='inner').to_df()

        # Should have 3 rows (user 1 has 2 orders, user 2 has 1 order)
        self.assertEqual(len(result), 3)
        # Should have columns from both tables
        self.assertIn('name', result.columns)
        self.assertIn('amount', result.columns)

    def test_left_join(self):
        """Left join should include all rows from left table."""
        left = pd.DataFrame({
            'id': [1, 2, 3],
            'value': ['a', 'b', 'c'],
        })

        right = pd.DataFrame({
            'id': [1, 2],
            'extra': ['x', 'y'],
        })

        left_store = ds.DataStore.from_df(left)
        right_store = ds.DataStore.from_df(right)

        result = left_store.join(right_store, on='id', how='left').to_df()

        # Should have 3 rows (all from left)
        self.assertEqual(len(result), 3)
        # Row with id=3 should have NaN for 'extra'
        self.assertTrue(pd.isna(result[result['id'] == 3]['extra'].iloc[0]))

    def test_join_with_different_column_names(self):
        """Join with left_on/right_on for different column names."""
        users = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
        })

        activities = pd.DataFrame({
            'activity_id': [1, 2, 3],
            'uid': [1, 2, 1],
            'action': ['login', 'buy', 'logout'],
        })

        users_store = ds.DataStore.from_df(users)
        activities_store = ds.DataStore.from_df(activities)

        result = users_store.join(
            activities_store,
            left_on='user_id',
            right_on='uid',
            how='inner'
        ).to_df()

        self.assertEqual(len(result), 3)
        self.assertIn('name', result.columns)
        self.assertIn('action', result.columns)


class TestUnion(unittest.TestCase):
    """Tests for DataStore.union() method."""

    def test_union_concatenates_rows(self):
        """union() should concatenate rows from both DataStores."""
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30],
        })
        df2 = pd.DataFrame({
            'id': [4, 5, 6],
            'value': [40, 50, 60],
        })

        store1 = ds.DataStore.from_df(df1)
        store2 = ds.DataStore.from_df(df2)

        result = store1.union(store2).to_df()

        # Should have 6 rows
        self.assertEqual(len(result), 6)
        # Should contain all values
        self.assertEqual(set(result['id']), {1, 2, 3, 4, 5, 6})

    def test_union_removes_duplicates(self):
        """union() without all=True should remove duplicates."""
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30],
        })
        df2 = pd.DataFrame({
            'id': [2, 3, 4],
            'value': [20, 30, 40],
        })

        store1 = ds.DataStore.from_df(df1)
        store2 = ds.DataStore.from_df(df2)

        result = store1.union(store2).to_df()

        # Should have 4 unique rows (duplicates for id 2, 3 removed)
        self.assertEqual(len(result), 4)

    def test_union_all_keeps_duplicates(self):
        """union(all=True) should keep all rows including duplicates."""
        df1 = pd.DataFrame({
            'id': [1, 2],
            'value': [10, 20],
        })
        df2 = pd.DataFrame({
            'id': [2, 3],
            'value': [20, 30],
        })

        store1 = ds.DataStore.from_df(df1)
        store2 = ds.DataStore.from_df(df2)

        result = store1.union(store2, all=True).to_df()

        # Should have 4 rows (duplicate row with id=2, value=20 kept)
        self.assertEqual(len(result), 4)


class TestWithColumn(unittest.TestCase):
    """Tests for DataStore.with_column() method."""

    def test_with_column_adds_computed_column(self):
        """with_column() should add a new computed column."""
        pdf = pd.DataFrame({
            'price': [100, 200, 300],
            'quantity': [2, 3, 1],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.with_column('total', store['price'] * store['quantity']).to_df()

        # Should have original columns plus 'total'
        self.assertIn('total', result.columns)
        # Verify computed values
        expected_totals = [200, 600, 300]
        self.assertEqual(list(result['total']), expected_totals)

    def test_with_column_scalar_value(self):
        """with_column() should support scalar values."""
        pdf = pd.DataFrame({
            'a': [1, 2, 3],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.with_column('constant', 42).to_df()

        self.assertIn('constant', result.columns)
        self.assertEqual(list(result['constant']), [42, 42, 42])

    def test_with_column_expression(self):
        """with_column() should support complex expressions."""
        pdf = pd.DataFrame({
            'value': [10, 20, 30],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.with_column('doubled', store['value'] * 2).to_df()
        result = store.with_column('doubled', store['value'] * 2).with_column(
            'plus_ten', store['value'] + 10
        ).to_df()

        self.assertIn('doubled', result.columns)
        self.assertIn('plus_ten', result.columns)


class TestSqlDirect(unittest.TestCase):
    """Tests for DataStore.sql() method."""

    def test_sql_filter(self):
        """sql() should execute filter queries correctly."""
        pdf = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.sql("SELECT * FROM __df__ WHERE age >= 30").to_df()

        # Should have 2 rows (Bob and Charlie)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result['name']), {'Bob', 'Charlie'})

    def test_sql_select_columns(self):
        """sql() should execute column selection correctly."""
        pdf = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.sql("SELECT a, c FROM __df__").to_df()

        self.assertEqual(list(result.columns), ['a', 'c'])

    def test_sql_aggregation(self):
        """sql() should execute aggregation queries correctly."""
        pdf = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.sql("SELECT category, SUM(value) as total FROM __df__ GROUP BY category").to_df()

        self.assertEqual(len(result), 2)


class TestChainedOperations(unittest.TestCase):
    """Tests for chained DataStore operations."""

    def test_filter_then_select(self):
        """filter() -> select() chain should work correctly."""
        pdf = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.filter(store['a'] > 2).select('a', 'b').to_df()

        self.assertEqual(len(result), 3)
        self.assertEqual(set(result['a']), {3, 4, 5})

    def test_groupby_then_orderby(self):
        """groupby() -> order_by() chain should work correctly."""
        pdf = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'C'],
            'value': [10, 20, 30, 40, 25],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.groupby('category').agg({'value': 'sum'}).order_by('value', ascending=False).to_df()

        # Check that results are sorted by value descending
        values = list(result['value'])
        self.assertEqual(values, sorted(values, reverse=True))

    def test_filter_distinct_limit(self):
        """filter() -> distinct() -> limit() chain should work correctly."""
        pdf = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
            'value': [1, 2, 1, 3, 2, 1, 3],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.filter(store['value'] >= 2).distinct().limit(2).to_df()

        self.assertEqual(len(result), 2)


class TestNullHandling(unittest.TestCase):
    """Tests for null value handling."""

    def test_is_not_null_filter(self):
        """filter with is_not_null() should work correctly."""
        pdf = pd.DataFrame({
            'a': [1, 2, None, 4, None],
            'b': [10, None, 30, None, 50],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.filter(store['a'].is_not_null()).to_df()

        # Should have 3 rows where 'a' is not null
        self.assertEqual(len(result), 3)
        self.assertFalse(result['a'].isna().any())


class TestStringOperations(unittest.TestCase):
    """Tests for string column operations."""

    def test_str_contains_filter(self):
        """filter with str.contains() should work correctly."""
        pdf = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'city': ['New York', 'Boston', 'Chicago', 'New Orleans'],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.filter(store['city'].str.contains('New')).to_df()

        # Should have 2 rows (New York, New Orleans)
        self.assertEqual(len(result), 2)
        self.assertTrue(all('New' in city for city in result['city']))


class TestInFilter(unittest.TestCase):
    """Tests for isin() filter."""

    def test_isin_filter(self):
        """filter with isin() should work correctly."""
        pdf = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D', 'E'],
            'value': [1, 2, 3, 4, 5],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.filter(store['category'].isin(['A', 'C', 'E'])).to_df()

        self.assertEqual(len(result), 3)
        self.assertEqual(set(result['category']), {'A', 'C', 'E'})


class TestBetweenFilter(unittest.TestCase):
    """Tests for between() filter."""

    def test_between_filter(self):
        """filter with between() should work correctly."""
        pdf = pd.DataFrame({
            'value': [5, 10, 15, 20, 25, 30],
        })

        store = ds.DataStore.from_df(pdf)
        result = store.filter(store['value'].between(10, 25)).to_df()

        self.assertEqual(len(result), 4)
        self.assertTrue(all(10 <= v <= 25 for v in result['value']))


if __name__ == '__main__':
    unittest.main()

