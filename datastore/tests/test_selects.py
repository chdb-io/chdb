"""
Test SELECT operations with chdb execution - migrated from pypika test_selects.py

Tests both SQL generation AND actual execution on chdb (ClickHouse).
"""

import unittest
import pandas as pd
from datastore import DataStore, Field, Count, Sum
from datastore.exceptions import QueryError
from datastore.lazy_result import LazySeries


class SelectTests(unittest.TestCase):
    """Basic SELECT tests with chdb execution"""

    def setUp(self):
        """Create test table and data"""
        self.ds = DataStore(table="abc")
        self.ds.connect()
        self.ds.create_table({"foo": "String", "bar": "String", "baz": "UInt32"}, drop_if_exists=True)
        self.ds.insert(
            [
                {"foo": "a", "bar": "x", "baz": 1},
                {"foo": "b", "bar": "y", "baz": 2},
                {"foo": "c", "bar": "z", "baz": 3},
            ]
        )

    def tearDown(self):
        """Clean up"""
        self.ds.close()

    def test_empty_datastore_default_select_star(self):
        """Test DataStore without select defaults to SELECT *"""
        ds = DataStore(table="abc")
        self.assertEqual('SELECT * FROM "abc"', ds.to_sql())

    def test_select__star(self):
        """Test SELECT * - SQL generation and execution"""
        result = self.ds.select("*").execute()

        # Verify SQL generation
        self.assertEqual('SELECT * FROM "abc"', self.ds.select("*").to_sql())

        # Verify execution
        self.assertEqual(3, len(result))
        self.assertEqual(3, len(result.column_names))

    def test_select__star__replacement(self):
        """Test that selecting * replaces previous selections"""
        ds_query = self.ds.select("foo").select("*")
        self.assertEqual('SELECT * FROM "abc"', ds_query.to_sql())

        result = ds_query.execute()
        self.assertEqual(3, len(result))

    def test_select__distinct__single(self):
        """Test SELECT DISTINCT with single column"""
        # Create table with duplicates
        ds = DataStore(table="test_distinct")
        ds.connect()
        ds.create_table({"city": "String"})
        ds.insert(
            [
                {"city": "NYC"},
                {"city": "LA"},
                {"city": "NYC"},
                {"city": "SF"},
                {"city": "LA"},
            ]
        )

        result = ds.select("city").distinct().execute()

        # Verify SQL
        self.assertEqual('SELECT DISTINCT "city" FROM "test_distinct"', ds.select("city").distinct().to_sql())

        # Verify execution - should have 3 unique cities
        self.assertEqual(3, len(result))
        cities = {row[0] for row in result.rows}
        self.assertEqual({"NYC", "LA", "SF"}, cities)

        ds.close()

    def test_select__distinct__multi(self):
        """Test SELECT DISTINCT with multiple columns"""
        ds = DataStore(table="test_distinct2")
        ds.connect()
        ds.create_table({"a": "String", "b": "String"})
        ds.insert(
            [
                {"a": "x", "b": "1"},
                {"a": "x", "b": "2"},
                {"a": "x", "b": "1"},  # duplicate
            ]
        )

        result = ds.select("a", "b").distinct().execute()

        self.assertEqual('SELECT DISTINCT "a", "b" FROM "test_distinct2"', ds.select("a", "b").distinct().to_sql())
        self.assertEqual(2, len(result))  # Only 2 unique combinations

        ds.close()

    def test_select__column__single__str(self):
        """Test SELECT single column as string"""
        result = self.ds.select("foo").execute()

        self.assertEqual('SELECT "foo" FROM "abc"', self.ds.select("foo").to_sql())
        self.assertEqual(3, len(result))
        self.assertEqual(["foo"], result.column_names)

    def test_select__column__single__field(self):
        """Test SELECT single column as Field object"""
        result = self.ds.select(self.ds.foo).execute()

        self.assertEqual('SELECT "foo" FROM "abc"', self.ds.select(self.ds.foo).to_sql())
        self.assertEqual(3, len(result))
        self.assertEqual(["foo"], result.column_names)

    def test_select__column__single__alias__str(self):
        """Test SELECT column with alias"""
        result = self.ds.select(self.ds.foo.as_("my_alias")).execute()

        self.assertEqual('SELECT "foo" AS "my_alias" FROM "abc"', self.ds.select(self.ds.foo.as_("my_alias")).to_sql())
        self.assertEqual(3, len(result))
        self.assertIn("my_alias", result.column_names)

    def test_select__columns__multi__str(self):
        """Test SELECT multiple columns as strings"""
        ds1 = self.ds.select("foo", "bar")
        ds2 = self.ds.select("foo").select("bar")

        self.assertEqual('SELECT "foo", "bar" FROM "abc"', ds1.to_sql())
        self.assertEqual('SELECT "foo", "bar" FROM "abc"', ds2.to_sql())

        # Execute both
        result1 = ds1.execute()
        result2 = ds2.execute()

        self.assertEqual(3, len(result1))
        self.assertEqual(3, len(result2))
        self.assertEqual(["foo", "bar"], result1.column_names)

    def test_select__columns__multi__field(self):
        """Test SELECT multiple columns as Field objects"""
        result1 = self.ds.select(self.ds.foo, self.ds.bar).execute()
        result2 = self.ds.select(self.ds.foo).select(self.ds.bar).execute()

        self.assertEqual(3, len(result1))
        self.assertEqual(3, len(result2))
        self.assertEqual(["foo", "bar"], result1.column_names)

    def test_select_with_limit(self):
        """Test SELECT with LIMIT using slice notation"""
        result = self.ds.select("foo")[:2].execute()

        self.assertEqual('SELECT "foo" FROM "abc" LIMIT 2', self.ds.select("foo")[:2].to_sql())
        self.assertEqual(2, len(result))

    def test_select_with_limit_zero(self):
        """Test SELECT with LIMIT 0"""
        result1 = self.ds.select("foo")[:0].execute()
        result2 = self.ds.select("foo").limit(0).execute()

        self.assertEqual('SELECT "foo" FROM "abc" LIMIT 0', self.ds.select("foo")[:0].to_sql())
        self.assertEqual('SELECT "foo" FROM "abc" LIMIT 0', self.ds.select("foo").limit(0).to_sql())

        self.assertEqual(0, len(result1))
        self.assertEqual(0, len(result2))

    def test_select_with_limit__func(self):
        """Test SELECT with LIMIT using method"""
        result = self.ds.select("foo").limit(2).execute()

        self.assertEqual('SELECT "foo" FROM "abc" LIMIT 2', self.ds.select("foo").limit(2).to_sql())
        self.assertEqual(2, len(result))

    def test_select_with_offset(self):
        """Test SELECT with OFFSET using slice notation"""
        result = self.ds.select("foo")[1:].execute()

        self.assertEqual('SELECT "foo" FROM "abc" OFFSET 1', self.ds.select("foo")[1:].to_sql())
        self.assertEqual(2, len(result))  # 3 total - 1 offset = 2

    def test_select_with_offset__func(self):
        """Test SELECT with OFFSET using method"""
        result = self.ds.select("foo").offset(1).execute()

        self.assertEqual('SELECT "foo" FROM "abc" OFFSET 1', self.ds.select("foo").offset(1).to_sql())
        self.assertEqual(2, len(result))

    def test_select_with_limit_and_offset(self):
        """Test SELECT with both LIMIT and OFFSET"""
        result = self.ds.select("foo")[1:2].execute()

        self.assertEqual('SELECT "foo" FROM "abc" LIMIT 1 OFFSET 1', self.ds.select("foo")[1:2].to_sql())
        self.assertEqual(1, len(result))


class WhereTests(unittest.TestCase):
    """WHERE clause tests with chdb execution"""

    def setUp(self):
        """Create test table"""
        self.ds = DataStore(table="abc")
        self.ds.connect()
        self.ds.create_table({"foo": "UInt32", "bar": "UInt32", "baz": "UInt32", "name": "String"}, drop_if_exists=True)
        self.ds.insert(
            [
                {"foo": 1, "bar": 10, "baz": 100, "name": "Alice"},
                {"foo": 5, "bar": 5, "baz": 50, "name": "Bob"},
                {"foo": 10, "bar": 1, "baz": 10, "name": "Charlie"},
            ]
        )

    def tearDown(self):
        """Clean up"""
        self.ds.close()

    def test_where_field_equals(self):
        """Test WHERE with field equality"""
        # Create a row where foo equals bar
        ds = DataStore(table="test_eq")
        ds.connect()
        ds.create_table({"foo": "UInt32", "bar": "UInt32"})
        ds.insert(
            [
                {"foo": 5, "bar": 5},
                {"foo": 3, "bar": 7},
            ]
        )

        result = ds.select("*").filter(ds.foo == ds.bar).execute()

        self.assertEqual(
            'SELECT * FROM "test_eq" WHERE "foo" = "bar"', ds.select("*").filter(ds.foo == ds.bar).to_sql()
        )
        self.assertEqual(1, len(result))

        ds.close()

    def test_where_field_equals_value(self):
        """Test WHERE field equals value"""
        result = self.ds.select("*").filter(self.ds.foo == 5).execute()

        self.assertEqual('SELECT * FROM "abc" WHERE "foo" = 5', self.ds.select("*").filter(self.ds.foo == 5).to_sql())
        self.assertEqual(1, len(result))
        self.assertEqual(5, result.rows[0][0])  # foo column

    def test_where_field_equals_where(self):
        """Test multiple WHERE conditions"""
        result = self.ds.select("*").filter(self.ds.foo == 5).filter(self.ds.bar == 5).execute()

        self.assertEqual(
            'SELECT * FROM "abc" WHERE ("foo" = 5 AND "bar" = 5)',
            self.ds.select("*").filter(self.ds.foo == 5).filter(self.ds.bar == 5).to_sql(),
        )
        self.assertEqual(1, len(result))

    def test_where_single_quote(self):
        """Test WHERE with string containing single quote"""
        ds = DataStore(table="test_quote")
        ds.connect()
        ds.create_table({"text": "String"})
        ds.insert([{"text": "bar'foo"}])

        result = ds.select("*").filter(ds.text == "bar'foo").execute()

        self.assertEqual(1, len(result))
        self.assertEqual("bar'foo", result.rows[0][0])

        ds.close()

    def test_where_field_equals_and(self):
        """Test WHERE with AND operator"""
        result = self.ds.select("*").filter((self.ds.foo == 5) & (self.ds.bar == 5)).execute()

        self.assertEqual(
            'SELECT * FROM "abc" WHERE ("foo" = 5 AND "bar" = 5)',
            self.ds.select("*").filter((self.ds.foo == 5) & (self.ds.bar == 5)).to_sql(),
        )
        self.assertEqual(1, len(result))

    def test_where_field_equals_or(self):
        """Test WHERE with OR operator"""
        result = self.ds.select("*").filter((self.ds.foo == 1) | (self.ds.foo == 10)).execute()

        self.assertEqual(
            'SELECT * FROM "abc" WHERE ("foo" = 1 OR "foo" = 10)',
            self.ds.select("*").filter((self.ds.foo == 1) | (self.ds.foo == 10)).to_sql(),
        )
        self.assertEqual(2, len(result))

    def test_where_nested_conditions(self):
        """Test WHERE with nested conditions"""
        result = self.ds.select("*").filter((self.ds.foo == 1) | (self.ds.bar == 5)).filter(self.ds.baz >= 50).execute()

        # (foo==1 OR bar==5) AND baz>=50
        # Alice: foo=1, bar=10, baz=100 - matches (foo==1 is true, baz>=50 is true)
        # Bob: foo=5, bar=5, baz=50 - matches (bar==5 is true, baz>=50 is true)
        # Charlie: foo=10, bar=1, baz=10 - doesn't match (neither condition in OR is true)
        self.assertEqual(2, len(result))  # Alice and Bob

    def test_where_field_gt(self):
        """Test WHERE with greater than"""
        result = self.ds.select("*").filter(self.ds.foo > 5).execute()

        self.assertEqual('SELECT * FROM "abc" WHERE "foo" > 5', self.ds.select("*").filter(self.ds.foo > 5).to_sql())
        self.assertEqual(1, len(result))  # Only Charlie (foo=10)

    def test_where_field_lt(self):
        """Test WHERE with less than"""
        result = self.ds.select("*").filter(self.ds.foo < 5).execute()

        self.assertEqual('SELECT * FROM "abc" WHERE "foo" < 5', self.ds.select("*").filter(self.ds.foo < 5).to_sql())
        self.assertEqual(1, len(result))  # Only Alice (foo=1)

    def test_where_field_gte(self):
        """Test WHERE with greater than or equal"""
        result = self.ds.select("*").filter(self.ds.foo >= 5).execute()

        self.assertEqual('SELECT * FROM "abc" WHERE "foo" >= 5', self.ds.select("*").filter(self.ds.foo >= 5).to_sql())
        self.assertEqual(2, len(result))  # Bob and Charlie

    def test_where_field_lte(self):
        """Test WHERE with less than or equal"""
        result = self.ds.select("*").filter(self.ds.foo <= 5).execute()

        self.assertEqual('SELECT * FROM "abc" WHERE "foo" <= 5', self.ds.select("*").filter(self.ds.foo <= 5).to_sql())
        self.assertEqual(2, len(result))  # Alice and Bob

    def test_where_field_ne(self):
        """Test WHERE with not equal"""
        result = self.ds.select("*").filter(self.ds.foo != 5).execute()

        self.assertEqual('SELECT * FROM "abc" WHERE "foo" != 5', self.ds.select("*").filter(self.ds.foo != 5).to_sql())
        self.assertEqual(2, len(result))  # Alice and Charlie


class GroupByTests(unittest.TestCase):
    """GROUP BY tests with chdb execution"""

    def setUp(self):
        """Create test table"""
        self.ds = DataStore(table="abc")
        self.ds.connect()
        self.ds.create_table({"foo": "String", "bar": "UInt32"}, drop_if_exists=True)
        self.ds.insert(
            [
                {"foo": "A", "bar": 10},
                {"foo": "A", "bar": 20},
                {"foo": "B", "bar": 30},
            ]
        )

    def tearDown(self):
        """Clean up"""
        self.ds.close()

    def test_groupby__single(self):
        """Test GROUP BY single column"""
        result = self.ds.select(self.ds.foo).groupby(self.ds.foo).execute()

        self.assertEqual(
            'SELECT "foo" FROM "abc" GROUP BY "foo"', self.ds.select(self.ds.foo).groupby(self.ds.foo).to_sql()
        )
        self.assertEqual(2, len(result))  # Two groups: A and B

    def test_groupby__multi(self):
        """Test GROUP BY multiple columns"""
        ds = DataStore(table="test_multi")
        ds.connect()
        ds.create_table({"a": "String", "b": "String", "c": "UInt32"})
        ds.insert(
            [
                {"a": "x", "b": "1", "c": 10},
                {"a": "x", "b": "2", "c": 20},
            ]
        )

        result = ds.select(ds.a, ds.b).groupby(ds.a, ds.b).execute()

        self.assertEqual(
            'SELECT "a", "b" FROM "test_multi" GROUP BY "a", "b"', ds.select(ds.a, ds.b).groupby(ds.a, ds.b).to_sql()
        )
        self.assertEqual(2, len(result))

        ds.close()

    def test_groupby__count_star(self):
        """Test GROUP BY with COUNT(*)"""
        result = self.ds.select(self.ds.foo, Count("*")).groupby(self.ds.foo).execute()

        # Note: Count("*") renders as COUNT('*') in our implementation
        sql = self.ds.select(self.ds.foo, Count("*")).groupby(self.ds.foo).to_sql()
        self.assertIn("COUNT", sql)
        self.assertIn("GROUP BY", sql)

        self.assertEqual(2, len(result))

        # Check counts
        results_dict = {row[0]: row[1] for row in result.rows}
        self.assertEqual(2, results_dict["A"])
        self.assertEqual(1, results_dict["B"])

    def test_groupby__count_field(self):
        """Test GROUP BY with COUNT(field)"""
        result = self.ds.select(self.ds.foo, Count(self.ds.bar)).groupby(self.ds.foo).execute()

        # COUNT is wrapped in toInt64() to match pandas int64 dtype
        self.assertEqual(
            'SELECT "foo", toInt64(COUNT("bar")) FROM "abc" GROUP BY "foo"',
            self.ds.select(self.ds.foo, Count(self.ds.bar)).groupby(self.ds.foo).to_sql(),
        )
        self.assertEqual(2, len(result))

    def test_groupby__str(self):
        """Test GROUP BY with string column name"""
        result = self.ds.select("foo", Count("*")).groupby("foo").execute()

        # Verify SQL contains key components
        sql = self.ds.select("foo", Count("*")).groupby("foo").to_sql()
        self.assertIn("COUNT", sql)
        self.assertIn("GROUP BY", sql)

        self.assertEqual(2, len(result))

    def test_groupby__size(self):
        """Test GROUP BY size() method - returns LazySeries (pd.Series compatible)"""
        result = self.ds.groupby("foo").size()

        # size() returns LazySeries, not DataStore
        self.assertIsInstance(result, LazySeries)

        # Natural trigger via index access - returns pd.Series
        self.assertEqual(result["A"], 2)  # Two rows with foo="A"
        self.assertEqual(result["B"], 1)  # One row with foo="B"

    def test_groupby__size_includes_nan(self):
        """Test that size() includes NaN values (unlike count())"""
        # Create a DataStore with NaN values
        df = pd.DataFrame({"category": ["X", "X", "Y", "Y", "Y"], "value": [1, None, 3, None, None]})  # Some NaN values
        ds_df = DataStore.from_dataframe(df)

        size_result = ds_df.groupby("category").size()  # Returns LazySeries (pd.Series compatible)
        count_result = ds_df.groupby("category").count()  # Returns lazy DataStore
        expected_count = df.groupby("category").count()

        # size() counts all rows (including NaN) - natural trigger via index access
        self.assertEqual(size_result["X"], 2)
        self.assertEqual(size_result["Y"], 3)

        # count() excludes NaN values - natural trigger via == comparison
        assert count_result == expected_count

    def test_groupby__size_multiple_columns(self):
        """Test size() with multiple groupby columns using pandas-style list argument"""
        df = pd.DataFrame({"a": ["x", "x", "x", "y"], "b": ["1", "1", "2", "1"], "c": [10, 20, 30, 40]})
        ds = DataStore.from_dataframe(df)

        # Use pandas-style list argument: groupby(["a", "b"])
        result = ds.groupby(["a", "b"]).size()

        # Returns LazySeries (pd.Series compatible with MultiIndex)
        self.assertIsInstance(result, LazySeries)
        # Natural trigger via index access
        self.assertEqual(result[("x", "1")], 2)
        self.assertEqual(result[("x", "2")], 1)
        self.assertEqual(result[("y", "1")], 1)

    def test_groupby__size_pandas_alignment(self):
        """Test size() result matches pandas exactly"""
        df = pd.DataFrame({"dept": ["A", "A", "B", "B", "B", "C"], "value": [1, 2, 3, 4, 5, 6]})
        ds = DataStore.from_dataframe(df)

        ds_result = ds.groupby("dept").size()
        pd_result = df.groupby("dept").size()

        # Natural trigger via .equals() for Series equality comparison
        assert ds_result.equals(pd_result)

    def test_groupby__size_single_group(self):
        """Test size() with all rows in single group"""
        df = pd.DataFrame({"category": ["same", "same", "same"], "value": [1, 2, 3]})
        ds = DataStore.from_dataframe(df)

        result = ds.groupby("category").size()

        # Natural trigger via len() and index access
        self.assertEqual(len(result), 1)
        self.assertEqual(result["same"], 3)

    def test_groupby__size_varargs_style(self):
        """Test size() with varargs groupby syntax: groupby('a', 'b')"""
        df = pd.DataFrame({"a": ["x", "x", "y"], "b": ["1", "2", "1"], "c": [10, 20, 30]})
        ds = DataStore.from_dataframe(df)

        # Use varargs style (also supported)
        result = ds.groupby("a", "b").size()

        # Returns LazySeries (pd.Series compatible with MultiIndex)
        self.assertIsInstance(result, LazySeries)
        # Natural trigger via index access
        self.assertEqual(result[("x", "1")], 1)
        self.assertEqual(result[("x", "2")], 1)
        self.assertEqual(result[("y", "1")], 1)

    def test_groupby__size_dtype(self):
        """Test size() returns int64 dtype like pandas"""
        df = pd.DataFrame({"category": ["A", "A", "B"], "value": [1, 2, 3]})
        ds = DataStore.from_dataframe(df)

        result = ds.groupby("category").size()

        # Natural trigger via .dtype property
        self.assertEqual(result.dtype, "int64")


class HavingTests(unittest.TestCase):
    """HAVING clause tests with chdb execution"""

    def setUp(self):
        """Create test table"""
        self.ds = DataStore(table="abc")
        self.ds.connect()
        self.ds.create_table({"foo": "String", "bar": "UInt32"}, drop_if_exists=True)
        self.ds.insert(
            [
                {"foo": "A", "bar": 10},
                {"foo": "A", "bar": 20},
                {"foo": "B", "bar": 30},
                {"foo": "C", "bar": 5},
            ]
        )

    def tearDown(self):
        """Clean up"""
        self.ds.close()

    def test_having_greater_than(self):
        """Test HAVING with greater than"""
        result = (
            self.ds.select(self.ds.foo, Sum(self.ds.bar)).groupby(self.ds.foo).having(Sum(self.ds.bar) > 20).execute()
        )

        self.assertEqual(
            'SELECT "foo", SUM("bar") FROM "abc" GROUP BY "foo" HAVING SUM("bar") > 20',
            self.ds.select(self.ds.foo, Sum(self.ds.bar)).groupby(self.ds.foo).having(Sum(self.ds.bar) > 20).to_sql(),
        )
        self.assertEqual(2, len(result))  # A (30) and B (30)

    def test_having_and(self):
        """Test HAVING with AND condition"""
        result = (
            self.ds.select(self.ds.foo, Sum(self.ds.bar))
            .groupby(self.ds.foo)
            .having((Sum(self.ds.bar) > 10) & (Sum(self.ds.bar) < 100))
            .execute()
        )

        self.assertEqual(2, len(result))  # A (30) and B (30), not C (5)


class OrderByTests(unittest.TestCase):
    """ORDER BY tests with chdb execution"""

    def setUp(self):
        """Create test table"""
        self.ds = DataStore(table="abc")
        self.ds.connect()
        self.ds.create_table({"foo": "String", "bar": "UInt32"}, drop_if_exists=True)
        self.ds.insert(
            [
                {"foo": "c", "bar": 3},
                {"foo": "a", "bar": 1},
                {"foo": "b", "bar": 2},
            ]
        )

    def tearDown(self):
        """Clean up"""
        self.ds.close()

    def test_orderby_single_field(self):
        """Test ORDER BY single field"""
        result = self.ds.select(self.ds.foo).sort(self.ds.foo).execute()

        self.assertEqual(
            'SELECT "foo" FROM "abc" ORDER BY "foo" ASC', self.ds.select(self.ds.foo).sort(self.ds.foo).to_sql()
        )
        self.assertEqual(3, len(result))

        # Check order
        values = [row[0] for row in result.rows]
        self.assertEqual(["a", "b", "c"], values)

    def test_orderby_multi_fields(self):
        """Test ORDER BY multiple fields"""
        result = self.ds.select(self.ds.foo, self.ds.bar).sort(self.ds.foo, self.ds.bar).execute()

        self.assertEqual(
            'SELECT "foo", "bar" FROM "abc" ORDER BY "foo" ASC, "bar" ASC',
            self.ds.select(self.ds.foo, self.ds.bar).sort(self.ds.foo, self.ds.bar).to_sql(),
        )
        self.assertEqual(3, len(result))

    def test_orderby_single_str(self):
        """Test ORDER BY with string column name"""
        result = self.ds.select("foo").sort("foo").execute()

        self.assertEqual('SELECT "foo" FROM "abc" ORDER BY "foo" ASC', self.ds.select("foo").sort("foo").to_sql())
        values = [row[0] for row in result.rows]
        self.assertEqual(["a", "b", "c"], values)

    def test_orderby_desc(self):
        """Test ORDER BY DESC"""
        result = self.ds.select(self.ds.foo).sort(self.ds.foo, ascending=False).execute()

        self.assertEqual(
            'SELECT "foo" FROM "abc" ORDER BY "foo" DESC',
            self.ds.select(self.ds.foo).sort(self.ds.foo, ascending=False).to_sql(),
        )
        values = [row[0] for row in result.rows]
        self.assertEqual(["c", "b", "a"], values)


class AliasTests(unittest.TestCase):
    """Alias tests with chdb execution"""

    def setUp(self):
        """Create test table"""
        self.ds = DataStore(table="abc")
        self.ds.connect()
        self.ds.create_table({"foo": "UInt32", "bar": "UInt32", "fiz": "String"}, drop_if_exists=True)
        self.ds.insert(
            [
                {"foo": 1, "bar": 2, "fiz": "test"},
            ]
        )

    def tearDown(self):
        """Clean up"""
        self.ds.close()

    def test_table_field(self):
        """Test field alias"""
        result = self.ds.select(self.ds.foo.as_("my_bar")).execute()

        self.assertEqual('SELECT "foo" AS "my_bar" FROM "abc"', self.ds.select(self.ds.foo.as_("my_bar")).to_sql())
        self.assertIn("my_bar", result.column_names)
        self.assertEqual(1, result.rows[0][0])

    def test_table_field__multi(self):
        """Test multiple field aliases"""
        result = self.ds.select(self.ds.foo.as_("alias1"), self.ds.fiz.as_("alias2")).execute()

        self.assertEqual(
            'SELECT "foo" AS "alias1", "fiz" AS "alias2" FROM "abc"',
            self.ds.select(self.ds.foo.as_("alias1"), self.ds.fiz.as_("alias2")).to_sql(),
        )
        self.assertIn("alias1", result.column_names)
        self.assertIn("alias2", result.column_names)

    def test_arithmetic_function(self):
        """Test alias on arithmetic expression"""
        result = self.ds.select((self.ds.foo + self.ds.bar).as_("sum")).execute()

        self.assertEqual(
            'SELECT ("foo"+"bar") AS "sum" FROM "abc"', self.ds.select((self.ds.foo + self.ds.bar).as_("sum")).to_sql()
        )
        self.assertIn("sum", result.column_names)
        self.assertEqual(3, result.rows[0][0])  # 1 + 2

    def test_functions_using_as(self):
        """Test function alias using as_()"""
        result = self.ds.select(Count("*").as_("total")).execute()

        # Verify SQL structure
        sql = self.ds.select(Count("*").as_("total")).to_sql()
        self.assertIn("COUNT", sql)
        self.assertIn("AS", sql)
        self.assertIn("total", sql)

        self.assertIn("total", result.column_names)
        self.assertEqual(1, result.rows[0][0])


if __name__ == '__main__':
    unittest.main()
