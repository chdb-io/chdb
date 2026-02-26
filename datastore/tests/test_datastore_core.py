"""
Test core DataStore functionality - converted from pypika test_query.py and test_selects.py
"""

import os
import unittest
import pandas as pd
from datastore import DataStore, Field, Sum, Count


class TestDataStoreBasics(unittest.TestCase):
    """Test basic DataStore operations."""

    def setUp(self):
        """Set up test DataStore."""
        self.ds = DataStore(table="customers")

    def test_create_datastore(self):
        """Test creating a DataStore."""
        ds = DataStore(table="test_table")
        self.assertEqual("test_table", ds.table_name)

    def test_datastore_repr(self):
        """Test DataStore string representation (lazy mode)."""
        # Create DataStore without operations - should show basic info
        ds = DataStore(source_type="file", table="data")

        # Since no SQL state or lazy ops, repr should show basic info
        repr_str = repr(ds)
        self.assertIn("file", repr_str)

        # Test with actual data source
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'users.csv')
        if os.path.exists(dataset_path):
            ds2 = DataStore.from_file(dataset_path)
            # repr triggers execution and shows DataFrame
            repr_str2 = repr(ds2)
            # Should show data or column names
            self.assertTrue('name' in repr_str2 or 'DataFrame' in repr_str2 or 'user_id' in repr_str2)

    def test_empty_datastore_sql(self):
        """Test SQL generation for empty DataStore."""
        ds = DataStore(table="test")
        sql = ds.to_sql()
        self.assertEqual('SELECT * FROM "test"', sql)


class TestSelect(unittest.TestCase):
    """Test SELECT operations."""

    def setUp(self):
        self.ds = DataStore(table="customers")

    def test_select_star(self):
        """Test SELECT *."""
        sql = self.ds.to_sql()
        self.assertEqual('SELECT * FROM "customers"', sql)

    def test_select_single_field(self):
        """Test SELECT single field."""
        sql = self.ds.select("name").to_sql()
        self.assertEqual('SELECT "name" FROM "customers"', sql)

    def test_select_multiple_fields(self):
        """Test SELECT multiple fields."""
        sql = self.ds.select("name", "age", "city").to_sql()
        self.assertEqual('SELECT "name", "age", "city" FROM "customers"', sql)

    def test_select_with_field_objects(self):
        """Test SELECT with Field objects."""
        sql = self.ds.select(Field("name"), Field("age")).to_sql()
        self.assertEqual('SELECT "name", "age" FROM "customers"', sql)

    def test_select_with_alias(self):
        """Test SELECT with field alias."""
        sql = self.ds.select(Field("name", alias="customer_name")).to_sql()
        self.assertEqual('SELECT "name" AS "customer_name" FROM "customers"', sql)


class TestWhere(unittest.TestCase):
    """Test WHERE clause."""

    def setUp(self):
        self.ds = DataStore(table="customers")

    def test_filter_equal(self):
        """Test WHERE with equality."""
        sql = self.ds.filter(Field("age") == 18).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE "age" = 18', sql)

    def test_filter_greater_than(self):
        """Test WHERE with greater than."""
        sql = self.ds.filter(Field("age") > 18).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE "age" > 18', sql)

    def test_filter_less_than(self):
        """Test WHERE with less than."""
        sql = self.ds.filter(Field("price") < 100).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE "price" < 100', sql)

    def test_filter_and(self):
        """Test WHERE with AND condition."""
        sql = self.ds.filter((Field("age") > 18) & (Field("city") == "NYC")).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE ("age" > 18 AND "city" = \'NYC\')', sql)

    def test_filter_or(self):
        """Test WHERE with OR condition."""
        sql = self.ds.filter((Field("status") == "active") | (Field("status") == "trial")).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE ("status" = \'active\' OR "status" = \'trial\')', sql)

    def test_multiple_filter_calls(self):
        """Test multiple filter() calls (should AND them)."""
        sql = self.ds.filter(Field("age") > 18).filter(Field("city") == "NYC").to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE ("age" > 18 AND "city" = \'NYC\')', sql)


class TestWhereAlias(unittest.TestCase):
    """Test WHERE clause using where() method (alias for filter())."""

    def setUp(self):
        self.ds = DataStore(table="customers")

    def test_where_equal(self):
        """Test WHERE with equality using where()."""
        sql = self.ds.where(Field("age") == 18).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE "age" = 18', sql)

    def test_where_greater_than(self):
        """Test WHERE with greater than using where()."""
        sql = self.ds.where(Field("age") > 18).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE "age" > 18', sql)

    def test_where_less_than(self):
        """Test WHERE with less than using where()."""
        sql = self.ds.where(Field("price") < 100).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE "price" < 100', sql)

    def test_where_and(self):
        """Test WHERE with AND condition using where()."""
        sql = self.ds.where((Field("age") > 18) & (Field("city") == "NYC")).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE ("age" > 18 AND "city" = \'NYC\')', sql)

    def test_where_or(self):
        """Test WHERE with OR condition using where()."""
        sql = self.ds.where((Field("status") == "active") | (Field("status") == "trial")).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE ("status" = \'active\' OR "status" = \'trial\')', sql)

    def test_multiple_where_calls(self):
        """Test multiple where() calls (should AND them)."""
        sql = self.ds.where(Field("age") > 18).where(Field("city") == "NYC").to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE ("age" > 18 AND "city" = \'NYC\')', sql)

    def test_where_same_as_filter(self):
        """Test that where() produces the same SQL as filter()."""
        filter_sql = self.ds.filter(Field("age") > 18).to_sql()
        where_sql = self.ds.where(Field("age") > 18).to_sql()
        self.assertEqual(filter_sql, where_sql)

    def test_where_and_filter_combined(self):
        """Test that where() and filter() can be used together."""
        sql = self.ds.where(Field("age") > 18).filter(Field("city") == "NYC").to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE ("age" > 18 AND "city" = \'NYC\')', sql)


class TestDynamicFieldAccess(unittest.TestCase):
    """Test dynamic field access (ds.column_name)."""

    def setUp(self):
        self.ds = DataStore(table="customers")

    def test_dynamic_field(self):
        """Test accessing field via ds.field_name."""
        from datastore import ColumnExpr
        field = self.ds.age
        # ds.field_name now returns ColumnExpr that wraps a Field
        self.assertIsInstance(field, ColumnExpr)
        self.assertIsInstance(field._expr, Field)
        self.assertEqual("age", field._expr.name)

    def test_dynamic_field_in_condition(self):
        """Test using dynamic field in condition."""
        sql = self.ds.filter(self.ds.age > 18).to_sql()
        self.assertEqual('SELECT * FROM "customers" WHERE "age" > 18', sql)

    def test_dynamic_field_in_select(self):
        """Test using dynamic field in select."""
        sql = self.ds.select(self.ds.name, self.ds.age).to_sql()
        self.assertEqual('SELECT "name", "age" FROM "customers"', sql)


class TestGroupBy(unittest.TestCase):
    """Test GROUP BY operations."""

    def setUp(self):
        self.ds = DataStore(table="orders")

    def test_groupby_single_field(self):
        """Test GROUP BY single field."""
        sql = self.ds.groupby("customer_id").to_sql()
        self.assertEqual('SELECT * FROM "orders" GROUP BY "customer_id"', sql)

    def test_groupby_multiple_fields(self):
        """Test GROUP BY multiple fields."""
        sql = self.ds.groupby("customer_id", "status").to_sql()
        self.assertEqual('SELECT * FROM "orders" GROUP BY "customer_id", "status"', sql)

    def test_groupby_with_aggregate(self):
        """Test GROUP BY with aggregate function."""
        sql = self.ds.groupby("customer_id").select(Field("customer_id"), Sum(Field("amount"), alias="total")).to_sql()
        self.assertEqual('SELECT "customer_id", SUM("amount") AS "total" FROM "orders" GROUP BY "customer_id"', sql)


class TestOrderBy(unittest.TestCase):
    """Test ORDER BY operations."""

    def setUp(self):
        self.ds = DataStore(table="customers")

    def test_sort_single_field_asc(self):
        """Test ORDER BY single field ascending."""
        sql = self.ds.sort("name").to_sql()
        self.assertEqual('SELECT * FROM "customers" ORDER BY "name" ASC', sql)

    def test_sort_single_field_desc(self):
        """Test ORDER BY single field descending."""
        sql = self.ds.sort("name", ascending=False).to_sql()
        self.assertEqual('SELECT * FROM "customers" ORDER BY "name" DESC', sql)

    def test_sort_multiple_fields(self):
        """Test ORDER BY multiple fields."""
        sql = self.ds.sort("city", "name").to_sql()
        self.assertEqual('SELECT * FROM "customers" ORDER BY "city" ASC, "name" ASC', sql)


class TestLimitOffset(unittest.TestCase):
    """Test LIMIT and OFFSET."""

    def setUp(self):
        self.ds = DataStore(table="customers")

    def test_limit(self):
        """Test LIMIT clause."""
        sql = self.ds.limit(10).to_sql()
        self.assertEqual('SELECT * FROM "customers" LIMIT 10', sql)

    def test_offset(self):
        """Test OFFSET clause."""
        sql = self.ds.offset(20).to_sql()
        self.assertEqual('SELECT * FROM "customers" OFFSET 20', sql)

    def test_limit_and_offset(self):
        """Test LIMIT and OFFSET together."""
        sql = self.ds.limit(10).offset(20).to_sql()
        self.assertEqual('SELECT * FROM "customers" LIMIT 10 OFFSET 20', sql)


class TestChaining(unittest.TestCase):
    """Test method chaining."""

    def setUp(self):
        self.ds = DataStore(table="orders")

    def test_complex_chain(self):
        """Test complex method chain."""
        sql = (
            self.ds.select("customer_id", Sum(Field("amount"), alias="total"))
            .filter(Field("status") == "completed")
            .groupby("customer_id")
            .sort("total", ascending=False)
            .limit(10)
            .to_sql()
        )

        expected = (
            'SELECT "customer_id", SUM("amount") AS "total" FROM "orders" '
            'WHERE "status" = \'completed\' '
            'GROUP BY "customer_id" '
            'ORDER BY "total" DESC '
            'LIMIT 10'
        )
        self.assertEqual(expected, sql)


class TestImmutability(unittest.TestCase):
    """Test immutability of operations."""

    def test_select_immutable(self):
        """Test that select() doesn't modify original."""
        ds1 = DataStore(table="test")
        ds2 = ds1.select("name")

        self.assertNotEqual(id(ds1), id(ds2))
        self.assertEqual('SELECT * FROM "test"', ds1.to_sql())
        self.assertEqual('SELECT "name" FROM "test"', ds2.to_sql())

    def test_filter_immutable(self):
        """Test that filter() doesn't modify original."""
        ds1 = DataStore(table="test")
        ds2 = ds1.filter(Field("age") > 18)

        self.assertNotEqual(id(ds1), id(ds2))
        self.assertEqual('SELECT * FROM "test"', ds1.to_sql())
        self.assertEqual('SELECT * FROM "test" WHERE "age" > 18', ds2.to_sql())

    def test_chaining_immutable(self):
        """Test that chaining creates new instances."""
        ds1 = DataStore(table="test")
        ds2 = ds1.select("name")
        ds3 = ds2.filter(Field("age") > 18)

        # All three should be different objects
        self.assertNotEqual(id(ds1), id(ds2))
        self.assertNotEqual(id(ds2), id(ds3))
        self.assertNotEqual(id(ds1), id(ds3))

        # Original should be unchanged
        self.assertEqual('SELECT * FROM "test"', ds1.to_sql())


class TestExecAlias(unittest.TestCase):
    """Test exec() method as an alias for execute()."""

    def test_exec_method_exists(self):
        """Test that exec() method exists."""
        ds = DataStore(table="test")
        self.assertTrue(hasattr(ds, 'exec'))
        self.assertTrue(callable(getattr(ds, 'exec')))

    def test_exec_same_as_execute(self):
        """Test that exec() calls execute()."""
        # We can't easily test the full execution without a real database,
        # but we can verify the method exists and has the same signature
        import inspect

        ds = DataStore(table="test")

        exec_method = getattr(ds, 'exec')
        execute_method = getattr(ds, 'execute')

        # Both should have the same return type annotation
        exec_sig = inspect.signature(exec_method)
        execute_sig = inspect.signature(execute_method)

        self.assertEqual(exec_sig.return_annotation, execute_sig.return_annotation)


class TestPandasIndexSupport(unittest.TestCase):
    """Test support for pandas Index in __getitem__."""

    def setUp(self):
        """Set up test DataStore with real data."""
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'users.csv')
        self.ds = DataStore.from_file(dataset_path)

    def test_getitem_with_pandas_index(self):
        """Test ds[pd.Index(['col1', 'col2'])] selects columns."""
        # Create a pandas Index
        cols = pd.Index(['user_id', 'name'])
        result = self.ds[cols]
        
        # Should return a DataStore with selected columns
        self.assertIsInstance(result, DataStore)
        
        # to_df() applies lazy ops including column selection
        df = result.to_df()
        self.assertListEqual(list(df.columns), ['user_id', 'name'])

    def test_getitem_with_columns_from_another_datastore(self):
        """Test ds[other_ds.columns] selects matching columns."""
        # Get columns from the DataStore
        cols = self.ds.columns
        
        # Select subset of columns
        subset_cols = cols[:2]  # Get first 2 columns
        result = self.ds[subset_cols]
        
        # to_df() applies lazy ops including column selection
        df = result.to_df()
        self.assertEqual(len(df.columns), 2)

    def test_getitem_with_pandas_index_from_dataframe(self):
        """Test using columns from a pandas DataFrame."""
        # Create a small DataFrame
        pdf = pd.DataFrame({'user_id': [1], 'name': ['test']})
        
        # Use its columns to select from DataStore
        result = self.ds[pdf.columns]
        
        # to_df() applies lazy ops including column selection
        df = result.to_df()
        self.assertListEqual(list(df.columns), ['user_id', 'name'])

    def test_getitem_index_immutability(self):
        """Test that using Index doesn't modify original DataStore."""
        cols = pd.Index(['user_id', 'name'])
        original_sql = self.ds.to_sql()
        
        result = self.ds[cols]
        
        # Original should be unchanged
        self.assertEqual(original_sql, self.ds.to_sql())
        self.assertNotEqual(id(self.ds), id(result))


class TestDataStoreIteration(unittest.TestCase):
    """Test __iter__ method for iterating over column names."""

    def setUp(self):
        """Set up test DataStore with real data."""
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'users.csv')
        self.ds = DataStore.from_file(dataset_path)

    def test_iter_yields_column_names(self):
        """Test that iterating over DataStore yields column names."""
        columns_from_iter = list(self.ds)
        columns_property = list(self.ds.columns)
        
        self.assertEqual(columns_from_iter, columns_property)

    def test_iter_in_for_loop(self):
        """Test using DataStore in a for loop."""
        column_list = []
        for col in self.ds:
            column_list.append(col)
        
        self.assertEqual(column_list, list(self.ds.columns))

    def test_iter_with_list_comprehension(self):
        """Test using DataStore in list comprehension."""
        upper_cols = [col.upper() for col in self.ds]
        expected = [col.upper() for col in self.ds.columns]
        
        self.assertEqual(upper_cols, expected)

    def test_iter_matches_pandas_behavior(self):
        """Test that iteration matches pandas DataFrame behavior."""
        # Execute to get DataFrame
        df = self.ds.execute().to_df()
        
        # Iterating over DataFrame yields column names
        df_cols = list(df.columns)
        ds_cols = list(self.ds)
        
        self.assertEqual(ds_cols, df_cols)


if __name__ == '__main__':
    unittest.main()
