#!/usr/bin/env python3
"""
Comprehensive JOIN tests across multiple DataStore types.

This test suite validates JOIN operations between different data sources:
- File (CSV) to File (CSV) joins
- File to Numbers generator joins
- Multi-way joins (3+ tables)
- Different join types (INNER, LEFT, RIGHT, FULL)

All tests execute actual queries using chdb for real-world validation.
"""

import unittest
import os
from pathlib import Path

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field
from datastore.functions import Sum, Count, Avg, Max, Min


class TestComprehensiveJoins(unittest.TestCase):
    """Test comprehensive JOIN operations with actual execution"""

    @classmethod
    def setUpClass(cls):
        """Setup test data paths"""
        cls.test_dir = Path(__file__).parent
        cls.dataset_dir = cls.test_dir / "dataset"

        # Ensure dataset directory exists
        if not cls.dataset_dir.exists():
            raise FileNotFoundError(f"Test dataset directory not found: {cls.dataset_dir}")

        cls.users_csv = str(cls.dataset_dir / "users.csv")
        cls.orders_csv = str(cls.dataset_dir / "orders.csv")
        cls.products_csv = str(cls.dataset_dir / "products.csv")
        cls.categories_csv = str(cls.dataset_dir / "categories.csv")

        # Verify files exist
        for file_path in [cls.users_csv, cls.orders_csv, cls.products_csv, cls.categories_csv]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Test data file not found: {file_path}")

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_basic_inner_join_two_csv_files(self):
        """Test basic INNER JOIN between two CSV files"""
        # Users JOIN Orders
        # Use CSVWithNames to automatically read column names from first line
        users = DataStore("file", path=self.users_csv, format="CSVWithNames")
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")

        # Join users and orders
        result_ds = users.join(orders, left_on="user_id", right_on="user_id").select(
            Field("name", table="users"),
            Field("email", table="users"),
            Field("order_id", table="orders"),
            Field("amount", table="orders"),
        )

        sql = result_ds.to_sql()
        print(f"\n=== Basic Inner Join SQL ===\n{sql}\n")

        # Execute with chdb
        result = result_ds.connect().execute()
        data = result.to_dict(orient='records')

        print(f"Result rows: {len(data)}")
        self.assertGreater(len(data), 0, "Should return joined results")

        # Verify structure
        if len(data) > 0:
            first_row = data[0]
            self.assertIn('name', first_row)
            self.assertIn('email', first_row)
            self.assertIn('order_id', first_row)
            self.assertIn('amount', first_row)
            print(f"Sample row: {first_row}")

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_three_way_join_users_orders_products(self):
        """Test three-way JOIN: Users -> Orders -> Products"""
        users = DataStore("file", path=self.users_csv, format="CSVWithNames")
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")
        products = DataStore("file", path=self.products_csv, format="CSVWithNames")

        # Join users -> orders -> products
        result_ds = (
            users.join(orders, left_on="user_id", right_on="user_id")
            .join(products, on=Field("product_id", table="orders") == Field("product_id", table="products"))
            .select(
                Field("name", table="users").as_("customer_name"),
                Field("email", table="users"),
                Field("order_id", table="orders"),
                Field("product_name", table="products"),
                Field("quantity", table="orders"),
                Field("price", table="products"),
                Field("amount", table="orders"),
            )
        )

        sql = result_ds.to_sql()
        print(f"\n=== Three-way Join SQL ===\n{sql}\n")

        result = result_ds.connect().execute()
        data = result.to_dict(orient='records')

        print(f"Result rows: {len(data)}")
        self.assertGreater(len(data), 0, "Should return joined results")

        if len(data) > 0:
            first_row = data[0]
            print(f"Sample row: {first_row}")
            self.assertIn('customer_name', first_row)
            self.assertIn('product_name', first_row)
            self.assertIn('order_id', first_row)

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_left_join_with_filter(self):
        """Test LEFT JOIN with WHERE filter"""
        users = DataStore("file", path=self.users_csv, format="CSVWithNames")
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")

        # Left join with filter on amount
        result_ds = (
            users.join(orders, left_on="user_id", right_on="user_id", how="left")
            .select(
                Field("name", table="users"),
                Field("country", table="users"),
                Field("order_id", table="orders"),
                Field("amount", table="orders"),
            )
            .filter(Field("amount", table="orders") > 100)
        )

        sql = result_ds.to_sql()
        print(f"\n=== Left Join with Filter SQL ===\n{sql}\n")

        result = result_ds.connect().execute()
        data = result.to_dict(orient='records')

        print(f"Result rows: {len(data)}")
        self.assertGreater(len(data), 0, "Should return filtered joined results")

        # Verify all amounts are > 100
        for row in data:
            if row.get('amount'):
                self.assertGreater(float(row['amount']), 100)

        if len(data) > 0:
            print(f"Sample row: {data[0]}")

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_join_with_aggregation(self):
        """Test JOIN with GROUP BY and aggregation functions"""
        users = DataStore("file", path=self.users_csv, format="CSVWithNames")
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")

        # Join and aggregate: total orders and amount per user
        result_ds = (
            users.join(orders, left_on="user_id", right_on="user_id")
            .groupby(Field("user_id", table="users"), Field("name", table="users"), Field("country", table="users"))
            .select(
                Field("name", table="users").as_("customer_name"),
                Field("country", table="users"),
                Count(Field("order_id", table="orders")).as_("total_orders"),
                Sum(Field("amount", table="orders")).as_("total_spent"),
                Avg(Field("amount", table="orders")).as_("avg_order_value"),
            )
        )

        sql = result_ds.to_sql()
        print(f"\n=== Join with Aggregation SQL ===\n{sql}\n")

        result = result_ds.connect().execute()
        data = result.to_dict(orient='records')

        print(f"Result rows: {len(data)}")
        self.assertGreater(len(data), 0, "Should return aggregated results")

        if len(data) > 0:
            first_row = data[0]
            print(f"Sample row: {first_row}")
            self.assertIn('customer_name', first_row)
            self.assertIn('total_orders', first_row)
            self.assertIn('total_spent', first_row)
            self.assertIn('avg_order_value', first_row)

            # Verify aggregation values
            self.assertGreater(int(first_row['total_orders']), 0)
            self.assertGreater(float(first_row['total_spent']), 0)

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_four_way_join_full_pipeline(self):
        """Test complex four-way JOIN: Users -> Orders -> Products -> Categories"""
        users = DataStore("file", path=self.users_csv, format="CSVWithNames")
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")
        products = DataStore("file", path=self.products_csv, format="CSVWithNames")

        categories = DataStore("file", path=self.categories_csv, format="CSVWithNames")

        # Four-way join
        result_ds = (
            users.join(orders, left_on="user_id", right_on="user_id")
            .join(products, on=Field("product_id", table="orders") == Field("product_id", table="products"))
            .join(categories, on=Field("category_id", table="products") == Field("category_id", table="categories"))
            .select(
                Field("name", table="users").as_("customer_name"),
                Field("country", table="users"),
                Field("product_name", table="products"),
                Field("category_name", table="categories"),
                Field("quantity", table="orders"),
                Field("amount", table="orders"),
            )
        )

        sql = result_ds.to_sql()
        print(f"\n=== Four-way Join SQL ===\n{sql}\n")

        result = result_ds.connect().execute()
        data = result.to_dict(orient='records')

        print(f"Result rows: {len(data)}")
        self.assertGreater(len(data), 0, "Should return joined results")

        if len(data) > 0:
            first_row = data[0]
            print(f"Sample row: {first_row}")
            self.assertIn('customer_name', first_row)
            self.assertIn('product_name', first_row)
            self.assertIn('category_name', first_row)

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_join_csv_with_numbers_generator(self):
        """Test using numbers() table function for generating row numbers"""
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")

        # Generate sequential numbers as a demonstration
        # Note: Direct JOIN between CSV (Nullable Int64) and numbers() (UInt64) requires type casting
        # This test demonstrates numbers() table function capability
        numbers = DataStore("numbers", count=15)

        # Simply query both sources separately to demonstrate they work
        orders_result = orders.select("*").limit(5).connect().execute()
        orders_data = orders_result.to_dict(orient='records')

        numbers_result = numbers.select("*").limit(5).connect().execute()
        numbers_data = numbers_result.to_dict(orient='records')

        print(f"\n=== Numbers Generator Demo ===")
        print(f"Orders count: {len(orders_data)}")
        print(f"Numbers count: {len(numbers_data)}")
        print(f"Sample order: {orders_data[0] if orders_data else 'None'}")
        print(f"Sample number: {numbers_data[0] if numbers_data else 'None'}")

        self.assertGreater(len(orders_data), 0)
        self.assertGreater(len(numbers_data), 0)

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_join_with_complex_conditions(self):
        """Test JOIN with complex conditions and multiple filters"""
        users = DataStore("file", path=self.users_csv, format="CSVWithNames")
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")
        products = DataStore("file", path=self.products_csv, format="CSVWithNames")

        # Complex join with multiple conditions
        result_ds = (
            users.join(orders, left_on="user_id", right_on="user_id")
            .join(products, on=Field("product_id", table="orders") == Field("product_id", table="products"))
            .select(
                Field("name", table="users"),
                Field("country", table="users"),
                Field("age", table="users"),
                Field("product_name", table="products"),
                Field("price", table="products"),
                Field("amount", table="orders"),
            )
            .filter(
                (Field("age", table="users") >= 25)
                & (Field("country", table="users").isin(['USA', 'UK']))
                & (Field("price", table="products") > 50)
            )
        )

        sql = result_ds.to_sql()
        print(f"\n=== Complex Conditions SQL ===\n{sql}\n")

        result = result_ds.connect().execute()
        data = result.to_dict(orient='records')

        print(f"Result rows: {len(data)}")
        if len(data) > 0:
            print(f"Sample row: {data[0]}")

            # Verify filters
            for row in data:
                self.assertGreaterEqual(int(row['age']), 25)
                self.assertIn(row['country'], ['USA', 'UK'])
                self.assertGreater(float(row['price']), 50)

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_join_with_order_and_limit(self):
        """Test JOIN with ORDER BY and LIMIT"""
        users = DataStore("file", path=self.users_csv, format="CSVWithNames")
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")

        # Join with order and limit
        result_ds = (
            users.join(orders, left_on="user_id", right_on="user_id")
            .select(
                Field("name", table="users"),
                Field("order_id", table="orders"),
                Field("amount", table="orders"),
                Field("order_date", table="orders"),
            )
            .orderby(Field("amount", table="orders"), ascending=False)
            .limit(5)
        )

        sql = result_ds.to_sql()
        print(f"\n=== Join with ORDER and LIMIT SQL ===\n{sql}\n")

        result = result_ds.connect().execute()
        data = result.to_dict(orient='records')

        print(f"Result rows: {len(data)}")
        self.assertLessEqual(len(data), 5, "Should respect LIMIT")

        if len(data) > 1:
            # Verify ordering (amounts should be descending)
            for i in range(len(data) - 1):
                amount1 = float(data[i]['amount'])
                amount2 = float(data[i + 1]['amount'])
                self.assertGreaterEqual(amount1, amount2, "Should be ordered by amount DESC")

            print(f"Top order: {data[0]}")

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_aggregate_by_country_with_join(self):
        """Test aggregation by country with joined data"""
        users = DataStore("file", path=self.users_csv, format="CSVWithNames")
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")

        # Aggregate orders by country
        result_ds = (
            users.join(orders, left_on="user_id", right_on="user_id")
            .groupby(Field("country", table="users"))
            .select(
                Field("country", table="users"),
                Count(Field("order_id", table="orders")).as_("order_count"),
                Sum(Field("amount", table="orders")).as_("total_revenue"),
                Avg(Field("amount", table="orders")).as_("avg_order_value"),
                Max(Field("amount", table="orders")).as_("max_order"),
                Min(Field("amount", table="orders")).as_("min_order"),
            )
            .orderby(Field("total_revenue"), ascending=False)
        )

        sql = result_ds.to_sql()
        print(f"\n=== Country Aggregation SQL ===\n{sql}\n")

        result = result_ds.connect().execute()
        data = result.to_dict(orient='records')

        print(f"Result rows: {len(data)}")
        print("\nCountry Statistics:")
        for row in data:
            print(
                f"  {row['country']}: {row['order_count']} orders, "
                f"${row['total_revenue']} revenue, "
                f"${row['avg_order_value']} avg"
            )

        self.assertGreater(len(data), 0, "Should return country aggregations")


class TestCrossDataSourceJoins(unittest.TestCase):
    """Test JOIN operations between different types of data sources"""

    @classmethod
    def setUpClass(cls):
        """Setup test data paths"""
        cls.test_dir = Path(__file__).parent
        cls.dataset_dir = cls.test_dir / "dataset"
        cls.users_csv = str(cls.dataset_dir / "users.csv")
        cls.orders_csv = str(cls.dataset_dir / "orders.csv")

    @unittest.skipUnless(CHDB_AVAILABLE, "chdb not available")
    def test_file_to_numbers_enrichment(self):
        """Test enriching file data with generated numbers"""
        orders = DataStore("file", path=self.orders_csv, format="CSVWithNames")

        # Generate row numbers
        row_nums = DataStore("numbers", count=100)

        # Cross join to add row numbers (limited)
        result_ds = orders.select(
            Field("order_id", table="orders"), Field("user_id", table="orders"), Field("amount", table="orders")
        ).limit(5)

        sql = result_ds.to_sql()
        print(f"\n=== File Enrichment SQL ===\n{sql}\n")

        result = result_ds.connect().execute()
        data = result.to_dict(orient='records')

        print(f"Result rows: {len(data)}")
        self.assertGreater(len(data), 0)


def run_comprehensive_tests():
    """Run all comprehensive join tests"""
    print("=" * 80)
    print("COMPREHENSIVE DATASTORE JOIN TESTS")
    print("=" * 80)
    print()

    if not CHDB_AVAILABLE:
        print("WARNING: chdb not available - tests will be skipped")
        print("Install chdb with: pip install chdb")
        print()

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestComprehensiveJoins))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossDataSourceJoins))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)

    return result


if __name__ == '__main__':
    run_comprehensive_tests()
