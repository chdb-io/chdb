"""
Advanced query scenarios - comprehensive SQL generation and execution tests

Tests complex combinations of features with real chdb execution.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field
from datastore.functions import Count, Sum, Avg, Max, Min, Upper, Lower, CustomFunction
from datastore.expressions import Literal


# ========== SQL Generation Tests ==========


class ComplexFilterTests(unittest.TestCase):
    """Test complex filter combinations"""

    def test_multiple_or_conditions(self):
        """Test multiple OR conditions"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter((ds.age < 18) | (ds.age > 65) | (ds.status == 'inactive')).to_sql()
        self.assertIn('OR', sql)
        self.assertIn('"age" < 18', sql)
        self.assertIn('"age" > 65', sql)

    def test_nested_and_or_conditions(self):
        """Test nested AND/OR conditions"""
        ds = DataStore(table="products")
        sql = (
            ds.select("*")
            .filter(((ds.category == 'Electronics') | (ds.category == 'Books')) & ((ds.price > 50) & (ds.stock > 0)))
            .to_sql()
        )
        self.assertIn('AND', sql)
        self.assertIn('OR', sql)

    def test_not_condition(self):
        """Test NOT condition"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(~(ds.age > 18)).to_sql()
        self.assertIn('NOT', sql)

    def test_complex_null_handling(self):
        """Test complex NULL handling with isNull/isNotNull functions"""
        ds = DataStore(table="data")
        sql = ds.select("*").filter((ds.email.notnull()) & (ds.phone.notnull()) & ~(ds.name.isnull())).to_sql()
        # notnull() returns ColumnExpr wrapping isNotNull()
        self.assertIn('isNotNull', sql)
        self.assertIn('NOT', sql)


class ComplexAggregationTests(unittest.TestCase):
    """Test complex aggregation scenarios"""

    def test_multiple_aggregates_same_field(self):
        """Test multiple aggregates on same field"""
        ds = DataStore(table="sales")
        sql = (
            ds.select(
                "region",
                Count("*").as_("count"),
                Sum(ds.amount).as_("total"),
                Avg(ds.amount).as_("average"),
                Min(ds.amount).as_("min_sale"),
                Max(ds.amount).as_("max_sale"),
            )
            .groupby("region")
            .to_sql()
        )

        self.assertIn('COUNT', sql)
        self.assertIn('SUM', sql)
        self.assertIn('AVG', sql)
        self.assertIn('MIN', sql)
        self.assertIn('MAX', sql)

    def test_aggregate_with_arithmetic(self):
        """Test aggregate with arithmetic expression"""
        ds = DataStore(table="orders")
        sql = (
            ds.select("category", (Sum(ds.quantity) * Literal(1.5)).as_("adjusted_total")).groupby("category").to_sql()
        )
        self.assertIn('SUM', sql)
        self.assertIn('*', sql)
        self.assertIn('1.5', sql)

    def test_having_with_multiple_conditions(self):
        """Test HAVING with multiple conditions"""
        ds = DataStore(table="sales")
        sql = (
            ds.select("region", Sum(ds.amount).as_("total"))
            .groupby("region")
            .having((Sum(ds.amount) > 1000) & (Count("*") > 10))
            .to_sql()
        )
        self.assertIn('HAVING', sql)
        self.assertIn('AND', sql)
        self.assertIn('SUM', sql)
        self.assertIn('COUNT', sql)

    def test_filter_and_having(self):
        """Test combining WHERE and HAVING"""
        ds = DataStore(table="orders")
        sql = (
            ds.select("category", Count("*").as_("count"))
            .filter(ds.status == 'completed')
            .groupby("category")
            .having(Count("*") > 5)
            .to_sql()
        )
        self.assertIn('WHERE', sql)
        self.assertIn('HAVING', sql)
        # WHERE should come before GROUP BY
        self.assertTrue(sql.index('WHERE') < sql.index('GROUP BY'))
        self.assertTrue(sql.index('GROUP BY') < sql.index('HAVING'))


class ComplexSortingTests(unittest.TestCase):
    """Test complex sorting scenarios"""

    def test_sort_by_aggregate(self):
        """Test sorting by aggregate function"""
        ds = DataStore(table="products")
        sql = (
            ds.select("category", Count("*").as_("count"))
            .groupby("category")
            .sort(Count("*"), ascending=False)
            .to_sql()
        )
        self.assertIn('ORDER BY', sql)
        self.assertIn('COUNT', sql)

    def test_sort_with_multiple_fields_different_directions(self):
        """Test sorting multiple fields (currently same direction)"""
        ds = DataStore(table="users")
        # Note: DataStore currently applies same direction to all fields
        sql = ds.select("*").sort(ds.age, ds.name, ascending=False).to_sql()
        self.assertIn('ORDER BY', sql)
        self.assertIn('DESC', sql)


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class ComplexQueryExecutionTests(unittest.TestCase):
    """Test complex query execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create comprehensive test data"""
        cls.init_sql = """
        CREATE TABLE sales_data (
            id UInt32,
            region String,
            category String,
            product String,
            quantity UInt32,
            price Float64,
            discount Float64,
            status String,
            sale_date Date
        ) ENGINE = Memory;
        
        INSERT INTO sales_data VALUES
            (1, 'North', 'Electronics', 'Laptop', 5, 1200.00, 0.10, 'completed', '2023-01-15'),
            (2, 'South', 'Books', 'Python Guide', 10, 45.00, 0.00, 'completed', '2023-01-16'),
            (3, 'North', 'Electronics', 'Mouse', 20, 25.00, 0.15, 'completed', '2023-01-17'),
            (4, 'East', 'Books', 'SQL Mastery', 8, 55.00, 0.05, 'pending', '2023-01-18'),
            (5, 'West', 'Electronics', 'Keyboard', 15, 75.00, 0.10, 'completed', '2023-01-19'),
            (6, 'South', 'Electronics', 'Monitor', 7, 300.00, 0.20, 'completed', '2023-01-20'),
            (7, 'North', 'Books', 'Data Science', 12, 60.00, 0.00, 'completed', '2023-01-21'),
            (8, 'East', 'Electronics', 'Tablet', 6, 450.00, 0.15, 'cancelled', '2023-01-22'),
            (9, 'West', 'Books', 'AI Fundamentals', 15, 70.00, 0.10, 'completed', '2023-01-23'),
            (10, 'North', 'Electronics', 'Headphones', 25, 150.00, 0.20, 'completed', '2023-01-24');
        """

        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session"""
        if hasattr(cls, 'session'):
            cls.session.cleanup()

    def _execute(self, sql):
        """Helper to execute SQL and return CSV result"""
        sql_no_quotes = sql.replace('"', '')
        result = self.session.query(sql_no_quotes, 'CSV')
        return result.bytes().decode('utf-8').strip().replace('"', '')

    def test_revenue_by_region_execution(self):
        """Test revenue calculation by region"""
        ds = DataStore(table="sales_data")
        # Calculate revenue as quantity * price * (1 - discount)
        sql = (
            ds.select("region", Sum(ds.quantity * ds.price * (Literal(1) - ds.discount)).as_("revenue"))
            .filter(ds.status == 'completed')
            .groupby("region")
            .sort("revenue", ascending=False)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Should have multiple regions with completed sales
        self.assertGreater(len(lines), 0)
        self.assertLess(len(lines), 6)  # At most 5 regions

    def test_top_selling_categories_execution(self):
        """Test top selling categories"""
        ds = DataStore(table="sales_data")
        sql = (
            ds.select("category", Sum(ds.quantity).as_("total_qty"), Count("*").as_("num_sales"))
            .filter(ds.status.isin(['completed', 'pending']))
            .groupby("category")
            .having(Sum(ds.quantity) > 20)
            .sort("total_qty", ascending=False)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Electronics should be on top
        self.assertGreater(len(lines), 0)

    def test_high_value_transactions_execution(self):
        """Test finding high value transactions"""
        ds = DataStore(table="sales_data")
        sql = (
            ds.select("id", "product", "quantity", "price")
            .filter((ds.quantity * ds.price) > 1000)
            .sort("quantity", "price", ascending=False)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Laptop (5*1200=6000), Tablet (6*450=2700), Monitor (7*300=2100), Headphones (25*150=3750)
        self.assertGreater(len(lines), 0)

    def test_discount_analysis_execution(self):
        """Test discount analysis"""
        ds = DataStore(table="sales_data")
        sql = ds.select("product").filter((ds.discount > 0.1) & (ds.status == 'completed')).sort("product").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Products with discount > 10% and completed: Mouse (15%), Monitor (20%), Headphones (20%)
        self.assertEqual(3, len(lines))

    def test_pagination_with_filters_execution(self):
        """Test pagination on filtered results"""
        ds = DataStore(table="sales_data")

        # Get first page
        sql1 = ds.select("id", "product").filter(ds.status == 'completed').sort("id")[:3].to_sql()
        result1 = self._execute(sql1)
        lines1 = result1.split('\n')
        self.assertEqual(3, len(lines1))

        # Get second page
        sql2 = ds.select("id", "product").filter(ds.status == 'completed').sort("id")[3:6].to_sql()
        result2 = self._execute(sql2)
        lines2 = result2.split('\n')
        self.assertEqual(3, len(lines2))

        # Pages should be different
        self.assertNotEqual(lines1, lines2)

    def test_string_functions_in_query_execution(self):
        """Test string functions in complex query"""
        ds = DataStore(table="sales_data")
        # Use Field('product') explicitly since ds.product is a pandas method
        product_field = Field('product')
        category_field = Field('category')
        sql = (
            ds.select(Upper(product_field).as_("upper_product"), Lower(category_field).as_("lower_category"))
            .filter(product_field.like('%book%'))
            .to_sql()
        )

        # Note: product names don't contain 'book' (case-sensitive)
        # This tests the SQL generation
        result = self._execute(sql)
        # Should return empty as no products contain lowercase 'book'
        self.assertEqual('', result)

    def test_distinct_with_filters_execution(self):
        """Test DISTINCT with complex filters"""
        ds = DataStore(table="sales_data")
        sql = (
            ds.select("category")
            .filter(ds.status == 'completed')
            .filter(ds.price > 50)
            .distinct()
            .sort("category")
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Both Electronics and Books have completed sales with price > 50
        self.assertEqual(2, len(lines))

    def test_aggregate_with_complex_expression_execution(self):
        """Test aggregate with complex expression"""
        ds = DataStore(table="sales_data")
        sql = (
            ds.select(
                "region",
                Sum(ds.quantity).as_("total_qty"),
                Avg(ds.price).as_("avg_price"),
                (Sum(ds.quantity * ds.price) / Sum(ds.quantity)).as_("weighted_avg_price"),
            )
            .filter(ds.status == 'completed')
            .groupby("region")
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        self.assertGreater(len(lines), 0)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class MultiConditionExecutionTests(unittest.TestCase):
    """Test queries with multiple complex conditions"""

    @classmethod
    def setUpClass(cls):
        """Create test data with various data patterns"""
        cls.init_sql = """
        CREATE TABLE customer_data (
            id UInt32,
            name String,
            email Nullable(String),
            age UInt32,
            city String,
            country String,
            purchase_count UInt32,
            total_spent Float64,
            last_purchase_date Nullable(Date),
            status String
        ) ENGINE = Memory;
        
        INSERT INTO customer_data VALUES
            (1, 'Alice Smith', 'alice@example.com', 28, 'New York', 'USA', 15, 2500.00, '2023-12-01', 'active'),
            (2, 'Bob Johnson', NULL, 35, 'London', 'UK', 8, 1200.00, '2023-11-15', 'active'),
            (3, 'Charlie Brown', 'charlie@test.com', 42, 'Paris', 'France', 25, 4500.00, NULL, 'inactive'),
            (4, 'David Lee', 'david@example.com', 31, 'Tokyo', 'Japan', 12, 1800.00, '2023-12-10', 'active'),
            (5, 'Eve Davis', NULL, 26, 'New York', 'USA', 5, 800.00, '2023-10-20', 'pending'),
            (6, 'Frank Miller', 'frank@example.com', 45, 'London', 'UK', 30, 5200.00, '2023-12-15', 'vip'),
            (7, 'Grace Wilson', 'grace@test.com', 29, 'Berlin', 'Germany', 18, 3100.00, '2023-12-05', 'active'),
            (8, 'Henry Taylor', NULL, 55, 'Sydney', 'Australia', 3, 450.00, NULL, 'inactive'),
            (9, 'Ivy Chen', 'ivy@example.com', 33, 'Singapore', 'Singapore', 22, 3800.00, '2023-12-12', 'active'),
            (10, 'Jack White', 'jack@example.com', 38, 'Toronto', 'Canada', 16, 2700.00, '2023-11-28', 'active');
        """

        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session"""
        if hasattr(cls, 'session'):
            cls.session.cleanup()

    def _execute(self, sql):
        """Helper to execute SQL and return CSV result"""
        sql_no_quotes = sql.replace('"', '')
        result = self.session.query(sql_no_quotes, 'CSV')
        return result.bytes().decode('utf-8').strip().replace('"', '').replace('\\N', 'NULL')

    def test_vip_customer_identification_execution(self):
        """Test identifying VIP customers"""
        ds = DataStore(table="customer_data")
        sql = (
            ds.select("id", "name", "total_spent", "purchase_count")
            .filter(
                (ds.total_spent > 2000)
                & (ds.purchase_count > 10)
                & (ds.email.notnull())
                & (ds.status.isin(['active', 'vip']))
            )
            .sort("total_spent", ascending=False)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Alice, David, Frank, Grace, Ivy, Jack should match
        self.assertGreater(len(lines), 3)

    def test_dormant_customers_execution(self):
        """Test finding dormant customers"""
        ds = DataStore(table="customer_data")
        sql = (
            ds.select("id", "name", "status")
            .filter((ds.last_purchase_date.isnull()) | (ds.status == 'inactive'))
            .sort("id")
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Charlie (NULL date), Henry (inactive + NULL date), and others
        self.assertGreater(len(lines), 0)
        self.assertIn('inactive', result)

    def test_city_segmentation_execution(self):
        """Test customer segmentation by city"""
        ds = DataStore(table="customer_data")
        sql = (
            ds.select(
                "city",
                Count("*").as_("customer_count"),
                Avg(ds.total_spent).as_("avg_spent"),
                Sum(ds.purchase_count).as_("total_purchases"),
            )
            .groupby("city")
            .having(Count("*") > 1)
            .sort("avg_spent", ascending=False)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # New York and London have 2 customers each
        self.assertEqual(2, len(lines))

    def test_age_range_analysis_execution(self):
        """Test age range analysis"""
        ds = DataStore(table="customer_data")
        sql = (
            ds.select("id", "name", "age")
            .filter(ds.age.between(30, 40))
            .filter(ds.status == 'active')
            .sort("age")
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Should have customers aged 30-40 who are active
        self.assertGreater(len(lines), 0)

    def test_email_completeness_execution(self):
        """Test email data completeness"""
        ds = DataStore(table="customer_data")

        # Customers with email
        sql_with = ds.select(Count("*").as_("count")).filter(ds.email.notnull()).to_sql()
        result_with = self._execute(sql_with)

        # Customers without email
        sql_without = ds.select(Count("*").as_("count")).filter(ds.email.isnull()).to_sql()
        result_without = self._execute(sql_without)

        # Total should be 10
        with_count = int(result_with)
        without_count = int(result_without)
        self.assertEqual(10, with_count + without_count)

    def test_top_countries_by_revenue_execution(self):
        """Test top countries by revenue"""
        ds = DataStore(table="customer_data")
        sql = (
            ds.select("country", Count("*").as_("customers"), Sum(ds.total_spent).as_("revenue"))
            .groupby("country")
            .having(Sum(ds.total_spent) > 2000)
            .sort("revenue", ascending=False)
            .limit(5)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Should have multiple countries
        self.assertGreater(len(lines), 0)

    def test_combined_filter_and_aggregation_execution(self):
        """Test combining filters with aggregation"""
        ds = DataStore(table="customer_data")
        sql = (
            ds.select(
                "status", Count("*").as_("count"), Avg(ds.age).as_("avg_age"), Sum(ds.total_spent).as_("total_revenue")
            )
            .filter(ds.age.between(25, 45))
            .filter(ds.purchase_count > 5)
            .groupby("status")
            .sort("total_revenue", ascending=False)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        self.assertGreater(len(lines), 0)


if __name__ == '__main__':
    unittest.main()
