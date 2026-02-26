"""
Real-world query scenarios - comprehensive tests with chdb execution

Tests realistic use cases combining multiple features.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field
from datastore.functions import Count, Sum, Avg, Max, Min, CustomFunction


# ========== SQL Generation Tests ==========


class ECommerceQueryTests(unittest.TestCase):
    """E-commerce query scenarios"""

    def test_product_search_query(self):
        """Test product search with multiple filters"""
        ds = DataStore(table="products")
        sql = (
            ds.select("id", "name", "price", "category")
            .filter(
                (ds.price.between(10, 100))
                & (ds.category.isin(['Electronics', 'Books']))
                & (ds.stock > 0)
                & (ds.name.like('%phone%'))
            )
            .sort("price")
            .limit(20)
            .to_sql()
        )

        self.assertIn('WHERE', sql)
        self.assertIn('BETWEEN', sql)
        self.assertIn('IN', sql)
        self.assertIn('LIKE', sql)
        self.assertIn('LIMIT 20', sql)

    def test_sales_summary_query(self):
        """Test sales summary with aggregation"""
        ds = DataStore(table="orders")
        sql = (
            ds.select(
                "category",
                Count("*").as_("order_count"),
                Sum(ds.amount).as_("total_sales"),
                Avg(ds.amount).as_("avg_order"),
            )
            .filter(ds.status.isin(['completed', 'shipped']))
            .groupby("category")
            .having(Count("*") > 10)
            .sort("total_sales", ascending=False)
            .to_sql()
        )

        self.assertIn('COUNT', sql)
        self.assertIn('SUM', sql)
        self.assertIn('AVG', sql)
        self.assertIn('GROUP BY', sql)
        self.assertIn('HAVING', sql)

    def test_user_activity_query(self):
        """Test user activity analysis"""
        ds = DataStore(table="users")
        sql = (
            ds.select("id", "username", "email", "last_login")
            .filter(ds.last_login.notnull())
            .filter(ds.status == 'active')
            .filter(ds.email.like('%@company.com'))
            .sort("last_login", ascending=False)
            .limit(100)
            .to_sql()
        )

        self.assertIn('isNotNull', sql)
        self.assertIn('LIKE', sql)
        self.assertIn('ORDER BY', sql)


class DataAnalysisQueryTests(unittest.TestCase):
    """Data analysis query scenarios"""

    def test_statistical_query(self):
        """Test statistical aggregation query"""
        ds = DataStore(table="measurements")
        sql = (
            ds.select(
                Count("*").as_("count"),
                Sum(ds.value).as_("total"),
                Avg(ds.value).as_("average"),
                Min(ds.value).as_("minimum"),
                Max(ds.value).as_("maximum"),
            )
            .filter(ds.value.notnull())
            .filter(ds.timestamp.between('2023-01-01', '2023-12-31'))
            .to_sql()
        )

        self.assertIn('COUNT', sql)
        self.assertIn('SUM', sql)
        self.assertIn('AVG', sql)
        self.assertIn('MIN', sql)
        self.assertIn('MAX', sql)

    def test_data_quality_check(self):
        """Test data quality checking query"""
        ds = DataStore(table="customer_data")
        sql = (
            ds.select("id", "name", "email", "phone")
            .filter(ds.email.isnull() | ds.phone.isnull() | ds.name.like('test%'))
            .sort("id")
            .to_sql()
        )

        self.assertIn('isNull', sql)
        self.assertIn('OR', sql)
        self.assertIn('LIKE', sql)

    def test_top_performers_query(self):
        """Test top performers query"""
        ds = DataStore(table="sales_reps")
        sql = (
            ds.select("name", Sum(ds.sales).as_("total_sales"), Count("*").as_("deal_count"))
            .groupby("name")
            .having(Sum(ds.sales) > 100000)
            .sort("total_sales", ascending=False)
            .limit(10)
            .to_sql()
        )

        self.assertIn('GROUP BY', sql)
        self.assertIn('HAVING', sql)
        self.assertIn('LIMIT 10', sql)


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class ECommerceExecutionTests(unittest.TestCase):
    """Real-world e-commerce query execution tests"""

    @classmethod
    def setUpClass(cls):
        """Create test tables with realistic data"""
        cls.session = chdb.session.Session()
        
        # Drop tables first to ensure clean state
        cls.session.query("DROP TABLE IF EXISTS products")
        cls.session.query("DROP TABLE IF EXISTS orders")
        
        cls.init_sql = """
        CREATE TABLE products (
            id UInt32,
            name String,
            category String,
            price Float64,
            stock UInt32
        ) ENGINE = Memory;
        
        INSERT INTO products VALUES
            (1, 'iPhone 14', 'Electronics', 899.99, 50),
            (2, 'Python Book', 'Books', 49.99, 100),
            (3, 'Samsung Phone', 'Electronics', 699.99, 30),
            (4, 'Java Book', 'Books', 39.99, 80),
            (5, 'Headphones', 'Electronics', 149.99, 0),
            (6, 'SQL Book', 'Books', 59.99, 60);
        
        CREATE TABLE orders (
            id UInt32,
            category String,
            amount Float64,
            status String
        ) ENGINE = Memory;
        
        INSERT INTO orders VALUES
            (1, 'Electronics', 899.99, 'completed'),
            (2, 'Books', 49.99, 'completed'),
            (3, 'Electronics', 699.99, 'shipped'),
            (4, 'Books', 39.99, 'completed'),
            (5, 'Electronics', 149.99, 'completed'),
            (6, 'Books', 59.99, 'shipped'),
            (7, 'Electronics', 899.99, 'completed'),
            (8, 'Electronics', 699.99, 'completed'),
            (9, 'Books', 49.99, 'completed'),
            (10, 'Books', 39.99, 'completed'),
            (11, 'Electronics', 149.99, 'shipped'),
            (12, 'Books', 59.99, 'completed');
        """

        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session and drop tables"""
        if hasattr(cls, 'session'):
            try:
                cls.session.query("DROP TABLE IF EXISTS products")
                cls.session.query("DROP TABLE IF EXISTS orders")
            except Exception:
                pass
            cls.session.cleanup()

    def _execute(self, sql):
        """Helper to execute SQL and return CSV result"""
        sql_no_quotes = sql.replace('"', '')
        result = self.session.query(sql_no_quotes, 'CSV')
        return result.bytes().decode('utf-8').strip().replace('"', '')

    def test_product_search_execution(self):
        """Test product search execution"""
        ds = DataStore(table="products")
        sql = (
            ds.select("id", "name", "price").filter((ds.price.between(40, 100)) & (ds.stock > 0)).sort("price").to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Products: Java Book (39.99), Python Book (49.99), SQL Book (59.99)
        # But price > 40, so only Python Book and SQL Book
        self.assertGreater(len(lines), 0)
        self.assertIn('49.99', result)
        self.assertIn('59.99', result)

    def test_sales_by_category_execution(self):
        """Test sales summary by category execution"""
        ds = DataStore(table="orders")
        sql = (
            ds.select("category", Count("*").as_("count"), Sum(ds.amount).as_("total"))
            .filter(ds.status.isin(['completed', 'shipped']))
            .groupby("category")
            .sort("total", ascending=False)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Should have 2 categories: Electronics and Books
        self.assertEqual(2, len(lines))

    def test_expensive_products_execution(self):
        """Test finding expensive products"""
        ds = DataStore(table="products")
        sql = ds.select("name", "price").filter(ds.price > 500).sort("price", ascending=False).to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # iPhone and Samsung Phone
        self.assertEqual(2, len(lines))
        self.assertIn('899.99', lines[0])  # iPhone (most expensive)

    def test_out_of_stock_products_execution(self):
        """Test finding out of stock products"""
        ds = DataStore(table="products")
        sql = ds.select("id", "name").filter(ds.stock == 0).to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Only Headphones is out of stock
        self.assertEqual(1, len(lines))
        self.assertIn('Headphones', result)

    def test_category_statistics_execution(self):
        """Test category statistics with HAVING"""
        ds = DataStore(table="orders")
        sql = ds.select("category", Count("*").as_("order_count")).groupby("category").having(Count("*") > 5).to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Both Electronics (6) and Books (6) have > 5 orders
        self.assertEqual(2, len(lines))
        self.assertIn('Electronics', result)
        self.assertIn('Books', result)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class DataAnalysisExecutionTests(unittest.TestCase):
    """Data analysis query execution tests"""

    @classmethod
    def setUpClass(cls):
        """Create analysis test data"""
        cls.session = chdb.session.Session()
        
        # Drop tables first to ensure clean state
        cls.session.query("DROP TABLE IF EXISTS metrics")
        
        cls.init_sql = """
        CREATE TABLE metrics (
            id UInt32,
            metric_name String,
            value Float64,
            timestamp Date
        ) ENGINE = Memory;
        
        INSERT INTO metrics VALUES
            (1, 'cpu_usage', 45.5, '2023-01-01'),
            (2, 'cpu_usage', 67.2, '2023-01-02'),
            (3, 'cpu_usage', 52.8, '2023-01-03'),
            (4, 'memory_usage', 72.1, '2023-01-01'),
            (5, 'memory_usage', 68.9, '2023-01-02'),
            (6, 'memory_usage', 75.3, '2023-01-03'),
            (7, 'disk_usage', 55.0, '2023-01-01'),
            (8, 'disk_usage', 58.2, '2023-01-02'),
            (9, 'disk_usage', 61.7, '2023-01-03');
        """

        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session and drop tables"""
        if hasattr(cls, 'session'):
            try:
                cls.session.query("DROP TABLE IF EXISTS metrics")
            except Exception:
                pass
            cls.session.cleanup()

    def _execute(self, sql):
        """Helper to execute SQL and return CSV result"""
        sql_no_quotes = sql.replace('"', '')
        result = self.session.query(sql_no_quotes, 'CSV')
        return result.bytes().decode('utf-8').strip().replace('"', '')

    def test_metric_summary_execution(self):
        """Test metric summary statistics"""
        ds = DataStore(table="metrics")
        sql = (
            ds.select(
                "metric_name",
                Count("*").as_("count"),
                Avg(ds.value).as_("avg"),
                Min(ds.value).as_("min"),
                Max(ds.value).as_("max"),
            )
            .groupby("metric_name")
            .sort("metric_name")
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # 3 metrics: cpu_usage, disk_usage, memory_usage
        self.assertEqual(3, len(lines))

    def test_high_usage_metrics_execution(self):
        """Test finding high usage metrics"""
        ds = DataStore(table="metrics")
        sql = (
            ds.select("metric_name", "value", "timestamp").filter(ds.value > 70).sort("value", ascending=False).to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Values > 70: 72.1, 75.3
        self.assertEqual(2, len(lines))

    def test_metric_aggregation_with_filter_execution(self):
        """Test aggregation with timestamp filter"""
        ds = DataStore(table="metrics")
        sql = (
            ds.select("metric_name", Avg(ds.value).as_("avg_value"))
            .filter(ds.timestamp.between('2023-01-01', '2023-01-02'))
            .groupby("metric_name")
            .having(Avg(ds.value) > 60)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # cpu_usage avg ~56, disk_usage avg ~56, memory_usage avg ~70
        # Only memory_usage > 60
        self.assertEqual(1, len(lines))
        self.assertIn('memory_usage', result)

    def test_distinct_metrics_execution(self):
        """Test distinct metric names"""
        ds = DataStore(table="metrics")
        sql = ds.select("metric_name").distinct().sort("metric_name").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # 3 unique metrics
        self.assertEqual(3, len(lines))
        self.assertEqual(['cpu_usage', 'disk_usage', 'memory_usage'], lines)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class ComplexQueryExecutionTests(unittest.TestCase):
    """Complex multi-condition query execution tests"""

    @classmethod
    def setUpClass(cls):
        """Create user activity data"""
        cls.session = chdb.session.Session()
        
        # Drop tables first to ensure clean state
        cls.session.query("DROP TABLE IF EXISTS user_activity")
        
        cls.init_sql = """
        CREATE TABLE user_activity (
            user_id UInt32,
            username String,
            email Nullable(String),
            age UInt32,
            city String,
            activity_score UInt32,
            last_active Date
        ) ENGINE = Memory;
        
        INSERT INTO user_activity VALUES
            (1, 'alice', 'alice@example.com', 25, 'NYC', 85, '2023-12-01'),
            (2, 'bob', NULL, 30, 'LA', 45, '2023-11-15'),
            (3, 'charlie', 'charlie@example.com', 35, 'NYC', 92, '2023-12-10'),
            (4, 'david', 'david@test.com', 28, 'Chicago', 67, '2023-12-05'),
            (5, 'eve', NULL, 42, 'NYC', 38, '2023-10-20'),
            (6, 'frank', 'frank@example.com', 31, 'LA', 78, '2023-12-08');
        """

        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session and drop tables"""
        if hasattr(cls, 'session'):
            try:
                cls.session.query("DROP TABLE IF EXISTS user_activity")
            except Exception:
                pass
            cls.session.cleanup()

    def _execute(self, sql):
        """Helper to execute SQL and return CSV result"""
        sql_no_quotes = sql.replace('"', '')
        result = self.session.query(sql_no_quotes, 'CSV')
        return result.bytes().decode('utf-8').strip().replace('"', '').replace('\\N', 'NULL')

    def test_active_users_in_city_execution(self):
        """Test finding active users in specific cities"""
        ds = DataStore(table="user_activity")
        sql = (
            ds.select("user_id", "username", "activity_score")
            .filter((ds.city.isin(['NYC', 'LA'])) & (ds.activity_score > 70) & (ds.email.notnull()))
            .sort("activity_score", ascending=False)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # charlie (NYC, 92), alice (NYC, 85), frank (LA, 78)
        self.assertEqual(3, len(lines))

    def test_user_segments_execution(self):
        """Test user segmentation"""
        ds = DataStore(table="user_activity")
        sql = (
            ds.select("city", Count("*").as_("user_count"), Avg(ds.activity_score).as_("avg_score"))
            .groupby("city")
            .having(Count("*") > 1)
            .sort("avg_score", ascending=False)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # NYC: 3 users, LA: 2 users, Chicago: 1 user (excluded)
        self.assertEqual(2, len(lines))

    def test_missing_email_users_execution(self):
        """Test finding users with missing emails"""
        ds = DataStore(table="user_activity")
        sql = ds.select("user_id", "username").filter(ds.email.isnull()).sort("user_id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # bob and eve have NULL emails
        self.assertEqual(2, len(lines))
        self.assertIn('bob', result)
        self.assertIn('eve', result)

    def test_pagination_execution(self):
        """Test query pagination with LIMIT and OFFSET"""
        ds = DataStore(table="user_activity")

        # Page 1
        sql1 = ds.select("user_id", "username").sort("user_id")[:3].to_sql()
        result1 = self._execute(sql1)
        lines1 = result1.split('\n')
        self.assertEqual(3, len(lines1))

        # Page 2
        sql2 = ds.select("user_id", "username").sort("user_id")[3:6].to_sql()
        result2 = self._execute(sql2)
        lines2 = result2.split('\n')
        self.assertEqual(3, len(lines2))

        # Results should not overlap
        self.assertNotEqual(lines1, lines2)


if __name__ == '__main__':
    unittest.main()
