"""
Test JOIN operations - migrated from pypika test_joins.py

Tests various JOIN types with SQL generation and chdb execution.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field, JoinType


# ========== SQL Generation Tests ==========


class BasicJoinTests(unittest.TestCase):
    """Test basic JOIN operations"""

    def test_inner_join_with_on(self):
        """Test INNER JOIN with ON condition"""
        ds1 = DataStore(table="customers")
        ds2 = DataStore(table="orders")

        sql = (
            ds1.join(ds2, on=Field("id", table="customers") == Field("customer_id", table="orders"))
            .select("*")
            .to_sql()
        )

        self.assertIn('INNER JOIN', sql)
        self.assertIn('"orders"', sql)
        self.assertIn('ON', sql)

    def test_left_join(self):
        """Test LEFT JOIN"""
        ds1 = DataStore(table="customers")
        ds2 = DataStore(table="orders")

        sql = (
            ds1.join(ds2, on=Field("id", table="customers") == Field("customer_id", table="orders"), how='left')
            .select("*")
            .to_sql()
        )

        self.assertIn('LEFT JOIN', sql)

    def test_right_join(self):
        """Test RIGHT JOIN"""
        ds1 = DataStore(table="customers")
        ds2 = DataStore(table="orders")

        sql = (
            ds1.join(ds2, on=Field("id", table="customers") == Field("customer_id", table="orders"), how='right')
            .select("*")
            .to_sql()
        )

        self.assertIn('RIGHT JOIN', sql)

    def test_outer_join(self):
        """Test FULL OUTER JOIN"""
        ds1 = DataStore(table="customers")
        ds2 = DataStore(table="orders")

        sql = (
            ds1.join(ds2, on=Field("id", table="customers") == Field("customer_id", table="orders"), how='outer')
            .select("*")
            .to_sql()
        )

        self.assertIn('FULL OUTER JOIN', sql)

    def test_join_with_left_on_right_on(self):
        """Test JOIN with left_on and right_on parameters"""
        ds1 = DataStore(table="customers")
        ds2 = DataStore(table="orders")

        sql = ds1.join(ds2, left_on='id', right_on='customer_id').select("*").to_sql()

        self.assertIn('JOIN', sql)
        self.assertIn('ON', sql)
        self.assertIn('"customers"."id"', sql)
        self.assertIn('"orders"."customer_id"', sql)


class JoinWithSelectTests(unittest.TestCase):
    """Test JOIN with SELECT specifications"""

    def test_join_select_from_both_tables(self):
        """Test selecting columns from both joined tables"""
        ds1 = DataStore(table="customers")
        ds2 = DataStore(table="orders")

        customer_name = Field("name", table="customers")
        order_amount = Field("amount", table="orders")

        sql = ds1.join(ds2, left_on='id', right_on='customer_id').select(customer_name, order_amount).to_sql()

        self.assertIn('"customers"."name"', sql)
        self.assertIn('"orders"."amount"', sql)

    def test_join_with_where(self):
        """Test JOIN with WHERE clause"""
        ds1 = DataStore(table="customers")
        ds2 = DataStore(table="orders")

        sql = (
            ds1.join(ds2, left_on='id', right_on='customer_id')
            .select("*")
            .filter(Field("amount", table="orders") > 100)
            .to_sql()
        )

        self.assertIn('JOIN', sql)
        self.assertIn('WHERE', sql)
        # JOIN should come before WHERE
        self.assertTrue(sql.index('JOIN') < sql.index('WHERE'))

    def test_join_with_groupby(self):
        """Test JOIN with GROUP BY"""
        ds1 = DataStore(table="customers")
        ds2 = DataStore(table="orders")

        from datastore.functions import Count, Sum

        customer_name = Field("name", table="customers")
        order_amount = Field("amount", table="orders")

        sql = (
            ds1.join(ds2, left_on='id', right_on='customer_id')
            .select(customer_name, Count("*").as_("order_count"), Sum(order_amount).as_("total"))
            .groupby(customer_name)
            .to_sql()
        )

        self.assertIn('JOIN', sql)
        self.assertIn('GROUP BY', sql)


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class JoinExecutionTests(unittest.TestCase):
    """Test JOIN execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test tables"""
        cls.session = chdb.session.Session()
        
        # Drop tables first to ensure clean state
        cls.session.query("DROP TABLE IF EXISTS customers")
        cls.session.query("DROP TABLE IF EXISTS orders")
        
        cls.init_sql = """
        CREATE TABLE customers (
            id UInt32,
            name String,
            city String
        ) ENGINE = Memory;
        
        CREATE TABLE orders (
            order_id UInt32,
            customer_id UInt32,
            amount Float64,
            status String
        ) ENGINE = Memory;
        
        INSERT INTO customers VALUES
            (1, 'Alice', 'NYC'),
            (2, 'Bob', 'LA'),
            (3, 'Charlie', 'Chicago'),
            (4, 'David', 'NYC');
        
        INSERT INTO orders VALUES
            (101, 1, 100.50, 'completed'),
            (102, 1, 75.00, 'completed'),
            (103, 2, 150.00, 'pending'),
            (104, 3, 200.00, 'completed'),
            (105, 1, 50.25, 'cancelled'),
            (106, 2, 125.00, 'completed');
        """

        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session and drop tables"""
        if hasattr(cls, 'session'):
            try:
                cls.session.query("DROP TABLE IF EXISTS customers")
                cls.session.query("DROP TABLE IF EXISTS orders")
            except Exception:
                pass
            cls.session.cleanup()

    def _execute(self, sql):
        """Helper to execute SQL and return CSV result"""
        sql_no_quotes = sql.replace('"', '')
        result = self.session.query(sql_no_quotes, 'CSV')
        return result.bytes().decode('utf-8').strip().replace('"', '')

    def test_inner_join_execution(self):
        """Test INNER JOIN execution"""
        ds_customers = DataStore(table="customers")
        ds_orders = DataStore(table="orders")

        sql = (
            ds_customers.join(ds_orders, left_on='id', right_on='customer_id')
            .select(Field("name", table="customers"), Field("amount", table="orders"))
            .sort(Field("amount", table="orders"))
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Should have 6 rows (all orders matched)
        self.assertEqual(6, len(lines))

    def test_join_with_aggregation_execution(self):
        """Test JOIN with aggregation"""
        from datastore.functions import Count, Sum

        ds_customers = DataStore(table="customers")
        ds_orders = DataStore(table="orders")

        customer_name = Field("name", table="customers")
        order_amount = Field("amount", table="orders")

        sql = (
            ds_customers.join(ds_orders, left_on='id', right_on='customer_id')
            .select(customer_name, Count("*").as_("count"), Sum(order_amount).as_("total"))
            .groupby(customer_name)
            .sort(customer_name)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Alice: 3 orders, Bob: 2 orders, Charlie: 1 order
        self.assertEqual(3, len(lines))
        self.assertIn('Alice', result)

    def test_join_with_filter_execution(self):
        """Test JOIN with WHERE filter"""
        ds_customers = DataStore(table="customers")
        ds_orders = DataStore(table="orders")

        sql = (
            ds_customers.join(ds_orders, left_on='id', right_on='customer_id')
            .select(Field("name", table="customers"))
            .filter(Field("amount", table="orders") > 100)
            .distinct()
            .sort(Field("name", table="customers"))
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Bob (150), Charlie (200), Alice has one >100 but distinct
        self.assertGreater(len(lines), 0)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class MultipleJoinExecutionTests(unittest.TestCase):
    """Test multiple JOIN operations"""

    @classmethod
    def setUpClass(cls):
        """Create test tables for complex joins"""
        cls.session = chdb.session.Session()
        
        # Drop tables first to ensure clean state
        cls.session.query("DROP TABLE IF EXISTS users")
        cls.session.query("DROP TABLE IF EXISTS posts")
        cls.session.query("DROP TABLE IF EXISTS comments")
        
        cls.init_sql = """
        CREATE TABLE users (
            user_id UInt32,
            username String,
            email String
        ) ENGINE = Memory;
        
        CREATE TABLE posts (
            post_id UInt32,
            user_id UInt32,
            title String,
            views UInt32
        ) ENGINE = Memory;
        
        CREATE TABLE comments (
            comment_id UInt32,
            post_id UInt32,
            user_id UInt32,
            text String
        ) ENGINE = Memory;
        
        INSERT INTO users VALUES
            (1, 'alice', 'alice@test.com'),
            (2, 'bob', 'bob@test.com'),
            (3, 'charlie', 'charlie@test.com');
        
        INSERT INTO posts VALUES
            (101, 1, 'First Post', 100),
            (102, 1, 'Second Post', 50),
            (103, 2, 'Bob Post', 200);
        
        INSERT INTO comments VALUES
            (1001, 101, 2, 'Nice post!'),
            (1002, 101, 3, 'Great!'),
            (1003, 102, 2, 'Interesting'),
            (1004, 103, 1, 'Cool');
        """

        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session and drop tables"""
        if hasattr(cls, 'session'):
            try:
                cls.session.query("DROP TABLE IF EXISTS users")
                cls.session.query("DROP TABLE IF EXISTS posts")
                cls.session.query("DROP TABLE IF EXISTS comments")
            except Exception:
                pass
            cls.session.cleanup()

    def _execute(self, sql):
        """Helper to execute SQL and return CSV result"""
        sql_no_quotes = sql.replace('"', '')
        result = self.session.query(sql_no_quotes, 'CSV')
        return result.bytes().decode('utf-8').strip().replace('"', '')

    def test_join_users_posts_execution(self):
        """Test joining users and posts"""
        from datastore.functions import Count

        ds_users = DataStore(table="users")
        ds_posts = DataStore(table="posts")

        username_field = Field("username", table="users")

        sql = (
            ds_users.join(ds_posts, left_on='user_id', right_on='user_id')
            .select(username_field, Count("*").as_("post_count"))
            .groupby(username_field)
            .sort(username_field)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # alice: 2 posts, bob: 1 post
        self.assertEqual(2, len(lines))
        self.assertIn('alice', result)
        self.assertIn('bob', result)

    def test_left_join_execution(self):
        """Test LEFT JOIN includes unmatched rows"""
        ds_users = DataStore(table="users")
        ds_posts = DataStore(table="posts")

        from datastore.functions import Count

        username_field = Field("username", table="users")

        sql = (
            ds_users.join(ds_posts, left_on='user_id', right_on='user_id', how='left')
            .select(username_field, Count(Field("post_id", table="posts")).as_("post_count"))
            .groupby(username_field)
            .sort(username_field)
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # alice, bob, charlie (charlie has 0 posts)
        self.assertEqual(3, len(lines))


if __name__ == '__main__':
    unittest.main()
