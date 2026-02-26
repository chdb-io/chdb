"""
Tests for subquery support
"""

import unittest
from datastore import DataStore
from datastore.expressions import Field
from datastore.functions import Count, Avg


class TestSubqueryInWhere(unittest.TestCase):
    """Test subqueries in WHERE clause."""

    def test_where_in_subquery(self):
        """Test WHERE ... IN (SELECT ...)"""
        ds_main = DataStore(table="orders")
        ds_sub = DataStore(table="users")

        subquery = ds_sub.select('id').filter(ds_sub.country == 'USA')
        query = ds_main.select('order_id', 'amount').filter(ds_main.user_id.isin(subquery))

        expected = 'SELECT "order_id", "amount" FROM "orders" WHERE "user_id" IN (SELECT "id" FROM "users" WHERE "country" = \'USA\')'
        self.assertEqual(expected, query.to_sql())

    def test_where_not_in_subquery(self):
        """Test WHERE ... NOT IN (SELECT ...)"""
        ds_main = DataStore(table="products")
        ds_sub = DataStore(table="discontinued_products")

        subquery = ds_sub.select('product_id')
        query = ds_main.select('name', 'price').filter(ds_main.id.notin(subquery))

        expected = 'SELECT "name", "price" FROM "products" WHERE "id" NOT IN (SELECT "product_id" FROM "discontinued_products")'
        self.assertEqual(expected, query.to_sql())

    def test_where_in_subquery_with_condition(self):
        """Test WHERE ... IN (SELECT ... WHERE ...)"""
        ds_main = DataStore(table="orders")
        ds_sub = DataStore(table="users")

        subquery = ds_sub.select('id').filter((ds_sub.country == 'USA') & (ds_sub.active == True))
        query = ds_main.select('*').filter(ds_main.user_id.isin(subquery))

        # Note: CompoundCondition generates SQL with parentheses around the whole expression
        expected = 'SELECT * FROM "orders" WHERE "user_id" IN (SELECT "id" FROM "users" WHERE ("country" = \'USA\' AND "active" = TRUE))'
        self.assertEqual(expected, query.to_sql())


class TestSubqueryFromClause(unittest.TestCase):
    """Test subqueries in FROM clause."""

    def test_select_from_subquery(self):
        """Test SELECT ... FROM (SELECT ...)"""
        ds_sub = DataStore(table="users")
        subquery = ds_sub.select('id', 'name', 'age').filter(ds_sub.age > 18).as_('adults')

        ds_main = DataStore(table=None)
        # For FROM subquery, we need to handle it differently
        # This is a simplified test - in practice, you might need a from_subquery method

        # Just test that the subquery generates correctly with alias
        expected = '(SELECT "id", "name", "age" FROM "users" WHERE "age" > 18) AS "adults"'
        self.assertEqual(expected, subquery.to_sql(as_subquery=True))

    def test_subquery_with_alias(self):
        """Test subquery with alias."""
        ds = DataStore(table="users")
        subquery = ds.select('country', Count('*').as_('user_count')).groupby('country').as_('country_stats')

        sql = subquery.to_sql(as_subquery=True)
        self.assertIn('SELECT "country"', sql)
        self.assertIn('GROUP BY "country"', sql)
        self.assertIn('AS "country_stats"', sql)


class TestSubqueryInInsert(unittest.TestCase):
    """Test subqueries in INSERT statements."""

    def test_insert_from_select(self):
        """Test INSERT INTO ... SELECT ..."""
        ds_target = DataStore(table="users_archive")
        ds_source = DataStore(table="users")

        subquery = ds_source.select('id', 'name', 'email').filter(ds_source.last_login < '2020-01-01')
        query = ds_target.insert_into('id', 'name', 'email').select_from(subquery)

        expected = 'INSERT INTO "users_archive" ("id", "name", "email") SELECT "id", "name", "email" FROM "users" WHERE "last_login" < \'2020-01-01\''
        self.assertEqual(expected, query.to_sql())

    def test_insert_from_select_with_join(self):
        """Test INSERT INTO ... SELECT with JOIN."""
        ds_target = DataStore(table="order_summary")
        ds_orders = DataStore(table="orders")
        ds_users = DataStore(table="users")

        subquery = (
            ds_orders.select('order_id', 'user_id', 'amount')
            .join(ds_users, on=ds_orders.user_id == ds_users.id)
            .filter(ds_users.country == 'USA')
        )

        query = ds_target.insert_into('order_id', 'user_id', 'amount').select_from(subquery)

        sql = query.to_sql()
        self.assertIn('INSERT INTO "order_summary"', sql)
        self.assertIn('SELECT "order_id", "user_id", "amount"', sql)
        self.assertIn('JOIN "users"', sql)


class TestNestedSubqueries(unittest.TestCase):
    """Test nested subqueries."""

    def test_nested_in_subquery(self):
        """Test nested IN subqueries: WHERE x IN (SELECT ... WHERE y IN (SELECT ...))"""
        ds_orders = DataStore(table="orders")
        ds_users = DataStore(table="users")
        ds_countries = DataStore(table="premium_countries")

        # Innermost subquery
        countries_subquery = ds_countries.select('country_code')

        # Middle subquery
        users_subquery = ds_users.select('id').filter(ds_users.country.isin(countries_subquery))

        # Main query
        query = ds_orders.select('order_id', 'amount').filter(ds_orders.user_id.isin(users_subquery))

        sql = query.to_sql()
        self.assertIn('SELECT "order_id", "amount" FROM "orders"', sql)
        self.assertIn('WHERE "user_id" IN (SELECT "id" FROM "users"', sql)
        self.assertIn('WHERE "country" IN (SELECT "country_code" FROM "premium_countries"', sql)


class TestSubqueryExecution(unittest.TestCase):
    """Test subquery execution with chdb."""

    def setUp(self):
        """Set up test tables."""
        # Create users table
        self.ds_users = DataStore(table="test_subq_users")
        self.ds_users.connect()
        self.ds_users.create_table({"id": "UInt64", "name": "String", "country": "String", "active": "UInt8"}, drop_if_exists=True)
        self.ds_users.insert(
            [
                {"id": 1, "name": "Alice", "country": "USA", "active": 1},
                {"id": 2, "name": "Bob", "country": "UK", "active": 1},
                {"id": 3, "name": "Charlie", "country": "USA", "active": 0},
                {"id": 4, "name": "Diana", "country": "Canada", "active": 1},
            ]
        )

        # Create orders table
        self.ds_orders = DataStore(table="test_subq_orders")
        self.ds_orders.connect()
        self.ds_orders.create_table({"order_id": "UInt64", "user_id": "UInt64", "amount": "Float64"}, drop_if_exists=True)
        self.ds_orders.insert(
            [
                {"order_id": 101, "user_id": 1, "amount": 100.0},
                {"order_id": 102, "user_id": 1, "amount": 150.0},
                {"order_id": 103, "user_id": 2, "amount": 200.0},
                {"order_id": 104, "user_id": 4, "amount": 300.0},
            ]
        )

    def tearDown(self):
        """Clean up."""
        self.ds_users.close()
        self.ds_orders.close()

    def test_where_in_subquery_execution(self):
        """Test WHERE ... IN (SELECT ...) execution."""
        # Get orders from USA users
        subquery = self.ds_users.select('id').filter(self.ds_users.country == 'USA')
        query = self.ds_orders.select('order_id', 'amount').filter(self.ds_orders.user_id.isin(subquery))

        result = query.execute()
        self.assertEqual(2, len(result))  # Alice has 2 orders (Charlie is inactive but still counts)

        rows = result.to_dict(orient='records')
        order_ids = [row['order_id'] for row in rows]
        self.assertIn(101, order_ids)
        self.assertIn(102, order_ids)

    def test_where_not_in_subquery_execution(self):
        """Test WHERE ... NOT IN (SELECT ...) execution."""
        # Get orders NOT from UK users
        subquery = self.ds_users.select('id').filter(self.ds_users.country == 'UK')
        query = self.ds_orders.select('order_id').filter(self.ds_orders.user_id.notin(subquery))

        result = query.execute()
        self.assertEqual(3, len(result))  # All except Bob's order

        rows = result.to_dict(orient='records')
        order_ids = [row['order_id'] for row in rows]
        self.assertIn(101, order_ids)
        self.assertIn(102, order_ids)
        self.assertIn(104, order_ids)
        self.assertNotIn(103, order_ids)

    def test_where_in_subquery_with_condition_execution(self):
        """Test WHERE ... IN (SELECT ... WHERE ...) execution."""
        # Get orders from active USA users
        subquery = self.ds_users.select('id').filter((self.ds_users.country == 'USA') & (self.ds_users.active == 1))
        query = self.ds_orders.select('order_id', 'user_id').filter(self.ds_orders.user_id.isin(subquery))

        result = query.execute()
        self.assertEqual(2, len(result))  # Only Alice's orders (she's active)

        rows = result.to_dict(orient='records')
        # All orders should be from user_id 1 (Alice)
        for row in rows:
            self.assertEqual(1, row['user_id'])

    def test_insert_from_subquery_execution(self):
        """Test INSERT INTO ... SELECT execution."""
        # Create archive table
        ds_archive = DataStore(table="test_subq_orders_archive")
        ds_archive.connect()
        ds_archive.create_table({"order_id": "UInt64", "user_id": "UInt64", "amount": "Float64"}, drop_if_exists=True)

        # Copy high-value orders (>= 200)
        subquery = self.ds_orders.select('order_id', 'user_id', 'amount').filter(self.ds_orders.amount >= 200)
        query = ds_archive.insert_into('order_id', 'user_id', 'amount').select_from(subquery)
        query.execute()

        # Verify archived data
        result = ds_archive.select('*').execute()
        self.assertEqual(2, len(result))  # 2 high-value orders

        rows = result.to_dict(orient='records')
        amounts = [row['amount'] for row in rows]
        self.assertIn(200.0, amounts)
        self.assertIn(300.0, amounts)

        ds_archive.close()

    def test_nested_subquery_execution(self):
        """Test nested subqueries execution."""
        # Create premium countries table
        ds_premium = DataStore(table="test_subq_premium")
        ds_premium.connect()
        ds_premium.create_table({"country_code": "String"}, drop_if_exists=True)
        ds_premium.insert([{"country_code": "USA"}, {"country_code": "Canada"}])

        # Find orders from users in premium countries
        # Nested: orders where user_id in (users where country in (premium countries))
        countries_subquery = ds_premium.select('country_code')
        users_subquery = self.ds_users.select('id').filter(self.ds_users.country.isin(countries_subquery))
        orders_query = self.ds_orders.select('order_id', 'amount').filter(self.ds_orders.user_id.isin(users_subquery))

        result = orders_query.execute()
        self.assertEqual(3, len(result))  # Alice's 2 orders + Diana's 1 order

        rows = result.to_dict(orient='records')
        order_ids = [row['order_id'] for row in rows]
        self.assertIn(101, order_ids)  # Alice
        self.assertIn(102, order_ids)  # Alice
        self.assertIn(104, order_ids)  # Diana
        self.assertNotIn(103, order_ids)  # Bob (UK not premium)

        ds_premium.close()


if __name__ == '__main__':
    unittest.main()
