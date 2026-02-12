"""
Complex Aggregate Function Tests for DataStore

Tests complex aggregation scenarios with focus on verifying actual SQL execution results.
Includes:
- Statistical functions (stddev, variance, median)
- Distinct counting (uniq, count_distinct)
- Array aggregation (group_array, group_uniq_array)
- Conditional aggregates
- Complex aggregate expressions
- Multiple group by with aggregates
- Window-like patterns
- Real-world business scenarios
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field, F, Sum, Count, Avg, Min, Max
from datastore.expressions import Literal


# ========== Execution Tests with chDB ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestStatisticalAggregatesExecution(unittest.TestCase):
    """Test statistical aggregate function execution with result verification."""

    @classmethod
    def setUpClass(cls):
        """Create test table with numeric data for statistics."""
        cls.init_sql = """
        CREATE TABLE stat_data (
            id UInt32,
            category String,
            value Float64
        ) ENGINE = Memory;
        
        INSERT INTO stat_data VALUES
            (1, 'A', 10.0),
            (2, 'A', 20.0),
            (3, 'A', 30.0),
            (4, 'A', 40.0),
            (5, 'A', 50.0),
            (6, 'B', 100.0),
            (7, 'B', 200.0),
            (8, 'B', 300.0),
            (9, 'C', 5.0),
            (10, 'C', 15.0);
        """
        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.cleanup()

    def _execute(self, sql):
        sql_no_quotes = sql.replace('"', "")
        result = self.session.query(sql_no_quotes, "CSV")
        return result.bytes().decode("utf-8").strip()

    def test_avg_execution_verify_value(self):
        """Test AVG calculation with exact value verification."""
        ds = DataStore(table="stat_data")
        sql = ds.select(F.avg(ds.value).as_("avg")).filter(ds.category == "A").to_sql()
        result = self._execute(sql)
        # Average of 10,20,30,40,50 = 150/5 = 30
        self.assertEqual(float(result), 30.0)

    def test_stddev_pop_execution_verify_value(self):
        """Test stddevPop calculation with exact value verification."""
        ds = DataStore(table="stat_data")
        sql = (
            ds.select(F.stddev_pop(ds.value).as_("std"))
            .filter(ds.category == "A")
            .to_sql()
        )
        result = self._execute(sql)
        # Standard deviation of 10,20,30,40,50:
        # mean = 30, deviations = -20,-10,0,10,20
        # variance = (400+100+0+100+400)/5 = 200
        # stddev = sqrt(200) ≈ 14.142
        self.assertAlmostEqual(float(result), 14.142, places=2)

    def test_stddev_samp_execution_verify_value(self):
        """Test stddevSamp (sample std) calculation with exact value verification."""
        ds = DataStore(table="stat_data")
        sql = (
            ds.select(F.stddev_samp(ds.value).as_("std"))
            .filter(ds.category == "A")
            .to_sql()
        )
        result = self._execute(sql)
        # Sample std: sqrt((400+100+0+100+400)/(5-1)) = sqrt(250) ≈ 15.811
        self.assertAlmostEqual(float(result), 15.811, places=2)

    def test_var_pop_execution_verify_value(self):
        """Test varPop calculation with exact value verification."""
        ds = DataStore(table="stat_data")
        sql = (
            ds.select(F.var_pop(ds.value).as_("var"))
            .filter(ds.category == "A")
            .to_sql()
        )
        result = self._execute(sql)
        # Variance of 10,20,30,40,50 = 200
        self.assertEqual(float(result), 200.0)

    def test_var_samp_execution_verify_value(self):
        """Test varSamp (sample variance) calculation with exact value verification."""
        ds = DataStore(table="stat_data")
        sql = (
            ds.select(F.var_samp(ds.value).as_("var"))
            .filter(ds.category == "A")
            .to_sql()
        )
        result = self._execute(sql)
        # Sample variance = 1000/(5-1) = 250
        self.assertEqual(float(result), 250.0)

    def test_median_execution_verify_value(self):
        """Test median calculation with exact value verification."""
        ds = DataStore(table="stat_data")
        sql = (
            ds.select(F.median(ds.value).as_("med")).filter(ds.category == "A").to_sql()
        )
        result = self._execute(sql)
        # Median of 10,20,30,40,50 = 30
        self.assertAlmostEqual(float(result), 30.0, places=1)

    def test_min_max_range_verify_values(self):
        """Test MIN, MAX and range calculation."""
        ds = DataStore(table="stat_data")
        sql = (
            ds.select(
                F.min(ds.value).as_("min_val"),
                F.max(ds.value).as_("max_val"),
                (F.max(ds.value) - F.min(ds.value)).as_("range"),
            )
            .filter(ds.category == "B")
            .to_sql()
        )
        result = self._execute(sql)
        values = result.split(",")
        # B: min=100, max=300, range=200
        self.assertEqual(float(values[0]), 100.0)
        self.assertEqual(float(values[1]), 300.0)
        self.assertEqual(float(values[2]), 200.0)

    def test_statistics_by_category_verify_all_rows(self):
        """Test statistical aggregates grouped by category with all values verified."""
        ds = DataStore(table="stat_data")
        sql = (
            ds.select(
                "category",
                F.avg(ds.value).as_("mean"),
                F.min(ds.value).as_("min_val"),
                F.max(ds.value).as_("max_val"),
                Count("*").as_("count"),
            )
            .groupby("category")
            .sort("category")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Category A: avg=30, min=10, max=50, count=5
        a_values = lines[0].replace('"', "").split(",")
        self.assertEqual(a_values[0], "A")
        self.assertEqual(float(a_values[1]), 30.0)
        self.assertEqual(float(a_values[2]), 10.0)
        self.assertEqual(float(a_values[3]), 50.0)
        self.assertEqual(int(a_values[4]), 5)

        # Category B: avg=200, min=100, max=300, count=3
        b_values = lines[1].replace('"', "").split(",")
        self.assertEqual(b_values[0], "B")
        self.assertEqual(float(b_values[1]), 200.0)
        self.assertEqual(float(b_values[2]), 100.0)
        self.assertEqual(float(b_values[3]), 300.0)
        self.assertEqual(int(b_values[4]), 3)

        # Category C: avg=10, min=5, max=15, count=2
        c_values = lines[2].replace('"', "").split(",")
        self.assertEqual(c_values[0], "C")
        self.assertEqual(float(c_values[1]), 10.0)
        self.assertEqual(float(c_values[2]), 5.0)
        self.assertEqual(float(c_values[3]), 15.0)
        self.assertEqual(int(c_values[4]), 2)

    def test_sum_total_verify_value(self):
        """Test SUM with exact value verification."""
        ds = DataStore(table="stat_data")
        sql = ds.select(Sum(ds.value).as_("total")).to_sql()
        result = self._execute(sql)
        # Total: 10+20+30+40+50+100+200+300+5+15 = 770
        self.assertEqual(float(result), 770.0)

    def test_count_all_verify_value(self):
        """Test COUNT(*) with exact value verification."""
        ds = DataStore(table="stat_data")
        sql = ds.select(Count("*").as_("cnt")).to_sql()
        result = self._execute(sql)
        self.assertEqual(int(result), 10)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestDistinctCountExecution(unittest.TestCase):
    """Test distinct counting with result verification."""

    @classmethod
    def setUpClass(cls):
        """Create test table with repeated values."""
        cls.init_sql = """
        CREATE TABLE event_data (
            id UInt32,
            user_id UInt32,
            category String,
            action String
        ) ENGINE = Memory;
        
        INSERT INTO event_data VALUES
            (1, 101, 'page', 'view'),
            (2, 101, 'page', 'view'),
            (3, 101, 'page', 'click'),
            (4, 102, 'page', 'view'),
            (5, 102, 'button', 'click'),
            (6, 103, 'page', 'view'),
            (7, 103, 'page', 'scroll'),
            (8, 103, 'link', 'click'),
            (9, 104, 'page', 'view'),
            (10, 104, 'page', 'view');
        """
        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.cleanup()

    def _execute(self, sql):
        sql_no_quotes = sql.replace('"', "")
        result = self.session.query(sql_no_quotes, "CSV")
        return result.bytes().decode("utf-8").strip()

    def test_uniq_users_verify_count(self):
        """Test counting unique users with exact value."""
        ds = DataStore(table="event_data")
        sql = ds.select(F.uniq(ds.user_id).as_("unique_users")).to_sql()
        result = self._execute(sql)
        # 4 unique users: 101, 102, 103, 104
        self.assertEqual(int(result), 4)

    def test_uniq_categories_verify_count(self):
        """Test counting unique categories with exact value."""
        ds = DataStore(table="event_data")
        sql = ds.select(F.uniq(ds.category).as_("unique_categories")).to_sql()
        result = self._execute(sql)
        # 3 unique categories: page, button, link
        self.assertEqual(int(result), 3)

    def test_uniq_actions_verify_count(self):
        """Test counting unique actions with exact value."""
        ds = DataStore(table="event_data")
        sql = ds.select(F.uniq(ds.action).as_("unique_actions")).to_sql()
        result = self._execute(sql)
        # 3 unique actions: view, click, scroll
        self.assertEqual(int(result), 3)

    def test_count_distinct_alias_same_as_uniq(self):
        """Test count_distinct returns same result as uniq."""
        ds = DataStore(table="event_data")
        sql_uniq = ds.select(F.uniq(ds.user_id).as_("cnt")).to_sql()
        sql_count_distinct = ds.select(F.count_distinct(ds.user_id).as_("cnt")).to_sql()
        result_uniq = self._execute(sql_uniq)
        result_count_distinct = self._execute(sql_count_distinct)
        self.assertEqual(result_uniq, result_count_distinct)
        self.assertEqual(int(result_uniq), 4)

    def test_uniq_per_user_verify_all_values(self):
        """Test unique actions per user with all values verified."""
        ds = DataStore(table="event_data")
        sql = (
            ds.select(
                "user_id",
                F.uniq(ds.action).as_("unique_actions"),
                Count("*").as_("total_events"),
            )
            .groupby("user_id")
            .sort("user_id")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # user 101: 2 unique actions (view, click), 3 events
        u101 = lines[0].split(",")
        self.assertEqual(int(u101[0]), 101)
        self.assertEqual(int(u101[1]), 2)
        self.assertEqual(int(u101[2]), 3)

        # user 102: 2 unique actions (view, click), 2 events
        u102 = lines[1].split(",")
        self.assertEqual(int(u102[0]), 102)
        self.assertEqual(int(u102[1]), 2)
        self.assertEqual(int(u102[2]), 2)

        # user 103: 3 unique actions (view, scroll, click), 3 events
        u103 = lines[2].split(",")
        self.assertEqual(int(u103[0]), 103)
        self.assertEqual(int(u103[1]), 3)
        self.assertEqual(int(u103[2]), 3)

        # user 104: 1 unique action (view), 2 events
        u104 = lines[3].split(",")
        self.assertEqual(int(u104[0]), 104)
        self.assertEqual(int(u104[1]), 1)
        self.assertEqual(int(u104[2]), 2)

    def test_uniq_with_filter_verify_value(self):
        """Test uniq with filter condition."""
        ds = DataStore(table="event_data")
        sql = (
            ds.select(F.uniq(ds.user_id).as_("cnt"))
            .filter(ds.action == "click")
            .to_sql()
        )
        result = self._execute(sql)
        # Users who clicked: 101, 102, 103 = 3
        self.assertEqual(int(result), 3)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestArrayAggregationExecution(unittest.TestCase):
    """Test array aggregation with result verification."""

    @classmethod
    def setUpClass(cls):
        """Create test table."""
        cls.init_sql = """
        CREATE TABLE team_data (
            id UInt32,
            team String,
            member String,
            skill String
        ) ENGINE = Memory;
        
        INSERT INTO team_data VALUES
            (1, 'Alpha', 'Alice', 'Python'),
            (2, 'Alpha', 'Bob', 'Java'),
            (3, 'Alpha', 'Alice', 'SQL'),
            (4, 'Beta', 'Charlie', 'Python'),
            (5, 'Beta', 'David', 'Python'),
            (6, 'Beta', 'Eve', 'Go'),
            (7, 'Gamma', 'Frank', 'Rust');
        """
        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.cleanup()

    def _execute(self, sql):
        sql_no_quotes = sql.replace('"', "")
        result = self.session.query(sql_no_quotes, "CSV")
        return result.bytes().decode("utf-8").strip()

    def test_group_array_count_elements(self):
        """Test groupArray collects correct number of elements."""
        ds = DataStore(table="team_data")
        sql = (
            ds.select(
                "team",
                F.group_array(ds.member).as_("members"),
                Count("*").as_("count"),
            )
            .groupby("team")
            .sort("team")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Alpha: 3 rows (Alice, Bob, Alice)
        self.assertIn("Alpha", lines[0])
        self.assertIn("3", lines[0])
        self.assertIn("Alice", lines[0])
        self.assertIn("Bob", lines[0])

        # Beta: 3 rows (Charlie, David, Eve)
        self.assertIn("Beta", lines[1])
        self.assertIn("3", lines[1])

        # Gamma: 1 row (Frank)
        self.assertIn("Gamma", lines[2])
        self.assertIn("1", lines[2])
        self.assertIn("Frank", lines[2])

    def test_group_uniq_array_unique_elements(self):
        """Test groupUniqArray only collects unique elements."""
        ds = DataStore(table="team_data")
        sql = (
            ds.select(
                "team",
                F.group_uniq_array(ds.member).as_("unique_members"),
            )
            .groupby("team")
            .sort("team")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Alpha: unique members = [Alice, Bob] (2 unique)
        alpha_line = lines[0]
        self.assertIn("Alpha", alpha_line)
        # Alice should appear only once
        self.assertEqual(alpha_line.count("Alice"), 1)

    def test_group_uniq_array_skills_per_team(self):
        """Test groupUniqArray for unique skills per team."""
        ds = DataStore(table="team_data")
        sql = (
            ds.select(
                "team",
                F.group_uniq_array(ds.skill).as_("unique_skills"),
                F.uniq(ds.skill).as_("skill_count"),
            )
            .groupby("team")
            .sort("team")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Alpha: Python, Java, SQL = 3 unique skills
        alpha_vals = lines[0].split(",")
        self.assertEqual(int(alpha_vals[-1]), 3)

        # Beta: Python, Go = 2 unique skills (Python appears twice)
        beta_vals = lines[1].split(",")
        self.assertEqual(int(beta_vals[-1]), 2)

        # Gamma: Rust = 1 skill
        gamma_vals = lines[2].split(",")
        self.assertEqual(int(gamma_vals[-1]), 1)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestConditionalAggregatesExecution(unittest.TestCase):
    """Test conditional aggregates with result verification."""

    @classmethod
    def setUpClass(cls):
        """Create test table for conditional aggregates."""
        cls.session = chdb.session.Session()
        # Drop table if exists from previous run
        cls.session.query("DROP TABLE IF EXISTS sales_data")
        cls.init_sql = """
        CREATE TABLE sales_data (
            id UInt32,
            region String,
            category String,
            amount Float64,
            status String
        ) ENGINE = Memory;
        
        INSERT INTO sales_data VALUES
            (1, 'North', 'Electronics', 1000.0, 'completed'),
            (2, 'North', 'Books', 50.0, 'completed'),
            (3, 'North', 'Electronics', 500.0, 'pending'),
            (4, 'South', 'Electronics', 800.0, 'completed'),
            (5, 'South', 'Books', 100.0, 'completed'),
            (6, 'South', 'Clothing', 200.0, 'cancelled'),
            (7, 'East', 'Electronics', 1200.0, 'completed'),
            (8, 'East', 'Books', 75.0, 'pending'),
            (9, 'West', 'Electronics', 600.0, 'completed'),
            (10, 'West', 'Clothing', 300.0, 'completed');
        """
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.query("DROP TABLE IF EXISTS sales_data")
            cls.session.cleanup()

    def _execute(self, sql):
        sql_no_quotes = sql.replace('"', "")
        result = self.session.query(sql_no_quotes, "CSV")
        return result.bytes().decode("utf-8").strip()

    def test_conditional_sum_by_status_verify_exact_values(self):
        """Test conditional sum based on status with exact values."""
        ds = DataStore(table="sales_data")
        sql = ds.select(
            F.sum(F.if_(ds.status == "completed", ds.amount, 0)).as_("completed_total"),
            F.sum(F.if_(ds.status == "pending", ds.amount, 0)).as_("pending_total"),
            F.sum(F.if_(ds.status == "cancelled", ds.amount, 0)).as_("cancelled_total"),
        ).to_sql()
        result = self._execute(sql)
        values = result.split(",")

        # completed: 1000+50+800+100+1200+600+300 = 4050
        self.assertEqual(float(values[0]), 4050.0)
        # pending: 500+75 = 575
        self.assertEqual(float(values[1]), 575.0)
        # cancelled: 200
        self.assertEqual(float(values[2]), 200.0)

    def test_conditional_sum_by_category_verify_exact_values(self):
        """Test conditional sum by category with exact values for each region."""
        ds = DataStore(table="sales_data")
        sql = (
            ds.select(
                "region",
                F.sum(F.if_(ds.category == "Electronics", ds.amount, 0)).as_(
                    "electronics"
                ),
                F.sum(F.if_(ds.category == "Books", ds.amount, 0)).as_("books"),
                F.sum(F.if_(ds.category == "Clothing", ds.amount, 0)).as_("clothing"),
                F.sum(ds.amount).as_("total"),
            )
            .groupby("region")
            .sort("region")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # East: Electronics=1200, Books=75, Clothing=0, Total=1275
        east = lines[0].replace('"', "").split(",")
        self.assertEqual(east[0], "East")
        self.assertEqual(float(east[1]), 1200.0)
        self.assertEqual(float(east[2]), 75.0)
        self.assertEqual(float(east[3]), 0.0)
        self.assertEqual(float(east[4]), 1275.0)

        # North: Electronics=1500, Books=50, Clothing=0, Total=1550
        north = lines[1].replace('"', "").split(",")
        self.assertEqual(north[0], "North")
        self.assertEqual(float(north[1]), 1500.0)
        self.assertEqual(float(north[2]), 50.0)
        self.assertEqual(float(north[3]), 0.0)
        self.assertEqual(float(north[4]), 1550.0)

        # South: Electronics=800, Books=100, Clothing=200, Total=1100
        south = lines[2].replace('"', "").split(",")
        self.assertEqual(south[0], "South")
        self.assertEqual(float(south[1]), 800.0)
        self.assertEqual(float(south[2]), 100.0)
        self.assertEqual(float(south[3]), 200.0)
        self.assertEqual(float(south[4]), 1100.0)

        # West: Electronics=600, Books=0, Clothing=300, Total=900
        west = lines[3].replace('"', "").split(",")
        self.assertEqual(west[0], "West")
        self.assertEqual(float(west[1]), 600.0)
        self.assertEqual(float(west[2]), 0.0)
        self.assertEqual(float(west[3]), 300.0)
        self.assertEqual(float(west[4]), 900.0)

    def test_conditional_count_verify_exact_values(self):
        """Test conditional counting with exact values."""
        ds = DataStore(table="sales_data")
        sql = (
            ds.select(
                "region",
                F.sum(F.if_(ds.status == "completed", 1, 0)).as_("completed_count"),
                F.sum(F.if_(ds.status == "pending", 1, 0)).as_("pending_count"),
                F.sum(F.if_(ds.status == "cancelled", 1, 0)).as_("cancelled_count"),
                Count("*").as_("total_count"),
            )
            .groupby("region")
            .sort("region")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # East: completed=1, pending=1, cancelled=0, total=2
        east = lines[0].replace('"', "").split(",")
        self.assertEqual(int(east[1]), 1)
        self.assertEqual(int(east[2]), 1)
        self.assertEqual(int(east[3]), 0)
        self.assertEqual(int(east[4]), 2)

        # North: completed=2, pending=1, cancelled=0, total=3
        north = lines[1].replace('"', "").split(",")
        self.assertEqual(int(north[1]), 2)
        self.assertEqual(int(north[2]), 1)
        self.assertEqual(int(north[3]), 0)
        self.assertEqual(int(north[4]), 3)

        # South: completed=2, pending=0, cancelled=1, total=3
        south = lines[2].replace('"', "").split(",")
        self.assertEqual(int(south[1]), 2)
        self.assertEqual(int(south[2]), 0)
        self.assertEqual(int(south[3]), 1)
        self.assertEqual(int(south[4]), 3)

        # West: completed=2, pending=0, cancelled=0, total=2
        west = lines[3].replace('"', "").split(",")
        self.assertEqual(int(west[1]), 2)
        self.assertEqual(int(west[2]), 0)
        self.assertEqual(int(west[3]), 0)
        self.assertEqual(int(west[4]), 2)

    def test_conditional_avg_verify_values(self):
        """Test conditional average calculation."""
        ds = DataStore(table="sales_data")
        sql = ds.select(
            F.avg(F.if_(ds.status == "completed", ds.amount, None)).as_(
                "avg_completed"
            ),
        ).to_sql()
        result = self._execute(sql)
        # completed amounts: 1000,50,800,100,1200,600,300 = 4050/7 ≈ 578.57
        self.assertAlmostEqual(float(result), 578.57, places=1)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestComplexAggregateExpressionsExecution(unittest.TestCase):
    """Test complex aggregate expressions with result verification."""

    @classmethod
    def setUpClass(cls):
        """Create test table for complex expressions."""
        cls.init_sql = """
        CREATE TABLE financial_data (
            id UInt32,
            department String,
            revenue Float64,
            cost Float64,
            employees UInt32
        ) ENGINE = Memory;
        
        INSERT INTO financial_data VALUES
            (1, 'Sales', 100000.0, 60000.0, 10),
            (2, 'Sales', 80000.0, 50000.0, 8),
            (3, 'Engineering', 50000.0, 40000.0, 20),
            (4, 'Engineering', 60000.0, 45000.0, 25),
            (5, 'Marketing', 40000.0, 35000.0, 5),
            (6, 'Marketing', 30000.0, 25000.0, 4);
        """
        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.cleanup()

    def _execute(self, sql):
        sql_no_quotes = sql.replace('"', "")
        result = self.session.query(sql_no_quotes, "CSV")
        return result.bytes().decode("utf-8").strip()

    def test_profit_calculation_verify_exact_values(self):
        """Test profit calculation with exact values per department."""
        ds = DataStore(table="financial_data")
        sql = (
            ds.select(
                "department",
                Sum(ds.revenue).as_("total_revenue"),
                Sum(ds.cost).as_("total_cost"),
                (Sum(ds.revenue) - Sum(ds.cost)).as_("profit"),
            )
            .groupby("department")
            .sort("department")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Engineering: revenue=110000, cost=85000, profit=25000
        eng = lines[0].replace('"', "").split(",")
        self.assertEqual(eng[0], "Engineering")
        self.assertEqual(float(eng[1]), 110000.0)
        self.assertEqual(float(eng[2]), 85000.0)
        self.assertEqual(float(eng[3]), 25000.0)

        # Marketing: revenue=70000, cost=60000, profit=10000
        mkt = lines[1].replace('"', "").split(",")
        self.assertEqual(mkt[0], "Marketing")
        self.assertEqual(float(mkt[1]), 70000.0)
        self.assertEqual(float(mkt[2]), 60000.0)
        self.assertEqual(float(mkt[3]), 10000.0)

        # Sales: revenue=180000, cost=110000, profit=70000
        sales = lines[2].replace('"', "").split(",")
        self.assertEqual(sales[0], "Sales")
        self.assertEqual(float(sales[1]), 180000.0)
        self.assertEqual(float(sales[2]), 110000.0)
        self.assertEqual(float(sales[3]), 70000.0)

    def test_profit_margin_percentage_verify_values(self):
        """Test profit margin percentage calculation."""
        ds = DataStore(table="financial_data")
        sql = (
            ds.select(
                "department",
                F.round(
                    (Sum(ds.revenue) - Sum(ds.cost)) / Sum(ds.revenue) * Literal(100), 2
                ).as_("margin_pct"),
            )
            .groupby("department")
            .sort("department")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Engineering: margin = 25000/110000*100 = 22.73%
        eng = lines[0].replace('"', "").split(",")
        self.assertAlmostEqual(float(eng[1]), 22.73, places=1)

        # Marketing: margin = 10000/70000*100 = 14.29%
        mkt = lines[1].replace('"', "").split(",")
        self.assertAlmostEqual(float(mkt[1]), 14.29, places=1)

        # Sales: margin = 70000/180000*100 = 38.89%
        sales = lines[2].replace('"', "").split(",")
        self.assertAlmostEqual(float(sales[1]), 38.89, places=1)

    def test_revenue_per_employee_verify_values(self):
        """Test revenue per employee calculation."""
        ds = DataStore(table="financial_data")
        sql = (
            ds.select(
                "department",
                Sum(ds.employees).as_("total_employees"),
                (Sum(ds.revenue) / Sum(ds.employees)).as_("revenue_per_employee"),
            )
            .groupby("department")
            .sort("department")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Engineering: 45 employees, 110000/45 = 2444.44
        eng = lines[0].replace('"', "").split(",")
        self.assertEqual(int(eng[1]), 45)
        self.assertAlmostEqual(float(eng[2]), 2444.44, places=1)

        # Marketing: 9 employees, 70000/9 = 7777.78
        mkt = lines[1].replace('"', "").split(",")
        self.assertEqual(int(mkt[1]), 9)
        self.assertAlmostEqual(float(mkt[2]), 7777.78, places=1)

        # Sales: 18 employees, 180000/18 = 10000
        sales = lines[2].replace('"', "").split(",")
        self.assertEqual(int(sales[1]), 18)
        self.assertEqual(float(sales[2]), 10000.0)

    def test_cost_efficiency_ratio_verify_values(self):
        """Test cost efficiency (revenue/cost) ratio."""
        ds = DataStore(table="financial_data")
        sql = (
            ds.select(
                "department",
                F.round(Sum(ds.revenue) / Sum(ds.cost), 2).as_("efficiency_ratio"),
            )
            .groupby("department")
            .sort("department")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Engineering: 110000/85000 = 1.29
        eng = lines[0].replace('"', "").split(",")
        self.assertAlmostEqual(float(eng[1]), 1.29, places=2)

        # Marketing: 70000/60000 = 1.17
        mkt = lines[1].replace('"', "").split(",")
        self.assertAlmostEqual(float(mkt[1]), 1.17, places=2)

        # Sales: 180000/110000 = 1.64
        sales = lines[2].replace('"', "").split(",")
        self.assertAlmostEqual(float(sales[1]), 1.64, places=2)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestMultipleGroupByExecution(unittest.TestCase):
    """Test multiple column GROUP BY with result verification."""

    @classmethod
    def setUpClass(cls):
        """Create test table for multi-column grouping."""
        cls.init_sql = """
        CREATE TABLE order_data (
            id UInt32,
            year UInt16,
            month UInt8,
            region String,
            category String,
            amount Float64
        ) ENGINE = Memory;
        
        INSERT INTO order_data VALUES
            (1, 2023, 1, 'North', 'Electronics', 1000.0),
            (2, 2023, 1, 'North', 'Books', 200.0),
            (3, 2023, 1, 'South', 'Electronics', 800.0),
            (4, 2023, 2, 'North', 'Electronics', 1200.0),
            (5, 2023, 2, 'North', 'Books', 150.0),
            (6, 2023, 2, 'South', 'Electronics', 900.0),
            (7, 2024, 1, 'North', 'Electronics', 1500.0),
            (8, 2024, 1, 'South', 'Electronics', 1100.0),
            (9, 2024, 1, 'South', 'Books', 300.0);
        """
        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.cleanup()

    def _execute(self, sql):
        sql_no_quotes = sql.replace('"', "")
        result = self.session.query(sql_no_quotes, "CSV")
        return result.bytes().decode("utf-8").strip()

    def test_year_month_grouping_verify_values(self):
        """Test grouping by year and month with exact values."""
        ds = DataStore(table="order_data")
        sql = (
            ds.select(
                "year",
                "month",
                Sum(ds.amount).as_("total"),
                Count("*").as_("orders"),
            )
            .groupby("year", "month")
            .sort("year", "month")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # 2023-1: total=2000 (1000+200+800), orders=3
        ym_2023_1 = lines[0].split(",")
        self.assertEqual(int(ym_2023_1[0]), 2023)
        self.assertEqual(int(ym_2023_1[1]), 1)
        self.assertEqual(float(ym_2023_1[2]), 2000.0)
        self.assertEqual(int(ym_2023_1[3]), 3)

        # 2023-2: total=2250 (1200+150+900), orders=3
        ym_2023_2 = lines[1].split(",")
        self.assertEqual(int(ym_2023_2[0]), 2023)
        self.assertEqual(int(ym_2023_2[1]), 2)
        self.assertEqual(float(ym_2023_2[2]), 2250.0)
        self.assertEqual(int(ym_2023_2[3]), 3)

        # 2024-1: total=2900 (1500+1100+300), orders=3
        ym_2024_1 = lines[2].split(",")
        self.assertEqual(int(ym_2024_1[0]), 2024)
        self.assertEqual(int(ym_2024_1[1]), 1)
        self.assertEqual(float(ym_2024_1[2]), 2900.0)
        self.assertEqual(int(ym_2024_1[3]), 3)

    def test_year_region_grouping_verify_values(self):
        """Test grouping by year and region with exact values."""
        ds = DataStore(table="order_data")
        sql = (
            ds.select(
                "year",
                "region",
                Sum(ds.amount).as_("total"),
            )
            .groupby("year", "region")
            .sort("year", "region")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # 2023-North: 1000+200+1200+150 = 2550
        yr_2023_north = lines[0].replace('"', "").split(",")
        self.assertEqual(int(yr_2023_north[0]), 2023)
        self.assertEqual(yr_2023_north[1], "North")
        self.assertEqual(float(yr_2023_north[2]), 2550.0)

        # 2023-South: 800+900 = 1700
        yr_2023_south = lines[1].replace('"', "").split(",")
        self.assertEqual(int(yr_2023_south[0]), 2023)
        self.assertEqual(yr_2023_south[1], "South")
        self.assertEqual(float(yr_2023_south[2]), 1700.0)

        # 2024-North: 1500
        yr_2024_north = lines[2].replace('"', "").split(",")
        self.assertEqual(int(yr_2024_north[0]), 2024)
        self.assertEqual(yr_2024_north[1], "North")
        self.assertEqual(float(yr_2024_north[2]), 1500.0)

        # 2024-South: 1100+300 = 1400
        yr_2024_south = lines[3].replace('"', "").split(",")
        self.assertEqual(int(yr_2024_south[0]), 2024)
        self.assertEqual(yr_2024_south[1], "South")
        self.assertEqual(float(yr_2024_south[2]), 1400.0)

    def test_multi_groupby_with_having_verify_filtered_results(self):
        """Test multi-column groupby with HAVING filtering."""
        ds = DataStore(table="order_data")
        sql = (
            ds.select(
                "year",
                "region",
                Sum(ds.amount).as_("total"),
            )
            .groupby("year", "region")
            .having(Sum(ds.amount) > 1500)
            .sort("total", ascending=False)
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Only rows with total > 1500: 2023-North=2550, 2023-South=1700
        self.assertEqual(len(lines), 2)

        # First (highest): 2023-North = 2550
        first = lines[0].replace('"', "").split(",")
        self.assertEqual(int(first[0]), 2023)
        self.assertEqual(first[1], "North")
        self.assertEqual(float(first[2]), 2550.0)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestAnyAnyLastExecution(unittest.TestCase):
    """Test any/anyLast function execution with result verification."""

    @classmethod
    def setUpClass(cls):
        """Create test table."""
        cls.init_sql = """
        CREATE TABLE log_data (
            id UInt32,
            user_id UInt32,
            session_id String,
            page String
        ) ENGINE = Memory;
        
        INSERT INTO log_data VALUES
            (1, 101, 'sess_a', '/home'),
            (2, 101, 'sess_a', '/products'),
            (3, 101, 'sess_b', '/cart'),
            (4, 102, 'sess_c', '/home'),
            (5, 102, 'sess_c', '/checkout');
        """
        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.cleanup()

    def _execute(self, sql):
        sql_no_quotes = sql.replace('"', "")
        result = self.session.query(sql_no_quotes, "CSV")
        return result.bytes().decode("utf-8").strip()

    def test_any_returns_valid_value(self):
        """Test any() returns a valid value from the group."""
        ds = DataStore(table="log_data")
        sql = (
            ds.select(
                "user_id",
                F.any(ds.page).as_("sample_page"),
            )
            .groupby("user_id")
            .sort("user_id")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # User 101 pages: /home, /products, /cart - any one of them
        u101 = lines[0].replace('"', "").split(",")
        self.assertEqual(int(u101[0]), 101)
        self.assertIn(u101[1], ["/home", "/products", "/cart"])

        # User 102 pages: /home, /checkout - any one of them
        u102 = lines[1].replace('"', "").split(",")
        self.assertEqual(int(u102[0]), 102)
        self.assertIn(u102[1], ["/home", "/checkout"])

    def test_any_last_returns_last_inserted_value(self):
        """Test anyLast() returns the last inserted value."""
        ds = DataStore(table="log_data")
        sql = (
            ds.select(
                "user_id",
                F.any_last(ds.page).as_("last_page"),
            )
            .groupby("user_id")
            .sort("user_id")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # User 101 last page: /cart (id=3)
        u101 = lines[0].replace('"', "").split(",")
        self.assertEqual(int(u101[0]), 101)
        self.assertEqual(u101[1], "/cart")

        # User 102 last page: /checkout (id=5)
        u102 = lines[1].replace('"', "").split(",")
        self.assertEqual(int(u102[0]), 102)
        self.assertEqual(u102[1], "/checkout")

    def test_uniq_sessions_per_user_verify_values(self):
        """Test unique sessions per user."""
        ds = DataStore(table="log_data")
        sql = (
            ds.select(
                "user_id",
                F.uniq(ds.session_id).as_("session_count"),
            )
            .groupby("user_id")
            .sort("user_id")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # User 101: 2 sessions (sess_a, sess_b)
        u101 = lines[0].split(",")
        self.assertEqual(int(u101[0]), 101)
        self.assertEqual(int(u101[1]), 2)

        # User 102: 1 session (sess_c)
        u102 = lines[1].split(",")
        self.assertEqual(int(u102[0]), 102)
        self.assertEqual(int(u102[1]), 1)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestComplexHavingExecution(unittest.TestCase):
    """Test complex HAVING conditions with result verification."""

    @classmethod
    def setUpClass(cls):
        """Create test table."""
        cls.init_sql = """
        CREATE TABLE product_sales (
            id UInt32,
            category String,
            product String,
            quantity UInt32,
            price Float64
        ) ENGINE = Memory;
        
        INSERT INTO product_sales VALUES
            (1, 'Electronics', 'Phone', 100, 500.0),
            (2, 'Electronics', 'Laptop', 50, 1000.0),
            (3, 'Electronics', 'Tablet', 75, 400.0),
            (4, 'Books', 'Novel', 200, 15.0),
            (5, 'Books', 'Guide', 150, 30.0),
            (6, 'Books', 'Comic', 300, 10.0),
            (7, 'Clothing', 'Shirt', 80, 25.0),
            (8, 'Clothing', 'Pants', 60, 40.0);
        """
        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.cleanup()

    def _execute(self, sql):
        sql_no_quotes = sql.replace('"', "")
        result = self.session.query(sql_no_quotes, "CSV")
        return result.bytes().decode("utf-8").strip()

    def test_having_filters_correctly(self):
        """Test HAVING filters groups correctly."""
        ds = DataStore(table="product_sales")
        sql = (
            ds.select(
                "category",
                Sum(ds.quantity).as_("total_qty"),
                Avg(ds.price).as_("avg_price"),
            )
            .groupby("category")
            .having((Sum(ds.quantity) > 100) & (Avg(ds.price) > 20))
            .sort("category")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Electronics: qty=225 > 100 ✓, avg_price=633.33 > 20 ✓
        # Books: qty=650 > 100 ✓, avg_price=18.33 > 20 ✗
        # Clothing: qty=140 > 100 ✓, avg_price=32.5 > 20 ✓
        self.assertEqual(len(lines), 2)

        # Should have Clothing and Electronics
        categories = [line.replace('"', "").split(",")[0] for line in lines]
        self.assertIn("Clothing", categories)
        self.assertIn("Electronics", categories)
        self.assertNotIn("Books", categories)

    def test_having_with_revenue_calculation_verify_values(self):
        """Test HAVING with calculated revenue filters correctly."""
        ds = DataStore(table="product_sales")
        sql = (
            ds.select(
                "category",
                (Sum(ds.quantity * ds.price)).as_("revenue"),
            )
            .groupby("category")
            .having(Sum(ds.quantity * ds.price) > 10000)
            .sort("revenue", ascending=False)
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Electronics: 100*500 + 50*1000 + 75*400 = 130000 > 10000 ✓
        # Books: 200*15 + 150*30 + 300*10 = 10500 > 10000 ✓
        # Clothing: 80*25 + 60*40 = 4400 > 10000 ✗
        self.assertEqual(len(lines), 2)

        # First is Electronics (highest revenue)
        first = lines[0].replace('"', "").split(",")
        self.assertEqual(first[0], "Electronics")
        self.assertEqual(float(first[1]), 130000.0)

        # Second is Books
        second = lines[1].replace('"', "").split(",")
        self.assertEqual(second[0], "Books")
        self.assertEqual(float(second[1]), 10500.0)

    def test_having_with_count_verify_values(self):
        """Test HAVING with COUNT condition."""
        ds = DataStore(table="product_sales")
        sql = (
            ds.select(
                "category",
                Count("*").as_("product_count"),
                Sum(ds.quantity).as_("total_qty"),
            )
            .groupby("category")
            .having(Count("*") >= 3)
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Electronics: 3 products ✓, Books: 3 products ✓, Clothing: 2 products ✗
        self.assertEqual(len(lines), 2)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestRealWorldAggregateScenarios(unittest.TestCase):
    """Test real-world aggregate scenarios with comprehensive result verification."""

    @classmethod
    def setUpClass(cls):
        """Create comprehensive test data."""
        cls.init_sql = """
        CREATE TABLE ecommerce_orders (
            order_id UInt32,
            customer_id UInt32,
            product_id UInt32,
            category String,
            region String,
            quantity UInt32,
            unit_price Float64,
            discount Float64,
            order_date Date,
            status String
        ) ENGINE = Memory;
        
        INSERT INTO ecommerce_orders VALUES
            (1, 1001, 1, 'Electronics', 'North', 2, 500.0, 0.1, '2024-01-15', 'completed'),
            (2, 1001, 2, 'Electronics', 'North', 1, 1200.0, 0.15, '2024-01-20', 'completed'),
            (3, 1002, 3, 'Books', 'South', 5, 25.0, 0.0, '2024-01-18', 'completed'),
            (4, 1002, 4, 'Books', 'South', 3, 35.0, 0.05, '2024-01-25', 'pending'),
            (5, 1003, 1, 'Electronics', 'North', 1, 500.0, 0.1, '2024-02-01', 'completed'),
            (6, 1003, 5, 'Clothing', 'East', 4, 45.0, 0.0, '2024-02-05', 'completed'),
            (7, 1004, 6, 'Clothing', 'West', 2, 80.0, 0.2, '2024-02-10', 'cancelled'),
            (8, 1005, 2, 'Electronics', 'East', 1, 1200.0, 0.1, '2024-02-15', 'completed'),
            (9, 1005, 7, 'Books', 'East', 10, 15.0, 0.1, '2024-02-18', 'completed'),
            (10, 1006, 8, 'Electronics', 'South', 3, 300.0, 0.05, '2024-02-20', 'completed');
        """
        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.cleanup()

    def _execute(self, sql):
        sql_no_quotes = sql.replace('"', "")
        result = self.session.query(sql_no_quotes, "CSV")
        return result.bytes().decode("utf-8").strip()

    def test_total_revenue_verify_value(self):
        """Test total gross revenue calculation."""
        ds = DataStore(table="ecommerce_orders")
        sql = ds.select(
            Sum(ds.quantity * ds.unit_price).as_("gross_revenue"),
        ).to_sql()
        result = self._execute(sql)
        # 2*500 + 1*1200 + 5*25 + 3*35 + 1*500 + 4*45 + 2*80 + 1*1200 + 10*15 + 3*300
        # = 1000 + 1200 + 125 + 105 + 500 + 180 + 160 + 1200 + 150 + 900 = 5520
        self.assertEqual(float(result), 5520.0)

    def test_net_revenue_with_discount_verify_value(self):
        """Test net revenue after discount calculation."""
        ds = DataStore(table="ecommerce_orders")
        sql = (
            ds.select(
                Sum(ds.quantity * ds.unit_price * (Literal(1) - ds.discount)).as_(
                    "net_revenue"
                ),
            )
            .filter(ds.status == "completed")
            .to_sql()
        )
        result = self._execute(sql)
        # completed orders net revenue:
        # 2*500*0.9 + 1*1200*0.85 + 5*25*1.0 + 1*500*0.9 + 4*45*1.0 + 1*1200*0.9 + 10*15*0.9 + 3*300*0.95
        # = 900 + 1020 + 125 + 450 + 180 + 1080 + 135 + 855 = 4745
        self.assertEqual(float(result), 4745.0)

    def test_orders_by_status_verify_counts(self):
        """Test order count by status."""
        ds = DataStore(table="ecommerce_orders")
        sql = (
            ds.select(
                "status",
                Count("*").as_("order_count"),
            )
            .groupby("status")
            .sort("status")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        status_counts = {}
        for line in lines:
            parts = line.replace('"', "").split(",")
            status_counts[parts[0]] = int(parts[1])

        self.assertEqual(status_counts["cancelled"], 1)
        self.assertEqual(status_counts["completed"], 8)
        self.assertEqual(status_counts["pending"], 1)

    def test_unique_customers_by_category_verify_values(self):
        """Test unique customer count per category."""
        ds = DataStore(table="ecommerce_orders")
        sql = (
            ds.select(
                "category",
                F.uniq(ds.customer_id).as_("unique_customers"),
            )
            .filter(ds.status == "completed")
            .groupby("category")
            .sort("category")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Books: customers 1002 (completed), 1005 = 2 unique (1002 has pending too but it's filtered)
        # Actually 1002's completed order is id=3, so unique = 1002, 1005 = 2
        books = lines[0].replace('"', "").split(",")
        self.assertEqual(books[0], "Books")
        self.assertEqual(int(books[1]), 2)

        # Clothing: customer 1003 = 1 unique
        clothing = lines[1].replace('"', "").split(",")
        self.assertEqual(clothing[0], "Clothing")
        self.assertEqual(int(clothing[1]), 1)

        # Electronics: customers 1001, 1003, 1005, 1006 = 4 unique
        electronics = lines[2].replace('"', "").split(",")
        self.assertEqual(electronics[0], "Electronics")
        self.assertEqual(int(electronics[1]), 4)

    def test_regional_summary_verify_all_values(self):
        """Test regional sales summary with all values verified."""
        ds = DataStore(table="ecommerce_orders")
        sql = (
            ds.select(
                "region",
                Count("*").as_("orders"),
                Sum(ds.quantity).as_("units"),
                Sum(ds.quantity * ds.unit_price).as_("revenue"),
            )
            .filter(ds.status == "completed")
            .groupby("region")
            .sort("region")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # East: orders 3 (6,8,9), units=4+1+10=15, revenue=180+1200+150=1530
        east = lines[0].replace('"', "").split(",")
        self.assertEqual(east[0], "East")
        self.assertEqual(int(east[1]), 3)
        self.assertEqual(int(east[2]), 15)
        self.assertEqual(float(east[3]), 1530.0)

        # North: orders 3 (1,2,5), units=2+1+1=4, revenue=1000+1200+500=2700
        north = lines[1].replace('"', "").split(",")
        self.assertEqual(north[0], "North")
        self.assertEqual(int(north[1]), 3)
        self.assertEqual(int(north[2]), 4)
        self.assertEqual(float(north[3]), 2700.0)

        # South: orders 2 (3,10), units=5+3=8, revenue=125+900=1025
        south = lines[2].replace('"', "").split(",")
        self.assertEqual(south[0], "South")
        self.assertEqual(int(south[1]), 2)
        self.assertEqual(int(south[2]), 8)
        self.assertEqual(float(south[3]), 1025.0)

    def test_category_avg_discount_verify_values(self):
        """Test average discount by category."""
        ds = DataStore(table="ecommerce_orders")
        sql = (
            ds.select(
                "category",
                F.round(F.avg(ds.discount) * Literal(100), 1).as_("avg_discount_pct"),
            )
            .groupby("category")
            .sort("category")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # Books: (0 + 0.05 + 0.1) / 3 * 100 = 5%
        books = lines[0].replace('"', "").split(",")
        self.assertEqual(books[0], "Books")
        self.assertAlmostEqual(float(books[1]), 5.0, places=1)

        # Clothing: (0 + 0.2) / 2 * 100 = 10%
        clothing = lines[1].replace('"', "").split(",")
        self.assertEqual(clothing[0], "Clothing")
        self.assertAlmostEqual(float(clothing[1]), 10.0, places=1)

        # Electronics: (0.1 + 0.15 + 0.1 + 0.1 + 0.05) / 5 * 100 = 10%
        electronics = lines[2].replace('"', "").split(",")
        self.assertEqual(electronics[0], "Electronics")
        self.assertAlmostEqual(float(electronics[1]), 10.0, places=1)

    def test_monthly_trend_verify_values(self):
        """Test monthly trend with exact values."""
        ds = DataStore(table="ecommerce_orders")
        sql = (
            ds.select(
                F.month(ds.order_date).as_("month"),
                Count("*").as_("orders"),
                Sum(ds.quantity * ds.unit_price).as_("revenue"),
            )
            .filter(ds.status == "completed")
            .groupby(F.month(ds.order_date))
            .sort("month")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # January (completed): orders 1,2,3 = 3 orders, revenue=1000+1200+125=2325
        jan = lines[0].split(",")
        self.assertEqual(int(jan[0]), 1)
        self.assertEqual(int(jan[1]), 3)
        self.assertEqual(float(jan[2]), 2325.0)

        # February (completed): orders 5,6,8,9,10 = 5 orders, revenue=500+180+1200+150+900=2930
        feb = lines[1].split(",")
        self.assertEqual(int(feb[0]), 2)
        self.assertEqual(int(feb[1]), 5)
        self.assertEqual(float(feb[2]), 2930.0)

    def test_customer_purchase_frequency_verify_values(self):
        """Test customer purchase frequency."""
        ds = DataStore(table="ecommerce_orders")
        sql = (
            ds.select(
                "customer_id",
                Count("*").as_("order_count"),
                F.uniq(ds.category).as_("categories"),
                Sum(ds.quantity * ds.unit_price).as_("total_spend"),
            )
            .filter(ds.status == "completed")
            .groupby("customer_id")
            .sort("customer_id")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split("\n")

        # customer 1001: 2 orders (1,2), 1 category (Electronics), spend=1000+1200=2200
        c1001 = lines[0].split(",")
        self.assertEqual(int(c1001[0]), 1001)
        self.assertEqual(int(c1001[1]), 2)
        self.assertEqual(int(c1001[2]), 1)
        self.assertEqual(float(c1001[3]), 2200.0)

        # customer 1002: 1 order (3), 1 category (Books), spend=125
        c1002 = lines[1].split(",")
        self.assertEqual(int(c1002[0]), 1002)
        self.assertEqual(int(c1002[1]), 1)
        self.assertEqual(int(c1002[2]), 1)
        self.assertEqual(float(c1002[3]), 125.0)

        # customer 1003: 2 orders (5,6), 2 categories, spend=500+180=680
        c1003 = lines[2].split(",")
        self.assertEqual(int(c1003[0]), 1003)
        self.assertEqual(int(c1003[1]), 2)
        self.assertEqual(int(c1003[2]), 2)
        self.assertEqual(float(c1003[3]), 680.0)


if __name__ == "__main__":
    if CHDB_AVAILABLE:
        print("✅ chDB is available - running execution tests")
    else:
        print("⚠️  chDB not available - skipping execution tests")
        print("   Install with: pip install chdb")

    unittest.main()
