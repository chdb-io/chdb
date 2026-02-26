"""
Test NULL condition operations - extended IS NULL/IS NOT NULL tests

Comprehensive NULL handling with various scenarios and chdb execution.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


# ========== SQL Generation Tests ==========


class NullBasicTests(unittest.TestCase):
    """Basic NULL tests"""

    def test_isnull_basic(self):
        """Test IS NULL"""
        cond = Field("email").isnull()
        self.assertEqual('"email" IS NULL', cond.to_sql())

    def test_notnull_basic(self):
        """Test IS NOT NULL"""
        cond = Field("email").notnull()
        self.assertEqual('"email" IS NOT NULL', cond.to_sql())

    def test_isnull_in_query(self):
        """Test isnull() in query - uses isNull function."""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.email.isnull()).to_sql()
        self.assertIn('isNull', sql)

    def test_notnull_in_query(self):
        """Test notnull() in query - uses isNotNull function."""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.phone.notnull()).to_sql()
        self.assertIn('isNotNull', sql)


class NullWithCombinationsTests(unittest.TestCase):
    """Test NULL combined with other conditions"""

    def test_null_and_other_condition(self):
        """Test isNull() AND another condition"""
        ds = DataStore(table="data")
        sql = ds.select("*").filter(ds.email.isnull() & (ds.status == 'active')).to_sql()
        self.assertIn('isNull', sql)
        self.assertIn('AND', sql)

    def test_null_or_other_condition(self):
        """Test isNull() OR another condition"""
        ds = DataStore(table="data")
        sql = ds.select("*").filter(ds.email.isnull() | ds.phone.isnull()).to_sql()
        self.assertIn('isNull', sql)
        self.assertIn('OR', sql)

    def test_not_null_and_in(self):
        """Test isNotNull() combined with IN"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.email.notnull() & ds.status.isin(['active', 'premium'])).to_sql()
        self.assertIn('isNotNull', sql)
        self.assertIn('IN', sql)

    def test_not_null_and_like(self):
        """Test isNotNull() combined with LIKE"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.email.notnull() & ds.email.like('%@company.com')).to_sql()
        self.assertIn('isNotNull', sql)
        self.assertIn('LIKE', sql)


class NullNegationTests(unittest.TestCase):
    """Test NOT operator with NULL conditions"""

    def test_not_isnull(self):
        """Test NOT (isNull()) - equivalent to isNotNull()"""
        ds = DataStore(table="data")
        sql = ds.select("*").filter(~ds.email.isnull()).to_sql()
        self.assertIn('NOT', sql)
        self.assertIn('isNull', sql)

    def test_not_notnull(self):
        """Test NOT (isNotNull()) - equivalent to isNull()"""
        ds = DataStore(table="data")
        sql = ds.select("*").filter(~ds.email.notnull()).to_sql()
        self.assertIn('NOT', sql)
        self.assertIn('isNotNull', sql)


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class NullExecutionTests(unittest.TestCase):
    """Test NULL condition execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table with NULL values"""
        cls.init_sql = """
        CREATE TABLE test_null_exec (
            id UInt32,
            name String,
            email Nullable(String),
            phone Nullable(String),
            address Nullable(String),
            status String
        ) ENGINE = Memory;
        
        INSERT INTO test_null_exec VALUES
            (1, 'Alice', 'alice@test.com', '555-1234', 'NYC', 'active'),
            (2, 'Bob', NULL, '555-5678', NULL, 'active'),
            (3, 'Charlie', 'charlie@test.com', NULL, 'LA', 'inactive'),
            (4, 'David', NULL, NULL, NULL, 'pending'),
            (5, 'Eve', 'eve@test.com', '555-9999', 'Chicago', 'active'),
            (6, 'Frank', NULL, '555-3333', 'NYC', 'deleted');
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

    def test_isnull_email_execution(self):
        """Test finding records with NULL email"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id", "name").filter(ds.email.isnull()).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Bob, David, Frank have NULL emails
        self.assertEqual(['2,Bob', '4,David', '6,Frank'], lines)

    def test_notnull_email_execution(self):
        """Test finding records with non-NULL email"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id").filter(ds.email.notnull()).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Alice, Charlie, Eve have emails
        self.assertEqual(['1', '3', '5'], lines)

    def test_null_or_null_execution(self):
        """Test OR of multiple NULL checks"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id").filter(ds.email.isnull() | ds.phone.isnull()).sort("id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Bob (NULL email), Charlie (NULL phone), David (both NULL), Frank (NULL email)
        self.assertEqual(['2', '3', '4', '6'], lines)

    def test_all_fields_null_execution(self):
        """Test finding records where all optional fields are NULL"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id", "name").filter(ds.email.isnull() & ds.phone.isnull() & ds.address.isnull()).to_sql()

        result = self._execute(sql)
        # Only David has all three fields NULL
        self.assertEqual('4,David', result)

    def test_null_and_status_execution(self):
        """Test NULL check combined with status filter"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id", "name").filter(ds.email.notnull() & (ds.status == 'active')).sort("id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Alice, Eve have email and are active
        self.assertEqual(['1,Alice', '5,Eve'], lines)

    def test_not_null_negation_execution(self):
        """Test NOT (IS NULL) execution"""
        ds = DataStore(table="test_null_exec")
        sql = ds.select("id").filter(~ds.email.isnull()).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Same as notnull: 1, 3, 5
        self.assertEqual(['1', '3', '5'], lines)


# ========== None Comparison Tests (pandas semantics) ==========


class NoneComparisonTests(unittest.TestCase):
    """
    Test == None and != None comparison operators.

    In pandas, element-wise comparison with Python's None singleton:
    - col == None returns False for ALL rows (no element equals None singleton)
    - col != None returns True for ALL rows (every element differs from None singleton)

    This is DIFFERENT from .isna()/.notna() which check for NA/NaN/None values.

    DataStore must match this pandas behavior for compatibility.
    """

    def test_ne_none_returns_all_rows(self):
        """ds['col'] != None should return ALL rows (like pandas)."""
        import pandas as pd

        df = pd.DataFrame({'s': ['abc', 'def', None, 'xyz']})
        ds = DataStore(df)

        pd_result = df[df['s'] != None]  # noqa: E711
        ds_result = ds[ds['s'] != None]  # noqa: E711

        # Use complete comparison instead of length-only check
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eq_none_returns_no_rows(self):
        """ds['col'] == None should return NO rows (like pandas)."""
        import pandas as pd

        df = pd.DataFrame({'s': ['abc', 'def', None, 'xyz']})
        ds = DataStore(df)

        pd_result = df[df['s'] == None]  # noqa: E711
        ds_result = ds[ds['s'] == None]  # noqa: E711

        # Use complete comparison instead of length-only check
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ne_none_mask_values(self):
        """ds['col'] != None should produce all True values."""
        import pandas as pd

        df = pd.DataFrame({'s': ['abc', 'def', None]})
        ds = DataStore(df)

        pd_mask = df['s'] != None  # noqa: E711
        ds_mask = ds['s'] != None  # noqa: E711

        # Duck Typing: use assert_series_equal to compare mask values
        # This triggers execution implicitly via .values access
        assert_series_equal(ds_mask, pd_mask, check_names=False)

    def test_eq_none_mask_values(self):
        """ds['col'] == None should produce all False values."""
        import pandas as pd

        df = pd.DataFrame({'s': ['abc', 'def', None]})
        ds = DataStore(df)

        pd_mask = df['s'] == None  # noqa: E711
        ds_mask = ds['s'] == None  # noqa: E711

        # Duck Typing: use assert_series_equal to compare mask values
        # This triggers execution implicitly via .values access
        assert_series_equal(ds_mask, pd_mask, check_names=False)

    def test_none_comparison_vs_isna(self):
        """Verify that == None is different from .isna()."""
        import pandas as pd

        df = pd.DataFrame({'s': ['abc', 'def', None, 'xyz']})
        ds = DataStore(df)

        # == None returns all False (no row equals None singleton)
        pd_eq_none_result = df[df['s'] == None]  # noqa: E711
        ds_eq_none_result = ds[ds['s'] == None]  # noqa: E711
        assert_datastore_equals_pandas(ds_eq_none_result, pd_eq_none_result)
        self.assertEqual(len(pd_eq_none_result), 0)

        # .isna() returns True for actual NA values
        pd_isna_result = df[df['s'].isna()]
        ds_isna_result = ds[ds['s'].isna()]
        assert_datastore_equals_pandas(ds_isna_result, pd_isna_result)
        self.assertEqual(len(pd_isna_result), 1)  # Only the None row

    def test_none_comparison_vs_notna(self):
        """Verify that != None is different from .notna()."""
        import pandas as pd

        df = pd.DataFrame({'s': ['abc', 'def', None, 'xyz']})
        ds = DataStore(df)

        # != None returns all True (every row differs from None singleton)
        pd_ne_none_result = df[df['s'] != None]  # noqa: E711
        ds_ne_none_result = ds[ds['s'] != None]  # noqa: E711
        assert_datastore_equals_pandas(ds_ne_none_result, pd_ne_none_result)
        self.assertEqual(len(pd_ne_none_result), 4)

        # .notna() returns True only for non-NA values
        pd_notna_result = df[df['s'].notna()]
        ds_notna_result = ds[ds['s'].notna()]
        assert_datastore_equals_pandas(ds_notna_result, pd_notna_result)
        self.assertEqual(len(pd_notna_result), 3)  # Excludes the None row

    def test_none_comparison_with_numeric_column(self):
        """Test None comparison with numeric columns containing NaN."""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0]})
        ds = DataStore(df)

        # != None returns all True
        pd_ne_result = df[df['x'] != None]  # noqa: E711
        ds_ne_result = ds[ds['x'] != None]  # noqa: E711
        assert_datastore_equals_pandas(ds_ne_result, pd_ne_result)

        # == None returns all False
        pd_eq_result = df[df['x'] == None]  # noqa: E711
        ds_eq_result = ds[ds['x'] == None]  # noqa: E711
        assert_datastore_equals_pandas(ds_eq_result, pd_eq_result)


if __name__ == "__main__":
    unittest.main()
