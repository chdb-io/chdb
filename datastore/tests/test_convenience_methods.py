"""
Tests for convenience methods like to_df() and to_dict() that execute queries directly.
"""

import unittest
import tempfile
import os

from datastore import DataStore
from tests.test_utils import assert_frame_equal


class TestConvenienceMethods(unittest.TestCase):
    """Test convenience methods for DataStore."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create a temporary CSV file for testing
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        # Write test data
        with open(cls.csv_file, "w") as f:
            f.write("id,name,age,city\n")
            f.write("1,Alice,25,NYC\n")
            f.write("2,Bob,30,LA\n")
            f.write("3,Charlie,35,Chicago\n")
            f.write("4,Diana,28,Boston\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def test_to_df_method_exists(self):
        """Test that to_df() method exists on DataStore."""
        ds = DataStore.from_file(self.csv_file)
        self.assertTrue(hasattr(ds, "to_df"))
        self.assertTrue(callable(ds.to_df))

    def test_to_dict_method_exists(self):
        """Test that to_dict() method exists on DataStore."""
        ds = DataStore.from_file(self.csv_file)
        self.assertTrue(hasattr(ds, "to_dict"))
        self.assertTrue(callable(ds.to_dict))

    def test_to_df_returns_dataframe(self):
        """Test that to_df() returns a pandas DataFrame."""
        ds = DataStore.from_file(self.csv_file)
        df = ds.select("*").to_df()

        # Check it's a DataFrame
        import pandas as pd

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        self.assertEqual(list(df.columns), ["id", "name", "age", "city"])

    def test_to_dict_returns_list_of_dicts(self):
        """Test that to_dict(orient='records') returns a list of dictionaries."""
        ds = DataStore.from_file(self.csv_file)
        records = ds.select("*").to_dict(orient='records')

        # Check it's a list of dicts
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 4)
        self.assertIsInstance(records[0], dict)
        self.assertIn("id", records[0])
        self.assertIn("name", records[0])

    def test_to_df_with_filter(self):
        """Test to_df() with filter applied."""
        ds = DataStore.from_file(self.csv_file)
        df = ds.select("*").filter(ds.age > 25).to_df()

        # Should only include Bob, Charlie, and Diana (age > 25)
        self.assertEqual(len(df), 3)

    def test_to_dict_with_filter(self):
        """Test to_dict(orient='records') with filter applied."""
        ds = DataStore.from_file(self.csv_file)
        records = ds.select("*").filter(ds.age > 25).to_dict(orient='records')

        # Should only include Bob, Charlie, and Diana (age > 25)
        self.assertEqual(len(records), 3)
        # Verify all ages are > 25
        for record in records:
            self.assertGreater(record["age"], 25)

    def test_to_df_with_limit(self):
        """Test to_df() with limit."""
        ds = DataStore.from_file(self.csv_file)
        df = ds.select("*").limit(2).to_df()

        # Should only include 2 rows
        self.assertEqual(len(df), 2)

    def test_to_dict_with_select_columns(self):
        """Test to_dict(orient='records') with specific column selection."""
        ds = DataStore.from_file(self.csv_file)
        records = ds.select("name", "age").to_dict(orient='records')

        # Should only include name and age columns
        self.assertEqual(len(records), 4)
        self.assertEqual(set(records[0].keys()), {"name", "age"})

    def test_to_df_same_as_execute_to_df(self):
        """Test that to_df() produces the same result as execute().to_df()."""
        ds = DataStore.from_file(self.csv_file)

        # Old way
        df1 = ds.select("*").filter(ds.age > 25).execute().to_df()

        # New way
        df2 = ds.select("*").filter(ds.age > 25).to_df()

        # Compare
        import pandas as pd

        assert_frame_equal(df1, df2)

    def test_to_dict_same_as_execute_to_dict(self):
        """Test that to_dict() produces the same result as execute().to_dict()."""
        ds = DataStore.from_file(self.csv_file)

        # Old way - pandas DataFrame.to_dict() default is 'dict' orient
        dict1 = ds.select("*").filter(ds.age > 25).execute().to_dict()

        # New way
        dict2 = ds.select("*").filter(ds.age > 25).to_dict()

        # Compare
        self.assertEqual(dict1, dict2)

    def test_to_df_with_complex_query(self):
        """Test to_df() with a complex query involving multiple operations."""
        ds = DataStore.from_file(self.csv_file)
        df = ds.select("name", "age").filter(ds.age >= 28).sort("age").limit(2).to_df()

        # Should return Diana (28) and Bob (30), sorted by age
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["name"]), ["Diana", "Bob"])

    def test_to_dict_with_complex_query(self):
        """Test to_dict(orient='records') with a complex query."""
        ds = DataStore.from_file(self.csv_file)
        records = ds.select("name", "city").filter(ds.age < 35).sort("name").to_dict(orient='records')

        # Should return Alice, Bob, Diana (all < 35), sorted by name
        self.assertEqual(len(records), 3)
        names = [r["name"] for r in records]
        self.assertEqual(names, ["Alice", "Bob", "Diana"])

    def test_describe_method_exists(self):
        """Test that describe() method exists on DataStore."""
        ds = DataStore.from_file(self.csv_file)
        self.assertTrue(hasattr(ds, "describe"))
        self.assertTrue(callable(ds.describe))

    def test_desc_method_exists(self):
        """Test that desc() method exists as shortcut for describe()."""
        ds = DataStore.from_file(self.csv_file)
        self.assertTrue(hasattr(ds, "desc"))
        self.assertTrue(callable(ds.desc))

    def test_head_method_exists(self):
        """Test that head() method exists on DataStore."""
        ds = DataStore.from_file(self.csv_file)
        self.assertTrue(hasattr(ds, "head"))
        self.assertTrue(callable(ds.head))

    def test_tail_method_exists(self):
        """Test that tail() method exists on DataStore."""
        ds = DataStore.from_file(self.csv_file)
        self.assertTrue(hasattr(ds, "tail"))
        self.assertTrue(callable(ds.tail))

    def test_describe_returns_dataframe(self):
        """Test that describe() returns a DataStore with statistics."""
        ds = DataStore.from_file(self.csv_file)
        stats = ds.select("*").describe()

        import pandas as pd

        # Should return DataStore, not DataFrame
        self.assertIsInstance(stats, DataStore)
        # Convert to DataFrame to check content
        stats_df = stats.to_df()
        # Should have standard statistics (count, mean, std, etc.)
        self.assertIn("count", stats_df.index)
        self.assertIn("mean", stats_df.index)
        self.assertIn("std", stats_df.index)
        # Should include numeric columns (id, age)
        self.assertIn("id", stats_df.columns)
        self.assertIn("age", stats_df.columns)

    def test_desc_same_as_describe(self):
        """Test that desc() produces the same result as describe()."""
        ds = DataStore.from_file(self.csv_file)

        stats1 = ds.select("*").describe()
        stats2 = ds.select("*").desc()

        import pandas as pd

        # Both should return DataStore
        self.assertIsInstance(stats1, DataStore)
        self.assertIsInstance(stats2, DataStore)
        # Compare the underlying DataFrames
        assert_frame_equal(stats1.to_df(), stats2.to_df())

    def test_describe_with_custom_percentiles(self):
        """Test describe() with custom percentiles."""
        ds = DataStore.from_file(self.csv_file)
        stats = ds.select("*").describe(percentiles=[0.1, 0.5, 0.9])

        # Should return DataStore
        self.assertIsInstance(stats, DataStore)
        # Convert to DataFrame to check percentiles
        stats_df = stats.to_df()
        # Should include custom percentiles
        self.assertIn("10%", stats_df.index)
        self.assertIn("50%", stats_df.index)
        self.assertIn("90%", stats_df.index)

    def test_head_default(self):
        """Test head() with default n=5."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select("*").head()

        import pandas as pd

        # Should return DataStore, not DataFrame
        self.assertIsInstance(result, DataStore)
        # We only have 4 rows, so should get all 4
        self.assertEqual(len(result), 4)

    def test_head_with_n(self):
        """Test head() with specific n value."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select("*").head(2)

        # Should return DataStore
        self.assertIsInstance(result, DataStore)
        self.assertEqual(len(result), 2)
        # Should be first 2 rows - convert to DataFrame to check values
        df = result.to_df()
        self.assertEqual(list(df["name"]), ["Alice", "Bob"])

    def test_head_with_filter(self):
        """Test head() with filter applied."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select("*").filter(ds.age > 25).head(2)

        # Should return DataStore
        self.assertIsInstance(result, DataStore)
        # Should get first 2 rows where age > 25
        self.assertEqual(len(result), 2)

    def test_tail_default(self):
        """Test tail() with default n=5."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select("*").tail()

        import pandas as pd

        # Should return DataStore, not DataFrame
        self.assertIsInstance(result, DataStore)
        # We only have 4 rows, so should get all 4
        self.assertEqual(len(result), 4)

    def test_tail_with_n(self):
        """Test tail() with specific n value."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select("*").tail(2)

        # Should return DataStore
        self.assertIsInstance(result, DataStore)
        self.assertEqual(len(result), 2)
        # Should be last 2 rows - convert to DataFrame to check values
        df = result.to_df()
        self.assertEqual(list(df["name"]), ["Charlie", "Diana"])

    def test_sample_method_exists(self):
        """Test that sample() method exists on DataStore."""
        ds = DataStore.from_file(self.csv_file)
        self.assertTrue(hasattr(ds, "sample"))
        self.assertTrue(callable(ds.sample))

    def test_sample_with_n(self):
        """Test sample() with specific n value."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select("*").sample(n=2, random_state=42)

        # Should return DataStore
        self.assertIsInstance(result, DataStore)
        self.assertEqual(len(result), 2)

    def test_sample_with_frac(self):
        """Test sample() with fraction."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select("*").sample(frac=0.5, random_state=42)

        # Should return DataStore
        self.assertIsInstance(result, DataStore)
        # 50% of 4 rows = 2 rows
        self.assertEqual(len(result), 2)

    def test_shape_property(self):
        """Test shape property returns (rows, cols) tuple."""
        ds = DataStore.from_file(self.csv_file)
        shape = ds.select("*").shape

        self.assertIsInstance(shape, tuple)
        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], 4)  # 4 rows
        self.assertEqual(shape[1], 4)  # 4 columns

    def test_columns_property(self):
        """Test columns property returns column names."""
        ds = DataStore.from_file(self.csv_file)
        cols = ds.select("*").columns

        import pandas as pd

        self.assertIsInstance(cols, pd.Index)
        self.assertEqual(list(cols), ["id", "name", "age", "city"])

    def test_count_method(self):
        """Test count() returns counts per column."""
        ds = DataStore.from_file(self.csv_file)
        counts = ds.select("*").count()

        import pandas as pd

        self.assertIsInstance(counts, pd.Series)
        # All columns should have 4 non-null values
        self.assertEqual(counts["id"], 4)
        self.assertEqual(counts["name"], 4)
        self.assertEqual(counts["age"], 4)
        self.assertEqual(counts["city"], 4)

    def test_count_rows_method(self):
        """Test count_rows() returns total row count using SQL COUNT(*)."""
        ds = DataStore.from_file(self.csv_file)
        total = ds.select("*").count_rows()

        # Should return integer
        self.assertIsInstance(total, int)
        self.assertEqual(total, 4)

    def test_count_rows_with_filter(self):
        """Test count_rows() respects filter conditions."""
        ds = DataStore.from_file(self.csv_file)
        total = ds.select("*").filter(ds.age > 28).count_rows()

        # Only Bob (30) and Charlie (35) should match
        self.assertEqual(total, 2)

    def test_count_uses_sql(self):
        """Test that count() uses SQL and produces same results as DataFrame.count()."""
        ds = DataStore.from_file(self.csv_file)

        # Get counts via optimized SQL method
        sql_counts = ds.select("*").count()

        # Get counts via DataFrame execution
        df_counts = ds.select("*").to_df().count()

        import pandas as pd

        # Both should be Series with same values
        self.assertIsInstance(sql_counts, pd.Series)
        self.assertIsInstance(df_counts, pd.Series)

        # Values should match
        for col in ["id", "name", "age", "city"]:
            self.assertEqual(sql_counts[col], df_counts[col])

    def test_count_with_filter(self):
        """Test count() respects filter conditions."""
        ds = DataStore.from_file(self.csv_file)
        counts = ds.select("*").filter(ds.age > 28).count()

        import pandas as pd

        self.assertIsInstance(counts, pd.Series)
        # Only Bob (30) and Charlie (35) should match
        self.assertEqual(counts["id"], 2)
        self.assertEqual(counts["name"], 2)
        self.assertEqual(counts["age"], 2)
        self.assertEqual(counts["city"], 2)

    def test_len_uses_count_rows(self):
        """Test len() uses efficient SQL-based count_rows()."""
        ds = DataStore.from_file(self.csv_file)

        # len() should use count_rows() internally
        self.assertEqual(len(ds.select("*")), 4)
        self.assertEqual(len(ds.select("*").filter(ds.age > 28)), 2)

    def test_info_method(self):
        """Test info() method exists and can be called."""
        ds = DataStore.from_file(self.csv_file)
        # info() prints to stdout, so just verify it doesn't raise
        result = ds.select("*").info()
        # info() returns None by default
        self.assertIsNone(result)

    def test_describe_same_as_to_df_describe(self):
        """Test that describe() produces the same result as to_df().describe()."""
        ds = DataStore.from_file(self.csv_file)

        stats1 = ds.select("*").to_df().describe()
        stats2 = ds.select("*").describe()

        import pandas as pd

        # stats2 is now a DataStore, convert to DataFrame to compare
        self.assertIsInstance(stats2, DataStore)
        assert_frame_equal(stats1, stats2.to_df())

    def test_head_same_as_to_df_head(self):
        """Test that head() produces the same result as limit().to_df()."""
        ds = DataStore.from_file(self.csv_file)

        df1 = ds.select("*").limit(3).to_df()
        df2 = ds.select("*").head(3)

        import pandas as pd

        # df2 is now a DataStore, convert to DataFrame to compare
        self.assertIsInstance(df2, DataStore)
        assert_frame_equal(df1, df2.to_df())




class TestSQLOptimizedMethods(unittest.TestCase):
    """Test SQL-optimized info/describe/sample that avoid full table loading."""

    @classmethod
    def setUpClass(cls):
        """Create a larger test CSV for SQL optimization tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_sql_opt.csv")

        import random

        random.seed(42)
        with open(cls.csv_file, "w") as f:
            f.write("id,name,age,score\n")
            for i in range(100):
                name = chr(65 + i % 26) + chr(97 + (i * 7) % 26)
                f.write(f"{i},{name},{20 + i % 50},{random.uniform(0, 100):.2f}\n")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    # ---- sample() SQL optimization tests ----

    def test_sample_sql_returns_correct_count(self):
        """sample(n) via SQL returns the right number of rows."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.sample(n=10)
        self.assertIsInstance(result, DataStore)
        self.assertEqual(len(result), 10)

    def test_sample_sql_returns_valid_rows(self):
        """All sampled rows must exist in the original data."""
        ds = DataStore.from_file(self.csv_file)
        sample_df = ds.sample(n=5).to_df()
        full_df = ds.to_df()
        for _, row in sample_df.iterrows():
            self.assertIn(row["id"], full_df["id"].values)

    def test_sample_sql_frac(self):
        """sample(frac) via SQL returns the correct fraction of rows."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.sample(frac=0.1)
        self.assertIsInstance(result, DataStore)
        self.assertEqual(len(result), 10)

    def test_sample_sql_with_filter(self):
        """sample() works correctly after filter (SQL pushdown)."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.filter(ds.age > 40).sample(n=3)
        self.assertEqual(len(result), 3)
        result_df = result.to_df()
        for _, row in result_df.iterrows():
            self.assertGreater(row["age"], 40)

    def test_sample_random_state_uses_pandas(self):
        """random_state forces pandas path for reproducibility."""
        ds = DataStore.from_file(self.csv_file)
        r1 = ds.sample(n=5, random_state=42).to_df()
        r2 = ds.sample(n=5, random_state=42).to_df()
        assert_frame_equal(r1, r2)

    def test_sample_replace_uses_pandas(self):
        """replace=True forces pandas path."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.sample(n=150, replace=True)
        self.assertEqual(len(result), 150)

    # ---- info() SQL optimization tests ----

    def test_info_sql_returns_none(self):
        """info() returns None (prints to stdout) like pandas."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.info()
        self.assertIsNone(result)

    def test_info_sql_with_buf(self):
        """info() writes to buffer and contains expected metadata."""
        import io

        ds = DataStore.from_file(self.csv_file)
        buf = io.StringIO()
        result = ds.info(buf=buf)
        self.assertIsNone(result)

        output = buf.getvalue()
        self.assertIn("100", output)  # 100 rows
        self.assertIn("4 columns", output)  # 4 columns
        self.assertIn("id", output)
        self.assertIn("name", output)
        self.assertIn("age", output)
        self.assertIn("score", output)

    def test_info_sql_with_filter(self):
        """info() respects filter and shows filtered row count."""
        import io

        ds = DataStore.from_file(self.csv_file)
        buf = io.StringIO()
        ds.filter(ds.age > 40).info(buf=buf)
        output = buf.getvalue()
        # Should NOT show "100 entries" since we filtered
        self.assertNotIn("100 entries", output)

    def test_info_sql_show_counts(self):
        """info(show_counts=True) includes non-null counts."""
        import io

        ds = DataStore.from_file(self.csv_file)
        buf = io.StringIO()
        ds.info(buf=buf, show_counts=True)
        output = buf.getvalue()
        self.assertIn("non-null", output)

    # ---- describe() SQL optimization tests ----

    def test_describe_sql_returns_datastore(self):
        """describe() via SQL returns a DataStore."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.describe()
        self.assertIsInstance(result, DataStore)

    def test_describe_sql_has_correct_stats(self):
        """describe() has all expected statistic rows."""
        ds = DataStore.from_file(self.csv_file)
        stats_df = ds.describe().to_df()
        for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            self.assertIn(stat, stats_df.index)

    def test_describe_sql_numeric_only(self):
        """describe() includes only numeric columns by default."""
        ds = DataStore.from_file(self.csv_file)
        stats_df = ds.describe().to_df()
        # id, age, score are numeric; name is string
        self.assertIn("id", stats_df.columns)
        self.assertIn("age", stats_df.columns)
        self.assertIn("score", stats_df.columns)
        self.assertNotIn("name", stats_df.columns)

    def test_describe_sql_matches_pandas(self):
        """describe() via SQL produces results matching pandas describe()."""
        ds = DataStore.from_file(self.csv_file)
        ds_stats = ds.describe().to_df()
        pd_stats = ds.to_df().describe()

        # Same columns and index
        self.assertEqual(list(ds_stats.columns), list(pd_stats.columns))
        self.assertEqual(list(ds_stats.index), list(pd_stats.index))

        # Values should match within tolerance
        for col in ds_stats.columns:
            for idx in ds_stats.index:
                ds_val = ds_stats.loc[idx, col]
                pd_val = pd_stats.loc[idx, col]
                import math
                if math.isnan(ds_val) and math.isnan(pd_val):
                    continue
                self.assertAlmostEqual(
                    ds_val,
                    pd_val,
                    places=4,
                    msg=f"Mismatch at [{idx}, {col}]: DS={ds_val}, PD={pd_val}",
                )

    def test_describe_sql_with_custom_percentiles(self):
        """describe() with custom percentiles works via SQL."""
        ds = DataStore.from_file(self.csv_file)
        stats = ds.describe(percentiles=[0.1, 0.5, 0.9])
        stats_df = stats.to_df()
        self.assertIn("10%", stats_df.index)
        self.assertIn("50%", stats_df.index)
        self.assertIn("90%", stats_df.index)

    def test_describe_sql_with_filter(self):
        """describe() works after filter (SQL pushdown)."""
        ds = DataStore.from_file(self.csv_file)
        stats = ds.filter(ds.age > 30).describe()
        stats_df = stats.to_df()

        # count should be less than 100
        self.assertLess(stats_df.loc["count", "age"], 100)
        # min age should be > 30
        self.assertGreater(stats_df.loc["min", "age"], 30)

    def test_describe_include_all_uses_pandas(self):
        """describe(include='all') falls back to pandas (includes non-numeric)."""
        ds = DataStore.from_file(self.csv_file)
        stats = ds.describe(include="all")
        stats_df = stats.to_df()
        # Should include non-numeric columns
        self.assertIn("name", stats_df.columns)

    def test_describe_sql_matches_pandas_mirror(self):
        """Mirror test: DataStore describe matches pandas describe exactly."""
        import pandas as pd

        # pandas path
        pd_df = pd.read_csv(self.csv_file)
        pd_result = pd_df.describe()

        # DataStore path (uses SQL optimization)
        ds = DataStore.from_file(self.csv_file)
        ds_result = ds.describe()

        assert_frame_equal(ds_result.to_df(), pd_result)


if __name__ == "__main__":
    unittest.main()
