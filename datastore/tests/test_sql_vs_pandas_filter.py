"""
Test to understand how SQL engine vs Pandas engine filtering works.

This test creates independent data and verifies both execution paths.
"""

import unittest
import tempfile
import os
import logging
import pandas as pd

from datastore import DataStore, config
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal


class TestSQLvsPandasFilter(unittest.TestCase):
    """Test SQL engine vs Pandas engine for filtering."""

    @classmethod
    def setUpClass(cls):
        """Create test data file."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        # Create simple test data
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.csv_file, index=False)

    def test_1_pure_sql_filter(self):
        """
        Test 1: Pure SQL filter - uses SQL engine only.

        When we only use SQL operations (select, filter), the entire query
        is compiled to SQL and executed by chdb/ClickHouse.
        """
        ds = DataStore.from_file(self.csv_file)

        # Pure SQL operations
        result = ds.select('name', 'age').filter(ds.age > 30)

        # Check what SQL is generated
        sql = result.to_sql()
        print(f"\n=== Test 1: Pure SQL Filter ===")
        print(f"Generated SQL: {sql}")

        # Execute and get results
        df = result.to_df()
        print(f"Result:\n{df}")

        # Verify: should have 3 rows (age > 30: Charlie=35, David=40, Eve=45)
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df['name']), ['Charlie', 'David', 'Eve'])
        self.assertListEqual(list(df['age']), [35, 40, 45])

    def test_2_pandas_operation_then_sql_filter(self):
        """
        Test 2: Pandas operation followed by SQL-like filter.

        When we mix pandas operations (like column assignment) with SQL operations,
        the execution becomes more complex:
        1. First, SQL operations before pandas ops are executed via SQL engine
        2. Then pandas operations are applied to the DataFrame
        3. Then SQL-like operations after pandas ops must be applied via pandas
           (because the data is already in DataFrame form)
        """
        ds = DataStore.from_file(self.csv_file)

        # SQL operation first
        result = ds.select('name', 'age', 'salary')

        # Pandas operation - this forces execution
        result['age_doubled'] = result['age'] * 2

        # Now filter AFTER pandas operation
        # This filter must be applied via pandas, not SQL!
        result = result.filter(result.age > 30)

        print(f"\n=== Test 2: Pandas Op then SQL-like Filter ===")
        print(f"Explain plan:")
        result.explain()

        # Execute and get results
        df = result.to_df()
        print(f"Result:\n{df}")

        # Verify: should have 3 rows with age_doubled column
        self.assertEqual(len(df), 3)
        self.assertIn('age_doubled', df.columns)
        # age_doubled should be 70, 80, 90 for ages 35, 40, 45
        self.assertListEqual(list(df['age_doubled']), [70, 80, 90])

    def test_3_filter_before_and_after_pandas(self):
        """
        Test 3: Filter before AND after pandas operation.

        This shows both execution paths:
        - First filter (age > 25) is compiled into SQL
        - Pandas operation (age_doubled) is applied to DataFrame
        - Second filter (age_doubled > 70) must use pandas engine
        """
        ds = DataStore.from_file(self.csv_file)

        # First filter - will be in SQL
        result = ds.select('name', 'age').filter(ds.age > 25)

        # Pandas operation
        result['age_doubled'] = result['age'] * 2

        # Second filter - must use pandas since data is already executed
        result = result.filter(result.age_doubled > 70)

        print(f"\n=== Test 3: Filter Before and After Pandas ===")
        print(f"Explain plan:")
        result.explain()

        df = result.to_df()
        print(f"Result:\n{df}")

        # First filter: age > 25 -> Bob(30), Charlie(35), David(40), Eve(45)
        # age_doubled: 60, 70, 80, 90
        # Second filter: age_doubled > 70 -> David(80), Eve(90)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df['name']), ['David', 'Eve'])

    def test_4_verify_lazy_sql_snapshot_execute(self):
        """
        Test 4: Directly verify that LazyRelationalOp.execute() works on DataFrames.

        This test shows what LazyRelationalOp.execute() is actually doing:
        It's the pandas-based execution of SQL-like operations on a DataFrame.
        """
        from datastore.lazy_ops import LazyRelationalOp
        from datastore.conditions import BinaryCondition
        from datastore.expressions import Field, Literal

        # Create a test DataFrame
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})

        # Create a condition: age > 28
        # BinaryCondition signature: (operator, left, right)
        condition = BinaryCondition('>', Field('age'), Literal(28))

        # Create a LazyRelationalOp for WHERE operation
        lazy_op = LazyRelationalOp('WHERE', 'age > 28', condition=condition)

        # Execute on DataFrame - this is the pandas execution path!
        result_df = lazy_op.execute(df, context=None)

        print(f"\n=== Test 4: LazyRelationalOp.execute() on DataFrame ===")
        print(f"Input DataFrame:\n{df}")
        print(f"After WHERE age > 28:\n{result_df}")

        # Should have 2 rows: Bob(30) and Charlie(35)
        self.assertEqual(len(result_df), 2)
        self.assertListEqual(list(result_df['name']), ['Bob', 'Charlie'])

    def test_5_execution_path_comparison(self):
        """
        Test 5: Compare execution paths and verify both give same results.
        """
        ds1 = DataStore.from_file(self.csv_file)
        ds2 = DataStore.from_file(self.csv_file)

        # Path 1: Pure SQL filter
        result1 = ds1.select('name', 'age').filter(ds1.age > 30)
        df1 = result1.to_df()

        # Path 2: Force pandas execution by adding a no-op pandas operation
        result2 = ds2.select('name', 'age')
        result2['temp'] = result2['age']  # Force execution
        result2 = result2.filter(result2.age > 30)  # This filter uses pandas
        result2 = result2[['name', 'age']]  # Remove temp column
        df2 = result2.to_df()

        print(f"\n=== Test 5: Execution Path Comparison ===")
        print(f"Pure SQL result:\n{df1}")
        print(f"Pandas execution result:\n{df2}")

        # Both should give same results
        self.assertEqual(len(df1), len(df2))
        self.assertListEqual(list(df1['name']), list(df2['name']))
        self.assertListEqual(list(df1['age']), list(df2['age']))


class TestComplexLazyPipelineExecution(unittest.TestCase):
    """
    Test cases for complex multi-step lazy pipeline execution.

    These tests are designed to catch bugs in the lazy pipeline execution,
    especially when mixing SQL operations with Pandas operations in complex ways.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data files."""
        cls.temp_dir = tempfile.mkdtemp()

        # Main test file
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")
        data = {
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
            'age': [25, 30, 35, 40, 45, 28, 33, 38, 42, 27],
            'salary': [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000, 85000, 52000],
            'department': ['A', 'B', 'A', 'B', 'A', 'C', 'C', 'B', 'A', 'C'],
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.csv_file, index=False)

        # Secondary file for joins
        cls.orders_file = os.path.join(cls.temp_dir, "orders.csv")
        orders_data = {
            'order_id': [101, 102, 103, 104, 105, 106],
            'user_id': [1, 2, 3, 1, 2, 4],
            'amount': [100, 200, 150, 300, 250, 175],
        }
        pd.DataFrame(orders_data).to_csv(cls.orders_file, index=False)

    def test_chained_arithmetic_assignments(self):
        """
        Test: Multiple chained arithmetic column assignments.

        Verify that each new column depends on previously computed columns correctly.
        """
        ds = DataStore.from_file(self.csv_file)

        # Chain of dependent arithmetic operations
        ds['step1'] = ds['age'] * 2  # step1 = age * 2
        ds['step2'] = ds['step1'] + 10  # step2 = step1 + 10
        ds['step3'] = ds['step2'] / 2  # step3 = step2 / 2
        ds['step4'] = ds['step3'] - ds['age']  # step4 = step3 - age
        ds['step5'] = ds['step4'] ** 2  # step5 = step4**2

        df = ds.to_df()

        # Verify each step
        self.assertTrue((df['step1'] == df['age'] * 2).all())
        self.assertTrue((df['step2'] == df['step1'] + 10).all())
        self.assertTrue((df['step3'] == df['step2'] / 2).all())
        self.assertTrue((df['step4'] == df['step3'] - df['age']).all())
        self.assertTrue((df['step5'] == df['step4'] ** 2).all())

        # Final formula verification: step5 = ((age*2+10)/2 - age)**2 = 5**2 = 25
        self.assertTrue((df['step5'] == 25).all())

    def test_filter_on_computed_column(self):
        """
        Test: Filter on a column computed via pandas assignment.

        This tests that lazy filter correctly applies to computed columns.
        """
        ds = DataStore.from_file(self.csv_file)

        # Compute column
        ds['age_doubled'] = ds['age'] * 2

        # Filter on computed column
        ds = ds.filter(ds['age_doubled'] > 60)

        df = ds.to_df()

        # Verify: age_doubled > 60 means age > 30
        self.assertTrue(all(df['age'] > 30))
        self.assertTrue(all(df['age_doubled'] > 60))
        self.assertEqual(len(df), 6)  # ages 35, 40, 45, 33, 38, 42

    def test_multiple_filters_on_computed_columns(self):
        """
        Test: Multiple filters on different computed columns.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['age_doubled'] = ds['age'] * 2
        ds = ds.filter(ds['age_doubled'] > 50)  # age > 25

        ds['salary_k'] = ds['salary'] / 1000
        ds = ds.filter(ds['salary_k'] > 60)  # salary > 60000

        df = ds.to_df()

        # Verify both conditions
        self.assertTrue(all(df['age'] > 25))
        self.assertTrue(all(df['salary'] > 60000))

    def test_sort_on_computed_column(self):
        """
        Test: Sort on a computed column.
        """
        ds = DataStore.from_file(self.csv_file)

        # Compute a column that reverses the natural order
        ds['inverse_age'] = 100 - ds['age']

        # Sort by computed column
        ds = ds.sort('inverse_age', ascending=True)

        df = ds.to_df()

        # Verify: inverse_age ascending means age descending
        ages = df['age'].tolist()
        self.assertEqual(ages, sorted(ages, reverse=True))

    def test_limit_after_computed_filter(self):
        """
        Test: LIMIT after filtering on computed column.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['age_plus_10'] = ds['age'] + 10
        ds = ds.filter(ds['age_plus_10'] > 40)  # age > 30
        ds = ds.sort('age', ascending=False)
        ds = ds.limit(3)

        df = ds.to_df()

        # Should have exactly 3 rows, sorted by age desc
        self.assertEqual(len(df), 3)
        self.assertEqual(df['age'].tolist(), [45, 42, 40])

    def test_complex_boolean_expression_filter(self):
        """
        Test: Complex boolean expressions with AND/OR.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['bonus'] = ds['salary'] * 0.1

        # Complex condition: (age > 30 AND salary > 60000) OR bonus > 8000
        ds = ds.filter(((ds['age'] > 30) & (ds['salary'] > 60000)) | (ds['bonus'] > 8000))

        df = ds.to_df()

        # Verify each row matches the condition
        for _, row in df.iterrows():
            condition = (row['age'] > 30 and row['salary'] > 60000) or row['bonus'] > 8000
            self.assertTrue(condition)

    def test_filter_assign_filter_assign_pattern(self):
        """
        Test: Alternating filter and assign pattern.

        This is a common pattern that tests the lazy pipeline.
        """
        ds = DataStore.from_file(self.csv_file)

        # Step 1: SQL filter
        ds = ds.filter(ds['age'] > 25)

        # Step 2: Pandas assign
        ds['step2_col'] = ds['age'] * 2

        # Step 3: Filter on original column
        ds = ds.filter(ds['salary'] > 55000)

        # Step 4: Pandas assign using step2_col
        ds['step4_col'] = ds['step2_col'] + ds['salary']

        # Step 5: Filter on computed column
        ds = ds.filter(ds['step2_col'] > 60)

        # Step 6: Final assignment
        ds['final'] = ds['step4_col'] / 1000

        df = ds.to_df()

        # Verify all conditions
        self.assertTrue(all(df['age'] > 25))
        self.assertTrue(all(df['salary'] > 55000))
        self.assertTrue(all(df['step2_col'] > 60))
        self.assertTrue((df['step2_col'] == df['age'] * 2).all())
        self.assertTrue((df['step4_col'] == df['step2_col'] + df['salary']).all())
        self.assertTrue((df['final'] == df['step4_col'] / 1000).all())

    def test_result_consistency_sql_vs_pandas_path(self):
        """
        Test: Verify SQL path and Pandas path give same results.

        Compare a pure SQL chain with a mixed SQL+Pandas chain.
        """
        # Path 1: Pure SQL (as much as possible)
        ds1 = DataStore.from_file(self.csv_file)
        ds1 = ds1.select('name', 'age', 'salary')
        ds1 = ds1.filter(ds1['age'] > 30)
        ds1 = ds1.filter(ds1['salary'] > 60000)
        ds1 = ds1.sort('age', ascending=True)
        df1 = ds1.to_df()

        # Path 2: Force Pandas path by inserting computed column
        ds2 = DataStore.from_file(self.csv_file)
        ds2 = ds2.select('name', 'age', 'salary')
        ds2['temp'] = ds2['age'] * 1  # Force execution (identity operation)
        ds2 = ds2.filter(ds2['age'] > 30)
        ds2 = ds2.filter(ds2['salary'] > 60000)
        ds2 = ds2.sort('age', ascending=True)
        df2 = ds2.to_df()

        # Compare core columns
        self.assertEqual(df1['name'].tolist(), df2['name'].tolist())
        self.assertEqual(df1['age'].tolist(), df2['age'].tolist())
        self.assertEqual(df1['salary'].tolist(), df2['salary'].tolist())

    def test_order_of_operations_matters(self):
        """
        Test: Verify that order of operations is preserved.

        Different orderings should produce different results.
        """
        # Order 1: Filter first, then limit
        ds1 = DataStore.from_file(self.csv_file)
        ds1['score'] = ds1['age'] + ds1['salary'] / 1000
        ds1 = ds1.sort('score', ascending=False)
        ds1 = ds1.filter(ds1['age'] > 30)
        ds1 = ds1.limit(3)
        df1 = ds1.to_df()

        # Order 2: Limit first, then filter (should give different result)
        ds2 = DataStore.from_file(self.csv_file)
        ds2['score'] = ds2['age'] + ds2['salary'] / 1000
        ds2 = ds2.sort('score', ascending=False)
        ds2 = ds2.limit(3)
        ds2 = ds2.filter(ds2['age'] > 30)  # Filter on already limited data
        df2 = ds2.to_df()

        # df1: First filter (6 rows with age > 30), then limit to 3
        self.assertEqual(len(df1), 3)

        # df2: First limit to 3, then filter those 3
        self.assertLessEqual(len(df2), 3)

    def test_column_overwrite(self):
        """
        Test: Overwriting a computed column multiple times.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['computed'] = ds['age'] * 1
        ds['computed'] = ds['age'] * 2
        ds['computed'] = ds['age'] * 3
        ds['computed'] = ds['age'] * 4

        df = ds.to_df()

        # Should have the last value
        self.assertTrue((df['computed'] == df['age'] * 4).all())

    def test_filter_with_between_values(self):
        """
        Test: Filter with BETWEEN-style condition using AND.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['score'] = ds['age'] * 2 + ds['salary'] / 1000

        # BETWEEN: 100 <= score <= 150
        ds = ds.filter((ds['score'] >= 100) & (ds['score'] <= 150))

        df = ds.to_df()

        # Verify all scores are in range
        self.assertTrue(all(df['score'] >= 100))
        self.assertTrue(all(df['score'] <= 150))

    def test_negative_and_division_operations(self):
        """
        Test: Negative numbers and division in computed columns.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['neg_age'] = -ds['age']
        ds['age_ratio'] = ds['age'] / 10
        ds['salary_percent'] = (ds['salary'] - 50000) / 50000 * 100

        df = ds.to_df()

        self.assertTrue((df['neg_age'] == -df['age']).all())
        self.assertTrue((df['age_ratio'] == df['age'] / 10).all())
        self.assertTrue(((df['salary_percent'] - (df['salary'] - 50000) / 50000 * 100).abs() < 0.01).all())

    def test_empty_result_after_computed_filter(self):
        """
        Test: Handle empty results after filtering on computed column.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['impossible'] = ds['age'] * 0 + 1  # All values are 1
        ds = ds.filter(ds['impossible'] > 100)  # No rows match

        df = ds.to_df()

        self.assertEqual(len(df), 0)

    def test_all_rows_match_computed_filter(self):
        """
        Test: All rows match the computed filter.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['always_true'] = ds['age'] * 0 + 100
        ds = ds.filter(ds['always_true'] > 0)

        df = ds.to_df()

        self.assertEqual(len(df), 10)  # All rows


class TestLazyPipelineEdgeCases(unittest.TestCase):
    """
    Edge case tests for lazy pipeline execution.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        data = {
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A'],
        }
        pd.DataFrame(data).to_csv(cls.csv_file, index=False)

    def test_single_row_after_filter(self):
        """
        Test: Single row result after filtering.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['doubled'] = ds['value'] * 2
        ds = ds.filter(ds['doubled'] == 80)  # Only value=40 matches

        df = ds.to_df()

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['value'], 40)

    def test_many_small_operations(self):
        """
        Test: Many small incremental operations.
        """
        ds = DataStore.from_file(self.csv_file)

        # 20 incremental additions
        ds['inc'] = ds['value']
        for i in range(20):
            ds['inc'] = ds['inc'] + 1

        df = ds.to_df()

        # inc should be value + 20
        self.assertTrue((df['inc'] == df['value'] + 20).all())

    def test_circular_dependency_detection(self):
        """
        Test: Ensure no circular dependency issues.

        Note: This tests that computed columns can reference each other linearly.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['a'] = ds['value'] * 2
        ds['b'] = ds['a'] + 10
        ds['c'] = ds['b'] - ds['a']  # Should be 10

        df = ds.to_df()

        self.assertTrue((df['c'] == 10).all())

    def test_large_expression_tree(self):
        """
        Test: Large expression tree in single assignment.
        """
        ds = DataStore.from_file(self.csv_file)

        # Complex expression: ((value * 2 + 10) / 2 - 5) ** 2
        ds['complex'] = ((ds['value'] * 2 + 10) / 2 - 5) ** 2

        df = ds.to_df()

        expected = ((df['value'] * 2 + 10) / 2 - 5) ** 2
        self.assertTrue(((df['complex'] - expected).abs() < 0.01).all())

    def test_select_subset_of_computed_columns(self):
        """
        Test: Select only some computed columns.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['col1'] = ds['value'] * 2
        ds['col2'] = ds['value'] * 3
        ds['col3'] = ds['value'] * 4

        # Select only some columns
        ds = ds[['id', 'col1', 'col3']]

        df = ds.to_df()

        self.assertIn('col1', df.columns)
        self.assertIn('col3', df.columns)
        self.assertNotIn('col2', df.columns)
        self.assertNotIn('value', df.columns)

    def test_filter_sort_filter_pattern(self):
        """
        Test: Filter -> Sort -> Filter pattern.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['score'] = ds['value'] * 10

        ds = ds.filter(ds['value'] > 10)  # Remove id=1
        ds = ds.sort('score', ascending=False)
        ds = ds.filter(ds['score'] < 500)  # Remove id=5

        df = ds.to_df()

        self.assertEqual(len(df), 3)  # ids 2, 3, 4
        # Should be sorted descending
        scores = df['score'].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_modulo_operation(self):
        """
        Test: Modulo operation in computed column.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['mod_3'] = ds['value'] % 3
        ds = ds.filter(ds['mod_3'] == 0)

        df = ds.to_df()

        # value = 30 is divisible by 3
        self.assertTrue(all(df['value'] % 3 == 0))


class TestPandasSQLResultVerification(unittest.TestCase):
    """
    Tests that verify results match expected values computed manually.

    These tests serve as regression tests for lazy pipeline correctness.
    """

    @classmethod
    def setUpClass(cls):
        """Create deterministic test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "verify_data.csv")

        # Create deterministic data for precise verification
        data = {
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
        pd.DataFrame(data).to_csv(cls.csv_file, index=False)

    def test_exact_values_after_computation(self):
        """
        Test: Verify exact output values.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['z'] = ds['x'] + ds['y']
        ds = ds.filter(ds['z'] > 50)

        df = ds.to_df()

        # Expected: x=4,y=40,z=44; x=5,y=50,z=55; ... x=10,y=100,z=110
        # Only z > 50: x>=5
        expected_x = [5, 6, 7, 8, 9, 10]
        expected_z = [55, 66, 77, 88, 99, 110]

        self.assertEqual(df['x'].tolist(), expected_x)
        self.assertEqual(df['z'].tolist(), expected_z)

    def test_multi_step_exact_verification(self):
        """
        Test: Multi-step pipeline with exact value verification.
        """
        ds = DataStore.from_file(self.csv_file)

        # Step 1: a = x * 10
        ds['a'] = ds['x'] * 10

        # Step 2: filter a > 30 (x > 3)
        ds = ds.filter(ds['a'] > 30)

        # Step 3: b = a + y
        ds['b'] = ds['a'] + ds['y']

        # Step 4: filter b > 100
        ds = ds.filter(ds['b'] > 100)

        # Step 5: sort by x descending
        ds = ds.sort('x', ascending=False)

        # Step 6: limit 3
        ds = ds.limit(3)

        df = ds.to_df()

        # Manual calculation:
        # After step 1: a = x*10 = [10,20,30,40,50,60,70,80,90,100]
        # After step 2: x > 3 -> x in [4,5,6,7,8,9,10]
        # After step 3: b = a + y = [40+40, 50+50, 60+60, 70+70, 80+80, 90+90, 100+100]
        #                         = [80, 100, 120, 140, 160, 180, 200]
        # After step 4: b > 100 -> x in [6,7,8,9,10], b in [120,140,160,180,200]
        # After step 5: sorted desc -> x = [10,9,8,7,6]
        # After step 6: limit 3 -> x = [10,9,8]

        self.assertEqual(len(df), 3)
        self.assertEqual(df['x'].tolist(), [10, 9, 8])
        self.assertEqual(df['b'].tolist(), [200, 180, 160])

    def test_aggregation_precondition(self):
        """
        Test: Verify data before and after filter for aggregation-like checks.
        """
        ds = DataStore.from_file(self.csv_file)

        ds['sum_xy'] = ds['x'] + ds['y']
        ds = ds.filter(ds['x'] <= 5)

        df = ds.to_df()

        # x in [1,2,3,4,5], sum_xy = [11,22,33,44,55]
        self.assertEqual(len(df), 5)
        self.assertEqual(df['sum_xy'].sum(), 11 + 22 + 33 + 44 + 55)


class TestMultipleToDF(unittest.TestCase):
    """
    Tests for multiple to_df() calls and caching behavior.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "cache_test.csv")

        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        pd.DataFrame(data).to_csv(cls.csv_file, index=False)

    def test_multiple_to_df_same_result(self):
        """
        Test: Multiple to_df() calls should return consistent results.
        """
        ds = DataStore.from_file(self.csv_file)
        ds['c'] = ds['a'] + ds['b']

        df1 = ds.to_df()
        df2 = ds.to_df()
        df3 = ds.to_df()

        # All should be identical
        self.assertTrue(df1.equals(df2))
        self.assertTrue(df2.equals(df3))

    def test_operations_after_to_df(self):
        """
        Test: Adding operations after to_df() should still work.
        """
        ds = DataStore.from_file(self.csv_file)
        ds['c'] = ds['a'] + ds['b']

        df1 = ds.to_df()
        self.assertIn('c', df1.columns)

        # Add more operations
        ds['d'] = ds['a'] * 2

        df2 = ds.to_df()
        self.assertIn('c', df2.columns)
        self.assertIn('d', df2.columns)


class TestInterleavedSQLPandas20PlusSteps(unittest.TestCase):
    """
    Stress tests with 20+ alternating SQL and Pandas operations.

    These tests are designed to catch bugs in lazy pipeline execution
    when SQL and Pandas operations are heavily interleaved.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "interleaved_test.csv")

        # Create data with 20 rows for meaningful filtering
        data = {
            'id': list(range(1, 21)),
            'x': list(range(10, 210, 10)),  # 10, 20, 30, ..., 200
            'y': list(range(100, 300, 10)),  # 100, 110, 120, ..., 290
            'category': ['A', 'B', 'C', 'D'] * 5,
        }
        pd.DataFrame(data).to_csv(cls.csv_file, index=False)

    def test_25_step_alternating_pipeline(self):
        """
        25 steps alternating between SQL and Pandas operations.

        Pattern: SQL -> Pandas -> SQL -> Pandas -> ... with exact verification.
        """
        ds = DataStore.from_file(self.csv_file)

        # Step 1: [SQL] filter x > 30
        ds = ds.filter(ds['x'] > 30)
        # After: ids 4-20 (17 rows)

        # Step 2: [Pandas] assign col_a = x * 2
        ds['col_a'] = ds['x'] * 2

        # Step 3: [SQL] filter col_a > 100
        ds = ds.filter(ds['col_a'] > 100)
        # After: x > 50, ids 6-20 (15 rows)

        # Step 4: [Pandas] assign col_b = col_a + y
        ds['col_b'] = ds['col_a'] + ds['y']

        # Step 5: [SQL] sort by col_b ascending
        ds = ds.sort('col_b', ascending=True)

        # Step 6: [Pandas] assign col_c = col_b / 10
        ds['col_c'] = ds['col_b'] / 10

        # Step 7: [SQL] filter col_c > 30
        ds = ds.filter(ds['col_c'] > 30)

        # Step 8: [Pandas] assign col_d = col_c - 20
        ds['col_d'] = ds['col_c'] - 20

        # Step 9: [SQL] filter y > 150
        ds = ds.filter(ds['y'] > 150)

        # Step 10: [Pandas] assign col_e = col_d * col_c
        ds['col_e'] = ds['col_d'] * ds['col_c']

        # Step 11: [SQL] sort by x descending
        ds = ds.sort('x', ascending=False)

        # Step 12: [Pandas] assign col_f = x + col_a
        ds['col_f'] = ds['x'] + ds['col_a']

        # Step 13: [SQL] limit 10
        ds = ds.limit(10)

        # Step 14: [Pandas] assign col_g = col_f - col_b
        ds['col_g'] = ds['col_f'] - ds['col_b']

        # Step 15: [SQL] filter col_f > 300
        ds = ds.filter(ds['col_f'] > 300)

        # Step 16: [Pandas] assign col_h = col_g + 100
        ds['col_h'] = ds['col_g'] + 100

        # Step 17: [SQL] sort by col_h ascending
        ds = ds.sort('col_h', ascending=True)

        # Step 18: [Pandas] assign col_i = col_h / 2
        ds['col_i'] = ds['col_h'] / 2

        # Step 19: [SQL] filter id > 10
        ds = ds.filter(ds['id'] > 10)

        # Step 20: [Pandas] assign col_j = col_i + col_e
        ds['col_j'] = ds['col_i'] + ds['col_e']

        # Step 21: [SQL] limit 5
        ds = ds.limit(5)

        # Step 22: [Pandas] assign col_k = col_j * 2
        ds['col_k'] = ds['col_j'] * 2

        # Step 23: [SQL] sort by id ascending
        ds = ds.sort('id', ascending=True)

        # Step 24: [Pandas] assign final_check = constant 1
        ds['final_check'] = ds['id'] * 0 + 1

        # Step 25: [SQL] filter final_check == 1 (should keep all)
        ds = ds.filter(ds['final_check'] == 1)

        df = ds.to_df()

        # Verify structure
        self.assertLessEqual(len(df), 5)
        self.assertIn('col_a', df.columns)
        self.assertIn('col_k', df.columns)
        self.assertIn('final_check', df.columns)

        # Verify computed values
        self.assertTrue((df['col_a'] == df['x'] * 2).all())
        self.assertTrue((df['col_b'] == df['col_a'] + df['y']).all())
        self.assertTrue((df['col_c'] == df['col_b'] / 10).all())
        self.assertTrue((df['final_check'] == 1).all())

        # Verify all filters applied
        self.assertTrue(all(df['x'] > 30))
        self.assertTrue(all(df['col_a'] > 100))
        self.assertTrue(all(df['y'] > 150))
        self.assertTrue(all(df['id'] > 10))

    def test_30_step_with_column_dependencies(self):
        """
        30 steps with complex column dependencies across steps.

        Each computed column depends on previously computed columns.
        """
        ds = DataStore.from_file(self.csv_file)

        # Phase 1: Initial setup (Steps 1-5)
        ds = ds.filter(ds['x'] >= 50)  # Step 1 [SQL]
        ds['p1'] = ds['x'] + 10  # Step 2 [Pandas]
        ds = ds.filter(ds['y'] > 120)  # Step 3 [SQL]
        ds['p2'] = ds['p1'] * 2  # Step 4 [Pandas] - depends on p1
        ds = ds.sort('x', ascending=True)  # Step 5 [SQL]

        # Phase 2: Build dependency chain (Steps 6-15)
        ds['p3'] = ds['p2'] - ds['p1']  # Step 6 [Pandas] - depends on p1, p2
        ds = ds.filter(ds['p3'] > 50)  # Step 7 [SQL]
        ds['p4'] = ds['p3'] + ds['x']  # Step 8 [Pandas]
        ds = ds.limit(15)  # Step 9 [SQL]
        ds['p5'] = ds['p4'] / 2  # Step 10 [Pandas]
        ds = ds.filter(ds['p5'] > 60)  # Step 11 [SQL]
        ds['p6'] = ds['p5'] + ds['p3']  # Step 12 [Pandas]
        ds = ds.sort('p6', ascending=False)  # Step 13 [SQL]
        ds['p7'] = ds['p6'] - ds['p4']  # Step 14 [Pandas]
        ds = ds.limit(10)  # Step 15 [SQL]

        # Phase 3: Continue chain (Steps 16-25)
        ds['p8'] = ds['p7'] * 2  # Step 16 [Pandas]
        ds = ds.filter(ds['p8'] > -100)  # Step 17 [SQL]
        ds['p9'] = ds['p8'] + ds['p6']  # Step 18 [Pandas]
        ds = ds.sort('id', ascending=True)  # Step 19 [SQL]
        ds['p10'] = ds['p9'] / 10  # Step 20 [Pandas]
        ds = ds.filter(ds['p10'] > 5)  # Step 21 [SQL]
        ds['p11'] = ds['p10'] + 100  # Step 22 [Pandas]
        ds = ds.limit(8)  # Step 23 [SQL]
        ds['p12'] = ds['p11'] * ds['p10']  # Step 24 [Pandas]
        ds = ds.filter(ds['p12'] > 500)  # Step 25 [SQL]

        # Phase 4: Final steps (Steps 26-30)
        ds['p13'] = ds['p12'] - ds['p11']  # Step 26 [Pandas]
        ds = ds.sort('p13', ascending=True)  # Step 27 [SQL]
        ds['p14'] = ds['p13'] + ds['p12']  # Step 28 [Pandas]
        ds = ds.limit(5)  # Step 29 [SQL]
        ds['final'] = ds['id'] * 0 + 30  # Step 30 [Pandas] - marker

        df = ds.to_df()

        # Verify
        self.assertLessEqual(len(df), 5)
        self.assertTrue((df['final'] == 30).all())

        # Verify dependency chain correctness
        if len(df) > 0:
            self.assertTrue((df['p1'] == df['x'] + 10).all())
            self.assertTrue((df['p2'] == df['p1'] * 2).all())
            self.assertTrue((df['p3'] == df['p2'] - df['p1']).all())
            # p3 = p1*2 - p1 = p1 = x + 10

    def test_20_step_filter_heavy(self):
        """
        20 steps with heavy filtering to test row reduction at each step.
        """
        ds = DataStore.from_file(self.csv_file)

        # Start with 20 rows, progressively filter down
        ds['f1'] = ds['x'] + ds['y']  # Step 1 [Pandas]
        ds = ds.filter(ds['x'] > 20)  # Step 2 [SQL] - removes ~2 rows
        ds['f2'] = ds['f1'] * 2  # Step 3 [Pandas]
        ds = ds.filter(ds['y'] > 110)  # Step 4 [SQL] - removes ~1 row
        ds['f3'] = ds['f2'] - ds['x']  # Step 5 [Pandas]
        ds = ds.filter(ds['f1'] > 150)  # Step 6 [SQL]
        ds['f4'] = ds['f3'] / 10  # Step 7 [Pandas]
        ds = ds.filter(ds['id'] > 3)  # Step 8 [SQL]
        ds['f5'] = ds['f4'] + ds['f2']  # Step 9 [Pandas]
        ds = ds.filter(ds['f5'] > 400)  # Step 10 [SQL]
        ds['f6'] = ds['f5'] - ds['f3']  # Step 11 [Pandas]
        ds = ds.filter(ds['x'] < 180)  # Step 12 [SQL]
        ds['f7'] = ds['f6'] * 0.5  # Step 13 [Pandas]
        ds = ds.filter(ds['f7'] > 100)  # Step 14 [SQL]
        ds['f8'] = ds['f7'] + ds['id']  # Step 15 [Pandas]
        ds = ds.filter(ds['y'] < 280)  # Step 16 [SQL]
        ds['f9'] = ds['f8'] - ds['f7']  # Step 17 [Pandas]
        ds = ds.filter(ds['f9'] > 5)  # Step 18 [SQL]
        ds['f10'] = ds['f9'] * 10  # Step 19 [Pandas]
        ds = ds.sort('f10', ascending=False)  # Step 20 [SQL]

        df = ds.to_df()

        # Verify all filters were applied
        self.assertTrue(all(df['x'] > 20))
        self.assertTrue(all(df['y'] > 110))
        self.assertTrue(all(df['id'] > 3))
        self.assertTrue(all(df['x'] < 180))
        self.assertTrue(all(df['y'] < 280))

        # Verify computed columns
        if len(df) > 0:
            self.assertTrue((df['f1'] == df['x'] + df['y']).all())
            self.assertTrue((df['f9'] == df['f8'] - df['f7']).all())
            # f9 = f8 - f7 = (f7 + id) - f7 = id
            self.assertTrue((df['f9'] == df['id']).all())

    def test_25_step_with_boolean_filters(self):
        """
        25 steps with complex boolean conditions interleaved with computations.
        """
        ds = DataStore.from_file(self.csv_file)

        # Step 1-5
        ds['b1'] = ds['x'] * 2  # Step 1 [Pandas]
        ds = ds.filter((ds['x'] > 30) & (ds['y'] > 120))  # Step 2 [SQL] AND
        ds['b2'] = ds['b1'] + ds['y']  # Step 3 [Pandas]
        ds = ds.filter((ds['b2'] > 200) | (ds['x'] > 100))  # Step 4 [SQL] OR
        ds['b3'] = ds['b2'] / 10  # Step 5 [Pandas]

        # Step 6-10
        ds = ds.filter(ds['b3'] > 15)  # Step 6 [SQL]
        ds['b4'] = ds['b3'] - ds['x'] / 10  # Step 7 [Pandas]
        ds = ds.filter((ds['id'] > 5) & (ds['id'] < 18))  # Step 8 [SQL] range
        ds['b5'] = ds['b4'] * 3  # Step 9 [Pandas]
        ds = ds.sort('b5', ascending=True)  # Step 10 [SQL]

        # Step 11-15
        ds['b6'] = ds['b5'] + 50  # Step 11 [Pandas]
        ds = ds.filter(ds['b6'] > 100)  # Step 12 [SQL]
        ds['b7'] = ds['b6'] - ds['b3']  # Step 13 [Pandas]
        ds = ds.limit(12)  # Step 14 [SQL]
        ds['b8'] = ds['b7'] * 2  # Step 15 [Pandas]

        # Step 16-20
        ds = ds.filter((ds['b8'] > 50) & (ds['b8'] < 500))  # Step 16 [SQL]
        ds['b9'] = ds['b8'] / 4  # Step 17 [Pandas]
        ds = ds.sort('b9', ascending=False)  # Step 18 [SQL]
        ds['b10'] = ds['b9'] + ds['b7']  # Step 19 [Pandas]
        ds = ds.filter(ds['b10'] > 50)  # Step 20 [SQL]

        # Step 21-25
        ds['b11'] = ds['b10'] - 25  # Step 21 [Pandas]
        ds = ds.limit(8)  # Step 22 [SQL]
        ds['b12'] = ds['b11'] + ds['id']  # Step 23 [Pandas]
        ds = ds.sort('id', ascending=True)  # Step 24 [SQL]
        ds['marker'] = ds['id'] * 0 + 25  # Step 25 [Pandas]

        df = ds.to_df()

        # Verify
        self.assertLessEqual(len(df), 8)
        self.assertTrue((df['marker'] == 25).all())

        # Verify range filter
        self.assertTrue(all(df['id'] > 5))
        self.assertTrue(all(df['id'] < 18))

    def test_22_step_aggregation_like_pattern(self):
        """
        22 steps simulating aggregation-like patterns with running totals.
        """
        ds = DataStore.from_file(self.csv_file)

        # Simulate running calculations
        ds['running'] = ds['x']  # Step 1 [Pandas]
        ds = ds.filter(ds['x'] > 30)  # Step 2 [SQL]
        ds['running'] = ds['running'] + ds['y']  # Step 3 [Pandas] accumulate
        ds = ds.sort('running', ascending=True)  # Step 4 [SQL]
        ds['running'] = ds['running'] * 2  # Step 5 [Pandas]
        ds = ds.filter(ds['running'] > 300)  # Step 6 [SQL]
        ds['running'] = ds['running'] - 100  # Step 7 [Pandas]
        ds = ds.limit(15)  # Step 8 [SQL]
        ds['running'] = ds['running'] / 2  # Step 9 [Pandas]
        ds = ds.filter(ds['running'] > 100)  # Step 10 [SQL]
        ds['running'] = ds['running'] + 50  # Step 11 [Pandas]
        ds = ds.sort('id', ascending=True)  # Step 12 [SQL]
        ds['running'] = ds['running'] * 1.5  # Step 13 [Pandas]
        ds = ds.filter(ds['id'] > 5)  # Step 14 [SQL]
        ds['running'] = ds['running'] - 25  # Step 15 [Pandas]
        ds = ds.limit(10)  # Step 16 [SQL]
        ds['running'] = ds['running'] / 1.5  # Step 17 [Pandas]
        ds = ds.filter(ds['running'] > 80)  # Step 18 [SQL]
        ds['running'] = ds['running'] + ds['x']  # Step 19 [Pandas]
        ds = ds.sort('running', ascending=False)  # Step 20 [SQL]
        ds['running'] = ds['running'] * 0 + 22  # Step 21 [Pandas] reset
        ds = ds.limit(5)  # Step 22 [SQL]

        df = ds.to_df()

        self.assertLessEqual(len(df), 5)
        self.assertTrue((df['running'] == 22).all())

    def test_exact_verification_20_steps(self):
        """
        20 steps with exact mathematical verification at each checkpoint.
        """
        ds = DataStore.from_file(self.csv_file)

        # Use simple deterministic data: x = id * 10, y = 100 + id * 10
        # So for id=1: x=10, y=110; id=2: x=20, y=120; etc.

        # Step 1: v1 = x + y (should be x + y = 110 + 2*id*10 = 110 + 20*id)
        ds['v1'] = ds['x'] + ds['y']

        # Step 2: filter id > 5
        ds = ds.filter(ds['id'] > 5)

        # Step 3: v2 = v1 - 100 (should be 10 + 20*id)
        ds['v2'] = ds['v1'] - 100

        # Step 4: filter v2 > 150 (20*id + 10 > 150 -> id > 7)
        ds = ds.filter(ds['v2'] > 150)

        # Step 5: v3 = v2 / 10 (should be 1 + 2*id)
        ds['v3'] = ds['v2'] / 10

        # Step 6: sort by v3 ascending
        ds = ds.sort('v3', ascending=True)

        # Step 7: v4 = v3 * 5 (should be 5 + 10*id)
        ds['v4'] = ds['v3'] * 5

        # Step 8: filter id < 18
        ds = ds.filter(ds['id'] < 18)

        # Step 9: v5 = v4 + id (should be 5 + 11*id)
        ds['v5'] = ds['v4'] + ds['id']

        # Step 10: limit 8
        ds = ds.limit(8)

        # Step 11: v6 = v5 - v3 (should be 5 + 11*id - 1 - 2*id = 4 + 9*id)
        ds['v6'] = ds['v5'] - ds['v3']

        # Step 12: filter v6 > 80 (4 + 9*id > 80 -> id > 8.44 -> id >= 9)
        ds = ds.filter(ds['v6'] > 80)

        # Step 13: v7 = v6 / 9 (should be 4/9 + id ≈ 0.44 + id)
        ds['v7'] = ds['v6'] / 9

        # Step 14: sort by id descending
        ds = ds.sort('id', ascending=False)

        # Step 15: v8 = v7 * 9 (should be v6 again = 4 + 9*id)
        ds['v8'] = ds['v7'] * 9

        # Step 16: limit 5
        ds = ds.limit(5)

        # Step 17: v9 = v8 - v6 (should be 0)
        ds['v9'] = ds['v8'] - ds['v6']

        # Step 18: filter v9 < 1 (should keep all since v9 ≈ 0)
        ds = ds.filter(ds['v9'] < 1)

        # Step 19: v10 = id * 0 + 20 (constant marker)
        ds['v10'] = ds['id'] * 0 + 20

        # Step 20: sort by id ascending
        ds = ds.sort('id', ascending=True)

        df = ds.to_df()

        # Verify exact values
        self.assertLessEqual(len(df), 5)
        self.assertTrue((df['v10'] == 20).all())

        # v9 should be approximately 0 (due to floating point)
        self.assertTrue((df['v9'].abs() < 0.01).all())

        # Verify filter conditions
        self.assertTrue(all(df['id'] > 5))
        self.assertTrue(all(df['id'] < 18))


class TestExecutionEngineVerification(unittest.TestCase):
    """
    Tests that verify the correct execution engine (SQL vs Pandas) is used.

    These tests use explain() and debug logging to confirm the expected
    execution path is taken for each operation.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "engine_test.csv")

        data = {
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        }
        pd.DataFrame(data).to_csv(cls.csv_file, index=False)

    def test_pure_sql_operations_explain(self):
        """
        Verify pure SQL operations show SQL icons (🚀 [chDB]) in explain.
        """
        ds = DataStore.from_file(self.csv_file)

        # Pure SQL operations
        ds = ds.select('id', 'value')
        ds = ds.filter(ds['value'] > 30)
        ds = ds.sort('value', ascending=False)
        ds = ds.limit(5)

        explain_output = ds.explain()
        print("\n=== Pure SQL Operations ===")
        print(explain_output)

        # All operations should be SQL (🚀 [chDB])
        self.assertIn('🚀 [chDB]', explain_output)
        self.assertIn('SELECT', explain_output)
        self.assertIn('WHERE', explain_output)
        self.assertIn('ORDER', explain_output)
        self.assertIn('LIMIT', explain_output)

        # Should indicate all via SQL engine (new format: single segment with chDB)
        self.assertIn('Segment 1 [chDB]', explain_output)
        # Should NOT have any Pandas segments
        self.assertNotIn('[Pandas]', explain_output)

        # Should NOT have Pandas icons (all operations are SQL/chDB)
        self.assertNotIn('🐼', explain_output)

        df = ds.to_df()
        self.assertEqual(len(df), 5)

    def test_pandas_operations_explain(self):
        """
        Verify Pandas operations show Pandas icons (🐼 [Pandas]) in explain.
        """
        ds = DataStore.from_file(self.csv_file)

        # Pandas operations
        ds['doubled'] = ds['value'] * 2
        ds['tripled'] = ds['value'] * 3
        ds = ds.add_prefix('col_')

        explain_output = ds.explain()
        print("\n=== Pandas Operations ===")
        print(explain_output)

        # Should have 🐼 [Pandas] for pandas assign operations
        self.assertIn('🐼 [Pandas]', explain_output)
        self.assertIn('Assign', explain_output)

        # Should indicate Pandas engine in segment info
        self.assertIn('[Pandas]', explain_output)

        df = ds.to_df()
        self.assertIn('col_doubled', df.columns)

    def test_mixed_operations_explain_shows_both_engines(self):
        """
        Verify mixed operations show both SQL (🔍) and Pandas (🐼/📝) icons.
        """
        ds = DataStore.from_file(self.csv_file)

        # Step 1: SQL filter
        ds = ds.filter(ds['value'] > 20)

        # Step 2: Pandas assign
        ds['computed'] = ds['value'] * 2

        # Step 3: SQL filter on computed column (becomes Pandas execution)
        ds = ds.filter(ds['computed'] > 80)

        # Step 4: Pandas assign
        ds['final'] = ds['computed'] + 10

        # Step 5: SQL sort
        ds = ds.sort('id', ascending=True)

        explain_output = ds.explain()
        print("\n=== Mixed SQL + Pandas Operations ===")
        print(explain_output)

        # Should have SQL engine and Pandas engine operations
        self.assertIn('🚀 [chDB]', explain_output)  # SQL operations
        self.assertIn('🐼 [Pandas]', explain_output)  # Pandas relational ops
        self.assertIn('🐼 [Pandas]', explain_output)  # Pandas assign ops

        # Verify segment info is shown (new format uses Segments instead of Phases)
        self.assertIn('Segment 1 [chDB]', explain_output)
        self.assertIn('[Pandas]', explain_output)

        # Verify WHERE, Assign, ORDER BY all appear
        self.assertIn('WHERE', explain_output)
        self.assertIn('Assign', explain_output)
        self.assertIn('ORDER', explain_output)

        df = ds.to_df()
        self.assertTrue(all(df['computed'] > 80))

    def test_25_step_explain_verification(self):
        """
        25-step pipeline with explain() verification of execution engine.
        """
        ds = DataStore.from_file(self.csv_file)

        # Build 25-step pipeline
        ds = ds.filter(ds['value'] > 10)  # 1 [SQL]
        ds['s2'] = ds['value'] * 2  # 2 [Pandas]
        ds = ds.filter(ds['s2'] > 30)  # 3 [Pandas - after pandas op]
        ds['s4'] = ds['s2'] + 10  # 4 [Pandas]
        ds = ds.sort('id', ascending=True)  # 5 [Pandas]
        ds['s6'] = ds['s4'] - 5  # 6 [Pandas]
        ds = ds.filter(ds['id'] > 2)  # 7 [Pandas]
        ds['s8'] = ds['s6'] * 2  # 8 [Pandas]
        ds = ds.limit(8)  # 9 [Pandas]
        ds['s10'] = ds['s8'] / 2  # 10 [Pandas]
        ds = ds.filter(ds['s10'] > 30)  # 11 [Pandas]
        ds['s12'] = ds['s10'] + 100  # 12 [Pandas]
        ds = ds.sort('s12', ascending=False)  # 13 [Pandas]
        ds['s14'] = ds['s12'] - 50  # 14 [Pandas]
        ds = ds.filter(ds['s14'] > 50)  # 15 [Pandas]
        ds['s16'] = ds['s14'] * 0.5  # 16 [Pandas]
        ds = ds.limit(5)  # 17 [Pandas]
        ds['s18'] = ds['s16'] + ds['id']  # 18 [Pandas]
        ds = ds.sort('id', ascending=True)  # 19 [Pandas]
        ds['s20'] = ds['s18'] - ds['s16']  # 20 [Pandas] = id
        ds = ds.filter(ds['s20'] > 3)  # 21 [Pandas]
        ds['s22'] = ds['s20'] * 10  # 22 [Pandas]
        ds = ds.limit(3)  # 23 [Pandas]
        ds['s24'] = ds['s22'] + 1  # 24 [Pandas]
        ds = ds.filter(ds['s24'] > 0)  # 25 [Pandas]

        explain_output = ds.explain()
        print("\n=== 25-Step Pipeline Explain ===")
        print(explain_output)

        # Count operations by engine
        sql_ops = explain_output.count('🚀 [chDB]')
        pandas_relational_ops = explain_output.count('🐼 [Pandas]')
        pandas_assign_ops = explain_output.count('🐼 [Pandas]')

        print(f"\nSQL operations (🚀 [chDB]): {sql_ops}")
        print(f"Pandas relational ops (🐼 [Pandas]): {pandas_relational_ops}")
        print(f"Pandas assign ops (🐼 [Pandas]): {pandas_assign_ops}")

        # Should have SQL ops (first filter) and Pandas ops
        self.assertGreater(sql_ops, 0, "Should have SQL operations")
        self.assertGreater(pandas_assign_ops, 0, "Should have Pandas assign operations")
        self.assertGreater(pandas_relational_ops, 0, "Should have Pandas relational operations")

        # Verify filter, sort, limit, assign all appear
        self.assertIn('WHERE', explain_output)
        self.assertIn('ORDER', explain_output)
        self.assertIn('LIMIT', explain_output)
        self.assertIn('Assign', explain_output)

        # Verify segment info (new format uses Segments instead of Phases)
        self.assertIn('Segment', explain_output)
        self.assertIn('[chDB]', explain_output)
        self.assertIn('[Pandas]', explain_output)

        # Execute and verify result
        df = ds.to_df()
        self.assertLessEqual(len(df), 3)

    def test_debug_logging_shows_execution_path(self):
        """
        Enable debug logging and verify execution path is logged.
        """
        import io
        import sys

        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Get the datastore logger and add handler
        logger = logging.getLogger('datastore')
        original_level = logger.level
        original_handlers = logger.handlers[:]
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            ds = DataStore.from_file(self.csv_file)

            # Mixed operations
            ds = ds.filter(ds['value'] > 30)
            ds['computed'] = ds['value'] * 2
            ds = ds.filter(ds['computed'] > 100)
            ds = ds.sort('id', ascending=True)

            # Execute
            df = ds.to_df()

            # Get log output
            log_output = log_capture.getvalue()
            print("\n=== Debug Log Output ===")
            print(log_output)

            # Verify some execution was logged
            # The log should contain LazyOp or Pandas execution messages
            self.assertGreater(len(log_output), 0, "Debug log should not be empty")

        finally:
            # Restore original logging state
            logger.setLevel(original_level)
            logger.removeHandler(handler)
            logger.handlers = original_handlers

    def test_explain_verbose_mode(self):
        """
        Test explain() with verbose=True for detailed output.
        """
        ds = DataStore.from_file(self.csv_file)

        ds = ds.select('id', 'value', 'name')
        ds = ds.filter(ds['value'] > 30)
        ds['doubled'] = ds['value'] * 2
        ds = ds.filter(ds['doubled'] > 80)
        ds = ds.sort('id', ascending=False)
        ds = ds.limit(5)

        # Non-verbose
        explain_normal = ds.explain(verbose=False)
        print("\n=== Explain (Normal) ===")
        print(explain_normal)

        # Verbose
        explain_verbose = ds.explain(verbose=True)
        print("\n=== Explain (Verbose) ===")
        print(explain_verbose)

        # Both should contain key operations
        for output in [explain_normal, explain_verbose]:
            self.assertIn('WHERE', output)
            self.assertIn('Assign', output)
            self.assertIn('ORDER', output)
            self.assertIn('LIMIT', output)

        # Execute
        df = ds.to_df()
        self.assertLessEqual(len(df), 5)

    def test_verify_filter_uses_correct_engine_after_pandas_op(self):
        """
        Critical test: Verify that filter AFTER pandas op uses Pandas engine.

        This is the key behavior that could have bugs in lazy execution.
        """
        ds = DataStore.from_file(self.csv_file)

        # Step 1: SQL filter (should use SQL engine)
        ds = ds.filter(ds['value'] > 20)

        # Step 2: Pandas column assignment (forces Phase 2)
        ds['computed'] = ds['value'] * 2

        # Step 3: Filter after pandas op (should use Pandas engine)
        ds = ds.filter(ds['computed'] > 80)

        explain_output = ds.explain()
        print("\n=== Filter After Pandas Op ===")
        print(explain_output)

        # Verify correct engine markers
        # First WHERE should be SQL (🚀 [chDB])
        self.assertIn('🚀 [chDB] WHERE', explain_output)

        # Second WHERE is also executed via chDB (via Python() table function on DataFrame)
        # With segmented execution, filters after pandas ops can still use SQL
        # Count 🚀 [chDB] WHERE operations (not counting WHERE in generated SQL)
        chdb_where_count = explain_output.count('🚀 [chDB] WHERE')
        self.assertEqual(chdb_where_count, 2, "Should have 2 chDB WHERE operations")

        # Assign should be Pandas
        self.assertIn('🐼 [Pandas] Assign', explain_output)

        # Verify segment info (new format uses Segments instead of Phases)
        self.assertIn('Segment 1 [chDB]', explain_output)  # First segment with initial filter
        self.assertIn('[Pandas]', explain_output)  # Should have Pandas segment for assign

        # Verify result is correct
        df = ds.to_df()
        self.assertTrue(all(df['value'] > 20))
        self.assertTrue(all(df['computed'] > 80))
        self.assertTrue((df['computed'] == df['value'] * 2).all())

    def test_20_step_with_execution_path_trace(self):
        """
        20-step pipeline with detailed execution path tracing.

        This test prints the execution plan and verifies the engine at each step.
        """
        ds = DataStore.from_file(self.csv_file)

        operations_log = []

        # Step 1: [SQL] filter
        ds = ds.filter(ds['value'] > 10)
        operations_log.append("Step 1: SQL filter (value > 10)")

        # Step 2: [Pandas] assign - triggers Phase 2
        ds['p2'] = ds['value'] * 2
        operations_log.append("Step 2: Pandas assign (p2 = value * 2)")

        # Step 3: [Pandas] filter on computed (after pandas op)
        ds = ds.filter(ds['p2'] > 40)
        operations_log.append("Step 3: Pandas filter on computed (p2 > 40)")

        # Step 4: [Pandas] assign
        ds['p4'] = ds['p2'] + ds['value']
        operations_log.append("Step 4: Pandas assign (p4 = p2 + value)")

        # Step 5: [Pandas] sort
        ds = ds.sort('p4', ascending=False)
        operations_log.append("Step 5: Pandas sort by computed (p4)")

        # Step 6: [Pandas] assign
        ds['p6'] = ds['p4'] / 10
        operations_log.append("Step 6: Pandas assign (p6 = p4 / 10)")

        # Step 7: [Pandas] limit
        ds = ds.limit(8)
        operations_log.append("Step 7: Pandas limit 8")

        # Step 8: [Pandas] assign
        ds['p8'] = ds['p6'] * 3
        operations_log.append("Step 8: Pandas assign (p8 = p6 * 3)")

        # Step 9: [Pandas] filter
        ds = ds.filter(ds['p8'] > 30)
        operations_log.append("Step 9: Pandas filter (p8 > 30)")

        # Step 10: [Pandas] assign
        ds['p10'] = ds['p8'] - ds['p6']
        operations_log.append("Step 10: Pandas assign (p10 = p8 - p6)")

        # Step 11: [Pandas] sort
        ds = ds.sort('id', ascending=True)
        operations_log.append("Step 11: Pandas sort by id")

        # Step 12: [Pandas] assign
        ds['p12'] = ds['p10'] + 10
        operations_log.append("Step 12: Pandas assign (p12 = p10 + 10)")

        # Step 13: [Pandas] filter
        ds = ds.filter(ds['id'] > 3)
        operations_log.append("Step 13: Pandas filter (id > 3)")

        # Step 14: [Pandas] assign
        ds['p14'] = ds['p12'] * 2
        operations_log.append("Step 14: Pandas assign (p14 = p12 * 2)")

        # Step 15: [Pandas] limit
        ds = ds.limit(5)
        operations_log.append("Step 15: Pandas limit 5")

        # Step 16: [Pandas] assign
        ds['p16'] = ds['p14'] / 4
        operations_log.append("Step 16: Pandas assign (p16 = p14 / 4)")

        # Step 17: [Pandas] filter
        ds = ds.filter(ds['p16'] > 10)
        operations_log.append("Step 17: Pandas filter (p16 > 10)")

        # Step 18: [Pandas] assign
        ds['p18'] = ds['p16'] + ds['id']
        operations_log.append("Step 18: Pandas assign (p18 = p16 + id)")

        # Step 19: [Pandas] sort
        ds = ds.sort('p18', ascending=False)
        operations_log.append("Step 19: Pandas sort by p18")

        # Step 20: [Pandas] assign marker
        ds['step_marker'] = ds['id'] * 0 + 20
        operations_log.append("Step 20: Pandas assign marker (= 20)")

        # Print operation log
        print("\n=== Operation Log ===")
        for op in operations_log:
            print(f"  {op}")

        # Print explain
        explain_output = ds.explain()
        print("\n=== Execution Plan ===")
        print(explain_output)

        # Verify structure - should have SQL and Pandas engine markers
        self.assertIn('🚀 [chDB]', explain_output)  # First filter uses SQL
        self.assertIn('🐼 [Pandas]', explain_output)  # Subsequent relational ops use Pandas
        self.assertIn('🐼 [Pandas]', explain_output)  # Assign ops use Pandas

        # Count operations by engine
        sql_count = explain_output.count('🚀 [chDB]')
        pandas_relational = explain_output.count('🐼 [Pandas]')
        pandas_assign = explain_output.count('🐼 [Pandas]')

        print(f"\n=== Engine Usage ===")
        print(f"SQL operations (🚀 [chDB]): {sql_count}")
        print(f"Pandas relational (🐼 [Pandas]): {pandas_relational}")
        print(f"Pandas assign (🐼 [Pandas]): {pandas_assign}")

        # Verify segment info (new format uses Segments instead of Phases)
        self.assertIn('Segment', explain_output)
        self.assertIn('[chDB]', explain_output)
        self.assertIn('[Pandas]', explain_output)

        # Execute and verify
        df = ds.to_df()

        print(f"\n=== Result ===")
        print(f"Rows: {len(df)}")
        print(df.to_string())

        # Verify marker
        self.assertTrue((df['step_marker'] == 20).all())

        # Verify computed columns are correct
        if len(df) > 0:
            self.assertTrue((df['p2'] == df['value'] * 2).all())
            self.assertTrue((df['p4'] == df['p2'] + df['value']).all())
            self.assertTrue(all(df['id'] > 3))


class TestLazyPipelineInterleaving(unittest.TestCase):
    """
    Tests for lazy pipeline interleaving using DataStore pandas-style API.

    This demonstrates the ability to:
    1. Filter data using pandas-style API
    2. Add computed columns lazily
    3. Chain filter/sort/limit operations
    4. Compare results with pure pandas (Mirror Code Pattern)
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "interleave_test.csv")

        cls.data = {
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        }
        pd.DataFrame(cls.data).to_csv(cls.csv_file, index=False)

    def test_filter_assign_filter_sort_limit_pipeline(self):
        """
        Test filter -> column assignment -> filter -> sort -> limit pipeline.

        Using pandas-style API throughout, comparing with pure pandas.
        """
        # ============ Pandas operations (reference) ============
        pd_df = pd.DataFrame(self.data)
        pd_df = pd_df[pd_df['value'] > 20]
        pd_df = pd_df[['id', 'value', 'category']].copy()
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df['score'] = pd_df['value'] + pd_df['id']
        pd_result = pd_df[pd_df['doubled'] > 100].sort_values('score', ascending=False).head(5)

        # ============ DataStore operations (mirror) ============
        ds = DataStore.from_file(self.csv_file)
        ds = ds.filter(ds['value'] > 20)
        ds = ds.select('id', 'value', 'category')
        ds['doubled'] = ds['value'] * 2
        ds['score'] = ds['value'] + ds['id']
        ds = ds.filter(ds['doubled'] > 100)
        ds = ds.sort_values('score', ascending=False)
        ds = ds.head(5)

        # ============ Compare results ============
        assert_datastore_equals_pandas(ds, pd_result.reset_index(drop=True))

    def test_multi_round_filter_assign_filter_assign_pattern(self):
        """
        Multiple rounds of filter -> assign -> filter -> assign.

        All operations through DataStore pandas-style API.
        """
        # ============ Pandas operations (reference) ============
        pd_df = pd.DataFrame(self.data)
        # Round 1: filter
        pd_df = pd_df[pd_df['value'] >= 20].copy()
        # Round 2: assign
        pd_df['r2_computed'] = pd_df['value'] * 3
        # Round 3: filter + assign
        pd_df = pd_df[pd_df['r2_computed'] > 100].copy()
        pd_df['r3_normalized'] = pd_df['r2_computed'] / 10
        # Round 4: assign
        pd_df['r4_final'] = pd_df['r3_normalized'] + pd_df['id']
        # Round 5: filter + sort + limit
        pd_result = (
            pd_df[pd_df['r4_final'] > 15][['id', 'value', 'r2_computed', 'r3_normalized', 'r4_final']]
            .sort_values('r4_final', ascending=False)
            .head(3)
        )

        # ============ DataStore operations (mirror) ============
        ds = DataStore.from_file(self.csv_file)
        # Round 1: filter
        ds = ds.filter(ds['value'] >= 20)
        # Round 2: assign
        ds['r2_computed'] = ds['value'] * 3
        # Round 3: filter + assign
        ds = ds.filter(ds['r2_computed'] > 100)
        ds['r3_normalized'] = ds['r2_computed'] / 10
        # Round 4: assign
        ds['r4_final'] = ds['r3_normalized'] + ds['id']
        # Round 5: filter + sort + limit
        ds = ds.filter(ds['r4_final'] > 15)
        ds = ds.select('id', 'value', 'r2_computed', 'r3_normalized', 'r4_final')
        ds = ds.sort_values('r4_final', ascending=False)
        ds = ds.head(3)

        # ============ Compare results ============
        assert_datastore_equals_pandas(ds, pd_result.reset_index(drop=True))

    def test_complex_expression_filter_on_computed_columns(self):
        """
        Test complex boolean expression filter on computed columns.
        """
        # ============ Pandas operations (reference) ============
        pd_df = pd.DataFrame(self.data)
        pd_df['value_squared'] = pd_df['value'] ** 2
        pd_df['ratio'] = pd_df['value_squared'] / pd_df['value']
        # Complex filter: value_squared > 2000 AND ratio > 50
        pd_result = pd_df[(pd_df['value_squared'] > 2000) & (pd_df['ratio'] > 50)]
        pd_result = pd_result[['id', 'value', 'value_squared', 'ratio']].sort_values('id')

        # ============ DataStore operations (mirror) ============
        ds = DataStore.from_file(self.csv_file)
        ds['value_squared'] = ds['value'] ** 2
        ds['ratio'] = ds['value_squared'] / ds['value']
        ds = ds.filter((ds['value_squared'] > 2000) & (ds['ratio'] > 50))
        ds = ds.select('id', 'value', 'value_squared', 'ratio')
        ds = ds.sort_values('id')

        # ============ Compare results ============
        assert_datastore_equals_pandas(ds, pd_result.reset_index(drop=True))

    def test_groupby_aggregation_on_computed_columns(self):
        """
        Test groupby aggregation on computed columns using named aggregation syntax.
        """
        # ============ Pandas operations (reference) ============
        pd_df = pd.DataFrame(self.data)
        pd_df['value_squared'] = pd_df['value'] ** 2
        pd_result = (
            pd_df.groupby('category')
            .agg(
                cnt=('id', 'count'),
                sum_value=('value', 'sum'),
                avg_squared=('value_squared', 'mean'),
                max_value=('value', 'max'),
                min_value=('value', 'min'),
            )
            .reset_index()
        )

        # ============ DataStore operations (mirror) ============
        ds = DataStore.from_df(pd.DataFrame(self.data))
        ds['value_squared'] = ds['value'] ** 2
        ds_result = ds.groupby('category').agg(
            cnt=('id', 'count'),
            sum_value=('value', 'sum'),
            avg_squared=('value_squared', 'mean'),
            max_value=('value', 'max'),
            min_value=('value', 'min'),
        ).reset_index()  # Match pandas which also calls reset_index()

        # ============ Compare results (order may differ) ============
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_after_transformation(self):
        """
        Test merge (join) between two transformed DataStores.
        """
        # ============ Pandas operations (reference) ============
        pd_df1 = pd.DataFrame(self.data)
        pd_df1['score'] = pd_df1['value'] * 2

        pd_df2 = pd_df1[pd_df1['category'] == 'A'][['id', 'value']].copy()
        pd_df2['bonus'] = pd_df2['value'] * 0.1
        pd_df2 = pd_df2[['id', 'bonus']]

        pd_result = pd_df1.merge(pd_df2, on='id', how='inner')
        pd_result['total'] = pd_result['score'] + pd_result['bonus']
        pd_result = pd_result[['id', 'value', 'score', 'bonus', 'total']].sort_values('total', ascending=False)

        # ============ DataStore operations (mirror) ============
        ds1 = DataStore.from_df(pd.DataFrame(self.data))
        ds1['score'] = ds1['value'] * 2

        ds2 = ds1.filter(ds1['category'] == 'A').select('id', 'value')
        ds2['bonus'] = ds2['value'] * 0.1
        ds2 = ds2.select('id', 'bonus')

        ds_result = ds1.merge(ds2, on='id', how='inner')
        ds_result['total'] = ds_result['score'] + ds_result['bonus']
        ds_result = ds_result.select('id', 'value', 'score', 'bonus', 'total')
        ds_result = ds_result.sort_values('total', ascending=False)

        # ============ Compare results ============
        assert_datastore_equals_pandas(ds_result, pd_result.reset_index(drop=True))

    def test_explain_shows_execution_plan(self):
        """
        Test that explain() shows the execution plan for the pipeline.
        """
        ds = DataStore.from_file(self.csv_file)
        ds = ds.filter(ds['value'] > 30)
        ds['computed'] = ds['value'] * 2
        ds = ds.filter(ds['computed'] > 120)
        ds = ds.sort_values('computed', ascending=False)
        ds = ds.head(3)

        explain_output = ds.explain()

        # Verify explain shows useful information
        self.assertIn('[chDB]', explain_output)
        # The explain should show filter/sort/limit operations
        self.assertIsInstance(explain_output, str)
        self.assertGreater(len(explain_output), 0)

        # Also verify the actual result is correct
        pd_df = pd.DataFrame(self.data)
        pd_df = pd_df[pd_df['value'] > 30].copy()
        pd_df['computed'] = pd_df['value'] * 2
        pd_result = pd_df[pd_df['computed'] > 120].sort_values('computed', ascending=False).head(3)

        assert_datastore_equals_pandas(ds, pd_result.reset_index(drop=True))


class TestSQLMethodInPipeline(unittest.TestCase):
    """
    Tests for the .sql() method that enables SQL-Pandas-SQL interleaving in the lazy pipeline.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "sql_method_test.csv")

        data = {
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        }
        pd.DataFrame(data).to_csv(cls.csv_file, index=False)

    def test_sql_method_basic(self):
        """
        Test basic .sql() method usage with SHORT FORM syntax.
        """
        ds = DataStore.from_file(self.csv_file)

        # Filter and compute
        ds = ds.filter(ds['value'] > 20)
        ds['doubled'] = ds['value'] * 2

        # Use .sql() with SHORT FORM - just the condition!
        # This expands to: SELECT * FROM __df__ WHERE doubled > 100 ORDER BY id
        ds = ds.sql("doubled > 100 ORDER BY id")

        df = ds.to_df()

        print("\n=== Basic .sql() method (short form) ===")
        print(df.to_string())

        # Verify
        self.assertGreater(len(df), 0)
        self.assertTrue(all(df['doubled'] > 100))

    def test_sql_method_short_form_variants(self):
        """
        Test various short form syntax variants.
        """
        ds = DataStore.from_file(self.csv_file)

        # Test 1: Condition only
        ds1 = ds.sql("value > 50")
        df1 = ds1.to_df()
        self.assertTrue(all(df1['value'] > 50))
        print(f"\n=== Short form 'value > 50' ===")
        print(f"Rows: {len(df1)}")

        # Test 2: Condition with ORDER BY
        ds2 = ds.sql("value > 30 ORDER BY value DESC")
        df2 = ds2.to_df()
        self.assertTrue(all(df2['value'] > 30))
        values = df2['value'].tolist()
        self.assertEqual(values, sorted(values, reverse=True))
        print(f"\n=== Short form 'value > 30 ORDER BY value DESC' ===")
        print(df2.to_string())

        # Test 3: ORDER BY only (no WHERE)
        ds3 = ds.sql("ORDER BY id DESC LIMIT 3")
        df3 = ds3.to_df()
        self.assertEqual(len(df3), 3)
        print(f"\n=== Short form 'ORDER BY id DESC LIMIT 3' ===")
        print(df3.to_string())

        # Test 4: LIMIT only
        ds4 = ds.sql("LIMIT 5")
        df4 = ds4.to_df()
        self.assertEqual(len(df4), 5)
        print(f"\n=== Short form 'LIMIT 5' ===")
        print(f"Rows: {len(df4)}")

    def test_sql_method_full_form_still_works(self):
        """
        Test that full SQL form still works.
        """
        ds = DataStore.from_file(self.csv_file)

        # Full SQL form should work as before
        ds = ds.sql("SELECT id, value, category FROM __df__ WHERE value > 50 ORDER BY value")

        df = ds.to_df()

        print("\n=== Full SQL form ===")
        print(df.to_string())

        # Verify
        self.assertTrue(all(df['value'] > 50))
        self.assertEqual(list(df.columns), ['id', 'value', 'category'])

    def test_sql_method_with_aggregation(self):
        """
        Test .sql() method with SQL aggregation.
        """
        ds = DataStore.from_file(self.csv_file)

        # Add computed column
        ds['score'] = ds['value'] * 2

        # Use SQL aggregation
        ds = ds.sql(
            """
            SELECT category, COUNT(*) as cnt, SUM(score) as total_score, AVG(value) as avg_value
            FROM __df__
            GROUP BY category
            ORDER BY total_score DESC
        """
        )

        df = ds.to_df()

        print("\n=== .sql() with aggregation ===")
        print(df.to_string())

        # Verify aggregation
        self.assertEqual(len(df), 2)  # 2 categories
        self.assertIn('cnt', df.columns)
        self.assertIn('total_score', df.columns)

    def test_sql_method_chain(self):
        """
        Test chaining .sql() with other operations.
        """
        ds = DataStore.from_file(self.csv_file)

        # SQL -> Pandas -> SQL -> Pandas chain
        ds = ds.filter(ds['value'] >= 30)  # SQL filter
        ds['computed'] = ds['value'] * 3  # Pandas assign

        # First SQL on DataFrame
        ds = ds.sql("SELECT id, value, computed, category FROM __df__ WHERE computed > 150")

        # Continue with Pandas
        ds['final'] = ds['computed'] + ds['id']

        # Second SQL on DataFrame
        ds = ds.sql("SELECT * FROM __df__ ORDER BY final DESC LIMIT 3")

        df = ds.to_df()

        print("\n=== .sql() chain ===")
        print(df.to_string())

        # Verify
        self.assertLessEqual(len(df), 3)
        self.assertIn('final', df.columns)

    def test_sql_method_explain(self):
        """
        Test that .sql() operations appear correctly in explain().
        """
        ds = DataStore.from_file(self.csv_file)

        ds = ds.filter(ds['value'] > 20)
        ds['doubled'] = ds['value'] * 2
        ds = ds.sql("SELECT * FROM __df__ WHERE doubled > 80")
        ds = ds.add_prefix('result_')

        explain_output = ds.explain()

        print("\n=== .sql() in explain() ===")
        print(explain_output)

        # Verify the SQL query operation appears with the correct icon
        self.assertIn('🚀 [chDB]', explain_output)
        self.assertIn('SQL Query:', explain_output)

    def test_sql_method_with_window_function(self):
        """
        Test .sql() with SQL window functions.
        """
        ds = DataStore.from_file(self.csv_file)

        # Use window function
        ds = ds.sql(
            """
            SELECT
                id, value, category,
                ROW_NUMBER() OVER (PARTITION BY category ORDER BY value DESC) as rank_in_category,
                SUM(value) OVER (PARTITION BY category) as category_total
            FROM __df__
            ORDER BY category, rank_in_category
        """
        )

        df = ds.to_df()

        print("\n=== .sql() with window function ===")
        print(df.to_string())

        # Verify window function results
        self.assertIn('rank_in_category', df.columns)
        self.assertIn('category_total', df.columns)

    def test_sql_method_multiple_sql_calls(self):
        """
        Test multiple .sql() calls in a pipeline.
        """
        ds = DataStore.from_file(self.csv_file)

        # First SQL
        ds = ds.sql("SELECT * FROM __df__ WHERE value > 20")

        # Pandas
        ds['step1'] = ds['value'] * 2

        # Second SQL
        ds = ds.sql("SELECT *, step1 / 10 as step2 FROM __df__ WHERE step1 > 60")

        # More Pandas
        ds['step3'] = ds['step2'] + ds['id']

        # Third SQL
        ds = ds.sql("SELECT id, value, step1, step2, step3 FROM __df__ ORDER BY step3 DESC LIMIT 5")

        df = ds.to_df()

        print("\n=== Multiple .sql() calls ===")
        print(df.to_string())

        # Verify
        self.assertLessEqual(len(df), 5)
        self.assertIn('step3', df.columns)

    def test_sql_method_with_where_prefix(self):
        """
        Test that .sql() handles queries starting with WHERE keyword.

        Regression test for issue where queries starting with WHERE
        would be double-prefixed, producing invalid SQL like:
        SELECT * FROM __df__ WHERE WHERE value > 100
        """
        ds = DataStore.from_file(self.csv_file)

        # Test 1: Query starting with WHERE keyword
        ds1 = ds.sql("WHERE value > 50")
        df1 = ds1.to_df()
        self.assertTrue(all(df1['value'] > 50))
        print(f"\n=== Query with WHERE prefix ===")
        print(f"Query: 'WHERE value > 50'")
        print(f"Rows: {len(df1)}")

        # Test 2: Verify it produces the same result as short form without WHERE
        ds2 = ds.sql("value > 50")
        df2 = ds2.to_df()
        # Row order is now preserved with the pre-added index column solution
        assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))
        print(f"Confirmed: 'WHERE value > 50' === 'value > 50'")

        # Test 3: WHERE with ORDER BY and LIMIT
        ds3 = ds.sql("WHERE value > 30 ORDER BY value DESC LIMIT 3")
        df3 = ds3.to_df()
        self.assertEqual(len(df3), 3)
        self.assertTrue(all(df3['value'] > 30))
        values = df3['value'].tolist()
        self.assertEqual(values, sorted(values, reverse=True))
        print(f"\n=== WHERE with ORDER BY and LIMIT ===")
        print(df3.to_string())


if __name__ == '__main__':
    unittest.main(verbosity=2)
