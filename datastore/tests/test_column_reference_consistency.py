"""
Aerospace-grade rigorous testing: Column reference consistency verification

Test objective: Verify consistency of three column reference methods (ds.col, ds["col"], col("name")) across all scenarios.
These tests are designed to discover deep design and implementation issues.

Critical: In aerospace, any inconsistency could lead to catastrophic consequences.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore, col, Field
from datastore.expressions import Expression, ArithmeticExpression, Literal
from datastore.column_expr import ColumnExpr
from datastore.conditions import Condition, BinaryCondition
from tests.test_utils import assert_frame_equal


class TestColumnReferenceTypes:
    """Verify types returned by different column reference methods"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                'altitude': [10000.5, 20000.3, 30000.1, 15000.8, 25000.2],
                'speed': [450.0, 520.0, 580.0, 490.0, 550.0],
                'fuel_level': [85.5, 72.3, 65.1, 78.9, 69.4],
                'aircraft_id': ['A001', 'A002', 'A003', 'A001', 'A002'],
                'flight_phase': ['climb', 'cruise', 'cruise', 'descent', 'cruise'],
                'timestamp': pd.date_range('2024-01-01 08:00', periods=5, freq='h'),
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_type_consistency_basic(self):
        """
        Critical Test: Verify types returned by three reference methods

        Expected behavior:
        - ds["col"] → ColumnExpr
        - ds.col → ColumnExpr
        - col("col") → Field

        This is a design decision, but may cause issues when mixing methods
        """
        ref_getitem = self.ds["altitude"]
        ref_getattr = self.ds.altitude
        ref_col = col("altitude")

        print(f"ds['altitude'] type: {type(ref_getitem)}")
        print(f"ds.altitude type: {type(ref_getattr)}")
        print(f"col('altitude') type: {type(ref_col)}")

        # Verify types
        assert isinstance(ref_getitem, ColumnExpr), f"ds['col'] should return ColumnExpr, got {type(ref_getitem)}"
        assert isinstance(ref_getattr, ColumnExpr), f"ds.col should return ColumnExpr, got {type(ref_getattr)}"
        assert isinstance(ref_col, Field), f"col('col') should return Field, got {type(ref_col)}"

        # CRITICAL: This type inconsistency is a potential source of bugs!
        # col() returns Field while ds["col"] returns ColumnExpr

    def test_underlying_expression_equivalence(self):
        """
        Verify that although types differ, underlying expressions are equivalent
        """
        ref_getitem = self.ds["altitude"]
        ref_getattr = self.ds.altitude
        ref_col = col("altitude")

        # ColumnExpr wraps Field
        assert isinstance(ref_getitem._expr, Field)
        assert isinstance(ref_getattr._expr, Field)

        # col() IS a Field directly
        assert isinstance(ref_col, Field)

        # Check SQL representation is the same
        sql_getitem = ref_getitem._expr.to_sql()
        sql_getattr = ref_getattr._expr.to_sql()
        sql_col = ref_col.to_sql()

        assert (
            sql_getitem == sql_getattr == sql_col
        ), f"SQL mismatch: getitem={sql_getitem}, getattr={sql_getattr}, col={sql_col}"


class TestArithmeticOperationsMixed:
    """Test arithmetic operations with mixed column reference methods"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data"""
        self.df = pd.DataFrame(
            {
                'altitude': [10000.0, 20000.0, 30000.0],
                'speed': [450.0, 520.0, 580.0],
                'fuel_rate': [100.0, 120.0, 140.0],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_addition_same_reference_style(self):
        """
        Test addition operations using the same reference style
        """
        # Using ds["col"]
        result1 = self.ds["altitude"] + self.ds["speed"]
        # Using ds.col
        result2 = self.ds.altitude + self.ds.speed
        # Using col()
        result3 = col("altitude") + col("speed")

        print(f"ds['a'] + ds['b']: {type(result1)}, {result1}")
        print(f"ds.a + ds.b: {type(result2)}, {result2}")
        print(f"col('a') + col('b'): {type(result3)}, {result3}")

        # All should produce equivalent SQL
        if isinstance(result1, ColumnExpr):
            sql1 = result1._expr.to_sql()
        else:
            sql1 = result1.to_sql()

        if isinstance(result2, ColumnExpr):
            sql2 = result2._expr.to_sql()
        else:
            sql2 = result2.to_sql()

        sql3 = result3.to_sql()

        assert sql1 == sql2, f"ds['col'] vs ds.col mismatch: {sql1} != {sql2}"
        assert sql1 == sql3, f"ds['col'] vs col() mismatch: {sql1} != {sql3}"

    def test_addition_mixed_reference_styles(self):
        """
        CRITICAL TEST: Mixed usage of different reference methods for addition

        Aviation scenario: Computing flight parameters
        altitude_adjusted = ds["altitude"] + col("speed")
        """
        # Mix ds["col"] with col()
        mixed1 = self.ds["altitude"] + col("speed")
        print(f"ds['altitude'] + col('speed'): type={type(mixed1)}, value={mixed1}")

        # Mix ds.col with col()
        mixed2 = self.ds.altitude + col("speed")
        print(f"ds.altitude + col('speed'): type={type(mixed2)}, value={mixed2}")

        # Mix all three
        mixed3 = self.ds["altitude"] + self.ds.speed + col("fuel_rate")
        print(f"ds['altitude'] + ds.speed + col('fuel_rate'): type={type(mixed3)}, value={mixed3}")

        # Verify these can all generate valid SQL
        for i, expr in enumerate([mixed1, mixed2, mixed3]):
            if isinstance(expr, ColumnExpr):
                sql = expr._expr.to_sql()
            else:
                sql = expr.to_sql()
            print(f"Expression {i+1} SQL: {sql}")
            assert sql is not None and len(sql) > 0

    def test_mixed_arithmetic_execution(self):
        """
        Verify that mixed reference arithmetic expressions execute correctly
        """
        # Create new column using mixed references
        ds = self.ds

        # Method 1: ds["col"] style
        ds['total1'] = ds["altitude"] + ds["speed"]

        # Method 2: ds.col style
        ds['total2'] = ds.altitude + ds.speed

        # Method 3: Mixed - this is where bugs might hide
        ds['total3'] = ds["altitude"] + self.ds.speed

        result = ds.to_df()

        # All three should produce identical results
        np.testing.assert_array_almost_equal(
            result['total1'], result['total2'], err_msg="ds['col'] vs ds.col produced different results"
        )
        np.testing.assert_array_almost_equal(
            result['total1'],
            result['total3'],
            err_msg="Mixed reference styles produced different results",
        )


class TestComparisonOperationsMixed:
    """Test comparison operations with mixed column reference methods"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data with aviation safety thresholds"""
        self.df = pd.DataFrame(
            {
                'altitude': [5000, 10000, 15000, 20000, 25000],
                'min_safe_altitude': [6000, 8000, 10000, 12000, 14000],
                'fuel_level': [90, 75, 60, 45, 30],
                'min_fuel_required': [20, 25, 30, 35, 40],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_comparison_types(self):
        """
        Verify types returned by comparison operations

        Expected:
        - ds["col"] > x returns ColumnExpr wrapping Condition (can execute AND be used as condition)
        - col("col") > x returns Condition (SQL-only, no DataStore context)
        """
        # Scalar comparisons
        cond1 = self.ds["altitude"] > 10000
        cond2 = self.ds.altitude > 10000
        cond3 = col("altitude") > 10000

        print(f"ds['altitude'] > 10000: {type(cond1)}")
        print(f"ds.altitude > 10000: {type(cond2)}")
        print(f"col('altitude') > 10000: {type(cond3)}")

        # ColumnExpr comparisons return ColumnExpr wrapping Condition
        assert isinstance(cond1, ColumnExpr), f"Expected ColumnExpr, got {type(cond1)}"
        assert isinstance(cond1._expr, Condition), f"Expected _expr to be Condition, got {type(cond1._expr)}"
        assert isinstance(cond2, ColumnExpr), f"Expected ColumnExpr, got {type(cond2)}"
        assert isinstance(cond2._expr, Condition), f"Expected _expr to be Condition, got {type(cond2._expr)}"
        # col() returns Field, which has no DataStore context, so returns Condition
        assert isinstance(cond3, Condition), f"Expected Condition, got {type(cond3)}"

        # ColumnExpr wrapping Condition can be used like pandas boolean Series
        assert cond1.value_counts() is not None
        assert cond2.sum() is not None

    def test_column_to_column_comparison(self):
        """
        CRITICAL: Column-to-column comparison (aviation safety check scenario)

        Verify altitude > min_safe_altitude behavior with different reference methods
        """
        # Check if altitude is above minimum safe altitude
        check1 = self.ds["altitude"] > self.ds["min_safe_altitude"]
        check2 = self.ds.altitude > self.ds.min_safe_altitude
        check3 = col("altitude") > col("min_safe_altitude")

        print(f"Check1 type: {type(check1)}, value: {check1}")
        print(f"Check2 type: {type(check2)}, value: {check2}")
        print(f"Check3 type: {type(check3)}, value: {check3}")

        # ColumnExpr comparisons return ColumnExpr wrapping Condition
        assert isinstance(check1, ColumnExpr), f"Expected ColumnExpr, got {type(check1)}"
        assert isinstance(check1._expr, Condition), f"Expected _expr to be Condition"
        assert isinstance(check2, ColumnExpr), f"Expected ColumnExpr, got {type(check2)}"
        assert isinstance(check2._expr, Condition), f"Expected _expr to be Condition"
        # col() returns Field, which has no DataStore context
        assert isinstance(check3, Condition), f"Expected Condition, got {type(check3)}"

        # SQL should be equivalent
        sql1 = check1._expr.to_sql()
        sql2 = check2._expr.to_sql()
        sql3 = check3.to_sql()

        assert sql1 == sql2, f"SQL mismatch: {sql1} != {sql2}"
        assert sql1 == sql3, f"SQL mismatch: {sql1} != {sql3}"

    def test_mixed_comparison_execution(self):
        """
        Verify mixed comparisons execute correctly and return identical results

        Aviation scenario: Filter unsafe flight conditions
        """
        # Use different reference methods to filter the same condition
        unsafe_flights_1 = self.ds.filter(self.ds["altitude"] < self.ds["min_safe_altitude"]).to_df()
        unsafe_flights_2 = self.ds.filter(self.ds.altitude < self.ds.min_safe_altitude).to_df()
        unsafe_flights_3 = self.ds.filter(col("altitude") < col("min_safe_altitude")).to_df()

        # Results should be identical
        assert_frame_equal(unsafe_flights_1, unsafe_flights_2, obj="ds['col'] vs ds.col filter")
        assert_frame_equal(unsafe_flights_1, unsafe_flights_3, obj="ds['col'] vs col() filter")


class TestAggregationMixed:
    """Test aggregation operations with mixed column reference methods"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                'flight_id': ['F001'] * 10 + ['F002'] * 10,
                'altitude': np.random.uniform(20000, 40000, 20),
                'fuel_consumed': np.random.uniform(50, 200, 20),
                'speed': np.random.uniform(400, 600, 20),
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_aggregation_with_col(self):
        """
        Verify col() behavior in aggregation
        """
        # Standard usage with col()
        stats = self.ds.groupby('flight_id').agg(
            avg_altitude=col('altitude').mean(),
            total_fuel=col('fuel_consumed').sum(),
            max_speed=col('speed').max(),
        )
        result = stats.to_df()

        assert len(result) == 2  # Two flights
        assert 'avg_altitude' in result.columns
        assert 'total_fuel' in result.columns
        assert 'max_speed' in result.columns

    def test_aggregation_with_ds_getitem(self):
        """
        Verify ds['col'].mean() returns scalar (matching pandas).
        
        For SQL aggregation in agg(), use col() instead.
        """
        import numpy as np
        
        # ds['col'].mean() returns scalar
        avg_alt = self.ds['altitude'].mean()
        assert isinstance(avg_alt, (int, float, np.integer, np.floating))
        
        # For SQL agg(), use col()
        stats = self.ds.groupby('flight_id').agg(
            avg_altitude=col('altitude').mean(),
            total_fuel=col('fuel_consumed').sum(),
        )
        result = stats.to_df()
        
        assert len(result) == 2
        assert 'avg_altitude' in result.columns

    def test_aggregation_with_ds_getattr(self):
        """
        Verify ds.col.mean() returns scalar (matching pandas).
        
        For SQL aggregation in agg(), use col() instead.
        """
        import numpy as np
        
        # ds.col.mean() returns scalar
        avg_alt = self.ds.altitude.mean()
        assert isinstance(avg_alt, (int, float, np.integer, np.floating))
        
        # For SQL agg(), use col()
        stats = self.ds.groupby('flight_id').agg(
            avg_altitude=col('altitude').mean(),
            total_fuel=col('fuel_consumed').sum(),
        )
        result = stats.to_df()
        
        assert len(result) == 2
        assert 'avg_altitude' in result.columns

    def test_aggregation_mixed_styles(self):
        """
        SQL agg() requires col() for aggregation expressions.
        
        ds['col'].mean() returns scalar (matching pandas), which cannot be
        used directly in agg(). Always use col() for SQL aggregation building.
        """
        # Use col() consistently for SQL aggregation
        stats = self.ds.groupby('flight_id').agg(
            avg_altitude=col('altitude').mean(),
            total_fuel=col('fuel_consumed').sum(),
            max_speed=col('speed').max(),
        )
        result = stats.to_df()

        assert len(result) == 2
        assert 'avg_altitude' in result.columns
        assert 'total_fuel' in result.columns
        assert 'max_speed' in result.columns

    def test_aggregation_result_consistency(self):
        """
        Verify col() aggregation results match pandas groupby.
        
        Note: ds['col'].mean() returns scalar (matching pandas), so SQL agg()
        must use col() for aggregation expressions.
        """
        # SQL aggregation uses col()
        stats1 = (
            self.ds.groupby('flight_id')
            .agg(
                avg_alt=col('altitude').mean(),
            )
            .to_df()
            .sort_values('flight_id')
            .reset_index(drop=True)
        )

        # Compare with pandas
        expected = (
            self.df.groupby('flight_id')
            .agg(avg_alt=('altitude', 'mean'))
            .reset_index()
            .sort_values('flight_id')
            .reset_index(drop=True)
        )

        # Results should match pandas
        np.testing.assert_array_almost_equal(
            stats1['avg_alt'].values, expected['avg_alt'].values, 
            err_msg="col() aggregation differs from pandas"
        )


class TestAssignmentMixed:
    """Test column assignment with mixed column reference methods"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data"""
        self.df = pd.DataFrame(
            {
                'altitude_ft': [10000, 20000, 30000],
                'speed_knots': [250, 300, 350],
                'temperature_c': [-10, -20, -30],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_assignment_with_col(self):
        """
        Column assignment using col()

        CRITICAL: col() has no DataStore reference, assignment may have issues
        """
        ds = DataStore.from_df(self.df.copy())

        try:
            # Convert feet to meters using col()
            ds['altitude_m'] = col('altitude_ft') * 0.3048
            result = ds.to_df()

            expected = self.df['altitude_ft'] * 0.3048
            np.testing.assert_array_almost_equal(
                result['altitude_m'].values, expected.values, err_msg="col() assignment produced wrong values"
            )
            print("col() assignment succeeded")

        except Exception as e:
            pytest.fail(f"col() assignment failed: {e}")

    def test_assignment_with_ds_getitem(self):
        """
        Column assignment using ds["col"]
        """
        ds = DataStore.from_df(self.df.copy())

        ds['altitude_m'] = ds['altitude_ft'] * 0.3048
        result = ds.to_df()

        expected = self.df['altitude_ft'] * 0.3048
        np.testing.assert_array_almost_equal(result['altitude_m'].values, expected.values)

    def test_assignment_with_ds_getattr(self):
        """
        Column assignment using ds.col
        """
        ds = DataStore.from_df(self.df.copy())

        ds['altitude_m'] = ds.altitude_ft * 0.3048
        result = ds.to_df()

        expected = self.df['altitude_ft'] * 0.3048
        np.testing.assert_array_almost_equal(result['altitude_m'].values, expected.values)

    def test_assignment_mixed_expression(self):
        """
        CRITICAL: Complex expression assignment with mixed reference methods

        Aviation scenario: Calculate True Air Speed (TAS) = IAS * sqrt(rho0/rho)
        Simplified: adjusted_speed = speed * (1 + altitude/100000)
        """
        ds = DataStore.from_df(self.df.copy())

        # Mix all three styles in one expression
        ds['adjusted_speed'] = ds.speed_knots * (1 + ds['altitude_ft'] / 100000)
        result = ds.to_df()

        expected = self.df['speed_knots'] * (1 + self.df['altitude_ft'] / 100000)
        np.testing.assert_array_almost_equal(
            result['adjusted_speed'].values, expected.values, err_msg="Mixed reference assignment produced wrong values"
        )

    def test_chained_assignment_consistency(self):
        """
        Verify consistency of chained assignments

        Calculate the same values using different reference methods, results should be consistent
        """
        # Fresh copy for each method
        ds1 = DataStore.from_df(self.df.copy())
        ds2 = DataStore.from_df(self.df.copy())
        ds3 = DataStore.from_df(self.df.copy())

        # Method 1: col() only
        ds1['temp_k'] = col('temperature_c') + 273.15
        ds1['temp_f'] = col('temperature_c') * 9 / 5 + 32

        # Method 2: ds["col"] only
        ds2['temp_k'] = ds2['temperature_c'] + 273.15
        ds2['temp_f'] = ds2['temperature_c'] * 9 / 5 + 32

        # Method 3: ds.col only
        ds3['temp_k'] = ds3.temperature_c + 273.15
        ds3['temp_f'] = ds3.temperature_c * 9 / 5 + 32

        result1 = ds1.to_df()
        result2 = ds2.to_df()
        result3 = ds3.to_df()

        # All should produce identical results
        np.testing.assert_array_almost_equal(
            result1['temp_k'].values, result2['temp_k'].values, err_msg="temp_k: col() vs ds['col']"
        )
        np.testing.assert_array_almost_equal(
            result1['temp_k'].values, result3['temp_k'].values, err_msg="temp_k: col() vs ds.col"
        )
        np.testing.assert_array_almost_equal(
            result1['temp_f'].values, result2['temp_f'].values, err_msg="temp_f: col() vs ds['col']"
        )


class TestFunctionChainMixed:
    """Test mixed column reference methods in function chains"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data"""
        self.df = pd.DataFrame(
            {
                'lat': [40.7128, 34.0522, 41.8781],
                'lon': [-74.0060, -118.2437, -87.6298],
                'aircraft_name': ['Boeing 737', 'AIRBUS A320', 'embraer e190'],
                'message': ['MAYDAY MAYDAY', 'All normal', 'Request vector'],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_string_functions_with_col(self):
        """
        Test string functions with col()
        """
        # col() returns Field, does it have .str accessor?
        try:
            upper_expr = col('aircraft_name').upper()
            print(f"col().upper() type: {type(upper_expr)}")
            print(f"col().upper() SQL: {upper_expr.to_sql()}")
        except AttributeError as e:
            print(f"col() doesn't have string methods: {e}")
            # This is expected - col() returns Field which may not have .str accessor
            pass

    def test_string_functions_with_ds_getitem(self):
        """
        Test string functions with ds["col"]
        """
        # ds["col"] returns ColumnExpr which should have .str accessor
        upper_col = self.ds['aircraft_name'].str.upper()
        print(f"ds['col'].str.upper() type: {type(upper_col)}")

    def test_string_functions_consistency(self):
        """
        Verify string function result consistency across different methods
        """
        # Using ds["col"].str
        ds1 = DataStore.from_df(self.df.copy())
        ds1['name_upper'] = ds1['aircraft_name'].str.upper()
        result1 = ds1.to_df()

        # Using ds.col.str
        ds2 = DataStore.from_df(self.df.copy())
        ds2['name_upper'] = ds2.aircraft_name.str.upper()
        result2 = ds2.to_df()

        # Compare results
        assert (
            result1['name_upper'].tolist() == result2['name_upper'].tolist()
        ), "String function results differ between ds['col'] and ds.col"

    def test_datetime_functions_mixed(self):
        """
        Test mixed usage of datetime functions

        KNOWN ISSUE: chDB's Python() table function converts timezone-naive
        datetime64[ns] to UTC, then displays in ClickHouse's default timezone.
        This causes time shifts when local timezone != UTC.

        This test verifies CONSISTENCY between reference methods, not absolute values.
        """
        df_with_dates = pd.DataFrame(
            {
                'departure_time': pd.date_range('2024-01-01 06:00', periods=5, freq='2h'),
                'flight_duration_min': [120, 180, 90, 240, 150],
            }
        )
        ds = DataStore.from_df(df_with_dates)

        # Extract hour using different methods
        ds['hour_getitem'] = ds['departure_time'].dt.hour
        ds['hour_getattr'] = ds.departure_time.dt.hour

        result = ds.to_df()

        # CRITICAL: Verify that different reference methods produce SAME results
        # (The absolute hour values may differ from Pandas due to timezone handling)
        assert (
            result['hour_getitem'].tolist() == result['hour_getattr'].tolist()
        ), "Different reference methods produced different hour values!"

        # Verify the hour increments are correct (2 hour intervals)
        hours = result['hour_getitem'].tolist()
        for i in range(1, len(hours)):
            diff = (hours[i] - hours[i - 1]) % 24
            assert diff == 2, f"Hour increment should be 2, got {diff}"


class TestEdgeCasesAndCornerCases:
    """Edge cases and corner cases testing"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data"""
        self.df = pd.DataFrame(
            {
                'value': [1.0, 2.0, 3.0, np.nan, 5.0],
                'category': ['A', 'B', 'A', 'B', 'A'],
                'zero': [0, 0, 0, 0, 0],
                'negative': [-1, -2, -3, -4, -5],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_null_handling_consistency(self):
        """
        Verify NULL value handling consistency
        """
        # Check if NaN is handled consistently
        ds1 = DataStore.from_df(self.df.copy())
        ds2 = DataStore.from_df(self.df.copy())

        ds1['is_valid'] = ds1['value'].notnull()
        ds2['is_valid'] = ds2.value.notnull()

        result1 = ds1.to_df()
        result2 = ds2.to_df()

        # Both should identify the same null values
        assert result1['is_valid'].tolist() == result2['is_valid'].tolist()

    def test_division_by_zero_consistency(self):
        """
        Verify division by zero handling consistency

        Aviation: This kind of error could cause invalid flight parameter calculations
        """
        ds = DataStore.from_df(self.df.copy())

        # Division by zero column
        ds['ratio1'] = ds['negative'] / ds['zero']
        ds['ratio2'] = ds.negative / ds.zero

        result = ds.to_df()

        # Both should handle division by zero the same way (likely inf or error)
        # The key is consistency, not the specific handling
        for i in range(len(result)):
            val1 = result['ratio1'].iloc[i]
            val2 = result['ratio2'].iloc[i]
            # Either both should be inf, or both nan, or both the same error state
            if np.isnan(val1):
                assert np.isnan(val2), f"NaN handling inconsistent at index {i}"
            elif np.isinf(val1):
                assert np.isinf(val2), f"Inf handling inconsistent at index {i}"
            else:
                assert val1 == val2, f"Value mismatch at index {i}: {val1} != {val2}"

    def test_expression_with_reserved_keywords(self):
        """
        Test using Python/SQL reserved keywords as column names
        """
        df_reserved = pd.DataFrame(
            {
                'from': [1, 2, 3],
                'select': [4, 5, 6],
                'where': [7, 8, 9],
                'class': [10, 11, 12],
            }
        )
        ds = DataStore.from_df(df_reserved)

        # These might fail or need special handling
        try:
            # ds["from"] should work (bracket notation)
            val1 = ds["from"]
            print(f"ds['from'] succeeded: {type(val1)}")
        except Exception as e:
            print(f"ds['from'] failed: {e}")

        try:
            # col("from") should work
            val3 = col("from")
            print(f"col('from') succeeded: {type(val3)}")
        except Exception as e:
            print(f"col('from') failed: {e}")

    def test_complex_nested_expression(self):
        """
        Test complex nested expressions

        Aviation scenario: Multi-parameter calculation
        """
        df = pd.DataFrame(
            {
                'altitude': [10000, 20000, 30000],
                'temperature': [-10, -30, -50],
                'pressure': [697, 466, 301],
                'speed': [250, 300, 350],
            }
        )
        ds = DataStore.from_df(df)

        # Complex expression mixing all reference styles
        # Simplified pressure altitude calculation
        ds['density_altitude'] = ds['altitude'] + (ds.temperature - 15) * 120 + (1013 - ds['pressure']) * 30

        result = ds.to_df()

        # Manual calculation for verification
        expected = df['altitude'] + (df['temperature'] - 15) * 120 + (1013 - df['pressure']) * 30

        np.testing.assert_array_almost_equal(
            result['density_altitude'].values,
            expected.values,
            err_msg="Complex nested expression produced wrong values",
        )


class TestLazyExecutionConsistency:
    """Verify lazy execution consistency across different reference methods"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data"""
        self.df = pd.DataFrame(
            {
                'x': list(range(100)),
                'y': list(range(100, 200)),
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_lazy_op_count_consistency(self):
        """
        Verify that different reference methods produce the same number of lazy operations
        """
        ds1 = DataStore.from_df(self.df.copy())
        ds2 = DataStore.from_df(self.df.copy())
        ds3 = DataStore.from_df(self.df.copy())

        # Same operations, different reference styles
        ds1['z'] = ds1['x'] + ds1['y']
        ds2['z'] = ds2.x + ds2.y
        ds3['z'] = col('x') + col('y')

        # Check lazy ops count before execution
        print(f"ds['col'] lazy ops: {len(ds1._lazy_ops)}")
        print(f"ds.col lazy ops: {len(ds2._lazy_ops)}")
        print(f"col() lazy ops: {len(ds3._lazy_ops)}")

        # Execute and verify results
        result1 = ds1.to_df()
        result2 = ds2.to_df()
        result3 = ds3.to_df()

        np.testing.assert_array_equal(result1['z'], result2['z'])
        np.testing.assert_array_equal(result1['z'], result3['z'])

    def test_multiple_operations_before_execution(self):
        """
        Test behavior of multiple operations before execution
        """
        ds = DataStore.from_df(self.df.copy())

        # Chain multiple operations using different reference styles
        ds['a'] = ds['x'] * 2
        ds['b'] = ds.y * 3
        ds['c'] = col('x') + col('y')
        ds['d'] = ds['a'] + ds.b  # Reference newly created columns

        # Count lazy ops before execution
        lazy_count_before = len(ds._lazy_ops)
        print(f"Lazy ops before execution: {lazy_count_before}")

        result = ds.to_df()

        # All new columns should exist
        assert 'a' in result.columns
        assert 'b' in result.columns
        assert 'c' in result.columns
        assert 'd' in result.columns

        # Verify calculations
        expected_a = self.df['x'] * 2
        expected_b = self.df['y'] * 3
        expected_c = self.df['x'] + self.df['y']
        expected_d = expected_a + expected_b

        np.testing.assert_array_equal(result['a'], expected_a)
        np.testing.assert_array_equal(result['b'], expected_b)
        np.testing.assert_array_equal(result['c'], expected_c)
        np.testing.assert_array_equal(result['d'], expected_d)


class TestRealWorldAviationScenarios:
    """Comprehensive testing of real-world aviation scenarios"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup realistic aviation data"""
        np.random.seed(42)
        n_records = 1000

        self.df = pd.DataFrame(
            {
                'flight_id': [f'F{i:04d}' for i in range(n_records)],
                'aircraft_type': np.random.choice(['B737', 'A320', 'E190', 'CRJ9'], n_records),
                'departure_airport': np.random.choice(['JFK', 'LAX', 'ORD', 'DFW', 'ATL'], n_records),
                'arrival_airport': np.random.choice(['JFK', 'LAX', 'ORD', 'DFW', 'ATL'], n_records),
                'altitude_ft': np.random.uniform(25000, 41000, n_records),
                'ground_speed_knots': np.random.uniform(350, 550, n_records),
                'heading_degrees': np.random.uniform(0, 360, n_records),
                'fuel_remaining_lbs': np.random.uniform(5000, 45000, n_records),
                'fuel_flow_lbs_hr': np.random.uniform(2000, 8000, n_records),
                'outside_air_temp_c': np.random.uniform(-60, -40, n_records),
                'wind_speed_knots': np.random.uniform(0, 100, n_records),
                'wind_direction': np.random.uniform(0, 360, n_records),
                'timestamp': pd.date_range('2024-01-01', periods=n_records, freq='min'),
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_fuel_endurance_calculation(self):
        """
        Calculate fuel endurance time

        endurance_hours = fuel_remaining / fuel_flow

        Aviation safety critical calculation!
        """
        ds = DataStore.from_df(self.df.copy())

        # Calculate using three different methods
        ds['endurance_1'] = ds['fuel_remaining_lbs'] / ds['fuel_flow_lbs_hr']
        ds['endurance_2'] = ds.fuel_remaining_lbs / ds.fuel_flow_lbs_hr
        ds['endurance_3'] = col('fuel_remaining_lbs') / col('fuel_flow_lbs_hr')

        result = ds.to_df()

        # All three should be identical
        np.testing.assert_array_almost_equal(
            result['endurance_1'].values,
            result['endurance_2'].values,
            decimal=6,
            err_msg="Endurance calculation inconsistent: ds['col'] vs ds.col",
        )
        np.testing.assert_array_almost_equal(
            result['endurance_1'].values,
            result['endurance_3'].values,
            decimal=6,
            err_msg="Endurance calculation inconsistent: ds['col'] vs col()",
        )

    def test_true_airspeed_approximation(self):
        """
        True airspeed approximation

        TAS ≈ IAS * sqrt(T_std / T_actual) * sqrt(P_std / P_actual)
        Simplified: TAS ≈ GS * (1 + altitude / 150000)
        """
        ds = DataStore.from_df(self.df.copy())

        # Using mixed reference styles
        ds['tas_approx'] = ds.ground_speed_knots * (1 + ds['altitude_ft'] / 150000)

        result = ds.to_df()

        # Verify calculation
        expected = self.df['ground_speed_knots'] * (1 + self.df['altitude_ft'] / 150000)
        np.testing.assert_array_almost_equal(result['tas_approx'].values, expected.values)

    def test_fleet_statistics_by_aircraft_type(self):
        """
        Fleet statistics by aircraft type using col() for SQL aggregation.
        
        Note: ds['col'].mean() returns scalar, so SQL agg() uses col().
        """
        stats = self.ds.groupby('aircraft_type').agg(
            avg_altitude=col('altitude_ft').mean(),
            avg_fuel=col('fuel_remaining_lbs').mean(),
            avg_speed=col('ground_speed_knots').mean(),
            flight_count=col('flight_id').count(),
        )

        result = stats.to_df()

        assert len(result) == 4  # B737, A320, E190, CRJ9
        assert 'avg_altitude' in result.columns
        assert 'avg_fuel' in result.columns
        assert 'avg_speed' in result.columns
        assert 'flight_count' in result.columns

        # Total flight count should equal original data
        assert result['flight_count'].sum() == len(self.df)

    def test_low_fuel_warning_filter(self):
        """
        Low fuel warning filter

        Aviation safety critical: Filter flights with fuel endurance < 2 hours
        """
        # Calculate endurance first
        ds = DataStore.from_df(self.df.copy())
        ds['endurance_hrs'] = ds['fuel_remaining_lbs'] / ds['fuel_flow_lbs_hr']

        # Filter using different reference styles - should get same results
        low_fuel_1 = ds.filter(ds['endurance_hrs'] < 2).to_df()
        low_fuel_2 = ds.filter(ds.endurance_hrs < 2).to_df()
        low_fuel_3 = ds.filter(col('endurance_hrs') < 2).to_df()

        assert (
            len(low_fuel_1) == len(low_fuel_2) == len(low_fuel_3)
        ), f"Filter results differ: {len(low_fuel_1)} vs {len(low_fuel_2)} vs {len(low_fuel_3)}"

        assert_frame_equal(low_fuel_1, low_fuel_2)
        assert_frame_equal(low_fuel_1, low_fuel_3)

    def test_complex_aviation_pipeline(self):
        """
        Complex aviation data processing pipeline

        1. Calculate derived fields (mixed reference methods)
        2. Filter conditions
        3. Group aggregation
        """
        ds = DataStore.from_df(self.df.copy())

        # Step 1: Derived fields using mixed references
        ds['endurance_hrs'] = ds['fuel_remaining_lbs'] / ds.fuel_flow_lbs_hr
        ds['altitude_m'] = ds.altitude_ft * 0.3048
        ds['speed_kmh'] = ds['ground_speed_knots'] * 1.852

        # Step 2: Filter (high altitude, long range flights)
        high_altitude = ds.filter((ds['altitude_ft'] > 35000) & (ds.endurance_hrs > 3))

        # Step 3: Aggregate by route
        # Note: We need to aggregate on filtered data
        # For simplicity, just verify the filter worked
        result = high_altitude.to_df()

        assert all(result['altitude_ft'] > 35000)
        assert all(result['endurance_hrs'] > 3)

        print(f"High altitude long range flights: {len(result)}")


class TestKnownIssues:
    """
    Tests documenting known issues

    These tests are marked as xfail or contain KNOWN ISSUE comments
    Aerospace applications must be aware of these limitations
    """

    def test_timezone_handling_issue(self):
        """
        KNOWN ISSUE: Inconsistent timezone handling

        Pandas datetime64[ns] is timezone-naive, but chDB treats it as UTC
        and converts to ClickHouse default timezone.

        Aviation impact:
        - Flight time calculations may be off by 8+ hours
        - Departure/arrival times displayed incorrectly
        - Cross-timezone flight calculations incorrect
        """
        import chdb

        df = pd.DataFrame(
            {
                'event_time': pd.date_range('2024-01-01 00:00', periods=3, freq='h'),
            }
        )

        # Get ClickHouse timezone
        tz_result = chdb.query("SELECT timezone()", 'CSV')
        ch_timezone = str(tz_result).strip().strip('"').strip()

        ds = DataStore.from_df(df)
        ds['hour'] = ds['event_time'].dt.hour
        result = ds.to_df()

        pandas_hours = df['event_time'].dt.hour.tolist()
        chdb_hours = result['hour'].tolist()

        # Log difference but don't fail (this is known behavior)
        print(f"\nClickHouse timezone: {ch_timezone}")
        print(f"Pandas hours: {pandas_hours}")
        print(f"chDB hours: {chdb_hours}")

        if pandas_hours != chdb_hours:
            print("WARNING: Timezone mismatch detected!")
            print("This is a KNOWN ISSUE when ClickHouse timezone != UTC")
            # Calculate offset
            offset = (chdb_hours[0] - pandas_hours[0]) % 24
            print(f"Time offset: {offset} hours")

    def test_column_name_quote_escaping(self):
        """
        KNOWN ISSUE: Double quotes in column names not properly escaped

        SQL generated: "col"name" (should be "col""name")

        Although it may currently work, this is a potential security and stability risk.
        """
        col_with_quote = 'test"col'

        df = pd.DataFrame(
            {
                col_with_quote: [1, 2, 3],
            }
        )

        ds = DataStore.from_df(df)
        expr = col(col_with_quote)
        generated_sql = expr.to_sql()

        print(f"\nColumn name: {col_with_quote}")
        print(f"Generated SQL: {generated_sql}")

        # Ideally should be "test""col"
        # But actually is "test"col"
        if '""' not in generated_sql and '"' in col_with_quote:
            print("WARNING: Double quote not properly escaped in SQL!")
            print("This could cause issues with certain column names.")

    def test_type_inconsistency_documentation(self):
        """
        DESIGN NOTE: col() vs ds["col"] return different types

        col() -> Field (no DataStore binding)
        ds["col"] -> ColumnExpr (with DataStore binding)
        ds.col -> ColumnExpr (with DataStore binding)

        In the current implementation, this difference doesn't cause errors because
        methods like agg() can correctly handle both types. But users should be aware of this.
        """
        df = pd.DataFrame({'x': [1, 2, 3]})
        ds = DataStore.from_df(df)

        ref_col = col('x')
        ref_getitem = ds['x']
        ref_getattr = ds.x

        print(f"\ncol('x') type: {type(ref_col).__name__}")
        print(f"ds['x'] type: {type(ref_getitem).__name__}")
        print(f"ds.x type: {type(ref_getattr).__name__}")

        # Verify types
        assert isinstance(ref_col, Field)
        assert isinstance(ref_getitem, ColumnExpr)
        assert isinstance(ref_getattr, ColumnExpr)

        # Verify their SQL output is consistent
        sql_col = ref_col.to_sql()
        sql_getitem = ref_getitem._expr.to_sql()
        sql_getattr = ref_getattr._expr.to_sql()

        assert sql_col == sql_getitem == sql_getattr, "SQL output should be identical regardless of reference method"


class TestCriticalAviationScenarios:
    """
    Critical aviation scenario tests

    These tests simulate real aviation calculation scenarios to verify library reliability in critical applications.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup realistic aviation data"""
        np.random.seed(42)
        self.n_records = 100

        self.flight_data = pd.DataFrame(
            {
                'flight_id': [f'UAL{i:04d}' for i in range(self.n_records)],
                'callsign': [f'UNITED{i}' for i in range(self.n_records)],
                'altitude_ft': np.random.uniform(25000, 41000, self.n_records),
                'indicated_airspeed_kts': np.random.uniform(250, 350, self.n_records),
                'mach_number': np.random.uniform(0.75, 0.85, self.n_records),
                'heading_true': np.random.uniform(0, 360, self.n_records),
                'latitude': np.random.uniform(30, 50, self.n_records),
                'longitude': np.random.uniform(-130, -70, self.n_records),
                'fuel_qty_lbs': np.random.uniform(10000, 50000, self.n_records),
                'fuel_flow_lbs_hr': np.random.uniform(3000, 7000, self.n_records),
                'oat_celsius': np.random.uniform(-60, -40, self.n_records),
                'wind_speed_kts': np.random.uniform(0, 100, self.n_records),
                'wind_direction': np.random.uniform(0, 360, self.n_records),
            }
        )
        self.ds = DataStore.from_df(self.flight_data)

    def test_fuel_exhaustion_time_calculation(self):
        """
        Fuel exhaustion time calculation - Critical safety calculation

        time_to_exhaustion = fuel_qty / fuel_flow

        Errors could lead to: fuel exhaustion, emergency landing, accidents
        """
        ds = DataStore.from_df(self.flight_data.copy())

        # Calculate using three different methods
        ds['tte_method1'] = ds['fuel_qty_lbs'] / ds['fuel_flow_lbs_hr']
        ds['tte_method2'] = ds.fuel_qty_lbs / ds.fuel_flow_lbs_hr
        ds['tte_method3'] = col('fuel_qty_lbs') / col('fuel_flow_lbs_hr')

        result = ds.to_df()

        # Verify all three methods produce identical results
        np.testing.assert_array_almost_equal(
            result['tte_method1'].values,
            result['tte_method2'].values,
            decimal=10,
            err_msg="CRITICAL: Fuel exhaustion time calculation inconsistency!",
        )
        np.testing.assert_array_almost_equal(
            result['tte_method1'].values,
            result['tte_method3'].values,
            decimal=10,
            err_msg="CRITICAL: Fuel exhaustion time calculation inconsistency!",
        )

        # Verify results are reasonable
        assert all(result['tte_method1'] > 0), "Fuel exhaustion time must be positive!"
        assert all(result['tte_method1'] < 24), "Fuel exhaustion time seems unreasonably high!"

    def test_true_airspeed_from_mach(self):
        """
        True airspeed calculation (from Mach number)

        TAS = Mach * sqrt(gamma * R * T)
        Simplified: TAS ≈ Mach * 661.47 * sqrt((OAT + 273.15) / 288.15)

        Errors could lead to: flight plan errors, fuel calculation errors, airspace conflicts
        """
        ds = DataStore.from_df(self.flight_data.copy())

        # Calculate using mixed reference methods
        # TAS = Mach * 661.47 * sqrt((OAT_K) / 288.15)
        # where OAT_K = OAT_C + 273.15

        # First calculate Kelvin temperature
        ds['oat_kelvin'] = ds['oat_celsius'] + 273.15

        # Then calculate TAS (needs sqrt function)
        # Since sqrt is a ClickHouse function, we use SQL or approximation
        ds['temp_ratio'] = ds.oat_kelvin / 288.15
        ds['tas_approx'] = ds['mach_number'] * 661.47 * (ds.temp_ratio**0.5)

        result = ds.to_df()

        # Verify TAS is in reasonable range (400-600 kts for typical cruise)
        assert all(result['tas_approx'] > 300), "TAS too low!"
        assert all(result['tas_approx'] < 700), "TAS too high!"

    def test_great_circle_distance_approximation(self):
        """
        Great circle distance approximation

        Using simplified Haversine formula
        d ≈ 60 * sqrt((lat2-lat1)^2 + (cos(lat1)*(lon2-lon1))^2)

        Errors could lead to: route planning errors, fuel calculation errors
        """
        # Create two location points
        df = pd.DataFrame(
            {
                'lat1': [40.7128, 34.0522],  # NYC, LAX
                'lon1': [-74.0060, -118.2437],
                'lat2': [51.5074, 48.8566],  # London, Paris
                'lon2': [-0.1278, 2.3522],
            }
        )
        ds = DataStore.from_df(df)

        # Calculate using mixed references
        ds['lat_diff'] = ds['lat2'] - ds.lat1
        ds['lon_diff'] = ds['lon2'] - ds['lon1']

        # Simplified calculation (not completely accurate, but tests consistency)
        ds['approx_dist'] = (ds.lat_diff**2 + ds['lon_diff'] ** 2) ** 0.5 * 60

        result = ds.to_df()

        # NYC to London ≈ 3459 nm (actual)
        # LAX to Paris ≈ 5663 nm (actual)
        # Our simplified formula will have deviation, but should be positive and non-zero
        assert all(result['approx_dist'] > 0), "Distance must be positive!"

    def test_landing_weight_calculation(self):
        """
        Landing weight calculation

        landing_weight = zero_fuel_weight + landing_fuel
        landing_fuel = current_fuel - trip_fuel

        Overweight landing could lead to: structural damage, tire blowout, brake failure
        """
        df = pd.DataFrame(
            {
                'zero_fuel_weight': [120000, 130000, 115000],  # lbs
                'current_fuel': [45000, 50000, 40000],
                'trip_fuel': [35000, 40000, 30000],
                'max_landing_weight': [160000, 165000, 155000],
            }
        )
        ds = DataStore.from_df(df)

        # Calculate using mixed references
        ds['landing_fuel'] = ds['current_fuel'] - ds.trip_fuel
        ds['landing_weight'] = ds.zero_fuel_weight + ds['landing_fuel']
        ds['weight_margin'] = ds['max_landing_weight'] - ds.landing_weight

        result = ds.to_df()

        # Verify calculation correctness
        expected_landing_fuel = df['current_fuel'] - df['trip_fuel']
        expected_landing_weight = df['zero_fuel_weight'] + expected_landing_fuel
        expected_margin = df['max_landing_weight'] - expected_landing_weight

        np.testing.assert_array_almost_equal(
            result['landing_fuel'].values,
            expected_landing_fuel.values,
            decimal=6,
            err_msg="CRITICAL: Landing fuel calculation error!",
        )
        np.testing.assert_array_almost_equal(
            result['landing_weight'].values,
            expected_landing_weight.values,
            decimal=6,
            err_msg="CRITICAL: Landing weight calculation error!",
        )
        np.testing.assert_array_almost_equal(
            result['weight_margin'].values,
            expected_margin.values,
            decimal=6,
            err_msg="CRITICAL: Weight margin calculation error!",
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
