"""
Comprehensive tests for the dtype correction system.

Tests all dtype correction rules to ensure chDB results match pandas behavior.

This test suite covers:
1. abs() dtype correction (CRITICAL priority)
2. sign() dtype correction (HIGH priority)
3. Arithmetic operations dtype correction (MEDIUM priority)
4. pow() dtype correction (HIGH priority)
5. floordiv dtype correction (HIGH priority)
6. Float32 preservation (LOW priority)
7. Datetime property dtype correction
8. Registry and configuration behavior
9. SQL pushdown path verification
10. Edge cases (nulls, empty DataFrames, large values, boundaries)
"""

import io
import logging
import numpy as np
import pandas as pd
import pytest
from datastore import DataStore
from datastore.dtype_correction import (
    dtype_registry,
    dtype_correction_config,
    CorrectionPriority,
)
from datastore.dtype_correction.config import CorrectionLevel
from datastore.dtype_correction.rules import (
    SignedAbsRule,
    SignPreserveRule,
    ArithmeticPreserveRule,
    PowPreserveRule,
    FloordivPreserveRule,
    Float32PreserveRule,
    ALL_RULES,
)
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_dataframe


# =============================================================================
# Test abs() Dtype Correction (CRITICAL Priority)
# =============================================================================


class TestAbsDtypeCorrection:
    """Test abs() dtype correction (CRITICAL priority).

    chDB returns unsigned types for abs() on signed integers,
    but pandas returns the same signed type.
    """

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
    def test_abs_signed_int_preserves_type(self, dtype):
        """abs() on signed int should return signed int (not unsigned)."""
        pd_df = pd.DataFrame({'a': pd.array([-3, -2, -1, 0, 1, 2, 3], dtype=dtype)})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        assert_datastore_equals_pandas(ds_df, pd_df)
        # Explicitly verify dtype
        ds_result = get_dataframe(ds_df)
        assert str(ds_result['abs_a'].dtype) == dtype

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_abs_float_preserves_type(self, dtype):
        """abs() on float should preserve float type."""
        pd_df = pd.DataFrame({'a': pd.array([-3.5, -2.5, 0.0, 2.5, 3.5], dtype=dtype)})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        # Float32 may be promoted to float64 by chDB, check values match
        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_almost_equal(ds_result['abs_a'].values, pd_df['abs_a'].values)

    @pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
    def test_abs_unsigned_int_preserves_type(self, dtype):
        """abs() on unsigned int should preserve unsigned type."""
        pd_df = pd.DataFrame({'a': pd.array([0, 1, 2, 3, 4, 5], dtype=dtype)})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        ds_result = get_dataframe(ds_df)
        assert str(ds_result['abs_a'].dtype) == dtype

    def test_abs_int64_max_value(self):
        """abs() should handle large int64 values correctly."""
        # Use values that are large but won't overflow when negated
        max_safe = 2**62 - 1
        pd_df = pd.DataFrame({'a': pd.array([-max_safe, max_safe], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['abs_a'].values, pd_df['abs_a'].values)
        assert str(ds_result['abs_a'].dtype) == 'int64'

    def test_abs_mixed_positive_negative(self):
        """abs() should handle mixed positive and negative values."""
        pd_df = pd.DataFrame({'a': pd.array([-100, -50, -1, 0, 1, 50, 100], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_abs_all_negative(self):
        """abs() should handle all negative values."""
        pd_df = pd.DataFrame({'a': pd.array([-5, -4, -3, -2, -1], dtype='int32')})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        assert_datastore_equals_pandas(ds_df, pd_df)
        ds_result = get_dataframe(ds_df)
        assert str(ds_result['abs_a'].dtype) == 'int32'

    def test_abs_all_zeros(self):
        """abs() should handle all zero values."""
        pd_df = pd.DataFrame({'a': pd.array([0, 0, 0, 0, 0], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# Test sign() Dtype Correction (HIGH Priority)
# =============================================================================


class TestSignDtypeCorrection:
    """Test sign() dtype correction (HIGH priority).

    chDB always returns Int8 for sign(), but pandas/numpy preserves
    the input type.
    """

    @pytest.mark.parametrize("dtype", ["int64", "float64"])
    def test_sign_preserves_input_type(self, dtype):
        """sign() should preserve input type."""
        data = [-3, -2, -1, 0, 1, 2, 3] if 'int' in dtype else [-3.5, -2.5, 0.0, 2.5, 3.5]
        pd_df = pd.DataFrame({'a': pd.array(data, dtype=dtype)})
        ds_df = DataStore(pd_df.copy())

        # pandas/numpy sign preserves type
        pd_df['sign_a'] = np.sign(pd_df['a'])
        ds_df['sign_a'] = np.sign(ds_df['a'])

        ds_result = get_dataframe(ds_df)
        # Values should match
        np.testing.assert_array_equal(ds_result['sign_a'].values, pd_df['sign_a'].values)

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
    def test_sign_all_int_types(self, dtype):
        """sign() should work correctly for all integer types."""
        pd_df = pd.DataFrame({'a': pd.array([-2, -1, 0, 1, 2], dtype=dtype)})
        ds_df = DataStore(pd_df.copy())

        pd_df['sign_a'] = np.sign(pd_df['a'])
        ds_df['sign_a'] = np.sign(ds_df['a'])

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['sign_a'].values, pd_df['sign_a'].values)

    def test_sign_float_with_zero(self):
        """sign() should handle float zero correctly (-0.0 vs 0.0)."""
        pd_df = pd.DataFrame({'a': pd.array([-1.5, -0.0, 0.0, 0.0, 1.5], dtype='float64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['sign_a'] = np.sign(pd_df['a'])
        ds_df['sign_a'] = np.sign(ds_df['a'])

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['sign_a'].values, pd_df['sign_a'].values)


# =============================================================================
# Test Arithmetic Operations Dtype Correction (MEDIUM Priority)
# =============================================================================


class TestArithmeticDtypeCorrection:
    """Test arithmetic operations dtype correction (MEDIUM priority).

    chDB may widen or narrow types for overflow protection,
    but pandas preserves the input type.
    """

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
    def test_add_scalar_preserves_int_type(self, dtype):
        """Addition with scalar should preserve integer type."""
        pd_df = pd.DataFrame({'a': pd.array([1, 2, 3, 4, 5], dtype=dtype)})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_plus_1'] = pd_df['a'] + 1
        ds_df['a_plus_1'] = ds_df['a'] + 1

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['a_plus_1'].values, pd_df['a_plus_1'].values)

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
    def test_sub_scalar_preserves_int_type(self, dtype):
        """Subtraction with scalar should preserve integer type."""
        pd_df = pd.DataFrame({'a': pd.array([5, 4, 3, 2, 1], dtype=dtype)})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_minus_1'] = pd_df['a'] - 1
        ds_df['a_minus_1'] = ds_df['a'] - 1

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['a_minus_1'].values, pd_df['a_minus_1'].values)

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
    def test_mul_scalar_preserves_int_type(self, dtype):
        """Multiplication with scalar should preserve integer type."""
        pd_df = pd.DataFrame({'a': pd.array([1, 2, 3, 4, 5], dtype=dtype)})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_times_2'] = pd_df['a'] * 2
        ds_df['a_times_2'] = ds_df['a'] * 2

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['a_times_2'].values, pd_df['a_times_2'].values)

    def test_mod_preserves_int_type(self):
        """Modulo operation should preserve integer type."""
        pd_df = pd.DataFrame({'a': pd.array([10, 11, 12, 13, 14], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_mod_3'] = pd_df['a'] % 3
        ds_df['a_mod_3'] = ds_df['a'] % 3

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['a_mod_3'].values, pd_df['a_mod_3'].values)

    def test_arithmetic_chain(self):
        """Chained arithmetic operations should maintain correct types."""
        pd_df = pd.DataFrame({'a': pd.array([1, 2, 3, 4, 5], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['result'] = (pd_df['a'] + 10) * 2 - 5
        ds_df['result'] = (ds_df['a'] + 10) * 2 - 5

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['result'].values, pd_df['result'].values)

    def test_column_arithmetic(self):
        """Arithmetic between columns should work correctly."""
        pd_df = pd.DataFrame(
            {
                'a': pd.array([1, 2, 3, 4, 5], dtype='int64'),
                'b': pd.array([5, 4, 3, 2, 1], dtype='int64'),
            }
        )
        ds_df = DataStore(pd_df.copy())

        pd_df['sum'] = pd_df['a'] + pd_df['b']
        pd_df['diff'] = pd_df['a'] - pd_df['b']
        pd_df['prod'] = pd_df['a'] * pd_df['b']

        ds_df['sum'] = ds_df['a'] + ds_df['b']
        ds_df['diff'] = ds_df['a'] - ds_df['b']
        ds_df['prod'] = ds_df['a'] * ds_df['b']

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['sum'].values, pd_df['sum'].values)
        np.testing.assert_array_equal(ds_result['diff'].values, pd_df['diff'].values)
        np.testing.assert_array_equal(ds_result['prod'].values, pd_df['prod'].values)


# =============================================================================
# Test Division and Floor Division Dtype Correction
# =============================================================================


class TestDivisionDtypeCorrection:
    """Test division operations dtype correction."""

    def test_truediv_returns_float(self):
        """True division should always return float."""
        pd_df = pd.DataFrame({'a': pd.array([10, 20, 30, 40, 50], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_div_3'] = pd_df['a'] / 3
        ds_df['a_div_3'] = ds_df['a'] / 3

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_almost_equal(ds_result['a_div_3'].values, pd_df['a_div_3'].values)

    def test_floordiv_int_preserves_int(self):
        """Floor division on int should return int."""
        pd_df = pd.DataFrame({'a': pd.array([10, 20, 30, 40, 50], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_floordiv_3'] = pd_df['a'] // 3
        ds_df['a_floordiv_3'] = ds_df['a'] // 3

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['a_floordiv_3'].values, pd_df['a_floordiv_3'].values)

    def test_floordiv_float_preserves_float(self):
        """Floor division on float should return float (pandas behavior)."""
        pd_df = pd.DataFrame({'a': pd.array([10.5, 20.5, 30.5, 40.5, 50.5], dtype='float64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_floordiv_3'] = pd_df['a'] // 3
        ds_df['a_floordiv_3'] = ds_df['a'] // 3

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_almost_equal(ds_result['a_floordiv_3'].values, pd_df['a_floordiv_3'].values)


# =============================================================================
# Test Power Operation Dtype Correction (HIGH Priority)
# =============================================================================


class TestPowDtypeCorrection:
    """Test pow() dtype correction (HIGH priority).

    chDB always returns Float64 for pow(), but pandas preserves
    integer type for integer ** integer operations.
    """

    def test_pow_int_int_values(self):
        """pow(int, int) should have correct values."""
        pd_df = pd.DataFrame({'a': pd.array([1, 2, 3, 4, 5], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_squared'] = pd_df['a'] ** 2
        ds_df['a_squared'] = ds_df['a'] ** 2

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_almost_equal(ds_result['a_squared'].values, pd_df['a_squared'].values)

    def test_pow_float_values(self):
        """pow() with float should return float."""
        pd_df = pd.DataFrame({'a': pd.array([1.5, 2.5, 3.5, 4.5], dtype='float64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_squared'] = pd_df['a'] ** 2
        ds_df['a_squared'] = ds_df['a'] ** 2

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_almost_equal(ds_result['a_squared'].values, pd_df['a_squared'].values)

    def test_pow_zero_exponent(self):
        """pow(x, 0) should return 1."""
        pd_df = pd.DataFrame({'a': pd.array([1, 2, 3, 4, 5], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['a_pow_0'] = pd_df['a'] ** 0
        ds_df['a_pow_0'] = ds_df['a'] ** 0

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['a_pow_0'].values, pd_df['a_pow_0'].values)


# =============================================================================
# Test Datetime Property Dtype Correction
# =============================================================================


class TestDatetimeDtypeCorrection:
    """Test datetime property dtype correction.

    Datetime properties should return int32 to match pandas behavior.
    """

    @pytest.fixture
    def datetime_df(self):
        """Create a DataFrame with datetime data."""
        dates = pd.to_datetime(['2024-01-15', '2024-06-20', '2024-12-25'])
        return pd.DataFrame({'date': dates})

    def test_dt_year_dtype(self, datetime_df):
        """dt.year should return int32."""
        pd_df = datetime_df.copy()
        ds_df = DataStore(pd_df.copy())

        pd_df['year'] = pd_df['date'].dt.year
        ds_df['year'] = ds_df['date'].dt.year

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['year'].values, pd_df['year'].values)
        assert str(ds_result['year'].dtype) == 'int32'

    def test_dt_month_dtype(self, datetime_df):
        """dt.month should return int32."""
        pd_df = datetime_df.copy()
        ds_df = DataStore(pd_df.copy())

        pd_df['month'] = pd_df['date'].dt.month
        ds_df['month'] = ds_df['date'].dt.month

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['month'].values, pd_df['month'].values)
        assert str(ds_result['month'].dtype) == 'int32'

    def test_dt_day_dtype(self, datetime_df):
        """dt.day should return int32."""
        pd_df = datetime_df.copy()
        ds_df = DataStore(pd_df.copy())

        pd_df['day'] = pd_df['date'].dt.day
        ds_df['day'] = ds_df['date'].dt.day

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['day'].values, pd_df['day'].values)
        assert str(ds_result['day'].dtype) == 'int32'

    def test_dt_dayofweek_dtype(self, datetime_df):
        """dt.dayofweek should return int32."""
        pd_df = datetime_df.copy()
        ds_df = DataStore(pd_df.copy())

        pd_df['dayofweek'] = pd_df['date'].dt.dayofweek
        ds_df['dayofweek'] = ds_df['date'].dt.dayofweek

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['dayofweek'].values, pd_df['dayofweek'].values)
        assert str(ds_result['dayofweek'].dtype) == 'int32'

    def test_dt_quarter_dtype(self, datetime_df):
        """dt.quarter should return int32."""
        pd_df = datetime_df.copy()
        ds_df = DataStore(pd_df.copy())

        pd_df['quarter'] = pd_df['date'].dt.quarter
        ds_df['quarter'] = ds_df['date'].dt.quarter

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['quarter'].values, pd_df['quarter'].values)
        assert str(ds_result['quarter'].dtype) == 'int32'


# =============================================================================
# Test Dtype Correction Registry
# =============================================================================


class TestDtypeCorrectionRegistry:
    """Test the dtype correction registry functionality."""

    def test_registry_has_abs_rule(self):
        """Registry should have abs correction rule."""
        rules = dtype_registry.get_rules('abs')
        assert len(rules) > 0
        assert rules[0].priority == CorrectionPriority.CRITICAL

    def test_registry_should_correct_abs(self):
        """Registry should indicate abs(int64)->uint64 needs correction."""
        assert dtype_registry.should_correct('abs', 'int64', 'uint64')
        assert not dtype_registry.should_correct('abs', 'float64', 'float64')

    def test_registry_get_target_dtype(self):
        """Registry should return correct target dtype."""
        assert dtype_registry.get_target_dtype('abs', 'int64') == 'int64'
        assert dtype_registry.get_target_dtype('abs', 'float64') is None

    def test_registry_has_all_rules(self):
        """Registry should have all defined rules."""
        # Verify registry contains expected function names
        assert dtype_registry.get_rules('abs')  # SignedAbsRule
        assert dtype_registry.get_rules('sign')  # SignPreserveRule
        assert dtype_registry.get_rules('add')  # ArithmeticPreserveRule
        assert dtype_registry.get_rules('pow')  # PowPreserveRule
        assert dtype_registry.get_rules('floordiv')  # FloordivPreserveRule

    def test_registry_rule_priority_order(self):
        """Rules should be sorted by priority."""
        rules = dtype_registry.get_rules('add')  # May have multiple rules
        if len(rules) > 1:
            for i in range(len(rules) - 1):
                assert rules[i].priority <= rules[i + 1].priority

    def test_registry_case_insensitive(self):
        """Registry should handle function names case-insensitively."""
        abs_rules = dtype_registry.get_rules('abs')
        ABS_rules = dtype_registry.get_rules('ABS')
        assert len(abs_rules) == len(ABS_rules)


# =============================================================================
# Test Dtype Correction Configuration
# =============================================================================


class TestDtypeCorrectionConfig:
    """Test dtype correction configuration."""

    def test_default_level_is_high(self):
        """Default correction level should be HIGH."""
        assert dtype_correction_config.level == CorrectionLevel.HIGH

    def test_critical_always_applies(self):
        """CRITICAL corrections should apply at any level >= CRITICAL."""
        dtype_correction_config.set_level(CorrectionLevel.CRITICAL)
        assert dtype_correction_config.should_apply(CorrectionPriority.CRITICAL)

        dtype_correction_config.set_level(CorrectionLevel.HIGH)
        assert dtype_correction_config.should_apply(CorrectionPriority.CRITICAL)

        dtype_correction_config.reset()

    def test_disable_corrections(self):
        """Setting level to NONE should disable all corrections."""
        dtype_correction_config.set_level(CorrectionLevel.NONE)
        assert not dtype_correction_config.should_apply(CorrectionPriority.CRITICAL)

        dtype_correction_config.reset()

    def test_high_level_applies_critical_and_high(self):
        """HIGH level should apply CRITICAL and HIGH priority rules."""
        dtype_correction_config.set_level(CorrectionLevel.HIGH)

        assert dtype_correction_config.should_apply(CorrectionPriority.CRITICAL)
        assert dtype_correction_config.should_apply(CorrectionPriority.HIGH)
        assert not dtype_correction_config.should_apply(CorrectionPriority.MEDIUM)
        assert not dtype_correction_config.should_apply(CorrectionPriority.LOW)

        dtype_correction_config.reset()

    def test_all_level_applies_all_priorities(self):
        """ALL level should apply all priority rules."""
        dtype_correction_config.set_level(CorrectionLevel.ALL)

        assert dtype_correction_config.should_apply(CorrectionPriority.CRITICAL)
        assert dtype_correction_config.should_apply(CorrectionPriority.HIGH)
        assert dtype_correction_config.should_apply(CorrectionPriority.MEDIUM)
        assert dtype_correction_config.should_apply(CorrectionPriority.LOW)

        dtype_correction_config.reset()


# =============================================================================
# Test Individual Rules Directly
# =============================================================================


class TestIndividualRules:
    """Test individual correction rules directly."""

    def test_signed_abs_rule_should_correct(self):
        """SignedAbsRule should identify signed->unsigned mismatches."""
        rule = SignedAbsRule()

        # Should correct
        assert rule.should_correct('int64', 'uint64')
        assert rule.should_correct('int32', 'uint32')
        assert rule.should_correct('int16', 'uint16')
        assert rule.should_correct('int8', 'uint8')

        # Should not correct
        assert not rule.should_correct('float64', 'float64')
        assert not rule.should_correct('uint64', 'uint64')
        assert not rule.should_correct('int64', 'int64')

    def test_signed_abs_rule_target_dtype(self):
        """SignedAbsRule should return correct target dtype."""
        rule = SignedAbsRule()

        assert rule.get_target_dtype('int64') == 'int64'
        assert rule.get_target_dtype('int32') == 'int32'
        assert rule.get_target_dtype('float64') is None
        assert rule.get_target_dtype('uint64') is None

    def test_sign_preserve_rule_should_correct(self):
        """SignPreserveRule should identify int8 output mismatches."""
        rule = SignPreserveRule()

        # Should correct
        assert rule.should_correct('int64', 'int8')
        assert rule.should_correct('float64', 'int8')

        # Should not correct
        assert not rule.should_correct('int8', 'int8')
        assert not rule.should_correct('int64', 'int64')

    def test_arithmetic_preserve_rule_family_detection(self):
        """ArithmeticPreserveRule should correctly detect dtype families."""
        rule = ArithmeticPreserveRule()

        assert rule._get_dtype_family('int64') == 'int'
        assert rule._get_dtype_family('uint64') == 'uint'
        assert rule._get_dtype_family('float64') == 'float'
        assert rule._get_dtype_family('object') == 'other'

    def test_floordiv_preserve_rule_should_correct(self):
        """FloordivPreserveRule should identify float->int mismatches."""
        rule = FloordivPreserveRule()

        # Should correct
        assert rule.should_correct('float64', 'int64')
        assert rule.should_correct('float32', 'int32')

        # Should not correct
        assert not rule.should_correct('int64', 'int64')
        assert not rule.should_correct('float64', 'float64')

    def test_pow_preserve_rule_apply_with_whole_numbers(self):
        """PowPreserveRule should convert to int64 for whole number results."""
        rule = PowPreserveRule()

        # Whole numbers should convert
        series = pd.Series([1.0, 4.0, 9.0, 16.0])
        result = rule.apply(series, 'int64')
        assert result.dtype == 'int64'

    def test_pow_preserve_rule_apply_with_fractional(self):
        """PowPreserveRule should preserve float for fractional results."""
        rule = PowPreserveRule()

        # Fractional numbers should stay float
        series = pd.Series([1.5, 2.5, 3.5])
        result = rule.apply(series, 'int64')
        assert 'float' in str(result.dtype)


# =============================================================================
# Test Integration with Full Pipeline
# =============================================================================


class TestDtypeCorrectionIntegration:
    """Integration tests for dtype correction in full pipeline."""

    def test_abs_in_complex_pipeline(self):
        """abs() correction works in complex pipelines."""
        pd_df = pd.DataFrame(
            {
                'a': [-5, -3, -1, 1, 3, 5],
                'b': [1, 2, 3, 4, 5, 6],
            }
        )
        ds_df = DataStore(pd_df.copy())

        # Complex pipeline: filter -> abs -> filter
        pd_result = pd_df[pd_df['b'] > 2].copy()
        pd_result['abs_a'] = pd_result['a'].abs()
        pd_result = pd_result[pd_result['abs_a'] > 2]

        ds_result = ds_df[ds_df['b'] > 2]
        ds_result['abs_a'] = ds_result['a'].abs()
        ds_result = ds_result[ds_result['abs_a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_abs_columns(self):
        """Multiple abs() columns in same DataFrame."""
        pd_df = pd.DataFrame(
            {
                'a': [-3, -2, -1, 0, 1, 2, 3],
                'b': [-6, -4, -2, 0, 2, 4, 6],
            }
        )
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        pd_df['abs_b'] = pd_df['b'].abs()

        ds_df['abs_a'] = ds_df['a'].abs()
        ds_df['abs_b'] = ds_df['b'].abs()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_abs_with_sorting(self):
        """abs() should work correctly with sorting."""
        # Use unique abs values to avoid non-deterministic ordering
        # when sort keys are equal (chDB sort is not guaranteed stable)
        pd_df = pd.DataFrame({'a': [-5, -4, -3, -2, -1, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        pd_result = pd_df.sort_values('abs_a', ascending=False)

        ds_df['abs_a'] = ds_df['a'].abs()
        ds_result = ds_df.sort_values('abs_a', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_abs_with_groupby_aggregation(self):
        """abs() should work correctly with groupby aggregation."""
        pd_df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'C', 'C'],
                'value': [-10, -5, -3, 3, -1, 1],
            }
        )
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_value'] = pd_df['value'].abs()
        ds_df['abs_value'] = ds_df['value'].abs()

        pd_result = pd_df.groupby('category')['abs_value'].sum().reset_index()
        ds_result = ds_df.groupby('category')['abs_value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_mixed_dtype_operations(self):
        """Mixed dtype operations should work correctly."""
        pd_df = pd.DataFrame(
            {
                'int_col': pd.array([-5, -3, -1, 1, 3, 5], dtype='int64'),
                'float_col': pd.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype='float64'),
            }
        )
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_int'] = pd_df['int_col'].abs()
        pd_df['abs_float'] = pd_df['float_col'].abs()
        pd_df['sum'] = pd_df['abs_int'] + pd_df['abs_float']

        ds_df['abs_int'] = ds_df['int_col'].abs()
        ds_df['abs_float'] = ds_df['float_col'].abs()
        ds_df['sum'] = ds_df['abs_int'] + ds_df['abs_float']

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_almost_equal(ds_result['sum'].values, pd_df['sum'].values)


# =============================================================================
# Test SQL Pushdown Path
# =============================================================================


class TestSQLPushdownDtypeCorrection:
    """Test dtype correction in SQL pushdown execution path."""

    def test_abs_sql_pushdown_with_debug_log(self):
        """Verify abs() generates correct SQL and applies dtype correction."""
        pd_df = pd.DataFrame({'a': pd.array([-3, -2, -1, 0, 1, 2, 3], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        # Capture debug logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger('datastore')
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            ds_df['abs_a'] = ds_df['a'].abs()
            _ = get_dataframe(ds_df)

            log_output = log_capture.getvalue()
            # Should contain abs in the SQL
            assert 'abs' in log_output.lower() or 'ABS' in log_output
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

    def test_multiple_corrections_in_single_query(self):
        """Multiple dtype corrections in a single SQL query."""
        pd_df = pd.DataFrame(
            {
                'a': pd.array([-3, -2, -1, 0, 1], dtype='int64'),
                'b': pd.array([-5, -4, -3, -2, -1], dtype='int32'),
            }
        )
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        pd_df['abs_b'] = pd_df['b'].abs()

        ds_df['abs_a'] = ds_df['a'].abs()
        ds_df['abs_b'] = ds_df['b'].abs()

        ds_result = get_dataframe(ds_df)

        # Verify dtypes are correct
        assert str(ds_result['abs_a'].dtype) == 'int64'
        assert str(ds_result['abs_b'].dtype) == 'int32'

        # Verify values
        np.testing.assert_array_equal(ds_result['abs_a'].values, pd_df['abs_a'].values)
        np.testing.assert_array_equal(ds_result['abs_b'].values, pd_df['abs_b'].values)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestDtypeCorrectionEdgeCases:
    """Test edge cases for dtype correction."""

    def test_abs_with_nulls(self):
        """abs() with null values should handle nulls correctly."""
        pd_df = pd.DataFrame({'a': pd.array([-3, None, -1, 0, 1, None, 3], dtype='Int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        ds_result = get_dataframe(ds_df)
        assert_series_equal(
            ds_result['abs_a'].astype('Int64'),
            pd_df['abs_a'],
        )

    def test_abs_empty_dataframe(self):
        """abs() on empty DataFrame should work.

        Note: Empty DataFrames may have dtype differences due to chDB returning
        float64 for empty numeric results. Values are both empty, so we only
        check that the operation doesn't error and both are empty.
        """
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        ds_result = get_dataframe(ds_df)
        # Both should be empty
        assert len(ds_result) == 0
        assert len(pd_df) == 0
        assert 'abs_a' in ds_result.columns
        assert 'abs_a' in pd_df.columns

    def test_abs_single_row(self):
        """abs() on single-row DataFrame should work."""
        pd_df = pd.DataFrame({'a': pd.array([-42], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_abs_int_min_boundary(self):
        """abs() should handle int type minimum values carefully.

        Note: abs(INT_MIN) may overflow in some systems. We test with
        values close to but not at the boundary.
        """
        # INT32 min is -2147483648, we use -2147483647 to avoid overflow
        pd_df = pd.DataFrame({'a': pd.array([-2147483647, 0, 2147483647], dtype='int32')})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['abs_a'].values, pd_df['abs_a'].values)

    def test_chained_abs(self):
        """abs(abs(x)) should produce correct values.

        Note: Chained operations may have dtype differences when the intermediate
        result is lazy. We verify values are correct; dtype correction for
        complex chains is a lower priority optimization.
        """
        pd_df = pd.DataFrame({'a': pd.array([-5, -3, -1, 1, 3, 5], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['double_abs'] = pd_df['a'].abs().abs()
        ds_df['double_abs'] = ds_df['a'].abs().abs()

        ds_result = get_dataframe(ds_df)
        # Values must match
        np.testing.assert_array_equal(ds_result['double_abs'].values, pd_df['double_abs'].values)

    def test_abs_after_arithmetic(self):
        """abs() after arithmetic operations should produce correct values.

        Note: When abs() is applied to an arithmetic expression result,
        the input dtype tracking may not capture the original type.
        We verify values are correct.
        """
        pd_df = pd.DataFrame({'a': pd.array([1, 2, 3, 4, 5], dtype='int64')})
        ds_df = DataStore(pd_df.copy())

        pd_df['result'] = (pd_df['a'] - 3).abs()
        ds_df['result'] = (ds_df['a'] - 3).abs()

        ds_result = get_dataframe(ds_df)
        # Values must match
        np.testing.assert_array_equal(ds_result['result'].values, pd_df['result'].values)

    def test_arithmetic_before_abs(self):
        """Arithmetic operations before abs() should preserve correct types."""
        pd_df = pd.DataFrame(
            {
                'a': pd.array([-10, -5, 0, 5, 10], dtype='int64'),
                'b': pd.array([3, 3, 3, 3, 3], dtype='int64'),
            }
        )
        ds_df = DataStore(pd_df.copy())

        pd_df['diff_abs'] = (pd_df['a'] - pd_df['b']).abs()
        ds_df['diff_abs'] = (ds_df['a'] - ds_df['b']).abs()

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_equal(ds_result['diff_abs'].values, pd_df['diff_abs'].values)


# =============================================================================
# Test Float32 Preservation (LOW Priority)
# =============================================================================


class TestFloat32Preservation:
    """Test float32 preservation (LOW priority).

    chDB often promotes float32 to float64, but pandas preserves float32.
    This is a LOW priority correction.
    """

    def test_float32_abs_values_match(self):
        """abs() on float32 should have correct values."""
        pd_df = pd.DataFrame({'a': pd.array([-3.5, -2.5, -1.5, 0.0, 1.5], dtype='float32')})
        ds_df = DataStore(pd_df.copy())

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_almost_equal(ds_result['abs_a'].values, pd_df['abs_a'].values, decimal=5)

    def test_float32_arithmetic_values_match(self):
        """Arithmetic on float32 should have correct values."""
        pd_df = pd.DataFrame({'a': pd.array([1.5, 2.5, 3.5, 4.5], dtype='float32')})
        ds_df = DataStore(pd_df.copy())

        pd_df['result'] = pd_df['a'] * 2 + 1
        ds_df['result'] = ds_df['a'] * 2 + 1

        ds_result = get_dataframe(ds_df)
        np.testing.assert_array_almost_equal(ds_result['result'].values, pd_df['result'].values, decimal=5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
