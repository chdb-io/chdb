"""
Tests for the Function Registry system.

This module tests the new FunctionRegistry, FunctionSpec, and related infrastructure
that provides Single Source of Truth for function definitions.
"""

import pytest

from datastore import (
    FunctionRegistry,
    FunctionType,
    FunctionCategory,
    FunctionSpec,
    register_function,
    WindowFunction,
)
from datastore.expressions import Field, Literal
from datastore.functions import Function, AggregateFunction


class TestFunctionRegistry:
    """Test FunctionRegistry basic functionality."""

    def test_registry_has_functions(self):
        """Registry should have functions registered."""
        stats = FunctionRegistry.stats()
        assert stats['total_functions'] > 0
        assert stats['total_aliases'] > 0

    def test_get_function_by_name(self):
        """Can get function spec by primary name."""
        spec = FunctionRegistry.get('to_datetime')
        assert spec is not None
        assert spec.name == 'to_datetime'
        assert spec.clickhouse_name == 'toDateTime'

    def test_get_function_by_alias(self):
        """Can get function spec by alias."""
        spec = FunctionRegistry.get('toDateTime')
        assert spec is not None
        assert spec.name == 'to_datetime'  # Returns canonical name

    def test_alias_returns_same_spec(self):
        """Alias lookup returns same spec as primary name."""
        spec1 = FunctionRegistry.get('to_datetime')
        spec2 = FunctionRegistry.get('toDateTime')
        assert spec1 is spec2

    def test_get_nonexistent_function(self):
        """Getting nonexistent function returns None."""
        spec = FunctionRegistry.get('nonexistent_function_xyz')
        assert spec is None

    def test_has_function(self):
        """Can check if function exists."""
        assert FunctionRegistry.has('upper')
        assert FunctionRegistry.has('UPPER') is False  # Case sensitive
        assert not FunctionRegistry.has('nonexistent_xyz')

    def test_get_by_category(self):
        """Can get functions by category."""
        string_funcs = FunctionRegistry.get_by_category(FunctionCategory.STRING)
        assert len(string_funcs) > 0
        for spec in string_funcs:
            assert spec.category == FunctionCategory.STRING

    def test_get_by_type(self):
        """Can get functions by type."""
        aggregates = FunctionRegistry.get_by_type(FunctionType.AGGREGATE)
        assert len(aggregates) > 0
        for spec in aggregates:
            assert spec.func_type == FunctionType.AGGREGATE

    def test_get_aggregates(self):
        """Convenience method for getting aggregates."""
        aggregates = FunctionRegistry.get_aggregates()
        assert len(aggregates) > 0
        assert all(spec.is_aggregate for spec in aggregates)

    def test_get_window_functions(self):
        """Convenience method for getting window functions."""
        windows = FunctionRegistry.get_window_functions()
        assert len(windows) > 0
        assert all(spec.is_window for spec in windows)


class TestFunctionSpec:
    """Test FunctionSpec properties and methods."""

    def test_is_aggregate_property(self):
        """Test is_aggregate property."""
        sum_spec = FunctionRegistry.get('sum')
        assert sum_spec.is_aggregate is True
        assert sum_spec.is_scalar is False
        assert sum_spec.is_window is False

    def test_is_window_property(self):
        """Test is_window property."""
        row_num_spec = FunctionRegistry.get('row_number')
        assert row_num_spec.is_window is True
        assert row_num_spec.is_aggregate is False
        assert row_num_spec.is_scalar is False

    def test_is_scalar_property(self):
        """Test is_scalar property."""
        upper_spec = FunctionRegistry.get('upper')
        assert upper_spec.is_scalar is True
        assert upper_spec.is_aggregate is False
        assert upper_spec.is_window is False

    def test_all_names_property(self):
        """Test all_names includes primary and aliases."""
        spec = FunctionRegistry.get('to_datetime')
        all_names = spec.all_names
        assert 'to_datetime' in all_names
        assert 'toDateTime' in all_names
        # Note: 'as_datetime' was removed from aliases in function_definitions.py

    def test_build_function(self):
        """Test building function via spec."""
        spec = FunctionRegistry.get('upper')
        expr = Field('name')
        result = spec.build(expr, alias='upper_name')
        
        assert isinstance(result, Function)
        assert result.to_sql() == 'upper("name")'
        assert result.alias == 'upper_name'

    def test_build_function_with_args(self):
        """Test building function with alias argument."""
        spec = FunctionRegistry.get('to_datetime')
        expr = Field('timestamp')
        result = spec.build(expr, alias='dt')
        
        sql = result.to_sql()
        assert 'toDateTime' in sql
        assert result.alias == 'dt'


class TestFunctionBuilding:
    """Test building functions from registry specs."""

    def test_build_scalar_function(self):
        """Build a scalar function."""
        spec = FunctionRegistry.get('length')
        result = spec.build(Field('name'))
        # length is wrapped in toInt64() to match pandas int64 dtype
        assert result.to_sql() == 'toInt64(length("name"))'

    def test_build_aggregate_function(self):
        """Build an aggregate function."""
        spec = FunctionRegistry.get('sum')
        result = spec.build(Field('amount'))
        assert isinstance(result, AggregateFunction)
        assert result.to_sql() == 'sum("amount")'
        assert result.is_aggregate is True

    def test_build_window_function(self):
        """Build a window function."""
        spec = FunctionRegistry.get('row_number')
        result = spec.build()
        assert isinstance(result, WindowFunction)
        assert result.to_sql() == 'row_number()'

    def test_build_function_with_alias(self):
        """Build function with alias."""
        spec = FunctionRegistry.get('count')
        result = spec.build(alias='total')
        assert result.alias == 'total'
        sql_with_alias = result.to_sql(with_alias=True)
        assert 'AS "total"' in sql_with_alias


class TestWindowFunction:
    """Test WindowFunction class and OVER clause."""

    def test_window_function_basic(self):
        """Basic window function without OVER."""
        wf = WindowFunction('row_number')
        assert wf.to_sql() == 'row_number()'

    def test_window_function_with_over_partition(self):
        """Window function with PARTITION BY."""
        wf = WindowFunction('row_number').over(partition_by='category')
        sql = wf.to_sql()
        assert 'OVER' in sql
        assert 'PARTITION BY "category"' in sql

    def test_window_function_with_over_order(self):
        """Window function with ORDER BY."""
        wf = WindowFunction('row_number').over(order_by='value DESC')
        sql = wf.to_sql()
        assert 'ORDER BY "value" DESC' in sql

    def test_window_function_with_over_both(self):
        """Window function with both PARTITION BY and ORDER BY."""
        wf = WindowFunction('row_number').over(
            partition_by='category',
            order_by='value DESC'
        )
        sql = wf.to_sql()
        assert 'PARTITION BY "category"' in sql
        assert 'ORDER BY "value" DESC' in sql

    def test_window_function_multi_partition(self):
        """Window function with multiple partition columns."""
        wf = WindowFunction('rank').over(
            partition_by=['region', 'category'],
            order_by='sales'
        )
        sql = wf.to_sql()
        assert 'PARTITION BY "region", "category"' in sql

    def test_window_function_multi_order(self):
        """Window function with multiple order columns."""
        wf = WindowFunction('dense_rank').over(
            partition_by='user_id',
            order_by=['year', 'month DESC']
        )
        sql = wf.to_sql()
        assert 'ORDER BY "year", "month" DESC' in sql

    def test_window_function_with_frame(self):
        """Window function with frame specification."""
        wf = WindowFunction('sum', Field('amount')).over(
            partition_by='user_id',
            order_by='date',
            frame='ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW'
        )
        sql = wf.to_sql()
        assert 'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW' in sql

    def test_window_function_with_alias(self):
        """Window function with alias."""
        wf = WindowFunction('row_number').over(
            partition_by='cat',
            order_by='val'
        ).as_('rn')
        sql = wf.to_sql(with_alias=True)
        assert 'AS "rn"' in sql

    def test_window_function_immutability(self):
        """OVER clause should create new instance."""
        wf1 = WindowFunction('row_number')
        wf2 = wf1.over(partition_by='category')
        
        # Original should be unchanged
        assert wf1._partition_by == []
        assert wf2._partition_by == ['category']

    def test_window_function_with_args(self):
        """Window function with arguments (like lead/lag)."""
        wf = WindowFunction('leadInFrame', Field('value'), Literal(1))
        sql = wf.to_sql()
        assert 'leadInFrame("value",1)' in sql

    def test_window_function_lead_with_over(self):
        """Build lead function with OVER from registry."""
        spec = FunctionRegistry.get('lead')
        lead_func = spec.build(Field('value'), offset=2, default=0)
        lead_with_over = lead_func.over(
            partition_by='user_id',
            order_by='timestamp'
        )
        sql = lead_with_over.to_sql()
        assert 'leadInFrame("value",2,0)' in sql
        assert 'PARTITION BY "user_id"' in sql
        assert 'ORDER BY "timestamp"' in sql


class TestFunctionAliases:
    """Test function aliases work correctly."""

    def test_mean_is_alias_for_avg(self):
        """mean should be an alias for avg."""
        avg_spec = FunctionRegistry.get('avg')
        mean_spec = FunctionRegistry.get('mean')
        assert avg_spec is mean_spec

    def test_uniq_is_alias_for_count_distinct(self):
        """uniq should be an alias for count_distinct."""
        cd_spec = FunctionRegistry.get('count_distinct')
        uniq_spec = FunctionRegistry.get('uniq')
        assert cd_spec is uniq_spec

    def test_toDateTime_is_alias_for_to_datetime(self):
        """toDateTime should be an alias for to_datetime."""
        spec1 = FunctionRegistry.get('to_datetime')
        spec2 = FunctionRegistry.get('toDateTime')
        assert spec1 is spec2

    def test_multiple_aliases(self):
        """Functions can have multiple aliases."""
        spec = FunctionRegistry.get('trim')
        assert 'strip' in spec.aliases


class TestFunctionCategories:
    """Test function categories are correctly assigned."""

    def test_string_category(self):
        """String functions have STRING category."""
        spec = FunctionRegistry.get('upper')
        assert spec.category == FunctionCategory.STRING

    def test_datetime_category(self):
        """DateTime functions have DATETIME category."""
        # Note: to_date/to_datetime are in TYPE_CONVERSION category
        # year/month/day are in DATETIME category
        spec = FunctionRegistry.get('year')
        assert spec.category == FunctionCategory.DATETIME

    def test_math_category(self):
        """Math functions have MATH category."""
        spec = FunctionRegistry.get('sqrt')
        assert spec.category == FunctionCategory.MATH

    def test_aggregate_category(self):
        """Aggregate functions have AGGREGATE category."""
        spec = FunctionRegistry.get('sum')
        assert spec.category == FunctionCategory.AGGREGATE

    def test_window_category(self):
        """Window functions have WINDOW category."""
        spec = FunctionRegistry.get('row_number')
        assert spec.category == FunctionCategory.WINDOW

    def test_conditional_category(self):
        """Conditional functions have CONDITIONAL category."""
        spec = FunctionRegistry.get('coalesce')
        assert spec.category == FunctionCategory.CONDITIONAL


class TestFunctionTypes:
    """Test function types are correctly assigned."""

    def test_scalar_type(self):
        """Scalar functions have SCALAR type."""
        spec = FunctionRegistry.get('upper')
        assert spec.func_type == FunctionType.SCALAR

    def test_aggregate_type(self):
        """Aggregate functions have AGGREGATE type."""
        spec = FunctionRegistry.get('sum')
        assert spec.func_type == FunctionType.AGGREGATE

    def test_window_type(self):
        """Window functions have WINDOW type."""
        spec = FunctionRegistry.get('row_number')
        assert spec.func_type == FunctionType.WINDOW


class TestRegistryStats:
    """Test registry statistics."""

    def test_stats_total_functions(self):
        """Stats should report total functions."""
        stats = FunctionRegistry.stats()
        assert 'total_functions' in stats
        assert stats['total_functions'] > 50  # We defined many functions

    def test_stats_by_type(self):
        """Stats should report by type."""
        stats = FunctionRegistry.stats()
        assert 'by_type' in stats
        assert 'SCALAR' in stats['by_type']
        assert 'AGGREGATE' in stats['by_type']
        assert 'WINDOW' in stats['by_type']

    def test_stats_by_category(self):
        """Stats should report by category."""
        stats = FunctionRegistry.stats()
        assert 'by_category' in stats
        assert 'STRING' in stats['by_category']
        assert 'DATETIME' in stats['by_category']


class TestAccessorOnlyFunctions:
    """Test accessor_only flag on functions."""

    def test_accessor_only_functions_exist(self):
        """Some functions are marked as accessor_only."""
        year_spec = FunctionRegistry.get('year')
        assert year_spec is not None
        assert year_spec.accessor_only is True

    def test_non_accessor_only_functions(self):
        """Most functions are not accessor_only."""
        upper_spec = FunctionRegistry.get('upper')
        assert upper_spec.accessor_only is False


class TestSupportsOverFlag:
    """Test supports_over flag for window functions."""

    def test_window_functions_support_over(self):
        """Window functions should have supports_over=True."""
        row_num = FunctionRegistry.get('row_number')
        assert row_num.supports_over is True

    def test_scalar_functions_dont_support_over(self):
        """Scalar functions should not have supports_over."""
        upper = FunctionRegistry.get('upper')
        assert upper.supports_over is False


