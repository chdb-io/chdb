"""
Unit tests for SQLBuilder - LazyColumnAssignment SQL Pushdown.

These tests verify the subquery wrapping strategy for computed columns.
Each test focuses on a specific scenario that was problematic in the
previous implementation attempt.

Test Categories:
1. Basic computed columns (SELECT *, expr AS col)
2. Computed column reference (auto subquery wrapping)
3. Column override (EXCEPT syntax)
4. Multiple computed columns
5. Column selection with computed columns
6. Complex chains (multiple computed + filter + sort)
"""

import unittest

from datastore.sql_builder import SQLBuilder, SQLLayer
from datastore.expressions import Field, ArithmeticExpression, Literal
from datastore.conditions import BinaryCondition


class TestSQLLayerBasic(unittest.TestCase):
    """Test SQLLayer basic functionality."""

    def test_simple_select_star(self):
        """Basic SELECT * FROM source."""
        layer = SQLLayer(source="file('data.csv', 'CSVWithNames')")
        sql = layer.to_sql()

        self.assertEqual(sql, "SELECT * FROM file('data.csv', 'CSVWithNames')")

    def test_select_with_computed_column(self):
        """SELECT *, (expr) AS col FROM source."""
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        layer = SQLLayer(
            source="file('data.csv', 'CSVWithNames')",
            computed_columns=[('c', expr)],
        )
        sql = layer.to_sql()

        self.assertIn('SELECT *,', sql)
        self.assertIn('AS "c"', sql)
        self.assertIn('"a"', sql)
        self.assertIn('"b"', sql)

    def test_select_with_except(self):
        """SELECT * EXCEPT(col), (expr) AS col FROM source."""
        expr = ArithmeticExpression('*', Field('value'), Literal(2))
        layer = SQLLayer(
            source="file('data.csv', 'CSVWithNames')",
            computed_columns=[('value', expr)],
            except_columns={'value'},
        )
        sql = layer.to_sql()

        self.assertIn('* EXCEPT("value")', sql)
        self.assertIn('AS "value"', sql)
        self.assertIn('"value"*2', sql)

    def test_select_explicit_columns(self):
        """SELECT col1, col2 FROM source."""
        layer = SQLLayer(
            source="file('data.csv', 'CSVWithNames')",
            explicit_columns=['a', 'b', 'c'],
        )
        sql = layer.to_sql()

        self.assertIn('SELECT "a", "b", "c"', sql)
        self.assertNotIn('*', sql)

    def test_where_clause(self):
        """SELECT * FROM source WHERE condition."""
        condition = BinaryCondition('>', Field('a'), Literal(10))
        layer = SQLLayer(
            source="file('data.csv', 'CSVWithNames')",
            where_conditions=[condition],
        )
        sql = layer.to_sql()

        self.assertIn('WHERE "a" > 10', sql)

    def test_orderby_clause(self):
        """SELECT * FROM source ORDER BY col ASC/DESC."""
        layer = SQLLayer(
            source="file('data.csv', 'CSVWithNames')",
            orderby_fields=[(Field('a'), True), (Field('b'), False)],
        )
        sql = layer.to_sql()

        self.assertIn('ORDER BY "a" ASC, "b" DESC', sql)

    def test_limit_offset(self):
        """SELECT * FROM source LIMIT n OFFSET m."""
        layer = SQLLayer(
            source="file('data.csv', 'CSVWithNames')",
            limit_value=10,
            offset_value=5,
        )
        sql = layer.to_sql()

        self.assertIn('LIMIT 10', sql)
        self.assertIn('OFFSET 5', sql)

    def test_nested_layer(self):
        """SELECT * FROM (inner_sql) AS alias."""
        inner_layer = SQLLayer(source="file('data.csv', 'CSVWithNames')")
        outer_layer = SQLLayer(source=inner_layer)

        sql = outer_layer.to_sql()

        self.assertIn('FROM (SELECT * FROM', sql)
        self.assertIn(') AS __subq1__', sql)


class TestSQLBuilderBasic(unittest.TestCase):
    """Test SQLBuilder basic functionality."""

    def test_simple_source(self):
        """Just source, no operations."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        sql = builder.build()

        self.assertEqual(sql, "SELECT * FROM file('data.csv', 'CSVWithNames')")

    def test_add_new_column(self):
        """Add a new computed column."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        builder.add_computed_column('c', expr)

        sql = builder.build()

        self.assertIn('SELECT *,', sql)
        self.assertIn('AS "c"', sql)
        self.assertIn('"a"+"b"', sql)

    def test_add_filter_no_computed_ref(self):
        """Add filter that doesn't reference computed column."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        builder.add_computed_column('c', expr)

        # Filter on original column 'a', not computed 'c'
        condition = BinaryCondition('>', Field('a'), Literal(10))
        builder.add_filter(condition)

        sql = builder.build()

        # Should NOT create subquery since filter doesn't reference computed column
        self.assertNotIn('__subq', sql)
        self.assertIn('WHERE "a" > 10', sql)
        self.assertIn('AS "c"', sql)


class TestSQLBuilderSubqueryWrapping(unittest.TestCase):
    """Test automatic subquery wrapping when referencing computed columns."""

    def test_filter_references_computed_column(self):
        """Filter referencing computed column should trigger subquery."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        builder.add_computed_column('c', expr)

        # Filter references computed column 'c'
        condition = BinaryCondition('>', Field('c'), Literal(100))
        builder.add_filter(condition)

        sql = builder.build()

        # Should create subquery
        self.assertIn('__subq', sql)
        # Inner query should have computed column
        self.assertIn('AS "c"', sql)
        self.assertIn('"a"+"b"', sql)
        # Outer query should have WHERE on 'c'
        self.assertIn('WHERE "c" > 100', sql)

    def test_orderby_references_computed_column(self):
        """ORDER BY referencing computed column should trigger subquery."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        builder.add_computed_column('c', expr)

        # Order by computed column 'c'
        builder.add_orderby([(Field('c'), False)])

        sql = builder.build()

        # Should create subquery
        self.assertIn('__subq', sql)
        # Outer query should have ORDER BY
        self.assertIn('ORDER BY "c" DESC', sql)

    def test_chained_computed_columns(self):
        """Computed column referencing another computed column."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")

        # c = a + b
        expr_c = ArithmeticExpression('+', Field('a'), Field('b'))
        builder.add_computed_column('c', expr_c)

        # d = c * 2 (references c)
        expr_d = ArithmeticExpression('*', Field('c'), Literal(2))
        builder.add_computed_column('d', expr_d)

        sql = builder.build()

        # Should create subquery because d references c
        self.assertIn('__subq', sql)
        # Inner query has 'c'
        self.assertIn('AS "c"', sql)
        self.assertIn('"a"+"b"', sql)
        # Outer query has 'd'
        self.assertIn('AS "d"', sql)
        self.assertIn('"c"*2', sql)

    def test_multiple_filters_on_computed(self):
        """Multiple filters, both referencing computed column."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        builder.add_computed_column('c', expr)

        condition1 = BinaryCondition('>', Field('c'), Literal(10))
        builder.add_filter(condition1)

        condition2 = BinaryCondition('<', Field('c'), Literal(100))
        builder.add_filter(condition2)

        sql = builder.build()

        # Should have subquery with both conditions in outer WHERE
        self.assertIn('__subq', sql)
        self.assertIn('"c" > 10', sql)
        self.assertIn('"c" < 100', sql)


class TestSQLBuilderColumnOverride(unittest.TestCase):
    """Test column override scenarios using EXCEPT."""

    def test_override_known_column(self):
        """Override a known original column uses explicit column list to preserve order."""
        # Known columns
        builder = SQLBuilder(
            "file('data.csv', 'CSVWithNames')",
            known_columns=['a', 'b', 'value'],
        )

        # Override 'value'
        expr = ArithmeticExpression('*', Field('value'), Literal(2))
        builder.add_computed_column('value', expr)

        sql = builder.build()

        # Should use explicit column list (preserves order) instead of EXCEPT
        # Old behavior: SELECT * EXCEPT("value"), ("value"*2) AS "value"
        # New behavior: SELECT "a", "b", ("value"*2) AS "value"
        self.assertIn('"a", "b"', sql)
        self.assertIn('AS "value"', sql)
        self.assertIn('"value"*2', sql)
        # Should NOT use EXCEPT (we use explicit list for order preservation)
        self.assertNotIn('EXCEPT', sql)

    def test_override_same_layer_computed(self):
        """Override a computed column in same layer should wrap first."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")

        # First computed
        expr1 = ArithmeticExpression('*', Field('a'), Literal(1))
        builder.add_computed_column('computed', expr1)

        # Override in same layer
        expr2 = ArithmeticExpression('*', Field('a'), Literal(2))
        builder.add_computed_column('computed', expr2)

        sql = builder.build()

        # Should wrap and use EXCEPT
        self.assertIn('__subq', sql)
        self.assertIn('* EXCEPT("computed")', sql)

    def test_multiple_overrides(self):
        """Multiple overrides of the same column."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")

        # Override 3 times
        for i in range(1, 4):
            expr = ArithmeticExpression('*', Field('a'), Literal(i))
            builder.add_computed_column('computed', expr)

        sql = builder.build()

        # Should have nested subqueries (2 subquery aliases for 3 assignments)
        self.assertGreaterEqual(sql.count('__subq'), 2)
        # Final expression should be a * 3
        self.assertIn('"a"*3', sql)
        self.assertIn('AS "computed"', sql)


class TestSQLBuilderColumnSelection(unittest.TestCase):
    """Test column selection with computed columns."""

    def test_select_with_computed(self):
        """Column selection after computed column should wrap."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")

        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        builder.add_computed_column('c', expr)

        builder.select_columns(['a', 'c'])

        sql = builder.build()

        # Should wrap computed column
        self.assertIn('__subq', sql)
        # Outer should have explicit columns
        self.assertIn('SELECT "a", "c"', sql)

    def test_select_without_computed(self):
        """Column selection without computed columns."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        builder.select_columns(['a', 'b'])

        sql = builder.build()

        # No subquery needed
        self.assertNotIn('__subq', sql)
        self.assertIn('SELECT "a", "b"', sql)


class TestSQLBuilderComplexScenarios(unittest.TestCase):
    """Test complex scenarios from the original failing tests."""

    def test_computed_filter_orderby_limit(self):
        """Computed column + filter on it + order by + limit."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")

        # Add computed column
        expr = ArithmeticExpression('*', Field('age'), Literal(2))
        builder.add_computed_column('age_doubled', expr)

        # Filter on computed column
        condition = BinaryCondition('>', Field('age_doubled'), Literal(60))
        builder.add_filter(condition)

        # Order by computed column
        builder.add_orderby([(Field('age_doubled'), False)])

        # Limit
        builder.add_limit(10)

        sql = builder.build()

        # Should have subquery
        self.assertIn('__subq', sql)
        # Computed column in inner
        self.assertIn('AS "age_doubled"', sql)
        self.assertIn('"age"*2', sql)
        # Filter in outer
        self.assertIn('WHERE "age_doubled" > 60', sql)
        # Order in outer
        self.assertIn('ORDER BY "age_doubled" DESC', sql)
        # Limit
        self.assertIn('LIMIT 10', sql)

    def test_chain_of_computed_columns_with_filter(self):
        """
        ds['step1'] = ds['age'] * 2
        ds['step2'] = ds['step1'] + 10
        ds['step3'] = ds['step2'] * 2
        ds.filter(ds['step3'] > 100)
        """
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")

        # step1 = age * 2
        expr1 = ArithmeticExpression('*', Field('age'), Literal(2))
        builder.add_computed_column('step1', expr1)

        # step2 = step1 + 10 (references step1 -> wraps)
        expr2 = ArithmeticExpression('+', Field('step1'), Literal(10))
        builder.add_computed_column('step2', expr2)

        # step3 = step2 * 2 (references step2 -> wraps)
        expr3 = ArithmeticExpression('*', Field('step2'), Literal(2))
        builder.add_computed_column('step3', expr3)

        # filter on step3 (references step3 -> wraps)
        condition = BinaryCondition('>', Field('step3'), Literal(100))
        builder.add_filter(condition)

        sql = builder.build()

        # Should have multiple subqueries
        self.assertIn('__subq', sql)
        # All computed columns should be present (checking key parts)
        self.assertIn('AS "step1"', sql)
        self.assertIn('"age"*2', sql)
        self.assertIn('AS "step2"', sql)
        self.assertIn('"step1"+10', sql)
        self.assertIn('AS "step3"', sql)
        self.assertIn('"step2"*2', sql)
        # Filter
        self.assertIn('WHERE "step3" > 100', sql)

    def test_select_subset_of_computed_columns(self):
        """
        ds['c1'] = ds['a'] + 1
        ds['c2'] = ds['b'] + 2
        ds['c3'] = ds['c1'] + ds['c2']
        ds = ds[['id', 'c1', 'c3']]
        """
        builder = SQLBuilder(
            "file('data.csv', 'CSVWithNames')",
            known_columns=['id', 'a', 'b'],
        )

        # c1 = a + 1
        expr1 = ArithmeticExpression('+', Field('a'), Literal(1))
        builder.add_computed_column('c1', expr1)

        # c2 = b + 2
        expr2 = ArithmeticExpression('+', Field('b'), Literal(2))
        builder.add_computed_column('c2', expr2)

        # c3 = c1 + c2 (references c1, c2 -> wraps)
        expr3 = ArithmeticExpression('+', Field('c1'), Field('c2'))
        builder.add_computed_column('c3', expr3)

        # Select subset
        builder.select_columns(['id', 'c1', 'c3'])

        sql = builder.build()

        # Should have nested subqueries
        self.assertIn('__subq', sql)
        # Final SELECT should be explicit columns
        self.assertIn('SELECT "id", "c1", "c3"', sql)

    def test_filter_on_original_then_computed_then_original(self):
        """
        ds.filter(ds['a'] > 10)  # filter on original
        ds['c'] = ds['a'] + ds['b']  # computed
        ds.filter(ds['c'] > 100)  # filter on computed
        ds.filter(ds['a'] < 50)  # filter on original again
        """
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")

        # Filter on original 'a'
        cond1 = BinaryCondition('>', Field('a'), Literal(10))
        builder.add_filter(cond1)

        # Computed column 'c'
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        builder.add_computed_column('c', expr)

        # Filter on computed 'c' -> wraps
        cond2 = BinaryCondition('>', Field('c'), Literal(100))
        builder.add_filter(cond2)

        # Filter on original 'a' again (no wrap needed, 'a' is not computed)
        cond3 = BinaryCondition('<', Field('a'), Literal(50))
        builder.add_filter(cond3)

        sql = builder.build()

        # Should have subquery for computed column reference
        self.assertIn('__subq', sql)
        # All filters should be present
        self.assertIn('"a" > 10', sql)
        self.assertIn('"c" > 100', sql)
        self.assertIn('"a" < 50', sql)


class TestSQLBuilderRowOrder(unittest.TestCase):
    """Test row order preservation."""

    def test_preserve_row_order_flag(self):
        """Test preserve_row_order adds ORDER BY rowNumberInAllBlocks()."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        builder.set_preserve_row_order(True)

        sql = builder.build()

        self.assertIn('ORDER BY rowNumberInAllBlocks()', sql)

    def test_explicit_orderby_overrides_row_order(self):
        """Explicit ORDER BY should override row order preservation."""
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        builder.set_preserve_row_order(True)
        builder.add_orderby([(Field('a'), True)])

        sql = builder.build()

        self.assertIn('ORDER BY "a" ASC', sql)
        self.assertNotIn('rowNumberInAllBlocks()', sql)


class TestExtractReferencedColumns(unittest.TestCase):
    """Test _extract_referenced_columns helper method."""

    def test_extract_from_field(self):
        """Extract column from simple Field."""
        builder = SQLBuilder("source")
        cols = builder._extract_referenced_columns(Field('my_column'))

        self.assertEqual(cols, {'my_column'})

    def test_extract_from_binary_condition(self):
        """Extract columns from BinaryCondition."""
        builder = SQLBuilder("source")
        condition = BinaryCondition('>', Field('a'), Literal(10))
        cols = builder._extract_referenced_columns(condition)

        self.assertEqual(cols, {'a'})

    def test_extract_from_arithmetic(self):
        """Extract columns from ArithmeticExpression."""
        builder = SQLBuilder("source")
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        cols = builder._extract_referenced_columns(expr)

        self.assertEqual(cols, {'a', 'b'})

    def test_extract_from_nested_arithmetic(self):
        """Extract columns from nested ArithmeticExpression."""
        builder = SQLBuilder("source")
        inner = ArithmeticExpression('+', Field('a'), Field('b'))
        outer = ArithmeticExpression('*', inner, Field('c'))
        cols = builder._extract_referenced_columns(outer)

        self.assertEqual(cols, {'a', 'b', 'c'})

    def test_extract_handles_quoted_names(self):
        """Extract strips quotes from column names."""
        builder = SQLBuilder("source")
        cols = builder._extract_referenced_columns(Field('"quoted_col"'))

        self.assertEqual(cols, {'quoted_col'})


class TestSQLBuilderEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_builder(self):
        """Builder with no operations."""
        builder = SQLBuilder("my_table")
        sql = builder.build()

        self.assertEqual(sql, "SELECT * FROM my_table")

    def test_only_limit(self):
        """Only LIMIT, no other operations."""
        builder = SQLBuilder("my_table")
        builder.add_limit(10)

        sql = builder.build()

        self.assertEqual(sql, "SELECT * FROM my_table LIMIT 10")

    def test_computed_column_with_literal_only(self):
        """Computed column using only literals."""
        builder = SQLBuilder("my_table")
        expr = ArithmeticExpression('+', Literal(1), Literal(2))
        builder.add_computed_column('three', expr)

        sql = builder.build()

        self.assertIn('AS "three"', sql)
        self.assertIn('1+2', sql)

    def test_multiple_override_columns_preserve_order(self):
        """Multiple column overrides preserve original column order."""
        builder = SQLBuilder("my_table", known_columns=['a', 'b', 'c'])

        expr_a = ArithmeticExpression('*', Field('a'), Literal(2))
        builder.add_computed_column('a', expr_a)

        expr_b = ArithmeticExpression('*', Field('b'), Literal(2))
        builder.add_computed_column('b', expr_b)

        sql = builder.build()

        # Should use explicit column list preserving order: a, b, c
        # Old behavior: SELECT * EXCEPT("a", "b"), ("a"*2) AS "a", ("b"*2) AS "b"
        # New behavior: SELECT ("a"*2) AS "a", ("b"*2) AS "b", "c"
        self.assertIn('AS "a"', sql)
        self.assertIn('AS "b"', sql)
        self.assertIn('"c"', sql)
        # Should NOT use EXCEPT (we use explicit list for order preservation)
        self.assertNotIn('EXCEPT', sql)
        # Verify order is preserved (a, b, c - not c, a, b or any other order)
        a_pos = sql.find('AS "a"')
        b_pos = sql.find('AS "b"')
        c_pos = sql.find('"c"')
        self.assertLess(a_pos, b_pos, "Column 'a' should come before 'b'")
        self.assertLess(b_pos, c_pos, "Column 'b' should come before 'c'")


class TestSQLBuilderWhereAliasConflict(unittest.TestCase):
    """
    Test handling of ClickHouse WHERE alias conflict.

    ClickHouse has a quirk: when SELECT has an alias that shadows an original
    column name, the WHERE clause may incorrectly use the aliased value instead
    of the original column value.

    Example (buggy):
        SELECT value*2 AS value FROM table WHERE value > 15
        # ClickHouse uses value*2 in WHERE instead of original value

    Fix: wrap to apply WHERE first, then compute in outer query.
    """

    def test_filter_before_same_name_assign_wraps_subquery(self):
        """
        Filter on original column, then assign same-name computed column.

        When the computed column name matches a column referenced in WHERE,
        we need to wrap to ensure WHERE uses the original column value.

        Expected SQL structure:
            SELECT * EXCEPT("value"), ("value"*2) AS "value"
            FROM (
                SELECT * FROM source WHERE "value" > 15
            ) AS __subq__
        """
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')", known_columns=['name', 'value'])

        # First add filter on 'value' (original column)
        condition = BinaryCondition('>', Field('value'), Literal(15))
        builder.add_filter(condition)

        # Then add computed column with same name 'value'
        expr = ArithmeticExpression('*', Field('value'), Literal(2))
        builder.add_computed_column('value', expr)

        sql = builder.build()

        # Should wrap: WHERE in inner subquery, computed column in outer SELECT
        self.assertIn('FROM (', sql, "Should have a subquery")
        self.assertIn('WHERE "value" > 15', sql, "WHERE should be present")
        self.assertIn('AS "value"', sql, "Computed column should be present")

        # Verify structure: WHERE is inside subquery (after FROM ()
        from_subq_pos = sql.find('FROM (')
        where_pos = sql.find('WHERE')
        self.assertGreater(where_pos, from_subq_pos, "WHERE should be inside subquery (after FROM ()")

        # Verify computed column is in outer SELECT (before FROM ()
        as_value_pos = sql.find('AS "value"')
        self.assertLess(as_value_pos, from_subq_pos, "AS value should be in outer SELECT (before FROM ()")

    def test_filter_before_different_name_assign_no_wrap(self):
        """
        Filter on one column, assign different-name computed column.

        No conflict - no wrapping needed.
        """
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')", known_columns=['name', 'value'])

        # Filter on 'value'
        condition = BinaryCondition('>', Field('value'), Literal(15))
        builder.add_filter(condition)

        # Computed column with different name 'doubled'
        expr = ArithmeticExpression('*', Field('value'), Literal(2))
        builder.add_computed_column('doubled', expr)

        sql = builder.build()

        # Should NOT have subquery wrapper (no FROM (...))
        # Note: The SQL structure should be: SELECT *, expr AS doubled ... WHERE
        self.assertNotIn('FROM (', sql, "Should not have a subquery wrapper")
        self.assertIn('WHERE', sql)
        self.assertIn('AS "doubled"', sql)

    def test_assign_before_filter_same_name_wraps_for_computed_ref(self):
        """
        Assign computed column, then filter on same name.

        Filter should use the computed value (not original), so we wrap to
        ensure the computed column is materialized before WHERE.
        """
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')", known_columns=['name', 'value'])

        # First add computed column 'value'
        expr = ArithmeticExpression('*', Field('value'), Literal(2))
        builder.add_computed_column('value', expr)

        # Then filter on 'value' (should use computed value)
        condition = BinaryCondition('>', Field('value'), Literal(15))
        builder.add_filter(condition)

        sql = builder.build()

        # Should wrap: computed in inner, WHERE in outer (uses computed value)
        self.assertIn('FROM (', sql, "Should have a subquery")
        self.assertIn('AS "value"', sql, "Computed column should be in inner query")
        self.assertIn('WHERE "value" > 15', sql, "WHERE should be in outer query")

    def test_multiple_filters_then_same_name_assign(self):
        """
        Multiple filters on same column, then assign same-name computed column.

        All filters should use original value.
        """
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')", known_columns=['name', 'value'])

        # Multiple filters on 'value'
        cond1 = BinaryCondition('>', Field('value'), Literal(10))
        cond2 = BinaryCondition('<', Field('value'), Literal(100))
        builder.add_filter(cond1)
        builder.add_filter(cond2)

        # Computed column same name
        expr = ArithmeticExpression('*', Field('value'), Literal(2))
        builder.add_computed_column('value', expr)

        sql = builder.build()

        # Both WHERE conditions should be in inner query
        self.assertIn('FROM (', sql, "Should have a subquery")
        self.assertIn('"value" > 10', sql)
        self.assertIn('"value" < 100', sql)
        self.assertIn('AS "value"', sql)


if __name__ == '__main__':
    unittest.main()

