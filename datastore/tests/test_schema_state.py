"""
Unit tests for SchemaState - Schema tracking for SQL pushdown.

Tests cover:
1. Creation from various sources
2. Adding computed columns
3. Pending column tracking
4. Wrap decision logic
5. Column selection
"""

import unittest

from datastore.schema_state import SchemaState, ColumnInfo
from datastore.expressions import Field, ArithmeticExpression, Literal


class TestColumnInfo(unittest.TestCase):
    """Test ColumnInfo data class."""

    def test_original_column(self):
        """Test original column creation."""
        info = ColumnInfo(name='id', source='original')

        self.assertEqual(info.name, 'id')
        self.assertTrue(info.is_original())
        self.assertFalse(info.is_computed())
        self.assertIsNone(info.expression)

    def test_computed_column(self):
        """Test computed column creation."""
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        info = ColumnInfo(
            name='c',
            source='computed',
            expression=expr,
            referenced_columns=frozenset({'a', 'b'}),
        )

        self.assertEqual(info.name, 'c')
        self.assertFalse(info.is_original())
        self.assertTrue(info.is_computed())
        self.assertEqual(info.expression, expr)
        self.assertEqual(info.referenced_columns, frozenset({'a', 'b'}))


class TestSchemaStateCreation(unittest.TestCase):
    """Test SchemaState creation methods."""

    def test_from_columns(self):
        """Create from list of column names."""
        state = SchemaState.from_columns(['id', 'name', 'value'])

        self.assertTrue(state.schema_known)
        self.assertEqual(set(state.get_column_names()), {'id', 'name', 'value'})
        self.assertEqual(len(state.pending_computed), 0)

        # All columns should be original
        for col in ['id', 'name', 'value']:
            self.assertTrue(state.is_original(col))
            self.assertFalse(state.is_computed(col))

    def test_from_schema_dict(self):
        """Create from schema dictionary."""
        schema = {'id': 'Int64', 'name': 'String', 'value': 'Float64'}
        state = SchemaState.from_schema_dict(schema)

        self.assertTrue(state.schema_known)
        self.assertEqual(set(state.get_column_names()), {'id', 'name', 'value'})

    def test_unknown_schema(self):
        """Create unknown schema state."""
        state = SchemaState.unknown()

        self.assertFalse(state.schema_known)
        self.assertEqual(len(state.get_column_names()), 0)
        self.assertEqual(len(state.pending_computed), 0)


class TestSchemaStateAddComputed(unittest.TestCase):
    """Test adding computed columns."""

    def test_add_new_computed(self):
        """Add a new computed column."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))

        new_state = state.add_computed('c', expr, {'a', 'b'})

        # Original state unchanged
        self.assertFalse(state.is_known_column('c'))

        # New state has computed column
        self.assertTrue(new_state.is_known_column('c'))
        self.assertTrue(new_state.is_computed('c'))
        self.assertIn('c', new_state.pending_computed)

    def test_add_override_original(self):
        """Override an original column."""
        state = SchemaState.from_columns(['id', 'value'])
        expr = ArithmeticExpression('*', Field('value'), Literal(2))

        new_state = state.add_computed('value', expr, {'value'})

        # 'value' is now computed
        self.assertTrue(new_state.is_computed('value'))
        self.assertIn('value', new_state.pending_computed)

    def test_add_chained_computed(self):
        """Add multiple computed columns in chain."""
        state = SchemaState.from_columns(['a', 'b'])

        # c = a + b
        expr_c = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr_c, {'a', 'b'})

        # d = c * 2
        expr_d = ArithmeticExpression('*', Field('c'), Literal(2))
        state = state.add_computed('d', expr_d, {'c'})

        self.assertTrue(state.is_computed('c'))
        self.assertTrue(state.is_computed('d'))
        self.assertEqual(state.pending_computed, {'c', 'd'})


class TestSchemaStatePendingTracking(unittest.TestCase):
    """Test pending computed column tracking."""

    def test_has_pending_computed(self):
        """Test has_pending_computed method."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})

        # 'c' is pending
        self.assertTrue(state.has_pending_computed({'c'}))
        self.assertTrue(state.has_pending_computed({'a', 'c'}))

        # 'a' is not pending (it's original)
        self.assertFalse(state.has_pending_computed({'a'}))
        self.assertFalse(state.has_pending_computed({'a', 'b'}))

    def test_materialize_pending(self):
        """Test materializing pending columns."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})

        # 'c' is pending
        self.assertIn('c', state.pending_computed)

        # Materialize
        new_state = state.materialize_pending()

        # Original state unchanged
        self.assertIn('c', state.pending_computed)

        # New state: 'c' is no longer pending, but is now original
        self.assertEqual(len(new_state.pending_computed), 0)
        self.assertTrue(new_state.is_original('c'))


class TestSchemaStateWrapDecisions(unittest.TestCase):
    """Test wrap decision logic."""

    def test_needs_wrap_for_reference(self):
        """Test wrap needed when referencing pending computed."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})

        # Referencing 'c' needs wrap (it's pending)
        self.assertTrue(state.needs_wrap_for_reference({'c'}))
        self.assertTrue(state.needs_wrap_for_reference({'a', 'c'}))

        # Referencing only original columns doesn't need wrap
        self.assertFalse(state.needs_wrap_for_reference({'a'}))
        self.assertFalse(state.needs_wrap_for_reference({'a', 'b'}))

    def test_needs_wrap_for_override(self):
        """Test wrap needed when overriding pending computed."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})

        # Overriding 'c' needs wrap (it's pending)
        self.assertTrue(state.needs_wrap_for_override('c'))

        # Overriding 'a' doesn't need wrap (it's original)
        self.assertFalse(state.needs_wrap_for_override('a'))

        # Overriding unknown column doesn't need wrap
        self.assertFalse(state.needs_wrap_for_override('unknown'))

    def test_no_wrap_after_materialize(self):
        """After materialize, no wrap needed for references."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})
        state = state.materialize_pending()

        # 'c' is now materialized, no wrap needed
        self.assertFalse(state.needs_wrap_for_reference({'c'}))
        self.assertFalse(state.needs_wrap_for_override('c'))


class TestSchemaStateColumnSelection(unittest.TestCase):
    """Test column selection."""

    def test_select_columns(self):
        """Test selecting subset of columns."""
        state = SchemaState.from_columns(['id', 'name', 'value'])

        new_state = state.select_columns(['id', 'value'])

        self.assertEqual(set(new_state.get_column_names()), {'id', 'value'})
        self.assertFalse(new_state.is_known_column('name'))

    def test_select_with_computed(self):
        """Test selecting columns including computed."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})

        new_state = state.select_columns(['a', 'c'])

        self.assertEqual(set(new_state.get_column_names()), {'a', 'c'})
        self.assertIn('c', new_state.pending_computed)
        # 'b' is removed, but 'c' still pending
        self.assertTrue(new_state.is_computed('c'))

    def test_select_removes_pending(self):
        """Selecting out a pending column removes it from pending."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})

        # Don't select 'c'
        new_state = state.select_columns(['a', 'b'])

        self.assertNotIn('c', new_state.pending_computed)
        self.assertFalse(new_state.is_known_column('c'))


class TestSchemaStateUnknownSchema(unittest.TestCase):
    """Test behavior with unknown schema."""

    def test_unknown_schema_add_computed(self):
        """Can add computed columns even with unknown schema."""
        state = SchemaState.unknown()
        expr = ArithmeticExpression('+', Field('a'), Field('b'))

        new_state = state.add_computed('c', expr, {'a', 'b'})

        self.assertTrue(new_state.is_computed('c'))
        self.assertIn('c', new_state.pending_computed)
        # Schema still unknown
        self.assertFalse(new_state.schema_known)

    def test_unknown_schema_is_checks(self):
        """is_original/is_computed return False for unknown columns."""
        state = SchemaState.unknown()

        # Unknown column is neither original nor computed
        self.assertFalse(state.is_original('unknown'))
        self.assertFalse(state.is_computed('unknown'))
        self.assertFalse(state.is_known_column('unknown'))


class TestSchemaStateMerge(unittest.TestCase):
    """Test merging schema states."""

    def test_merge_states(self):
        """Test merging two schema states."""
        state1 = SchemaState.from_columns(['a', 'b'])
        state2 = SchemaState.from_columns(['c', 'd'])

        merged = state1.merge(state2)

        self.assertEqual(set(merged.get_column_names()), {'a', 'b', 'c', 'd'})

    def test_merge_with_computed(self):
        """Test merging states with computed columns."""
        state1 = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state1 = state1.add_computed('c', expr, {'a', 'b'})

        state2 = SchemaState.from_columns(['d'])

        merged = state1.merge(state2)

        self.assertTrue(merged.is_computed('c'))
        self.assertIn('c', merged.pending_computed)

    def test_merge_unknown_with_known(self):
        """Merging unknown with known results in unknown."""
        state1 = SchemaState.unknown()
        state2 = SchemaState.from_columns(['a', 'b'])

        merged = state1.merge(state2)

        self.assertFalse(merged.schema_known)


class TestSchemaStateGetters(unittest.TestCase):
    """Test getter methods."""

    def test_get_original_columns(self):
        """Get only original columns."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})

        original = state.get_original_columns()

        self.assertEqual(set(original), {'a', 'b'})
        self.assertNotIn('c', original)

    def test_get_computed_columns(self):
        """Get only computed columns."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})

        computed = state.get_computed_columns()

        self.assertEqual(computed, ['c'])

    def test_get_expression(self):
        """Get expression for computed column."""
        state = SchemaState.from_columns(['a', 'b'])
        expr = ArithmeticExpression('+', Field('a'), Field('b'))
        state = state.add_computed('c', expr, {'a', 'b'})

        result = state.get_expression('c')

        self.assertEqual(result, expr)

        # Original column has no expression
        self.assertIsNone(state.get_expression('a'))


if __name__ == '__main__':
    unittest.main()

