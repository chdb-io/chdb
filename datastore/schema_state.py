"""
Schema State Tracking for SQL Pushdown.

This module provides schema state tracking for the lazy operation chain,
enabling correct SQL pushdown decisions by tracking:

1. Column origin (original from source vs computed by operation)
2. Column expressions (SQL expressions for computed columns)
3. Pending computed columns (columns added in current SQL layer)

Key Design Decisions:
- SchemaState is immutable-friendly (returns new instances on modification)
- Supports unknown schema scenarios (graceful degradation)
- Tracks both column names and their SQL expressions for subquery building

Example Usage:
    # Initialize from known schema
    state = SchemaState.from_columns(['id', 'name', 'value'])

    # Add computed column
    state = state.add_computed('doubled', expr, ['value'])

    # Check if operation references pending computed columns
    if state.has_pending_computed({'doubled'}):
        # Need to wrap subquery first
        state = state.materialize_pending()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Literal, Any
from copy import copy

from .expressions import Expression


@dataclass(frozen=True)
class ColumnInfo:
    """
    Information about a single column.

    Attributes:
        name: Column name
        source: Where this column came from
            - 'original': From the data source
            - 'computed': Created by a LazyColumnAssignment
        expression: SQL expression for computed columns (None for original)
        referenced_columns: Columns referenced by the expression
    """

    name: str
    source: Literal['original', 'computed']
    expression: Optional[Expression] = None
    referenced_columns: frozenset = field(default_factory=frozenset)

    def is_computed(self) -> bool:
        """Check if this column is computed."""
        return self.source == 'computed'

    def is_original(self) -> bool:
        """Check if this column is from the original source."""
        return self.source == 'original'


@dataclass
class SchemaState:
    """
    Tracks schema state through the lazy operation chain.

    This class maintains:
    1. Column metadata (name, source, expression)
    2. Pending computed columns (not yet materialized in a subquery)
    3. Schema known flag (whether we know all columns)

    The 'pending_computed' set tracks columns that have been added in the
    current SQL layer but not yet materialized. When an operation references
    a pending computed column, we need to wrap the current layer as a subquery
    first.

    Attributes:
        columns: Mapping from column name to ColumnInfo
        pending_computed: Set of column names pending in current layer
        schema_known: Whether we know the complete schema
    """

    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    pending_computed: Set[str] = field(default_factory=set)
    schema_known: bool = False

    @classmethod
    def from_columns(cls, column_names: List[str]) -> 'SchemaState':
        """
        Create SchemaState from a list of known column names.

        Args:
            column_names: List of column names from the source

        Returns:
            New SchemaState instance with known schema
        """
        columns = {name: ColumnInfo(name=name, source='original') for name in column_names}
        return cls(columns=columns, pending_computed=set(), schema_known=True)

    @classmethod
    def from_schema_dict(cls, schema_dict: Dict[str, str]) -> 'SchemaState':
        """
        Create SchemaState from a schema dictionary.

        Args:
            schema_dict: Mapping from column name to data type

        Returns:
            New SchemaState instance with known schema
        """
        columns = {name: ColumnInfo(name=name, source='original') for name in schema_dict.keys()}
        return cls(columns=columns, pending_computed=set(), schema_known=True)

    @classmethod
    def unknown(cls) -> 'SchemaState':
        """
        Create SchemaState for unknown schema scenario.

        Returns:
            New SchemaState with schema_known=False
        """
        return cls(columns={}, pending_computed=set(), schema_known=False)

    def copy(self) -> 'SchemaState':
        """Create a copy of this state."""
        return SchemaState(
            columns=dict(self.columns),
            pending_computed=set(self.pending_computed),
            schema_known=self.schema_known,
        )

    def add_computed(
        self,
        name: str,
        expression: Expression,
        referenced_columns: Set[str] = None,
    ) -> 'SchemaState':
        """
        Add a computed column to the schema.

        If the column already exists (override scenario), it's updated.
        The column is added to pending_computed set.

        Args:
            name: Column name
            expression: SQL expression for the column
            referenced_columns: Columns referenced by the expression

        Returns:
            New SchemaState with the computed column added
        """
        new_state = self.copy()
        refs = frozenset(referenced_columns) if referenced_columns else frozenset()

        new_state.columns[name] = ColumnInfo(
            name=name,
            source='computed',
            expression=expression,
            referenced_columns=refs,
        )
        new_state.pending_computed.add(name)
        return new_state

    def materialize_pending(self) -> 'SchemaState':
        """
        Materialize pending computed columns.

        Called when wrapping current layer as subquery. After materialization,
        computed columns become "original" columns of the subquery output.

        Returns:
            New SchemaState with pending columns cleared
        """
        new_state = self.copy()

        # Convert pending computed columns to "original" (they're now
        # part of the subquery output)
        for name in new_state.pending_computed:
            if name in new_state.columns:
                old_info = new_state.columns[name]
                new_state.columns[name] = ColumnInfo(
                    name=name,
                    source='original',  # Now materialized
                    expression=None,
                    referenced_columns=frozenset(),
                )

        new_state.pending_computed = set()
        return new_state

    def select_columns(self, columns: List[str]) -> 'SchemaState':
        """
        Apply column selection, keeping only specified columns.

        Args:
            columns: Column names to keep

        Returns:
            New SchemaState with only selected columns
        """
        new_state = self.copy()
        new_state.columns = {name: info for name, info in new_state.columns.items() if name in columns}
        new_state.pending_computed &= set(columns)
        return new_state

    def get_column_names(self) -> List[str]:
        """Get list of all known column names."""
        return list(self.columns.keys())

    def get_original_columns(self) -> List[str]:
        """Get list of original (non-computed) column names."""
        return [name for name, info in self.columns.items() if info.is_original()]

    def get_computed_columns(self) -> List[str]:
        """Get list of computed column names."""
        return [name for name, info in self.columns.items() if info.is_computed()]

    def get_pending_computed(self) -> Set[str]:
        """Get set of pending computed column names."""
        return set(self.pending_computed)

    def has_pending_computed(self, names: Set[str]) -> bool:
        """
        Check if any of the given names are pending computed columns.

        Args:
            names: Set of column names to check

        Returns:
            True if any name is in pending_computed
        """
        return bool(names & self.pending_computed)

    def is_computed(self, name: str) -> bool:
        """Check if a column is computed."""
        if name in self.columns:
            return self.columns[name].is_computed()
        return False

    def is_original(self, name: str) -> bool:
        """Check if a column is original."""
        if name in self.columns:
            return self.columns[name].is_original()
        return False

    def is_known_column(self, name: str) -> bool:
        """Check if a column is known in the schema."""
        return name in self.columns

    def get_expression(self, name: str) -> Optional[Expression]:
        """Get the expression for a computed column."""
        if name in self.columns:
            return self.columns[name].expression
        return None

    def needs_wrap_for_reference(self, referenced_columns: Set[str]) -> bool:
        """
        Check if referencing these columns requires a subquery wrap.

        A wrap is needed if any referenced column is pending computed.

        Args:
            referenced_columns: Columns that would be referenced

        Returns:
            True if a wrap is needed
        """
        return bool(referenced_columns & self.pending_computed)

    def needs_wrap_for_override(self, column_name: str) -> bool:
        """
        Check if overriding this column requires a subquery wrap.

        A wrap is needed if the column is pending computed (would create
        duplicate alias in same SQL layer).

        Args:
            column_name: Column to be overridden

        Returns:
            True if a wrap is needed
        """
        return column_name in self.pending_computed

    def merge(self, other: 'SchemaState') -> 'SchemaState':
        """
        Merge another SchemaState into this one.

        Used when combining operation chains.

        Args:
            other: SchemaState to merge

        Returns:
            New merged SchemaState
        """
        new_state = self.copy()
        new_state.columns.update(other.columns)
        new_state.pending_computed.update(other.pending_computed)
        new_state.schema_known = self.schema_known and other.schema_known
        return new_state

    def __repr__(self) -> str:
        cols = list(self.columns.keys())[:5]
        if len(self.columns) > 5:
            cols.append('...')
        pending = list(self.pending_computed)[:3]
        if len(self.pending_computed) > 3:
            pending.append('...')
        return f"SchemaState(columns={cols}, " f"pending={pending}, " f"known={self.schema_known})"
