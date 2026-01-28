"""
Test tolist() return type consistency between DataStore and pandas.

This module verifies that ColumnExpr.tolist() returns Python list type,
matching pandas Series.tolist() behavior exactly.

The key principle:
- pandas Series.tolist() returns a Python list (not numpy array)
- DataStore ColumnExpr.tolist() MUST return Python list (not numpy array)

"Values are the same" is NOT sufficient - the type must be identical.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from datastore.column_expr import ColumnExpr


class TestTolistReturnType:
    """Test that tolist() returns Python list type (Mirror Code Pattern)."""

    def test_tolist_returns_python_list_integers(self):
        """tolist() must return Python list for integer column."""
        # pandas
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        pd_result = pd_df['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})
        ds_result = ds_df['col'].tolist()

        # Type check - must be identical
        assert type(pd_result) is list, f"pandas returned {type(pd_result)}, expected list"
        assert type(ds_result) is list, f"DataStore returned {type(ds_result)}, expected list"

        # Value check
        assert ds_result == pd_result

    def test_tolist_returns_python_list_floats(self):
        """tolist() must return Python list for float column."""
        # pandas
        pd_df = pd.DataFrame({'col': [1.1, 2.2, 3.3]})
        pd_result = pd_df['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': [1.1, 2.2, 3.3]})
        ds_result = ds_df['col'].tolist()

        # Type check
        assert type(pd_result) is list
        assert type(ds_result) is list

        # Value check
        assert ds_result == pd_result

    def test_tolist_returns_python_list_strings(self):
        """tolist() must return Python list for string column."""
        # pandas
        pd_df = pd.DataFrame({'col': ['a', 'b', 'c']})
        pd_result = pd_df['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': ['a', 'b', 'c']})
        ds_result = ds_df['col'].tolist()

        # Type check
        assert type(pd_result) is list
        assert type(ds_result) is list

        # Value check
        assert ds_result == pd_result

    def test_tolist_returns_python_list_mixed_types(self):
        """tolist() must return Python list for mixed-type column."""
        # pandas
        pd_df = pd.DataFrame({'col': [1, 'two', 3.0, None]})
        pd_result = pd_df['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': [1, 'two', 3.0, None]})
        ds_result = ds_df['col'].tolist()

        # Type check - must be list
        assert type(pd_result) is list
        assert type(ds_result) is list

    def test_tolist_returns_python_list_empty(self):
        """tolist() must return Python list for empty column."""
        # pandas
        pd_df = pd.DataFrame({'col': []})
        pd_result = pd_df['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': []})
        ds_result = ds_df['col'].tolist()

        # Type check
        assert type(pd_result) is list
        assert type(ds_result) is list

        # Value check
        assert ds_result == pd_result == []

    def test_tolist_returns_python_list_single_element(self):
        """tolist() must return Python list for single element column."""
        # pandas
        pd_df = pd.DataFrame({'col': [42]})
        pd_result = pd_df['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': [42]})
        ds_result = ds_df['col'].tolist()

        # Type check
        assert type(pd_result) is list
        assert type(ds_result) is list

        # Value check
        assert ds_result == pd_result == [42]

    def test_tolist_returns_python_list_with_nulls(self):
        """tolist() must return Python list for column with nulls."""
        # pandas
        pd_df = pd.DataFrame({'col': [1.0, None, 3.0, None, 5.0]})
        pd_result = pd_df['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': [1.0, None, 3.0, None, 5.0]})
        ds_result = ds_df['col'].tolist()

        # Type check
        assert type(pd_result) is list
        assert type(ds_result) is list

    def test_tolist_not_numpy_array(self):
        """tolist() must NOT return numpy array."""
        # DataStore
        ds_df = DataStore({'col': [1, 2, 3]})
        ds_result = ds_df['col'].tolist()

        # Explicit check that it's not numpy array
        assert not isinstance(ds_result, np.ndarray), \
            f"tolist() returned numpy array, must return Python list"

    def test_tolist_element_types_match_pandas(self):
        """Elements in tolist() result should have consistent types with pandas."""
        # pandas
        pd_df = pd.DataFrame({'col': [1, 2, 3]})
        pd_result = pd_df['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': [1, 2, 3]})
        ds_result = ds_df['col'].tolist()

        # Check element types
        for i, (pd_elem, ds_elem) in enumerate(zip(pd_result, ds_result)):
            assert type(pd_elem) == type(ds_elem), \
                f"Element {i} type mismatch: pandas={type(pd_elem)}, DataStore={type(ds_elem)}"


class TestToListAliasReturnType:
    """Test that to_list() alias also returns Python list."""

    def test_to_list_returns_python_list(self):
        """to_list() (alias) must return Python list."""
        # pandas
        pd_df = pd.DataFrame({'col': [1, 2, 3]})
        pd_result = pd_df['col'].to_list()

        # DataStore (mirror)
        ds_df = DataStore({'col': [1, 2, 3]})
        ds_result = ds_df['col'].to_list()

        # Type check
        assert type(pd_result) is list
        assert type(ds_result) is list

        # Value check
        assert ds_result == pd_result


class TestTolistAfterOperations:
    """Test tolist() return type after various operations."""

    def test_tolist_after_filter(self):
        """tolist() after filter must return Python list."""
        # pandas
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        pd_result = pd_df[pd_df['col'] > 2]['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})
        ds_result = ds_df[ds_df['col'] > 2]['col'].tolist()

        # Type check
        assert type(pd_result) is list
        assert type(ds_result) is list

        # Value check
        assert ds_result == pd_result == [3, 4, 5]

    def test_tolist_after_head(self):
        """tolist() after head() must return Python list."""
        # pandas
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        pd_result = pd_df.head(3)['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})
        ds_result = ds_df.head(3)['col'].tolist()

        # Type check
        assert type(pd_result) is list
        assert type(ds_result) is list

        # Value check
        assert ds_result == pd_result == [1, 2, 3]

    def test_tolist_after_sort(self):
        """tolist() after sort must return Python list."""
        # pandas
        pd_df = pd.DataFrame({'col': [3, 1, 2]})
        pd_result = pd_df.sort_values('col')['col'].tolist()

        # DataStore (mirror)
        ds_df = DataStore({'col': [3, 1, 2]})
        ds_result = ds_df.sort_values('col')['col'].tolist()

        # Type check
        assert type(pd_result) is list
        assert type(ds_result) is list

        # Value check
        assert ds_result == pd_result == [1, 2, 3]
