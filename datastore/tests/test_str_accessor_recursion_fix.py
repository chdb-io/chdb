"""
Tests for the string accessor recursion fix.

This test file verifies that string accessor methods (str.extract, str.extractall,
str.split, str.wrap) do not cause RecursionError when:
1. Used in column assignment
2. The resulting column is then used in operations like pd.crosstab

The root cause was that the executor closure would call col_expr._execute(),
which triggered the parent DataStore's _execute() again, causing infinite recursion.

The fix stores metadata on the ColumnExpr (_str_accessor_method, _str_source_expr, etc.)
and ExpressionEvaluator handles this by evaluating the source from the current df
instead of calling _execute().
"""

import unittest
import pandas as pd
import numpy as np

from datastore import DataStore
import datastore.pandas_api as ds_pd
from tests.test_utils import assert_datastore_equals_pandas


class TestStrExtractRecursionFix(unittest.TestCase):
    """Test that str.extract doesn't cause recursion when used with column assignment."""

    def test_str_extract_column_assignment_no_recursion(self):
        """str.extract used in column assignment should not cause RecursionError."""
        # This is the exact scenario from the bug report
        data = {
            'Name': [
                'Braund, Mr. Owen Harris',
                'Cumings, Mrs. John',
                'Heikkinen, Miss. Laina',
                'Futrelle, Mrs. Jacques',
                'Allen, Mr. William',
            ],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
        }

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['Title'] = pd_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['Title'] = ds_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Compare results - this triggers execution
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_extract_with_crosstab_no_recursion(self):
        """str.extract result used in pd.crosstab should not cause RecursionError."""
        data = {
            'Name': [
                'Braund, Mr. Owen Harris',
                'Cumings, Mrs. John',
                'Heikkinen, Miss. Laina',
                'Futrelle, Mrs. Jacques',
                'Allen, Mr. William',
            ],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
        }

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['Title'] = pd_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
        pd_result = pd.crosstab(pd_df['Title'], pd_df['Sex'])

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['Title'] = ds_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds_result = ds_pd.crosstab(ds_df['Title'], ds_df['Sex'])

        # Compare crosstab results (ds_pd.crosstab returns DataStore)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_extract_expand_true_column_assignment(self):
        """str.extract with expand=True in column assignment should work."""
        data = {'text': ['a1b2', 'c3d4', 'e5f6']}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_extracted = pd_df['text'].str.extract(r'([a-z])(\d)', expand=True)
        pd_df['letter'] = pd_extracted[0]
        pd_df['digit'] = pd_extracted[1]

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_extracted = ds_df['text'].str.extract(r'([a-z])(\d)', expand=True)
        ds_df['letter'] = ds_extracted[0]
        ds_df['digit'] = ds_extracted[1]

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_extract_multiple_columns_assignment(self):
        """Multiple str.extract column assignments should work without recursion."""
        data = {
            'Name': ['John-Smith', 'Jane-Doe', 'Bob-Johnson'],
            'Code': ['A1-B2', 'C3-D4', 'E5-F6'],
        }

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['FirstName'] = pd_df['Name'].str.extract(r'^([A-Za-z]+)-', expand=False)
        pd_df['LastName'] = pd_df['Name'].str.extract(r'-([A-Za-z]+)$', expand=False)
        pd_df['CodePrefix'] = pd_df['Code'].str.extract(r'^([A-Z]\d)-', expand=False)

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['FirstName'] = ds_df['Name'].str.extract(r'^([A-Za-z]+)-', expand=False)
        ds_df['LastName'] = ds_df['Name'].str.extract(r'-([A-Za-z]+)$', expand=False)
        ds_df['CodePrefix'] = ds_df['Code'].str.extract(r'^([A-Z]\d)-', expand=False)

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)


class TestStrExtractallRecursionFix(unittest.TestCase):
    """Test that str.extractall doesn't cause recursion when used with column assignment."""

    def test_str_extractall_execution(self):
        """str.extractall should execute without recursion."""
        data = {'text': ['a1b2c3', 'd4e5', 'f6']}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_result = pd_df['text'].str.extractall(r'([a-z])(\d)')

        # DataStore operations
        ds_df = DataStore(data)
        ds_result = ds_df['text'].str.extractall(r'([a-z])(\d)')

        # Trigger execution and compare
        # extractall returns a DataFrame with MultiIndex
        pd.testing.assert_frame_equal(
            ds_result._execute() if hasattr(ds_result, '_execute') else ds_result,
            pd_result,
        )


class TestStrSplitExpandRecursionFix(unittest.TestCase):
    """Test that str.split with expand=True doesn't cause recursion."""

    def test_str_split_expand_true_column_assignment(self):
        """str.split with expand=True in column assignment should work."""
        data = {'name': ['John Smith', 'Jane Doe', 'Bob Johnson']}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_split = pd_df['name'].str.split(' ', expand=True)
        pd_df['first_name'] = pd_split[0]
        pd_df['last_name'] = pd_split[1]

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_split = ds_df['name'].str.split(' ', expand=True)
        ds_df['first_name'] = ds_split[0]
        ds_df['last_name'] = ds_split[1]

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_split_expand_true_with_n_limit(self):
        """str.split with expand=True and n limit should work."""
        data = {'path': ['a/b/c/d', 'e/f/g', 'h/i']}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_split = pd_df['path'].str.split('/', n=2, expand=True)
        pd_df['part1'] = pd_split[0]
        pd_df['part2'] = pd_split[1]

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_split = ds_df['path'].str.split('/', n=2, expand=True)
        ds_df['part1'] = ds_split[0]
        ds_df['part2'] = ds_split[1]

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)


class TestStrWrapRecursionFix(unittest.TestCase):
    """Test that str.wrap doesn't cause recursion when used with column assignment."""

    def test_str_wrap_column_assignment(self):
        """str.wrap in column assignment should work without recursion."""
        data = {'text': ['hello world this is a test', 'another long text here', 'short']}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['wrapped'] = pd_df['text'].str.wrap(10)

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['wrapped'] = ds_df['text'].str.wrap(10)

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)


class TestStrAccessorMetadataPreservation(unittest.TestCase):
    """Test that str accessor metadata is properly preserved on ColumnExpr."""

    def test_str_extract_has_metadata(self):
        """str.extract result should have operation descriptor metadata."""
        ds = DataStore({'text': ['a1', 'b2']})
        result = ds['text'].str.extract(r'([a-z])(\d)', expand=False)

        # Check new operation descriptor metadata exists
        self.assertEqual(result._op_type, 'accessor')
        self.assertEqual(result._op_accessor, 'str')
        self.assertEqual(result._op_method, 'extract')
        self.assertIsNotNone(result._op_source)
        self.assertEqual(result._op_args, (r'([a-z])(\d)',))
        self.assertEqual(result._op_kwargs, {'flags': 0, 'expand': False})

    def test_str_extractall_executes_correctly(self):
        """str.extractall should execute correctly and match pandas behavior."""
        data = {'text': ['a1b2', 'c3d4']}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_result = pd_df['text'].str.extractall(r'([a-z])(\d)')

        # DataStore operations
        ds_df = DataStore(data)
        ds_result = ds_df['text'].str.extractall(r'([a-z])(\d)')

        # extractall returns a DataStore - compare column values
        # Note: MultiIndex may not be preserved, so we compare data content
        np.testing.assert_array_equal(ds_result[0].values, pd_result[0].values)
        np.testing.assert_array_equal(ds_result[1].values, pd_result[1].values)

    def test_str_split_expand_true_has_metadata(self):
        """str.split with expand=True should have operation descriptor metadata."""
        ds = DataStore({'text': ['a b', 'c d']})
        result = ds['text'].str.split(' ', expand=True)

        # Check new operation descriptor metadata exists
        self.assertEqual(result._op_type, 'accessor')
        self.assertEqual(result._op_accessor, 'str')
        self.assertEqual(result._op_method, 'split')
        self.assertIsNotNone(result._op_source)
        self.assertIn('expand', result._op_kwargs)
        self.assertTrue(result._op_kwargs['expand'])

    def test_str_wrap_has_metadata(self):
        """str.wrap result should have operation descriptor metadata."""
        ds = DataStore({'text': ['hello world']})
        result = ds['text'].str.wrap(5)

        # Check new operation descriptor metadata exists
        self.assertEqual(result._op_type, 'accessor')
        self.assertEqual(result._op_accessor, 'str')
        self.assertEqual(result._op_method, 'wrap')
        self.assertIsNotNone(result._op_source)
        self.assertEqual(result._op_args, (5,))


class TestGroupbyNthRecursionFix(unittest.TestCase):
    """Test that groupby().nth() doesn't cause recursion when used with column assignment."""

    def test_groupby_nth_column_assignment_no_recursion(self):
        """groupby().nth() in column assignment should not cause RecursionError."""
        data = {'category': ['A', 'A', 'B', 'B', 'A'], 'value': [1, 2, 3, 4, 5]}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['nth_val'] = pd_df.groupby('category')['value'].nth(0)

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['nth_val'] = ds_df.groupby('category')['value'].nth(0)

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_groupby_nth_negative_index(self):
        """groupby().nth() with negative index should work."""
        data = {'category': ['A', 'A', 'B', 'B', 'A'], 'value': [1, 2, 3, 4, 5]}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['last_val'] = pd_df.groupby('category')['value'].nth(-1)

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['last_val'] = ds_df.groupby('category')['value'].nth(-1)

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_groupby_nth_multiple_groups(self):
        """groupby().nth() with multiple groupby columns should work."""
        data = {
            'cat1': ['A', 'A', 'A', 'B', 'B', 'B'],
            'cat2': ['X', 'X', 'Y', 'X', 'Y', 'Y'],
            'value': [1, 2, 3, 4, 5, 6],
        }

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['nth_val'] = pd_df.groupby(['cat1', 'cat2'])['value'].nth(0)

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['nth_val'] = ds_df.groupby(['cat1', 'cat2'])['value'].nth(0)

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_groupby_nth_has_metadata(self):
        """groupby().nth() result should have operation descriptor metadata."""
        ds = DataStore({'category': ['A', 'B'], 'value': [1, 2]})
        result = ds.groupby('category')['value'].nth(0)

        # Check operation descriptor metadata exists
        self.assertEqual(result._op_type, 'groupby_nth')
        self.assertIsNotNone(result._op_source)
        self.assertEqual(result._op_groupby_cols, ['category'])


class TestGroupbySizeRecursionFix(unittest.TestCase):
    """Test that groupby().size() doesn't cause recursion when used with column assignment."""

    def test_groupby_size_column_assignment_no_recursion(self):
        """groupby().size() in column assignment should not cause RecursionError.

        Note: size() returns a Series indexed by groupby cols, so column assignment
        results in NaN for non-matching indices (expected pandas behavior).
        """
        data = {'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['size_col'] = pd_df.groupby('category').size()

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['size_col'] = ds_df.groupby('category').size()

        # Compare results - both should have NaN because size() index doesn't match
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_groupby_size_standalone_execution(self):
        """groupby().size() standalone should work without recursion."""
        data = {'category': ['A', 'A', 'B', 'B', 'A'], 'value': [1, 2, 3, 4, 5]}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('category').size()

        # DataStore operations
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('category').size()

        # Compare results
        pd.testing.assert_series_equal(
            ds_result._execute() if hasattr(ds_result, '_execute') else ds_result,
            pd_result,
        )


class TestGroupbyCumcountRecursionFix(unittest.TestCase):
    """Test that groupby().cumcount() doesn't cause recursion when used with column assignment."""

    def test_groupby_cumcount_column_assignment_no_recursion(self):
        """groupby().cumcount() in column assignment should not cause RecursionError."""
        data = {'category': ['A', 'A', 'B', 'B', 'A'], 'value': [1, 2, 3, 4, 5]}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['cumcount'] = pd_df.groupby('category').cumcount()

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['cumcount'] = ds_df.groupby('category').cumcount()

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_groupby_cumcount_descending(self):
        """groupby().cumcount(ascending=False) should work without recursion."""
        data = {'category': ['A', 'A', 'B', 'B', 'A'], 'value': [1, 2, 3, 4, 5]}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['cumcount_desc'] = pd_df.groupby('category').cumcount(ascending=False)

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['cumcount_desc'] = ds_df.groupby('category').cumcount(ascending=False)

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)


class TestGroupbyTransformRecursionFix(unittest.TestCase):
    """Test that groupby().transform() doesn't cause recursion when used with column assignment."""

    def test_groupby_transform_column_assignment_no_recursion(self):
        """groupby().transform() in column assignment should not cause RecursionError."""
        data = {'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['mean_val'] = pd_df.groupby('category')['value'].transform('mean')

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['mean_val'] = ds_df.groupby('category')['value'].transform('mean')

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_groupby_transform_sum(self):
        """groupby().transform('sum') should work without recursion."""
        data = {'category': ['A', 'A', 'B', 'B', 'A'], 'value': [1, 2, 3, 4, 5]}

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['group_sum'] = pd_df.groupby('category')['value'].transform('sum')

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['group_sum'] = ds_df.groupby('category')['value'].transform('sum')

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_groupby_transform_has_metadata(self):
        """groupby().transform() result should have operation descriptor metadata."""
        ds = DataStore({'category': ['A', 'B'], 'value': [1, 2]})
        result = ds.groupby('category')['value'].transform('mean')

        # Check operation descriptor metadata exists
        self.assertEqual(result._op_type, 'groupby_transform')
        self.assertIsNotNone(result._op_source)
        self.assertEqual(result._op_groupby_cols, ['category'])


class TestStrAccessorChainedWithOtherOps(unittest.TestCase):
    """Test str accessor with other operations to ensure no recursion."""

    def test_str_extract_then_filter(self):
        """str.extract result used in filter should work."""
        data = {
            'Name': ['Mr. John', 'Mrs. Jane', 'Miss Alice', 'Dr. Bob'],
            'Age': [30, 25, 35, 40],
        }

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['Title'] = pd_df['Name'].str.extract(r'^([A-Za-z]+)\.', expand=False)
        pd_result = pd_df[pd_df['Title'] == 'Mr']

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['Title'] = ds_df['Name'].str.extract(r'^([A-Za-z]+)\.', expand=False)
        ds_result = ds_df[ds_df['Title'] == 'Mr']

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_extract_then_groupby(self):
        """str.extract result used in groupby should work."""
        data = {
            'Name': ['Mr. John', 'Mrs. Jane', 'Miss Alice', 'Mr. Bob'],
            'Age': [30, 25, 35, 40],
        }

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['Title'] = pd_df['Name'].str.extract(r'^([A-Za-z]+)\.', expand=False)
        pd_result = pd_df.groupby('Title')['Age'].mean().reset_index()

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['Title'] = ds_df['Name'].str.extract(r'^([A-Za-z]+)\.', expand=False)
        ds_result = ds_df.groupby('Title')['Age'].mean().reset_index()

        # Compare results (groupby order may differ)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_str_extract_then_sort(self):
        """str.extract result used in sort should work."""
        data = {
            'Name': ['Mr. John', 'Mrs. Jane', 'Miss Alice', 'Dr. Bob'],
        }

        # pandas operations
        pd_df = pd.DataFrame(data)
        pd_df['Title'] = pd_df['Name'].str.extract(r'^([A-Za-z]+)\.', expand=False)
        pd_result = pd_df.sort_values('Title')

        # DataStore operations (mirror of pandas)
        ds_df = DataStore(data)
        ds_df['Title'] = ds_df['Name'].str.extract(r'^([A-Za-z]+)\.', expand=False)
        ds_result = ds_df.sort_values('Title')

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    unittest.main()
