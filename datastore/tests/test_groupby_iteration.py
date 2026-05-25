"""
Tests for pandas-style iteration over LazyGroupBy.

These tests verify that ``LazyGroupBy`` mirrors the iteration / lookup parts
of the pandas ``DataFrameGroupBy`` API:

- ``for key, group in gb:``                - iteration over (key, sub_df) pairs
- ``gb.get_group(key)``                    - direct access to one group
- ``gb.groups``                            - label-based index of each group
- ``gb.indices``                           - positional index of each group
- ``len(gb)`` / ``key in gb``              - count & membership

Each test follows the Mirror Code Pattern (CLAUDE.md): the pandas and
DataStore expressions are written verbatim side-by-side so a reader can spot
divergences instantly.

Regression motivation: a user reported that
``for (date, code), group in ds.groupby(['date', 'code']):`` raised
``TypeError: Expected str or list, got int``. The root cause was that
``LazyGroupBy`` had no ``__iter__``, so Python's iter protocol fell back to
``__getitem__(0)`` which rejected ``int``. The misleading error message
disguised a missing-feature problem as a wrong-arg-type problem.
"""

import unittest

import numpy as np
import pandas as pd

from datastore import DataStore


class TestGroupByIterationSingleColumn(unittest.TestCase):
    """``for key, group in ds.groupby(col):`` mirrors pandas exactly."""

    def setUp(self):
        self.pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A', 'C'],
                'value': [10, 20, 30, 40, 50, 60],
                'score': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        self.ds = DataStore(self.pdf.copy())

    def test_iter_yields_scalar_key_and_sub_dataframe(self):
        pd_groupby = self.pdf.groupby('category')
        ds_groupby = self.ds.groupby('category')

        pd_items = list(pd_groupby)
        ds_items = list(ds_groupby)

        self.assertEqual(len(ds_items), len(pd_items))

        for (pd_key, pd_group), (ds_key, ds_group) in zip(pd_items, ds_items):
            self.assertEqual(ds_key, pd_key)
            self.assertIsInstance(ds_group, pd.DataFrame)
            pd.testing.assert_frame_equal(ds_group, pd_group)

    def test_iter_preserves_source_index_within_each_group(self):
        custom_index = [100, 200, 300, 400, 500, 600]
        pdf = self.pdf.copy()
        pdf.index = custom_index
        ds = DataStore(pdf.copy())

        pd_items = list(pdf.groupby('category'))
        ds_items = list(ds.groupby('category'))

        for (pd_key, pd_group), (ds_key, ds_group) in zip(pd_items, ds_items):
            self.assertEqual(ds_key, pd_key)
            self.assertEqual(list(ds_group.index), list(pd_group.index))
            pd.testing.assert_frame_equal(ds_group, pd_group)

    def test_iter_total_rows_equal_source(self):
        ds_total = sum(len(g) for _, g in self.ds.groupby('category'))
        self.assertEqual(ds_total, len(self.pdf))

    def test_iter_sort_false_uses_first_occurrence_order(self):
        pd_keys = [k for k, _ in self.pdf.groupby('category', sort=False)]
        ds_keys = [k for k, _ in self.ds.groupby('category', sort=False)]

        self.assertEqual(ds_keys, pd_keys)
        self.assertEqual(ds_keys, ['A', 'B', 'C'])

    def test_iter_sort_true_default_uses_sorted_key_order(self):
        pd_keys = [k for k, _ in self.pdf.groupby('category')]
        ds_keys = [k for k, _ in self.ds.groupby('category')]

        self.assertEqual(ds_keys, sorted(pd_keys))
        self.assertEqual(ds_keys, ['A', 'B', 'C'])


class TestGroupByIterationMultiColumn(unittest.TestCase):
    """Verbatim mirror of the user-reported case (`for (date, code), group in ...`)."""

    def setUp(self):
        self.pdf = pd.DataFrame(
            {
                'date': [
                    '2026-05-22',
                    '2026-05-22',
                    '2026-05-23',
                    '2026-05-23',
                    '2026-05-23',
                ],
                'code': ['000001', '000002', '000001', '000001', '000002'],
                'price': [10.0, 20.0, 10.5, 11.0, 21.0],
                'volume': [100, 200, 150, 400, 250],
            }
        )
        self.ds = DataStore(self.pdf.copy())

    def test_iter_yields_tuple_key_and_sub_dataframe(self):
        pd_items = list(self.pdf.groupby(['date', 'code']))
        ds_items = list(self.ds.groupby(['date', 'code']))

        self.assertEqual(len(ds_items), len(pd_items))

        for (pd_key, pd_group), (ds_key, ds_group) in zip(pd_items, ds_items):
            self.assertIsInstance(ds_key, tuple)
            self.assertEqual(len(ds_key), 2)
            self.assertEqual(ds_key, pd_key)
            self.assertIsInstance(ds_group, pd.DataFrame)
            pd.testing.assert_frame_equal(ds_group, pd_group)

    def test_iter_unpacking_matches_user_screenshot_pattern(self):
        """Exact mirror of `for (date, code), group in ds.groupby(['date', 'code']):`."""
        pd_groupby = self.pdf.groupby(['date', 'code'])
        ds_groupby = self.ds.groupby(['date', 'code'])

        pd_collected = []
        for (date, code), group in pd_groupby:
            pd_collected.append((date, code, len(group), list(group['price'])))

        ds_collected = []
        for (date, code), group in ds_groupby:
            ds_collected.append((date, code, len(group), list(group['price'])))

        self.assertEqual(ds_collected, pd_collected)

    def test_iter_does_not_drop_or_duplicate_rows(self):
        ds_concat = pd.concat(
            [g for _, g in self.ds.groupby(['date', 'code'])], axis=0
        )
        ds_concat_sorted = ds_concat.sort_index()
        pd.testing.assert_frame_equal(ds_concat_sorted, self.pdf)


class TestGroupByIterationDropna(unittest.TestCase):
    """``dropna`` parameter is honoured by iteration just like pandas."""

    def setUp(self):
        self.pdf = pd.DataFrame(
            {
                'category': ['A', 'B', None, 'A', np.nan, 'B'],
                'value': [10, 20, 30, 40, 50, 60],
            }
        )
        self.ds = DataStore(self.pdf.copy())

    def test_iter_dropna_true_default_excludes_na_group(self):
        pd_keys = [k for k, _ in self.pdf.groupby('category')]
        ds_keys = [k for k, _ in self.ds.groupby('category')]

        self.assertEqual(ds_keys, pd_keys)
        self.assertNotIn(np.nan, ds_keys)
        self.assertNotIn(None, ds_keys)

    def test_iter_dropna_false_includes_na_group(self):
        pd_keys = [k for k, _ in self.pdf.groupby('category', dropna=False)]
        ds_keys = [k for k, _ in self.ds.groupby('category', dropna=False)]

        self.assertEqual(len(ds_keys), len(pd_keys))
        ds_has_na = any(pd.isna(k) for k in ds_keys)
        pd_has_na = any(pd.isna(k) for k in pd_keys)
        self.assertTrue(ds_has_na)
        self.assertEqual(ds_has_na, pd_has_na)


class TestGroupByIterationWithColumnSelection(unittest.TestCase):
    """``gb[['c1','c2']]`` then iterate yields sub-DataFrames with only those columns."""

    def setUp(self):
        self.pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B'],
                'v1': [1, 2, 3, 4],
                'v2': [10.0, 20.0, 30.0, 40.0],
                'v3': ['x', 'y', 'z', 'w'],
            }
        )
        self.ds = DataStore(self.pdf.copy())

    def test_iter_after_multi_column_selection_restricts_columns(self):
        pd_items = list(self.pdf.groupby('category')[['v1', 'v2']])
        ds_items = list(self.ds.groupby('category')[['v1', 'v2']])

        self.assertEqual(len(ds_items), len(pd_items))

        for (pd_key, pd_group), (ds_key, ds_group) in zip(pd_items, ds_items):
            self.assertEqual(ds_key, pd_key)
            self.assertEqual(list(ds_group.columns), ['v1', 'v2'])
            pd.testing.assert_frame_equal(ds_group, pd_group)


class TestGroupByGetGroup(unittest.TestCase):
    """``gb.get_group(key)`` mirrors pandas exactly."""

    def setUp(self):
        self.pdf = pd.DataFrame(
            {
                'date': ['2026-05-22', '2026-05-22', '2026-05-23', '2026-05-23'],
                'code': ['000001', '000002', '000001', '000001'],
                'price': [10.0, 20.0, 10.5, 11.0],
                'volume': [100, 200, 150, 400],
            }
        )
        self.ds = DataStore(self.pdf.copy())

    def test_get_group_single_column_returns_subframe(self):
        pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A'],
                'value': [1, 2, 3, 4, 5],
            }
        )
        ds = DataStore(pdf.copy())

        pd_group = pdf.groupby('category').get_group('A')
        ds_group = ds.groupby('category').get_group('A')

        self.assertIsInstance(ds_group, pd.DataFrame)
        pd.testing.assert_frame_equal(ds_group, pd_group)

    def test_get_group_multi_column_tuple_key(self):
        pd_group = self.pdf.groupby(['date', 'code']).get_group(
            ('2026-05-23', '000001')
        )
        ds_group = self.ds.groupby(['date', 'code']).get_group(
            ('2026-05-23', '000001')
        )

        self.assertEqual(len(ds_group), 2)
        pd.testing.assert_frame_equal(ds_group, pd_group)

    def test_get_group_with_column_selection_restricts_columns(self):
        pd_group = self.pdf.groupby('date')[['code', 'price']].get_group(
            '2026-05-23'
        )
        ds_group = self.ds.groupby('date')[['code', 'price']].get_group(
            '2026-05-23'
        )

        self.assertEqual(list(ds_group.columns), ['code', 'price'])
        pd.testing.assert_frame_equal(ds_group, pd_group)

    def test_get_group_missing_key_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.ds.groupby('date').get_group('1999-01-01')

    def test_get_group_accepts_legacy_obj_kwarg_for_pandas_api_parity(self):
        """Legacy pandas signature ``get_group(name, obj=...)`` should not
        raise ``TypeError: unexpected keyword``. The ``obj`` parameter was
        deprecated/removed in pandas >=2.0; we accept and ignore it so that
        callers carrying old pandas code don't break when swapping to
        DataStore.
        """
        pdf = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4]})
        ds = DataStore(pdf.copy())

        ds_group_no_kwarg = ds.groupby('category').get_group('A')
        ds_group_obj_none = ds.groupby('category').get_group('A', obj=None)
        ds_group_obj_unused = ds.groupby('category').get_group('A', obj=pdf)

        pd.testing.assert_frame_equal(ds_group_obj_none, ds_group_no_kwarg)
        pd.testing.assert_frame_equal(ds_group_obj_unused, ds_group_no_kwarg)


class TestGroupByGroupsAndIndices(unittest.TestCase):
    """``gb.groups`` and ``gb.indices`` mirror pandas semantics."""

    def setUp(self):
        self.pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A'],
                'value': [10, 20, 30, 40, 50],
            },
            index=[10, 20, 30, 40, 50],
        )
        self.ds = DataStore(self.pdf.copy())

    def test_groups_returns_label_index_per_group(self):
        pd_groups = self.pdf.groupby('category').groups
        ds_groups = self.ds.groupby('category').groups

        self.assertEqual(set(ds_groups.keys()), set(pd_groups.keys()))
        for key in pd_groups:
            self.assertEqual(list(ds_groups[key]), list(pd_groups[key]))

    def test_indices_returns_positional_locations_per_group(self):
        pd_indices = self.pdf.groupby('category').indices
        ds_indices = self.ds.groupby('category').indices

        self.assertEqual(set(ds_indices.keys()), set(pd_indices.keys()))
        for key in pd_indices:
            np.testing.assert_array_equal(ds_indices[key], pd_indices[key])

    def test_groups_multi_column_uses_tuple_keys(self):
        pdf = pd.DataFrame(
            {
                'a': ['x', 'x', 'y', 'y'],
                'b': [1, 2, 1, 1],
                'v': [10, 20, 30, 40],
            }
        )
        ds = DataStore(pdf.copy())

        pd_groups = pdf.groupby(['a', 'b']).groups
        ds_groups = ds.groupby(['a', 'b']).groups

        self.assertEqual(set(ds_groups.keys()), set(pd_groups.keys()))
        for key in pd_groups:
            self.assertIsInstance(key, tuple)
            self.assertEqual(list(ds_groups[key]), list(pd_groups[key]))

    def test_indices_multi_column_uses_tuple_keys(self):
        pdf = pd.DataFrame(
            {
                'a': ['x', 'x', 'y', 'y'],
                'b': [1, 2, 1, 1],
                'v': [10, 20, 30, 40],
            }
        )
        ds = DataStore(pdf.copy())

        pd_indices = pdf.groupby(['a', 'b']).indices
        ds_indices = ds.groupby(['a', 'b']).indices

        self.assertEqual(set(ds_indices.keys()), set(pd_indices.keys()))
        for key in pd_indices:
            self.assertIsInstance(key, tuple)
            np.testing.assert_array_equal(ds_indices[key], pd_indices[key])


class TestGroupByLenAndContains(unittest.TestCase):
    """``len(gb)`` and ``key in gb`` mirror pandas."""

    def setUp(self):
        self.pdf = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'C'],
                'value': [1, 2, 3, 4, 5],
            }
        )
        self.ds = DataStore(self.pdf.copy())

    def test_len_equals_ngroups_equals_pandas(self):
        pd_len = len(self.pdf.groupby('category'))
        ds_len = len(self.ds.groupby('category'))
        ds_ngroups = self.ds.groupby('category').ngroups

        self.assertEqual(ds_len, pd_len)
        self.assertEqual(ds_len, ds_ngroups)
        self.assertEqual(ds_len, 3)

    def test_contains_existing_key_single_column(self):
        gb = self.ds.groupby('category')
        self.assertIn('A', gb)
        self.assertIn('B', gb)
        self.assertIn('C', gb)

    def test_contains_missing_key_single_column(self):
        gb = self.ds.groupby('category')
        self.assertNotIn('Z', gb)

    def test_contains_existing_key_multi_column_uses_tuple(self):
        pdf = pd.DataFrame(
            {
                'a': ['x', 'x', 'y'],
                'b': [1, 2, 1],
                'v': [10, 20, 30],
            }
        )
        ds = DataStore(pdf.copy())

        gb = ds.groupby(['a', 'b'])
        self.assertIn(('x', 1), gb)
        self.assertIn(('y', 1), gb)
        self.assertNotIn(('y', 2), gb)


class TestGroupByPandasCompatibilityEdges(unittest.TestCase):
    """Pandas 2.x / 3.x compatibility edges around iteration & lookup.

    Each test mirrors the matching pandas expression verbatim so divergence is
    immediately visible. These cover scenarios that real users hit but which
    aren't exercised by the core iteration tests:

    1. ``as_index=False`` must not change iter / get_group / groups results
       (it only affects agg output shape - we verify our implementation
       doesn't accidentally couple it to iteration).
    2. ``datetime64`` group keys round-trip correctly (the user's real-world
       case typically has actual datetime, not strings).
    3. ``dropna=False`` makes ``np.nan`` a valid lookup key for both
       ``get_group()`` and ``groups``.
    4. Degenerate inputs (empty / single-row / all-same-value) iterate
       without raising.
    """

    # ----- as_index=False does not affect iter / get_group / groups -----

    def test_iter_unaffected_by_as_index_false(self):
        pdf = pd.DataFrame(
            {'cat': ['A', 'B', 'A', 'B'], 'v': [1, 2, 3, 4]}
        )
        ds = DataStore(pdf.copy())

        pd_items = list(pdf.groupby('cat', as_index=False))
        ds_items = list(ds.groupby('cat', as_index=False))

        self.assertEqual(len(ds_items), len(pd_items))
        for (pk, pg), (dk, dg) in zip(pd_items, ds_items):
            self.assertEqual(dk, pk)
            pd.testing.assert_frame_equal(dg, pg)

    def test_get_group_unaffected_by_as_index_false(self):
        pdf = pd.DataFrame(
            {'cat': ['A', 'B', 'A', 'B'], 'v': [1, 2, 3, 4]}
        )
        ds = DataStore(pdf.copy())

        pd_group = pdf.groupby('cat', as_index=False).get_group('A')
        ds_group = ds.groupby('cat', as_index=False).get_group('A')

        pd.testing.assert_frame_equal(ds_group, pd_group)

    def test_groups_unaffected_by_as_index_false(self):
        pdf = pd.DataFrame(
            {'cat': ['A', 'B', 'A', 'B'], 'v': [1, 2, 3, 4]}
        )
        ds = DataStore(pdf.copy())

        pd_groups = pdf.groupby('cat', as_index=False).groups
        ds_groups = ds.groupby('cat', as_index=False).groups

        self.assertEqual(set(ds_groups.keys()), set(pd_groups.keys()))
        for k in pd_groups:
            self.assertEqual(list(ds_groups[k]), list(pd_groups[k]))

    # ----- datetime64 group keys -----

    def test_iter_with_datetime_groupby_key_single_column(self):
        pdf = pd.DataFrame(
            {
                'date': pd.to_datetime(
                    ['2026-05-22', '2026-05-22', '2026-05-23']
                ),
                'v': [1, 2, 3],
            }
        )
        ds = DataStore(pdf.copy())

        pd_items = list(pdf.groupby('date'))
        ds_items = list(ds.groupby('date'))

        self.assertEqual(len(ds_items), len(pd_items))
        for (pk, pg), (dk, dg) in zip(pd_items, ds_items):
            self.assertEqual(dk, pk)
            self.assertIsInstance(dk, pd.Timestamp)
            pd.testing.assert_frame_equal(dg, pg)

    def test_iter_with_datetime_groupby_key_multi_column(self):
        """Real-world variant of the user's screenshot, with date as datetime64."""
        pdf = pd.DataFrame(
            {
                'date': pd.to_datetime(
                    [
                        '2026-05-22',
                        '2026-05-22',
                        '2026-05-23',
                        '2026-05-23',
                    ]
                ),
                'code': ['000001', '000002', '000001', '000001'],
                'price': [10.0, 20.0, 10.5, 11.0],
            }
        )
        ds = DataStore(pdf.copy())

        pd_items = list(pdf.groupby(['date', 'code']))
        ds_items = list(ds.groupby(['date', 'code']))

        self.assertEqual(len(ds_items), len(pd_items))
        for (pk, pg), (dk, dg) in zip(pd_items, ds_items):
            self.assertIsInstance(dk, tuple)
            self.assertIsInstance(dk[0], pd.Timestamp)
            self.assertEqual(dk, pk)
            pd.testing.assert_frame_equal(dg, pg)

    def test_get_group_with_datetime_tuple_key(self):
        pdf = pd.DataFrame(
            {
                'date': pd.to_datetime(
                    ['2026-05-22', '2026-05-23', '2026-05-23']
                ),
                'code': ['000001', '000001', '000002'],
                'price': [10.0, 10.5, 21.0],
            }
        )
        ds = DataStore(pdf.copy())

        target_key = (pd.Timestamp('2026-05-23'), '000001')
        pd_group = pdf.groupby(['date', 'code']).get_group(target_key)
        ds_group = ds.groupby(['date', 'code']).get_group(target_key)

        pd.testing.assert_frame_equal(ds_group, pd_group)

    # ----- dropna=False + NaN as lookup key -----

    # NB: ``get_group(np.nan)`` is intentionally not tested - pandas 2.x
    # raises KeyError on NaN lookup (NaN != NaN in hash table) while
    # pandas 3.x supports it, so any single assertion would diverge between
    # supported pandas versions. NaN-group presence is still covered via
    # ``.groups`` and ``iter`` (both version-stable APIs).

    def test_groups_includes_nan_key_when_dropna_false(self):
        pdf = pd.DataFrame(
            {'cat': ['A', None, 'B', np.nan], 'v': [1, 2, 3, 4]}
        )
        ds = DataStore(pdf.copy())

        pd_groups = pdf.groupby('cat', dropna=False).groups
        ds_groups = ds.groupby('cat', dropna=False).groups

        self.assertEqual(len(ds_groups), len(pd_groups))
        pd_nan_keys = [k for k in pd_groups if pd.isna(k)]
        ds_nan_keys = [k for k in ds_groups if pd.isna(k)]
        self.assertEqual(len(pd_nan_keys), 1)
        self.assertEqual(len(ds_nan_keys), 1)
        self.assertEqual(
            list(ds_groups[ds_nan_keys[0]]),
            list(pd_groups[pd_nan_keys[0]]),
        )

    # NB: ``get_group(('x', np.nan))`` for multi-column dropna=False groupby
    # is intentionally not tested - pandas 2.x raises KeyError on NaN-bearing
    # tuple keys (NaN != NaN in hash lookup) while pandas 3.x supports it,
    # so any single assertion would diverge between supported pandas versions.
    # Single-column NaN-key behaviour is covered by the two tests above.

    # ----- degenerate inputs: empty / single-row / all-same -----

    def test_iter_on_empty_dataframe_yields_no_groups(self):
        pdf = pd.DataFrame(
            {
                'cat': pd.Series([], dtype='object'),
                'v': pd.Series([], dtype='int64'),
            }
        )
        ds = DataStore(pdf.copy())

        self.assertEqual(list(ds.groupby('cat')), list(pdf.groupby('cat')))
        self.assertEqual(len(ds.groupby('cat')), 0)
        self.assertEqual(ds.groupby('cat').ngroups, 0)

    def test_iter_on_single_row_dataframe_yields_one_group(self):
        pdf = pd.DataFrame({'cat': ['A'], 'v': [42]})
        ds = DataStore(pdf.copy())

        pd_items = list(pdf.groupby('cat'))
        ds_items = list(ds.groupby('cat'))

        self.assertEqual(len(ds_items), 1)
        self.assertEqual(ds_items[0][0], 'A')
        pd.testing.assert_frame_equal(ds_items[0][1], pd_items[0][1])

    def test_iter_on_all_same_value_dataframe_yields_one_group(self):
        pdf = pd.DataFrame({'cat': ['A', 'A', 'A'], 'v': [1, 2, 3]})
        ds = DataStore(pdf.copy())

        pd_items = list(pdf.groupby('cat'))
        ds_items = list(ds.groupby('cat'))

        self.assertEqual(len(ds_items), 1)
        self.assertEqual(ds_items[0][0], 'A')
        pd.testing.assert_frame_equal(ds_items[0][1], pd_items[0][1])


class TestSeriesGroupByIteration(unittest.TestCase):
    """``ds.groupby(...)[col]`` returns a ColumnExpr that iterates like pandas
    ``SeriesGroupBy``: yields ``(group_key, sub_series)`` pairs.

    Before the fix, ``for k, s in ds.groupby('cat')['v']:`` would iterate over
    the raw values of the executed column (5 ints), completely ignoring the
    grouping. Now it mirrors pandas SeriesGroupBy exactly.
    """

    def test_series_groupby_yields_key_and_subseries_single_column(self):
        pdf = pd.DataFrame(
            {'cat': ['A', 'B', 'A', 'B', 'C'], 'v': [10, 20, 30, 40, 50]}
        )
        ds = DataStore(pdf.copy())

        pd_items = list(pdf.groupby('cat')['v'])
        ds_items = list(ds.groupby('cat')['v'])

        self.assertEqual(len(ds_items), len(pd_items))
        for (pk, ps), (dk, ds_s) in zip(pd_items, ds_items):
            self.assertEqual(dk, pk)
            self.assertIsInstance(ds_s, pd.Series)
            self.assertEqual(ds_s.tolist(), ps.tolist())
            self.assertEqual(list(ds_s.index), list(ps.index))

    def test_series_groupby_yields_tuple_key_for_multi_column(self):
        pdf = pd.DataFrame(
            {
                'a': ['x', 'x', 'y', 'y'],
                'b': [1, 2, 1, 1],
                'v': [10, 20, 30, 40],
            }
        )
        ds = DataStore(pdf.copy())

        pd_items = list(pdf.groupby(['a', 'b'])['v'])
        ds_items = list(ds.groupby(['a', 'b'])['v'])

        self.assertEqual(len(ds_items), len(pd_items))
        for (pk, ps), (dk, ds_s) in zip(pd_items, ds_items):
            self.assertIsInstance(dk, tuple)
            self.assertEqual(dk, pk)
            self.assertEqual(ds_s.tolist(), ps.tolist())

    def test_plain_column_iteration_still_yields_values(self):
        """Regression: plain ds['col'] iteration must keep yielding scalars."""
        pdf = pd.DataFrame({'v': [10, 20, 30]})
        ds = DataStore(pdf.copy())

        self.assertEqual(list(ds['v']), [10, 20, 30])

    def test_series_groupby_iter_respects_dropna(self):
        pdf = pd.DataFrame(
            {'cat': ['A', None, 'B', np.nan], 'v': [1, 2, 3, 4]}
        )
        ds = DataStore(pdf.copy())

        pd_items = list(pdf.groupby('cat', dropna=False)['v'])
        ds_items = list(ds.groupby('cat', dropna=False)['v'])

        self.assertEqual(len(ds_items), len(pd_items))
        for (pk, ps), (dk, ds_s) in zip(pd_items, ds_items):
            # NaN keys aren't equal even to themselves, so compare via isna
            if pd.isna(pk):
                self.assertTrue(pd.isna(dk))
            else:
                self.assertEqual(dk, pk)
            self.assertEqual(ds_s.tolist(), ps.tolist())

    def test_iter_after_aggregation_yields_scalar_values_not_subseries(self):
        """Aggregated ColumnExprs propagate _groupby_fields downstream (for
        SQL pushdown bookkeeping), but their iter must keep pandas Series
        semantics - yielding scalar values, not (key, sub_series) pairs.
        Regression for the CI failure of ``test_arithmetic_on_groupby_result``.
        """
        pdf = pd.DataFrame(
            {'group': ['A', 'B', 'C'], 'value': [10, 20, 30]}
        )
        ds = DataStore(pdf.copy())

        pd_result = pdf.groupby('group')['value'].sum() * 2 + 5
        ds_result = ds.groupby('group')['value'].sum() * 2 + 5

        self.assertEqual(list(ds_result), list(pd_result))

    def test_iter_after_groupby_transform_yields_scalar_values(self):
        """``gb['col'].transform(...)`` returns an op-mode ColumnExpr that
        manually re-copies ``_expr=Field`` and ``_groupby_fields`` from its
        source. The judgement that selects SeriesGroupBy iteration must look
        at the derived-mode entry fields (``_source``, ``_op_type``,
        ``_agg_func_name``) and skip SeriesGroupBy iteration here.

        Regression for the CI failure of ``test_groupby_transform`` -
        previously this fell into the SeriesGroupBy branch and yielded
        ``[('bar', Series), ('foo', Series)]`` instead of transform values.
        """
        pdf = pd.DataFrame(
            {'A': ['foo', 'foo', 'bar', 'bar'], 'B': [1, 2, 3, 4]}
        )
        ds = DataStore(pdf.copy())

        pd_transform = pdf.groupby('A')['B'].transform('sum')
        ds_transform = ds.groupby('A')['B'].transform('sum')

        self.assertEqual(list(ds_transform), pd_transform.tolist())


class TestGroupByGetItemErrorPaths(unittest.TestCase):
    """``gb[<illegal>]`` raises TypeError with a clear, type-specific message.

    The legacy message ``Expected str or list, got int`` was misleading because
    it surfaced from Python's iter-protocol fallback (``gb[0]`` when ``__iter__``
    was missing), making users believe they had passed the wrong argument to
    ``groupby()``. We now split the error into two branches and verify both.
    """

    def setUp(self):
        self.ds = DataStore(
            pd.DataFrame({'category': ['A', 'B'], 'value': [1, 2]})
        )

    def test_int_indexing_error_mentions_iteration_and_get_group(self):
        gb = self.ds.groupby('category')

        with self.assertRaises(TypeError) as ctx:
            _ = gb[0]  # type: ignore[index]  # intentional wrong type to test error path

        msg = str(ctx.exception)
        self.assertIn('integer indexing', msg)
        self.assertIn('get_group', msg)
        self.assertIn('iterate', msg.lower())

    def test_float_indexing_uses_generic_column_selection_error(self):
        gb = self.ds.groupby('category')

        with self.assertRaises(TypeError) as ctx:
            _ = gb[3.14]  # type: ignore[index]  # intentional wrong type

        msg = str(ctx.exception)
        self.assertIn('str or list of str', msg)
        self.assertIn('float', msg)
        self.assertNotIn('integer indexing', msg)

    def test_none_indexing_uses_generic_column_selection_error(self):
        gb = self.ds.groupby('category')

        with self.assertRaises(TypeError) as ctx:
            _ = gb[None]  # type: ignore[index]  # intentional wrong type

        msg = str(ctx.exception)
        self.assertIn('str or list of str', msg)
        self.assertIn('NoneType', msg)


if __name__ == '__main__':
    unittest.main()
