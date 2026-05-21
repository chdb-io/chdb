"""
Real user journey: treating an aggregation result as a DataFrame again.

Pandas users naturally think of ``groupby().agg(...)`` as producing a
DataFrame that they can immediately keep operating on - filter it, sort
it, project a column out, aggregate it again, merge it, etc. This is the
"result-as-input" mental model.

Most of chdb-ds's pre-Mark test suite tested aggregations as the END of
a chain. This file tests the OTHER half of the mental model: every
aggregation is a STARTING POINT for further work.

Each test mirrors the same code on pandas and chdb-ds and demands
identical output. The cross product is 5 aggregations x 5 follow-ons =
25 user paths covered.
"""

import unittest
from itertools import product

import numpy as np
import pandas as pd

from datastore import DataStore

from tests.test_utils import assert_datastore_equals_pandas


def _make_dataset():
    np.random.seed(7)
    n = 2000
    return pd.DataFrame(
        {
            'category': np.random.choice(['A', 'B', 'C', 'D'], n),
            'subcat': np.random.choice(['x', 'y', 'z'], n),
            'value': np.random.randint(0, 1000, n),
            'score': np.random.uniform(0, 100, n),
        }
    )


# Aggregations a user reaches for first: each returns a DataStore/DataFrame
# with ``category`` as the index. Lambda takes the source, returns the
# aggregated result.
AGG_BUILDERS = [
    ('sum_on_value',  lambda df: df.groupby('category').agg({'value': 'sum'})),
    ('mean_on_score', lambda df: df.groupby('category').agg({'score': 'mean'})),
    ('count_on_value', lambda df: df.groupby('category').agg({'value': 'count'})),
    ('min_on_score',  lambda df: df.groupby('category').agg({'score': 'min'})),
    ('max_on_value',  lambda df: df.groupby('category').agg({'value': 'max'})),
]

# Follow-ons a user does AFTER seeing the aggregate. Each lambda takes
# the result and the column name produced by the aggregation.
FOLLOWUPS = [
    (
        'filter_by_mask',
        lambda r, c: r[r[c] > r[c].median()],
    ),
    (
        'sort_descending',
        lambda r, c: r.sort_values(c, ascending=False),
    ),
    (
        'head_top_2',
        lambda r, c: r.sort_values(c, ascending=False).head(2),
    ),
    (
        'select_only_agg_col',
        lambda r, c: r[[c]],
    ),
    (
        'filter_then_head',
        # Use the median rather than a literal threshold like ``> 0``.
        # Reason: when the aggregation output column name matches a
        # source column name (e.g. ``count(value) AS value``), a literal
        # threshold can accidentally resolve against the source column's
        # value range and silently mask a name-shadowing bug. Using the
        # result's own median guarantees we filter against the aggregate
        # output. See ``test_agg_output_name_shadows_source_column`` for
        # the explicit regression covering that bug.
        lambda r, c: r[r[c] > r[c].median()].head(2),
    ),
]


def _agg_col(agg_name: str) -> str:
    """Return the column name the aggregation produces."""
    # All current builders aggregate a single column with a single func,
    # so the output column name = the source column name.
    return agg_name.split('_on_')[1]


class TestAggThenUseResult(unittest.TestCase):
    """``result = df.groupby().agg(...); use(result)`` for many shapes."""

    @classmethod
    def setUpClass(cls):
        cls.pd_df = _make_dataset()

    def _run_case(self, agg_name, agg_fn, follow_name, follow_fn):
        col = _agg_col(agg_name)
        pd_result = follow_fn(agg_fn(self.pd_df), col)
        ds = DataStore(self.pd_df)
        ds_result = follow_fn(agg_fn(ds), col)
        # When the follow-on returns a Series-shaped result we still want
        # value-by-value equality; use the helper which delegates.
        assert_datastore_equals_pandas(
            ds_result,
            pd_result,
            check_index=True,
            check_row_order=isinstance(pd_result, pd.DataFrame) and pd_result.index.is_monotonic_increasing
            or isinstance(pd_result, pd.Series),
        )


# Dynamically attach a test method per (agg, follow) pair so failures
# point at the exact combination. We deliberately do NOT bind the helper
# or the loop variables at module scope (avoids accidental pytest
# collection of the leftover names as tests).
def _attach_combinations():
    def _make_test(agg_name, agg_fn, follow_name, follow_fn):
        def _runner(self):
            self._run_case(agg_name, agg_fn, follow_name, follow_fn)

        _runner.__name__ = f'test_{agg_name}__then__{follow_name}'
        _runner.__doc__ = (
            f'After {agg_name}, apply {follow_name} and verify match with pandas.'
        )
        return _runner

    for (agg_name, agg_fn), (follow_name, follow_fn) in product(
        AGG_BUILDERS, FOLLOWUPS
    ):
        method = _make_test(agg_name, agg_fn, follow_name, follow_fn)
        setattr(TestAggThenUseResult, method.__name__, method)


_attach_combinations()


# ---------------------------------------------------------------------------
# Permanent regression for the agg-output-shadows-source-column bug found
# via this journey file. ``filter_then_head`` originally used ``r[r[c] > 0]``,
# which on ``count(value)`` against a source column also called ``value``
# (range 0..999) accidentally exercised the SOURCE column instead of the
# AGG output column - dropping the ~2 source rows that had value == 0
# and causing every group's count to drop by exactly 1. Fixed by routing
# DataFrame-source single-layer plans with ``LazyGroupByAgg`` through
# ``_build_layered_sql`` (which splits WHEREs by GROUP BY position and
# emits HAVING for post-agg conditions, rewriting Field refs to the
# agg's emitted ``__agg_*__`` temp aliases).
# ---------------------------------------------------------------------------


class TestAggOutputShadowsSourceColumn(unittest.TestCase):
    """``result = df.groupby('cat').agg({'col': 'count'})`` followed by
    ``result[result['col'] > 0]`` correctly binds the WHERE to the
    aggregated output, not the source column of the same name.
    """

    def test_agg_output_name_shadows_source_column(self):
        np.random.seed(7)
        n = 2000
        df = pd.DataFrame(
            {
                'category': np.random.choice(['A', 'B', 'C', 'D'], n),
                # ``value`` has a few literal zeros - the smoking gun.
                'value': np.random.randint(0, 1000, n),
            }
        )

        pd_agg = df.groupby('category').agg({'value': 'count'})
        pd_filtered = pd_agg[pd_agg['value'] > 0]

        ds = DataStore(df)
        ds_agg = ds.groupby('category').agg({'value': 'count'})
        ds_filtered = ds_agg[ds_agg['value'] > 0]

        # When the bug is fixed both sides return the same row counts.
        assert_datastore_equals_pandas(
            ds_filtered, pd_filtered, check_index=True
        )


if __name__ == '__main__':
    unittest.main()
