"""
Cartesian product test: every realistic ``result_builder`` x every
realistic ``follow_up`` combination is mechanically generated and
compared to pandas.

Each individual chain is short, but together they cover the
"result-as-input" combinatorial space that the per-feature unit tests
do not exercise.

Failure modes typically caught here:

- dispatcher silently drops an op in a nested subquery (the Mark bug
  class)
- alias rename does not propagate across a wrapper layer
- result of a builder is a Series/DataFrame mismatch
- post-aggregation operations bind to source columns instead of
  aggregation outputs

A failure is named ``test_<builder>__then__<follow>`` so the offending
combination is obvious from the test id.
"""

import unittest
from itertools import product

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore

from tests.test_utils import assert_datastore_equals_pandas


@pytest.fixture(scope='module')
def fixture_df():
    """A small but typed dataset shared across all parametrized cases."""
    rng = np.random.default_rng(2026)
    n = 500
    return pd.DataFrame(
        {
            'cat': rng.choice(['A', 'B', 'C', 'D'], n),
            'sub': rng.choice(['x', 'y', 'z'], n),
            'v': rng.integers(1, 200, n),
            'w': rng.uniform(0, 100, n),
        }
    )


# ---------------------------------------------------------------------------
# Result builders: each takes a DataFrame-like (works on both pandas and
# DataStore) and returns a result that the user would treat as a new
# DataFrame. Each tuple is ``(name, builder_fn, output_col, output_is_index)``
# where ``output_col`` is the most-relevant column the follow-up can act on.
# ---------------------------------------------------------------------------

RESULT_BUILDERS = [
    (
        'groupby_agg_sum',
        lambda df: df.groupby('cat').agg({'v': 'sum'}),
        'v',
    ),
    (
        'groupby_agg_mean',
        lambda df: df.groupby('cat').agg({'w': 'mean'}),
        'w',
    ),
    (
        'groupby_agg_max',
        lambda df: df.groupby('cat').agg({'v': 'max'}),
        'v',
    ),
    (
        'groupby_agg_sum_then_head',
        lambda df: df.groupby('cat').agg({'v': 'sum'}).sort_values('v').head(3),
        'v',
    ),
    (
        'filter_then_head',
        lambda df: df[df['v'] > 50].head(50),
        'v',
    ),
    (
        # ``w`` is uniform float, near-unique so tie-breaking does not
        # leak between pandas and DataStore. Don't use ``v`` (integer)
        # because tied ``v`` values surface a known stable-sort semantic
        # difference that is out of scope for this Cartesian.
        'sort_head',
        lambda df: df.sort_values('w', ascending=False).head(20),
        'w',
    ),
    (
        'head_then_filter',
        lambda df: df.head(100)[lambda x: x['v'] > 30],
        'v',
    ),
    (
        'filter_select',
        lambda df: df[df['v'] > 50][['cat', 'v']],
        'v',
    ),
]


# ---------------------------------------------------------------------------
# Follow-ups: each takes the builder's result and the relevant output
# column name and produces a final result for comparison.
# ---------------------------------------------------------------------------

FOLLOW_UPS = [
    (
        'bool_mask_above_median',
        lambda r, c: r[r[c] > r[c].median()],
    ),
    (
        'loc_above_median',
        lambda r, c: r.loc[r[c] > r[c].median()],
    ),
    (
        'sort_descending_by_col',
        lambda r, c: r.sort_values(c, ascending=False),
    ),
    (
        'head_two',
        lambda r, c: r.head(2),
    ),
    (
        'select_only_col',
        lambda r, c: r[[c]] if c in r.columns else r,
    ),
    (
        'filter_then_head',
        lambda r, c: r[r[c] > r[c].median()].head(2),
    ),
]


# Some combinations are known to need additional work or have semantic
# differences. Tag them as xfail / skip at collection time so the suite
# stays green where it is supposed to.
KNOWN_XFAIL = set()  # populated below as we discover them


def _ids():
    """Generate human-readable parameter ids."""
    ids = []
    for b in RESULT_BUILDERS:
        for f in FOLLOW_UPS:
            ids.append(f'{b[0]}__then__{f[0]}')
    return ids


_PARAMS = [
    (b[0], b[1], b[2], f[0], f[1])
    for b in RESULT_BUILDERS
    for f in FOLLOW_UPS
]


@pytest.mark.parametrize(
    'builder_name,builder_fn,output_col,follow_name,follow_fn',
    _PARAMS,
    ids=_ids(),
)
def test_builder_followup_matches_pandas(
    builder_name, builder_fn, output_col, follow_name, follow_fn, fixture_df
):
    """Mirror the same code on pandas and DataStore; compare results."""
    pd_result = follow_fn(builder_fn(fixture_df), output_col)

    ds = DataStore(fixture_df)
    ds_result = follow_fn(builder_fn(ds), output_col)

    # Aggregation-style outputs often have set indexes; comparing rows by
    # value+index is sufficient. Sort within column for stable comparison
    # when the order isn't guaranteed by an explicit ORDER BY in the chain.
    sorted_chain = (
        'sort' in builder_name or 'head' in builder_name
        or 'sort' in follow_name or 'head' in follow_name
    )
    assert_datastore_equals_pandas(
        ds_result,
        pd_result,
        check_index=True,
        check_row_order=sorted_chain,
    )
