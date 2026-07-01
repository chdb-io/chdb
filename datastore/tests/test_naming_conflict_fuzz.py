"""
Systematic enumeration of naming-conflict scenarios.

Most chdb-ds bugs around aggregation alias handling boil down to a
small set of name-collision shapes:

- the agg's output alias equals a source column name
- a source column is named after an agg function ('count', 'sum', ...)
- a WHERE column equals the agg's output alias
- a sort column equals the agg's output alias
- chained aggregations re-aggregate a column with the same name

This file enumerates each of these shapes and compares to pandas.

Naming the parametrized test ids after the shape keeps failing tests
self-documenting.
"""

import unittest
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore

from tests.test_utils import assert_datastore_equals_pandas


# Each tuple is (shape_id, data_factory, op_chain_callable, post_compare_kwargs)
# - data_factory takes no args and returns a pandas DataFrame
# - op_chain_callable takes a DataFrame-like and returns the final result
# - post_compare_kwargs forwards to assert_datastore_equals_pandas
CONFLICT_PATTERNS: List[Tuple[str, Callable, Callable, dict]] = [
    # ---- agg alias equals source column name -----------------------------
    (
        'agg_alias_equals_source_col_sum',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B', 'A', 'B'] * 25, 'count': list(range(100))}
        ),
        lambda df: df.groupby('cat').agg({'count': 'sum'}),
        dict(check_index=True),
    ),
    (
        'agg_alias_equals_source_col_mean',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B', 'A', 'B'] * 25, 'sum': list(range(100))}
        ),
        lambda df: df.groupby('cat').agg({'sum': 'mean'}),
        dict(check_index=True),
    ),
    (
        'agg_alias_equals_source_col_max',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B'] * 50, 'amount': list(range(100))}
        ),
        lambda df: df.groupby('cat').agg({'amount': 'max'}),
        dict(check_index=True),
    ),

    # ---- source col named like an agg function ----------------------------
    (
        'source_col_named_count_simple_agg',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B'] * 50, 'count': list(range(100))}
        ),
        lambda df: df.groupby('cat').count(),
        dict(check_index=True),
    ),
    (
        'source_col_named_sum_simple_agg',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B'] * 50, 'sum': list(range(100))}
        ),
        lambda df: df.groupby('cat').sum(),
        dict(check_index=True),
    ),

    # ---- WHERE on source col then agg with same alias --------------------
    (
        'where_then_agg_alias_collision_sum',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B'] * 100, 'amount': list(range(200))}
        ),
        lambda df: df[df['amount'] > 50].groupby('cat').agg({'amount': 'sum'}),
        dict(check_index=True),
    ),
    (
        'where_then_agg_alias_collision_count',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B', 'C'] * 50, 'val': list(range(150))}
        ),
        lambda df: df[df['val'] > 30].groupby('cat').agg({'val': 'count'}),
        dict(check_index=True),
    ),

    # ---- multiple agg funcs on same col (multi-index) -------------------
    (
        'multi_agg_same_col',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B'] * 50, 'value': list(range(100))}
        ),
        lambda df: df.groupby('cat').agg({'value': ['mean', 'max']}),
        dict(check_index=True),
    ),

    # ---- sort by aggregated column with conflict-named alias -----------
    (
        'sort_by_agg_alias_equal_source_col',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B', 'C'] * 50, 'amount': list(range(150))}
        ),
        lambda df: (
            df[df['amount'] > 20]
            .groupby('cat')
            .agg({'amount': 'sum'})
            .sort_values('amount', ascending=False)
        ),
        dict(check_index=True),
    ),

    # ---- post-LIMIT WHERE on alias-conflicted col --------------------
    (
        'post_limit_filter_on_conflict_col',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B', 'C', 'D'] * 50, 'value': list(range(200))}
        ),
        lambda df: (
            df[df['value'] > 50]
            .groupby('cat')
            .agg({'value': 'sum'})
            .sort_values('value', ascending=False)
            .head(5)
        ),
        dict(check_index=True),
    ),

    # ---- ColumnExpr.agg(['count']) flat naming ------------------------
    (
        'columnexpr_single_col_agg_count',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B', 'C'] * 30, 'rating': list(range(90))}
        ),
        lambda df: df.groupby('cat')['rating'].agg(['count']),
        dict(check_index=True),
    ),
    (
        'columnexpr_single_col_agg_mean_count',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B', 'C'] * 30, 'rating': list(range(90))}
        ),
        lambda df: df.groupby('cat')['rating'].agg(['mean', 'count']),
        dict(check_index=True),
    ),

    # ---- WHERE on cat with agg on cat as the only output ---------------
    (
        'where_on_groupkey_then_agg',
        lambda: pd.DataFrame(
            {'cat': list('ABCD') * 25, 'v': list(range(100))}
        ),
        lambda df: df[df['cat'] != 'D'].groupby('cat').agg({'v': 'sum'}),
        dict(check_index=True),
    ),

    # ---- agg col name happens to be a SQL keyword-like word ------------
    (
        'agg_col_named_order',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B'] * 50, 'order': list(range(100))}
        ),
        lambda df: df.groupby('cat').agg({'order': 'sum'}),
        dict(check_index=True),
    ),
    (
        'agg_col_named_limit',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B'] * 50, 'limit': list(range(100))}
        ),
        lambda df: df.groupby('cat').agg({'limit': 'max'}),
        dict(check_index=True),
    ),

    # ---- chained groupby on result with same column name --------------
    (
        'agg_then_groupby_again_same_col',
        lambda: pd.DataFrame(
            {
                'cat': list('ABCD') * 25,
                'sub': list('xy') * 50,
                'value': list(range(100)),
            }
        ),
        lambda df: (
            df.groupby(['cat', 'sub'])
            .agg({'value': 'sum'})
            .reset_index()
            .groupby('cat')
            .agg({'value': 'max'})
        ),
        dict(check_index=True),
    ),

    # ---- source col name matches the aggregate alias used internally ---
    (
        'source_col_named_agg_v',
        lambda: pd.DataFrame(
            {'cat': ['A', 'B'] * 50, '__agg_v__': list(range(100))}
        ),
        lambda df: df.groupby('cat').agg({'__agg_v__': 'sum'}),
        dict(check_index=True),
    ),
]


@pytest.mark.parametrize(
    'shape_id,data_factory,op_chain,compare_kwargs',
    CONFLICT_PATTERNS,
    ids=[p[0] for p in CONFLICT_PATTERNS],
)
def test_naming_conflict_matches_pandas(
    shape_id, data_factory, op_chain, compare_kwargs
):
    """Each shape mirrors pandas; mismatch indicates an alias-handling bug."""
    pd_df = data_factory()
    pd_result = op_chain(pd_df)

    ds = DataStore(pd_df)
    ds_result = op_chain(ds)

    # Aggregations without explicit sort have undefined row order.
    compare_kwargs = dict(compare_kwargs)
    compare_kwargs.setdefault('check_row_order', False)

    assert_datastore_equals_pandas(ds_result, pd_result, **compare_kwargs)
