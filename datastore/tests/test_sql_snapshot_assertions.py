"""
SQL snapshot tests: for each chain shape, assert that the SQL emitted by
the planner+builder contains/excludes specific substrings.

These tests are designed to catch "dispatcher failed silently but the
SQL accidentally still ran" bugs that pure value-comparison tests miss.
For example: if the planner declares ``plan.groupby_agg is not None``
but the SQL does NOT contain ``GROUP BY``, that is a dispatch bug
regardless of whether the result happened to be empty (so no value
mismatch was detected).

Snapshot strategy: don't snapshot the whole SQL string (that would
break on any cosmetic change); only assert specific substrings that
correspond to *structural* properties of the SQL.

Add a new entry whenever a regression bug is fixed - the corresponding
"must contain" / "must not contain" set is the simplest description of
what the bug was.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.query_planner import QueryPlanner
from datastore.sql_executor import SQLExecutionEngine


@pytest.fixture(scope='module')
def parquet_source():
    """A persistent parquet file used by every test."""
    rng = np.random.default_rng(2026)
    n = 1000
    df = pd.DataFrame(
        {
            'cat': rng.choice(['A', 'B', 'C', 'D'], n),
            'sub': rng.choice(['x', 'y', 'z'], n),
            'v': rng.integers(1, 200, n),
            'w': rng.uniform(0, 100, n),
        }
    )
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'src.parquet')
    df.to_parquet(path)
    yield path
    import shutil

    shutil.rmtree(tmp)


def _build_sql(ds_chain) -> str:
    """Run the planner+SQL builder and return the SQL string for the
    first (single) SQL segment."""
    planner = QueryPlanner()
    exec_plan = planner.plan_segments(
        ds_chain._lazy_ops, has_sql_source=True, schema=None
    )
    sql_segments = [s for s in exec_plan.segments if s.is_sql() and s.plan]
    assert sql_segments, 'expected at least one SQL segment'
    seg = sql_segments[0]
    engine = SQLExecutionEngine(ds_chain)
    return engine.build_sql_from_plan(seg.plan, schema={}).sql


# ---------------------------------------------------------------------------
# Shape table: (id, chain_builder, must_contain, must_not_contain)
# Each chain_builder takes a DataStore and returns a chained DataStore.
# ``must_contain`` is a list of substrings every generated SQL must include.
# ``must_not_contain`` is a list of substrings the SQL must NOT include.
# ---------------------------------------------------------------------------

SHAPES = [
    (
        # The original Mark bug: GROUP BY must appear in inner subquery
        # when the chain ends with a post-LIMIT WHERE on the agg column.
        'mark_pattern_layer_n_groupby_kept',
        lambda ds: (
            ds[ds['cat'] == 'A']
            .groupby('sub')
            .agg({'v': 'sum'})
            .sort_values('v', ascending=False)
            .head(5)[lambda r: r['v'] > 0]
        ),
        ['GROUP BY', 'sum("v")', 'LIMIT 5', '__subq'],
        [],
    ),
    (
        'simple_groupby_emits_group_by',
        lambda ds: ds.groupby('cat').agg({'v': 'sum'}),
        ['GROUP BY', 'sum("v")'],
        [],
    ),
    (
        'filter_then_groupby_keeps_where_and_group_by',
        lambda ds: ds[ds['v'] > 50].groupby('cat').agg({'v': 'sum'}),
        ['WHERE', 'GROUP BY', '> 50'],
        [],
    ),
    (
        'no_groupby_pure_filter_does_not_emit_group_by',
        lambda ds: ds[ds['v'] > 50],
        ['WHERE', '> 50'],
        ['GROUP BY'],
    ),
    (
        'no_groupby_pure_sort_does_not_emit_group_by',
        lambda ds: ds.sort_values('v', ascending=False).head(10),
        ['ORDER BY', 'LIMIT 10'],
        ['GROUP BY'],
    ),
    (
        'limit_then_filter_creates_nested_subquery',
        lambda ds: ds.head(50)[lambda r: r['v'] > 30],
        ['LIMIT 50', '__subq', 'WHERE'],
        ['GROUP BY'],
    ),
    (
        'post_limit_filter_on_groupby_result_uses_temp_alias',
        # post-LIMIT WHERE on the agg column ``v`` (which collides with
        # source col ``v``) - SQL must use the ``__agg_v__`` temp alias
        # in the outer WHERE.
        lambda ds: (
            ds[ds['v'] > 100]
            .groupby('cat')
            .agg({'v': 'sum'})
            .sort_values('v', ascending=False)
            .head(5)[lambda r: r['v'] > 1000]
        ),
        ['GROUP BY', '__agg_v__'],
        [],
    ),
    (
        'two_step_filter_keeps_both_where_clauses',
        lambda ds: ds[ds['v'] > 50][ds['cat'] == 'A'],
        ['"v" > 50', '"cat" = '],
        [],
    ),
    (
        'sort_then_head_keeps_order_by',
        lambda ds: ds.sort_values('v').head(5),
        ['ORDER BY "v"', 'LIMIT 5'],
        [],
    ),
    (
        'select_then_filter_only_selected_cols',
        lambda ds: ds[['cat', 'v']][lambda r: r['v'] > 50],
        ['WHERE', '"v" > 50'],
        [],
    ),
    (
        'multi_func_agg_emits_multiple_aggregates',
        lambda ds: ds.groupby('cat').agg({'v': ['sum', 'mean']}),
        ['sum("v")', 'avg("v")', 'GROUP BY'],
        [],
    ),
    (
        'single_column_agg_via_columnexpr',
        lambda ds: ds.groupby('cat')['v'].agg(['count', 'mean']),
        ['count', 'avg', 'GROUP BY'],
        [],
    ),
    (
        'limit_then_groupby_groupby_emitted',
        # head(K) followed by groupby - GROUP BY should still emit;
        # the previous nested SQL builder dropped this case.
        lambda ds: ds.head(200).groupby('cat').agg({'v': 'sum'}),
        ['GROUP BY', 'LIMIT 200', 'sum("v")'],
        [],
    ),
    (
        'filter_head_filter_groupby_layer_n_dispatch',
        # GroupByAgg lands in a non-zero layer; the dispatcher must
        # still emit GROUP BY in that wrapper.
        lambda ds: (
            ds[ds['v'] > 20]
            .head(100)[lambda r: r['w'] > 30]
            .groupby('cat')
            .agg({'v': 'sum'})
        ),
        ['GROUP BY', 'sum("v")', '__subq'],
        [],
    ),
    (
        'filter_groupby_orderby_limit_no_post_filter',
        # baseline: simple agg pipeline without nested wrapper layers
        lambda ds: (
            ds[ds['v'] > 30]
            .groupby('cat')
            .agg({'v': 'sum'})
            .sort_values('v', ascending=False)
            .head(3)
        ),
        ['GROUP BY', 'ORDER BY', 'LIMIT 3'],
        ['__subq'],
    ),
    (
        'pure_select_columns',
        lambda ds: ds[['cat', 'v']],
        ['SELECT', '"cat"', '"v"', 'FROM'],
        ['GROUP BY', 'ORDER BY'],
    ),
    (
        'distinct_pattern_keeps_distinct',
        lambda ds: ds[['cat', 'sub']].drop_duplicates(),
        ['DISTINCT'],
        [],
    ),
]


@pytest.mark.parametrize(
    'shape_id,builder,must_contain,must_not_contain',
    SHAPES,
    ids=[s[0] for s in SHAPES],
)
def test_sql_structural_assertions(
    shape_id, builder, must_contain, must_not_contain, parquet_source
):
    ds = DataStore.from_file(parquet_source)
    ds_chain = builder(ds)
    sql = _build_sql(ds_chain)

    for needle in must_contain:
        assert needle in sql, (
            f'[{shape_id}] expected SQL to contain {needle!r}, '
            f'got SQL:\n{sql}'
        )
    for needle in must_not_contain:
        assert needle not in sql, (
            f'[{shape_id}] expected SQL NOT to contain {needle!r}, '
            f'got SQL:\n{sql}'
        )
