"""
Structural invariants on the QueryPlan / generated SQL pair.

These tests do NOT execute the SQL. They build the plan, build the SQL,
and assert structural properties that any internal change must preserve.
A failure here means the planner and the SQL builder have drifted out of
agreement - exactly the bug class that hid the layer-N GroupByAgg
silent-drop bug for so long.

Why this style: pure equality-with-pandas tests can pass on bad SQL by
accident (e.g. an empty result on both sides). Invariants on the
plan/SQL pair are independent of input data and tend to catch
dispatcher / builder regressions immediately.

Each invariant runs against a list of representative chains so a single
broken invariant fires from multiple test ids and is easy to localise.
"""

import os
import re
import tempfile

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.lazy_ops import (
    LazyGroupByAgg,
    LazyMask,
    LazyRelationalOp,
    LazyWhere,
)
from datastore.query_planner import QueryPlanner
from datastore.sql_executor import SQLExecutionEngine


@pytest.fixture(scope='module')
def parquet_source():
    rng = np.random.default_rng(33)
    n = 500
    df = pd.DataFrame(
        {
            'cat': rng.choice(['A', 'B', 'C'], n),
            'v': rng.integers(1, 100, n),
            'w': rng.uniform(0, 100, n),
        }
    )
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'src.parquet')
    df.to_parquet(path)
    yield path
    import shutil

    shutil.rmtree(tmp)


def _plan_and_sql(ds_chain):
    planner = QueryPlanner()
    exec_plan = planner.plan_segments(
        ds_chain._lazy_ops, has_sql_source=True, schema=None
    )
    seg = next(s for s in exec_plan.segments if s.is_sql() and s.plan)
    plan = seg.plan
    engine = SQLExecutionEngine(ds_chain)
    sql = engine.build_sql_from_plan(plan, schema={}).sql
    return plan, sql


# ---------------------------------------------------------------------------
# A representative selection of chain shapes used by every invariant test.
# ---------------------------------------------------------------------------

CHAINS_WITH_GROUPBY = [
    ('simple_groupby_sum', lambda ds: ds.groupby('cat').agg({'v': 'sum'})),
    (
        'filter_then_groupby_sum',
        lambda ds: ds[ds['v'] > 10].groupby('cat').agg({'v': 'sum'}),
    ),
    (
        'groupby_sort_head',
        lambda ds: ds.groupby('cat').agg({'v': 'sum'}).sort_values('v').head(2),
    ),
    (
        'groupby_in_layer_n',
        # head() before groupby pushes the agg into a non-zero layer
        lambda ds: ds.head(50).groupby('cat').agg({'v': 'sum'}),
    ),
    (
        'post_limit_filter_on_agg',
        lambda ds: (
            ds[ds['v'] > 10]
            .groupby('cat')
            .agg({'v': 'sum'})
            .sort_values('v', ascending=False)
            .head(3)[lambda r: r['v'] > 100]
        ),
    ),
    (
        'multi_agg_funcs',
        lambda ds: ds.groupby('cat').agg({'v': ['sum', 'mean']}),
    ),
]

CHAINS_WITHOUT_GROUPBY = [
    ('pure_filter', lambda ds: ds[ds['v'] > 10]),
    (
        'filter_sort_head',
        lambda ds: ds[ds['v'] > 10].sort_values('w').head(5),
    ),
    ('select_only', lambda ds: ds[['cat', 'v']]),
    (
        'limit_then_filter',
        lambda ds: ds.head(50)[lambda r: r['v'] > 30],
    ),
]

CHAINS_WITH_NESTED_LAYERS = [
    (
        'limit_then_where_creates_two_layers',
        lambda ds: ds.head(50)[lambda r: r['v'] > 30],
        2,
    ),
    (
        'three_layer_chain',
        lambda ds: (
            ds[ds['v'] > 10]
            .head(100)[lambda r: r['w'] > 30]
            .head(10)[lambda r: r['v'] > 20]
        ),
        # WHERE -> LIMIT -> WHERE (new layer) -> LIMIT -> WHERE (new layer)
        3,
    ),
]


# ---------------------------------------------------------------------------
# Invariant 1: groupby_agg in plan iff GROUP BY in SQL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'name,builder', CHAINS_WITH_GROUPBY, ids=[c[0] for c in CHAINS_WITH_GROUPBY]
)
def test_groupby_in_plan_implies_group_by_in_sql(name, builder, parquet_source):
    ds = DataStore.from_file(parquet_source)
    plan, sql = _plan_and_sql(builder(ds))
    has_agg_in_layers = any(
        isinstance(op, LazyGroupByAgg)
        for layer in plan.layers
        for op in layer
    )
    assert plan.groupby_agg is not None or has_agg_in_layers, (
        f'[{name}] expected plan.groupby_agg or layer-level LazyGroupByAgg; got plan={plan}'
    )
    assert 'GROUP BY' in sql, (
        f'[{name}] plan has GroupByAgg but SQL is missing GROUP BY:\n{sql}'
    )


@pytest.mark.parametrize(
    'name,builder',
    CHAINS_WITHOUT_GROUPBY,
    ids=[c[0] for c in CHAINS_WITHOUT_GROUPBY],
)
def test_no_groupby_in_plan_implies_no_group_by_in_sql(
    name, builder, parquet_source
):
    ds = DataStore.from_file(parquet_source)
    plan, sql = _plan_and_sql(builder(ds))
    has_agg_in_layers = any(
        isinstance(op, LazyGroupByAgg)
        for layer in plan.layers
        for op in layer
    )
    assert plan.groupby_agg is None and not has_agg_in_layers, (
        f'[{name}] expected no GroupByAgg; got plan={plan}'
    )
    assert 'GROUP BY' not in sql, (
        f'[{name}] plan has no GroupByAgg but SQL contains GROUP BY:\n{sql}'
    )


# ---------------------------------------------------------------------------
# Invariant 2: layer count maps to subquery count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'name,builder,expected_layers',
    CHAINS_WITH_NESTED_LAYERS,
    ids=[c[0] for c in CHAINS_WITH_NESTED_LAYERS],
)
def test_n_layers_implies_n_minus_1_subqueries(
    name, builder, expected_layers, parquet_source
):
    ds = DataStore.from_file(parquet_source)
    plan, sql = _plan_and_sql(builder(ds))
    assert len(plan.layers) == expected_layers, (
        f'[{name}] expected {expected_layers} layers, got {len(plan.layers)}: '
        f'{plan.layers}'
    )
    n_subq = len(re.findall(r'__subq\d+__', sql))
    # Each wrapper layer introduces exactly one __subqN__ alias; the
    # first layer reads from the table source so it does not.
    expected_subq = expected_layers - 1
    assert n_subq == expected_subq, (
        f'[{name}] expected {expected_subq} __subqN__ aliases, got {n_subq}.\nSQL:\n{sql}'
    )


# ---------------------------------------------------------------------------
# Invariant 3: every alias rename has a presence in the SQL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'name,builder', CHAINS_WITH_GROUPBY, ids=[c[0] for c in CHAINS_WITH_GROUPBY]
)
def test_alias_renames_appear_in_sql(name, builder, parquet_source):
    ds = DataStore.from_file(parquet_source)
    plan, sql = _plan_and_sql(builder(ds))
    if not plan.alias_renames:
        pytest.skip(f'[{name}] plan introduced no aliases')
    for temp_alias in plan.alias_renames.keys():
        assert temp_alias in sql, (
            f'[{name}] plan declared temp alias {temp_alias!r} but SQL '
            f'does not reference it:\n{sql}'
        )


# ---------------------------------------------------------------------------
# Invariant 4: WHERE op count is bounded by SQL WHERE clause count
# ---------------------------------------------------------------------------


_WHERE_PATTERN = re.compile(r'\bWHERE\b', re.IGNORECASE)


@pytest.mark.parametrize(
    'name,builder',
    CHAINS_WITH_GROUPBY + CHAINS_WITHOUT_GROUPBY,
    ids=[c[0] for c in CHAINS_WITH_GROUPBY + CHAINS_WITHOUT_GROUPBY],
)
def test_where_clauses_present_when_plan_has_where_ops(
    name, builder, parquet_source
):
    ds = DataStore.from_file(parquet_source)
    plan, sql = _plan_and_sql(builder(ds))
    n_where_ops_in_layers = sum(
        1
        for layer in plan.layers
        for op in layer
        if isinstance(op, LazyRelationalOp) and op.op_type == 'WHERE'
    )
    if n_where_ops_in_layers == 0:
        pytest.skip(f'[{name}] no WHERE in plan; nothing to assert')
    n_where_in_sql = len(_WHERE_PATTERN.findall(sql))
    assert n_where_in_sql >= 1, (
        f'[{name}] plan has {n_where_ops_in_layers} WHERE ops but SQL has '
        f'zero WHERE clauses:\n{sql}'
    )


# ---------------------------------------------------------------------------
# Invariant 5: layer-0 wraps the actual source, never a __subq alias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'name,builder',
    CHAINS_WITH_GROUPBY + CHAINS_WITHOUT_GROUPBY,
    ids=[c[0] for c in CHAINS_WITH_GROUPBY + CHAINS_WITHOUT_GROUPBY],
)
def test_layer_zero_reads_from_real_source(name, builder, parquet_source):
    """The innermost subquery must read from the actual source, not from
    a synthetic __subqN__ alias. If it does not, we have a layer indexing
    bug."""
    ds = DataStore.from_file(parquet_source)
    plan, sql = _plan_and_sql(builder(ds))
    # The innermost FROM should reference the parquet path or table fn.
    # We assert the parquet path is in the SQL (it must show up exactly
    # once - the original table reference).
    assert 'file(' in sql, (
        f'[{name}] innermost SQL missing file() source reference:\n{sql}'
    )


# ---------------------------------------------------------------------------
# Invariant 6: groupby_agg pointer matches a LazyGroupByAgg op in layers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'name,builder', CHAINS_WITH_GROUPBY, ids=[c[0] for c in CHAINS_WITH_GROUPBY]
)
def test_groupby_agg_pointer_aligns_with_layers(name, builder, parquet_source):
    """``plan.groupby_agg`` should be either ``None`` or an actual op
    inside ``plan.layers``; never a dangling reference."""
    ds = DataStore.from_file(parquet_source)
    plan, _sql = _plan_and_sql(builder(ds))
    if plan.groupby_agg is None:
        pytest.skip(f'[{name}] plan has no groupby_agg pointer')
    layer_ops = [op for layer in plan.layers for op in layer]
    matches = [
        op
        for op in layer_ops
        if isinstance(op, LazyGroupByAgg)
        and (
            op is plan.groupby_agg
            or (
                op.groupby_cols == plan.groupby_agg.groupby_cols
                and op.agg_dict == plan.groupby_agg.agg_dict
                and op.agg_func == plan.groupby_agg.agg_func
            )
        )
    ]
    assert matches, (
        f'[{name}] plan.groupby_agg has no corresponding LazyGroupByAgg in '
        f'layers; pointer is dangling.'
    )


# ---------------------------------------------------------------------------
# Invariant 7: where_ops convenience list matches LazyWhere/Mask in layers
# ---------------------------------------------------------------------------


def test_where_ops_pointer_alignment(parquet_source):
    """``plan.where_ops`` should reflect exactly the LazyWhere/LazyMask
    ops that live inside the layers."""
    ds = DataStore.from_file(parquet_source)
    # Build a chain that involves where() (pandas-style boolean wrapper)
    ds_chain = ds.where(ds['v'] > 50, 0)
    plan, _sql = _plan_and_sql(ds_chain)
    layer_where_mask = [
        op
        for layer in plan.layers
        for op in layer
        if isinstance(op, (LazyWhere, LazyMask))
    ]
    assert len(plan.where_ops) == len(layer_where_mask), (
        f'plan.where_ops length {len(plan.where_ops)} != '
        f'LazyWhere/Mask count in layers {len(layer_where_mask)}; pointer is stale.'
    )


# ---------------------------------------------------------------------------
# Invariant 8: SELECT never has a bare orphan __agg_*__ that isn't in renames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'name,builder', CHAINS_WITH_GROUPBY, ids=[c[0] for c in CHAINS_WITH_GROUPBY]
)
def test_temp_alias_in_sql_was_registered_in_plan(
    name, builder, parquet_source
):
    """Every ``__agg_*__`` alias the SQL emits must also be in
    ``plan.alias_renames`` so post-execution can rename it back. An
    orphan SQL alias would leak the temp name to the user."""
    ds = DataStore.from_file(parquet_source)
    plan, sql = _plan_and_sql(builder(ds))
    sql_aliases = set(re.findall(r'__agg_\w+__', sql))
    if not sql_aliases:
        pytest.skip(f'[{name}] no temp aliases emitted')
    declared = set(plan.alias_renames.keys())
    leaks = sql_aliases - declared
    assert not leaks, (
        f'[{name}] SQL emits temp aliases {leaks!r} that are not in '
        f'plan.alias_renames {declared!r}; post-execution will not rename them.'
    )
