"""
Property-based exploration on file sources: random op-chains over a
Parquet and a CSV file must match pandas on the same data, including
the row labels pandas gives the rows it reads from a file.

Uses the chain DSL of ``test_property_based_chains`` (filter / sort /
head / select / agg), which runs on DataFrame sources. Labels are
compared when the chain defines the row order and doesn't aggregate.
"""

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings

from datastore import DataStore

from tests.test_property_based_chains import (
    _DF,
    _chain_uses_dropped_col,
    apply_chain,
    op_chain,
)
from tests.test_utils import assert_datastore_equals_pandas


@pytest.fixture(scope="module", params=["parquet", "csv"])
def file_path(request, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("chains") / f"chains.{request.param}")
    if request.param == "parquet":
        _DF.to_parquet(path)
    else:
        _DF.to_csv(path, index=False)
    return path


@given(chain=op_chain(min_steps=3, max_steps=6))
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)
def test_random_file_chain_matches_pandas(file_path, chain):
    if _chain_uses_dropped_col(chain):
        return
    try:
        pd_result = apply_chain(_DF, chain)
    except (KeyError, ValueError, TypeError):
        return
    try:
        ds_result = apply_chain(DataStore.from_file(file_path), chain)
    except (KeyError, ValueError, TypeError):
        return

    # Empty results after an aggregation diverge from pandas, as in
    # test_property_based_chains (journeys/test_property_chain_followups.py
    # ::TestEmptyGroupbyColumnDrift).
    if len(pd_result) == 0:
        return

    last_two = [s['op'] for s in chain[-2:]]
    deterministic_tail = last_two[-1] in {'head', 'sort'} and (
        last_two[-1] != 'head' or (len(last_two) >= 2 and last_two[-2] == 'sort')
    )
    assert_datastore_equals_pandas(
        ds_result,
        pd_result,
        check_row_order=deterministic_tail,
        check_nullable_dtype=False,
        msg=str(chain),
    )
    if deterministic_tail and not any(s['op'] == 'agg' for s in chain):
        pd.testing.assert_index_equal(
            ds_result.index, pd_result.index, exact=False, obj=str(chain)
        )
