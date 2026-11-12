"""
join() and merge() read every join type the same way on the SQL JOIN path
and the pandas merge path (JoinType.parse).

Mirror tests against pandas merge on DataFrame, Parquet and CSV sources,
and on mixed sources, where a SQL-backed join runs as a pandas merge.
"""

import pandas as pd
import pytest

from datastore import DataStore, JoinType
from datastore.exceptions import JoinTypeError, QueryError
from datastore.lazy_ops import LazyJoin
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_datastore_equals_pandas_chdb_compat,
)


LEFT = pd.DataFrame({"id": [1, 2, 3, 4], "a": [10, 20, 30, 40]})
RIGHT = pd.DataFrame({"id": [4, 2, 5], "b": [400, 200, 500]})

# Spelling -> pandas merge how
HOWS = {
    "inner": "inner",
    "left": "left",
    "left outer": "left",
    "LEFT_OUTER": "left",
    "right": "right",
    "right outer": "right",
    "outer": "outer",
    "full": "outer",
    "FULL": "outer",
    "full outer": "outer",
    "full_outer": "outer",
    "cross": "cross",
    "left_anti": "left_anti",
    "left anti": "left_anti",
    "right_anti": "right_anti",
    JoinType.left: "left",
    JoinType.full_outer: "outer",
}

# Both sides from files join in SQL; with a DataFrame side the join is a
# pandas merge
SOURCES = ["dataframe", "parquet", "csv", "parquet+dataframe", "dataframe+parquet"]


@pytest.fixture
def make_sides(tmp_path):
    def write(df, name, fmt):
        path = str(tmp_path / f"{name}.{fmt}")
        if fmt == "parquet":
            df.to_parquet(path)
        else:
            df.to_csv(path, index=False)
        return path

    def side(df, name, kind):
        if kind == "dataframe":
            return DataStore(df.copy())
        return DataStore.from_file(write(df, name, kind))

    def make(source):
        left_kind, _, right_kind = source.partition("+")
        return side(LEFT, "left", left_kind), side(RIGHT, "right", right_kind or left_kind)

    return make


def _keys(pandas_how):
    return {} if pandas_how == "cross" else {"on": "id"}


def _runs_in_sql(source, pandas_how):
    return source in ("parquet", "csv") and pandas_how not in ("cross", "left_anti", "right_anti")


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("how", list(HOWS), ids=[str(h) for h in HOWS])
@pytest.mark.parametrize("api", ["join", "merge"])
def test_join_type_matches_pandas_merge(request, make_sides, source, how, api):
    pandas_how = HOWS[how]
    pd_result = LEFT.merge(RIGHT, how=pandas_how, **_keys(pandas_how))

    left, right = make_sides(source)
    ds_result = getattr(left, api)(right, how=how, **_keys(pandas_how))

    in_sql = _runs_in_sql(source, pandas_how)
    if in_sql:
        assert [join_type.pandas_how for _, join_type, _ in ds_result._joins] == [pandas_how]
    else:
        assert not ds_result._joins
        joins = [op for op in ds_result._lazy_ops if isinstance(op, LazyJoin)]
        # merge() with a DataFrame side, or without keys, merges right away
        if api == "join":
            assert [op.how for op in joins] == [pandas_how]
    if in_sql and pandas_how in ("right", "outer"):
        request.applymarker(
            pytest.mark.xfail(
                reason="SQL RIGHT/FULL JOIN rows are not in pandas merge order",
                strict=False,
            )
        )
    if in_sql:
        # A SQL outer join reads unmatched columns as nullable (Int64 with
        # NA), pandas as float64 with NaN
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)
        assert list(ds_result.index) == list(pd_result.index)
    else:
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("how", ["full", "outer", "LEFT_OUTER"])
def test_join_type_after_lazy_ops_on_right(make_sides, source, how):
    """A right side whose pipeline runs in pandas joins as a pandas merge."""
    pandas_how = HOWS[how]
    pd_right = RIGHT.assign(b=RIGHT["b"].apply(lambda v: v + 1))
    pd_result = LEFT.merge(pd_right, how=pandas_how, on="id")

    left, right = make_sides(source)
    right = right.assign(b=right["b"].apply(lambda v: v + 1))
    ds_result = left.join(right, how=how, on="id")

    assert not ds_result._joins
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("how", ["left semi", "right_semi", "bogus", "full inner"])
@pytest.mark.parametrize("api", ["join", "merge"])
def test_unsupported_join_type_raises(make_sides, source, how, api):
    with pytest.raises(ValueError):
        LEFT.merge(RIGHT, how=how, on="id")

    left, right = make_sides(source)
    with pytest.raises(JoinTypeError, match="join type"):
        getattr(left, api)(right, how=how, on="id")


@pytest.mark.parametrize("source", SOURCES)
def test_cross_join_takes_no_keys(make_sides, source):
    left, right = make_sides(source)
    with pytest.raises(QueryError, match="cross join takes no join keys"):
        left.join(right, how="cross", on="id")
