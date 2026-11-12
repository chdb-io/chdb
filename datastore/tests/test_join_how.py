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


def _expected_merge(how, right=RIGHT):
    if int(pd.__version__.split(".")[0]) >= 3 or how not in ("left_anti", "right_anti"):
        return LEFT.merge(right, how=how, **_keys(how))
    # Explicit pandas 3 anti-join results for this fixture. pandas 2 cannot
    # provide an oracle for these operations; still check values and labels.
    if how == "left_anti":
        return pd.DataFrame(
            {"id": [1, 3], "a": [10, 30], "b": [float("nan"), float("nan")]},
            index=[0, 2],
        )
    return pd.DataFrame({"id": [5], "a": [float("nan")], "b": [right.loc[2, "b"]]}, index=[2])


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("how", list(HOWS), ids=[str(h) for h in HOWS])
@pytest.mark.parametrize("api", ["join", "merge"])
def test_join_type_matches_pandas_merge(request, make_sides, source, how, api):
    pandas_how = HOWS[how]
    pd_result = _expected_merge(pandas_how)

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
@pytest.mark.parametrize("how", ["full", "outer", "LEFT_OUTER", "left_anti", "right_anti"])
def test_join_type_after_lazy_ops_on_right(make_sides, source, how):
    """A right side whose pipeline runs in pandas joins as a pandas merge."""
    pandas_how = HOWS[how]
    pd_right = RIGHT.assign(b=RIGHT["b"].apply(lambda v: v + 1))
    pd_result = _expected_merge(pandas_how, pd_right)

    left, right = make_sides(source)
    right = right.assign(b=right["b"].apply(lambda v: v + 1))
    ds_result = left.join(right, how=how, on="id")

    assert not ds_result._joins
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("how", ["left semi", "right_semi", "bogus", "full inner"])
@pytest.mark.parametrize("api", ["join", "merge"])
def test_unsupported_join_type_raises(make_sides, source, how, api):
    # pandas 2 can raise an internal KeyError/UnboundLocalError for an
    # invalid how; DataStore must consistently raise its public JoinTypeError.
    pandas_errors = ValueError if int(pd.__version__.split(".")[0]) >= 3 else (
        ValueError, KeyError, UnboundLocalError,
    )
    with pytest.raises(pandas_errors):
        LEFT.merge(RIGHT, how=how, on="id")

    left, right = make_sides(source)
    with pytest.raises(JoinTypeError, match="join type"):
        getattr(left, api)(right, how=how, on="id")


@pytest.mark.parametrize("source", SOURCES)
def test_cross_join_takes_no_keys(make_sides, source):
    left, right = make_sides(source)
    with pytest.raises(QueryError, match="cross join takes no join keys"):
        left.join(right, how="cross", on="id")


@pytest.mark.parametrize("how", ["left_anti", "right_anti"])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "duplicates_and_nulls", "all_matched", "empty_left", "empty_right",
        "different_keys", "multiple_keys", "index", "multiindex", "mixed_index",
        "array_keys", "indicator", "marker_names", "multiindex_columns", "series",
    ],
)
def test_anti_join_compat_matches_native(monkeypatch, how, sort, case):
    """Check the pandas 2 fallback against pandas 3's native implementation.

    The main API matrix above also runs on pandas 2 with explicit results;
    these cases use pandas 3 as an independent oracle for the fallback's
    less common merge options, dtypes, ordering and exact index labels.
    """
    import datastore.pandas_compat as compat

    if not compat._PANDAS_3_PLUS:
        pytest.skip("native anti-join oracle requires pandas 3")

    left = pd.DataFrame({"key": [2., 1., 2., None, 3., 1.], "value": range(6)})
    right = pd.DataFrame({"key": [2., 2., None, 4.], "value": range(4)})
    kwargs = {"on": "key", "sort": sort, "suffixes": ("_left", "_right")}
    if case == "all_matched":
        right = left.copy()
    elif case == "empty_left":
        left = left.iloc[:0]
    elif case == "empty_right":
        right = right.iloc[:0]
    elif case == "different_keys":
        right = right.rename(columns={"key": "other"})
        kwargs = {"left_on": "key", "right_on": "other", "sort": sort}
    elif case == "multiple_keys":
        left["second"] = [0, 0, 1, 0, 1, 0]
        right["second"] = [0, 0, 0, 1]
        kwargs["on"] = ["key", "second"]
    elif case in ("index", "multiindex", "mixed_index"):
        if case == "multiindex":
            left["second"] = [0, 0, 1, 0, 1, 0]
            right["second"] = [0, 0, 0, 1]
        keys = ["key", "second"] if case == "multiindex" else ["key"]
        left = left.set_index(keys)
        right = right.set_index(keys)
        kwargs.pop("on")
        kwargs.update(left_index=True, right_index=True)
        if case == "mixed_index":
            left = left.reset_index()
            kwargs.pop("left_index")
            kwargs["left_on"] = "key"
    elif case == "array_keys":
        kwargs.pop("on")
        kwargs.update(left_on=left["key"].to_numpy(), right_on=right["key"].to_numpy())
    elif case == "indicator":
        kwargs["indicator"] = "origin"
    elif case == "marker_names":
        for name in ("_merge", "_left_indicator", "_right_indicator", "__anti_join_match__"):
            left[name] = 1
            right[name] = 2
    elif case == "multiindex_columns":
        left.columns = pd.MultiIndex.from_tuples([("key", ""), ("value", "")])
        right.columns = left.columns
        kwargs["on"] = [("key", "")]
    elif case == "series":
        right = right.set_index("key")["value"]
        kwargs.pop("on")
        kwargs.update(left_on="key", right_index=True)

    expected = pd.merge(left, right, how=how, **kwargs)
    monkeypatch.setattr(compat, "_PANDAS_HAS_ANTI_JOIN", False)
    result = compat._pandas_merge(left, right, how=how, **kwargs)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["left_anti", "right_anti"])
def test_anti_join_merge_options_and_validation(how):
    """The eager merge fallback supports indicators and validates full inputs."""
    left = pd.DataFrame({"key": [1, 1, 2], "a": [10, 11, 20]})
    right = pd.DataFrame({"key": [1, 3], "b": [100, 300]})
    with pytest.raises(pd.errors.MergeError):
        DataStore(left).merge(DataStore(right), how=how, on="key", validate="one_to_one")

    result = DataStore(left).merge(
        DataStore(right), how=how, on="key", indicator="origin", validate="many_to_one",
    )
    if how == "left_anti":
        expected = pd.DataFrame({"key": [2], "a": [20], "b": [float("nan")]}, index=[2])
        origin = "left_only"
    else:
        expected = pd.DataFrame({"key": [3], "a": [float("nan")], "b": [300]}, index=[2])
        origin = "right_only"
    expected["origin"] = pd.Categorical([origin], categories=["left_only", "right_only", "both"])
    assert_datastore_equals_pandas(result, expected, check_index=True)
