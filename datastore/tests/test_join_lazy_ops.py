"""
join() and merge() must carry the right side's lazy operations (issue #695).

The right DataStore joins with its complete pipeline as it is when join()
or merge() is called, like the DataFrame pandas merges with. Covers both
sides being files (one SQL JOIN with the right pipeline as a subquery),
DataFrame sources on either side (pandas merge), and a right pipeline that
needs pandas.

All tests use the mirror pattern: DataStore result == pandas result,
including the row order pandas merge defines.
"""

import pandas as pd
import pytest

from datastore import DataStore, Field
from datastore.exceptions import QueryError
from tests.test_utils import assert_datastore_equals_pandas


LEFT = pd.DataFrame({"k": [1, 2, 3, 4], "a": [10, 20, 30, 40]})
RIGHT = pd.DataFrame({"k": [1, 2, 3, 4], "w": [1, 2, 3, 4]})

KINDS = [("file", "file"), ("dataframe", "file"), ("file", "dataframe"), ("dataframe", "dataframe")]


@pytest.fixture(params=KINDS, ids=["-".join(k) for k in KINDS])
def sides(request, tmp_path):
    """Return (left, right) factories building DataStores of each kind."""

    def factory(kind, df, name):
        if kind == "dataframe":
            return lambda: DataStore(df.copy())
        path = str(tmp_path / f"{name}.parquet")
        df.to_parquet(path)
        return lambda: DataStore.from_file(path)

    left_kind, right_kind = request.param
    return factory(left_kind, LEFT, "l"), factory(right_kind, RIGHT, "r")


def assert_joined_equal(ds_result, pd_result):
    assert_datastore_equals_pandas(ds_result, pd_result)


class TestJoinCarriesRightLazyOps:
    @pytest.mark.parametrize("method", ["join", "merge"])
    def test_issue_right_assigned_column(self, sides, method):
        make_left, make_right = sides
        pr = RIGHT.copy()
        pr["w"] = pr["w"] * 100
        pd_result = LEFT.merge(pr, on="k")

        r = make_right()
        r["w"] = r["w"] * 100
        ds_result = getattr(make_left(), method)(r, on="k")

        assert_joined_equal(ds_result, pd_result)

    @pytest.mark.parametrize("method", ["join", "merge"])
    def test_issue_right_filtered(self, sides, method):
        make_left, make_right = sides
        pd_result = LEFT.merge(RIGHT[RIGHT["w"] > 2], on="k")

        r = make_right()
        r = r[r["w"] > 2]
        ds_result = getattr(make_left(), method)(r, on="k")

        assert_joined_equal(ds_result, pd_result)

    def test_right_filtered_and_assigned(self, sides):
        make_left, make_right = sides
        pr = RIGHT[RIGHT["w"] > 1].copy()
        pr["w"] = pr["w"] + 0.5
        pd_result = LEFT.merge(pr, on="k")

        r = make_right()
        r = r[r["w"] > 1]
        r["w"] = r["w"] + 0.5
        ds_result = make_left().merge(r, on="k")

        assert_joined_equal(ds_result, pd_result)

    def test_right_head(self, sides):
        make_left, make_right = sides
        pd_result = LEFT.merge(RIGHT.head(2), on="k")

        ds_result = make_left().merge(make_right().head(2), on="k")

        assert_joined_equal(ds_result, pd_result)

    def test_right_sorted_head(self, sides):
        make_left, make_right = sides
        pd_result = LEFT.merge(RIGHT.sort_values("w", ascending=False).head(2), on="k")

        r = make_right().sort_values("w", ascending=False).head(2)
        ds_result = make_left().merge(r, on="k")

        assert_joined_equal(ds_result, pd_result)

    def test_right_new_column_then_projection(self, sides):
        make_left, make_right = sides
        pr = RIGHT.copy()
        pr["z"] = pr["w"] * 2
        pd_result = LEFT.merge(pr[["k", "z"]], on="k")

        r = make_right()
        r["z"] = r["w"] * 2
        ds_result = make_left().merge(r[["k", "z"]], on="k")

        assert_joined_equal(ds_result, pd_result)

    def test_right_filtered_left_on_right_on(self, sides):
        make_left, make_right = sides
        pr = RIGHT.rename(columns={"k": "rk"})
        pd_result = LEFT.merge(pr[pr["w"] > 2], left_on="k", right_on="rk")

        r = make_right().rename(columns={"k": "rk"})
        r = r[r["w"] > 2]
        ds_result = make_left().merge(r, left_on="k", right_on="rk")

        assert_joined_equal(ds_result, pd_result)

    def test_left_and_right_filtered(self, sides):
        make_left, make_right = sides
        pd_result = LEFT[LEFT["a"] < 40].merge(RIGHT[RIGHT["w"] > 1], on="k")

        left = make_left()
        r = make_right()
        r = r[r["w"] > 1]
        ds_result = left[left["a"] < 40].merge(r, on="k")

        assert_joined_equal(ds_result, pd_result)

    def test_filter_after_join_with_filtered_right(self, sides):
        make_left, make_right = sides
        pd_joined = LEFT.merge(RIGHT[RIGHT["w"] > 1], on="k")
        pd_result = pd_joined[pd_joined["a"] > 20]

        r = make_right()
        r = r[r["w"] > 1]
        ds_joined = make_left().merge(r, on="k")
        ds_result = ds_joined[ds_joined["a"] > 20]

        assert_joined_equal(ds_result, pd_result)

    def test_right_ops_after_merge_do_not_leak(self, sides):
        make_left, make_right = sides
        pd_result = LEFT.merge(RIGHT[RIGHT["w"] > 1], on="k")

        r = make_right()
        r = r[r["w"] > 1]
        ds_result = make_left().merge(r, on="k")
        r["w"] = r["w"] * 100

        assert_joined_equal(ds_result, pd_result)

    def test_right_pipeline_needs_pandas(self, sides):
        make_left, make_right = sides
        pr = RIGHT.copy()
        pr["w"] = pr["w"].apply(lambda v: v * 3)
        pd_result = LEFT.merge(pr, on="k")

        r = make_right()
        r["w"] = r["w"].apply(lambda v: v * 3)
        ds_result = make_left().merge(r, on="k")

        assert_joined_equal(ds_result, pd_result)


@pytest.fixture
def files(tmp_path):
    lp, rp = str(tmp_path / "l.parquet"), str(tmp_path / "r.parquet")
    LEFT.to_parquet(lp)
    RIGHT.to_parquet(rp)
    return lp, rp


def test_condition_join_with_filtered_right_file(files):
    lp, rp = files
    pd_result = LEFT.merge(RIGHT[RIGHT["w"] > 2], on="k")

    r = DataStore.from_file(rp)
    r = r[r["w"] > 2]
    joined = DataStore.from_file(lp).join(r, on=Field("k", table="l") == Field("k", table="r"))
    ds_result = joined.select(Field("k", table="l"), "a", "w")

    assert_joined_equal(ds_result, pd_result)


def test_condition_join_with_pandas_right_pipeline_raises(files):
    lp, rp = files
    r = DataStore.from_file(rp)
    r["w"] = r["w"].apply(lambda v: v * 3)

    with pytest.raises(QueryError, match="join on column names"):
        DataStore.from_file(lp).join(r, on=Field("k", table="l") == Field("k", table="r"))


def test_right_file_ops_in_join_sql(files):
    """Both sides files: one SQL query whose JOIN reads the right pipeline."""
    lp, rp = files
    r = DataStore.from_file(rp)
    r = r[r["w"] > 2]
    sql = DataStore.from_file(lp).merge(r, on="k").to_sql()

    assert sql.count("JOIN") == 1
    assert '"w" > 2' in sql
