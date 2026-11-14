"""
A boolean mask holding a window (shift, cumsum, rank) keeps the rows it was
computed over: applied after a later sort, filter or head - by where, mask,
ds[mask], loc or filter - it is aligned like the pandas Series it stands for,
not recomputed on the later rows.
"""

import warnings

import pandas as pd
import pytest

from datastore import DataStore, F
from datastore.column_expr import ColumnExpr
from tests.test_utils import assert_datastore_equals_pandas


def _frame():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})


MASKS = {
    "shift": lambda s: s["x"].shift(1) > 1,
    "cumsum": lambda s: s["x"].cumsum() > 3,
    "rank": lambda s: s["x"].rank(method="first") > 2,
    "isin_shift": lambda s: s["x"].shift(1).isin([2.0, 3.0]),
    "row_wise": lambda s: s["x"] > 2,
}
LATER = {
    "sort": lambda s: s.sort_values("x", ascending=False),
    "filter": lambda s: s[s["x"] > 1],
    "head": lambda s: s.head(3),
}
APPLY = {
    "where": (lambda f, m: f.where(m, 0), lambda f, m: f.where(m, 0)),
    "mask": (lambda f, m: f.mask(m, 0), lambda f, m: f.mask(m, 0)),
    "getitem": (lambda f, m: f[m], lambda f, m: f[m]),
    "loc": (lambda f, m: f.loc[m], lambda f, m: f.loc[m]),
    "filter": (lambda f, m: f.filter(m), lambda f, m: f[m]),
}

# A file source numbers its rows from 0 again after a filter, so a mask
# computed on the unfiltered rows is aligned to the wrong rows; pandas keeps
# the source labels.
FILE_SOURCE_RENUMBERS_ROWS = pytest.mark.xfail(
    strict=True, reason="a filtered file source renumbers its rows instead of keeping its labels"
)


def _cases():
    for source in ("dataframe", "parquet"):
        for mask in MASKS:
            for later in LATER:
                for apply in APPLY:
                    marks = ()
                    if source == "parquet" and later == "filter" and mask != "row_wise":
                        marks = (FILE_SOURCE_RENUMBERS_ROWS,)
                    yield pytest.param(source, mask, later, apply, marks=marks, id=f"{source}-{mask}-{later}-{apply}")


@pytest.fixture
def parquet_path(tmp_path):
    path = str(tmp_path / "frame.parquet")
    _frame().to_parquet(path)
    return path


@pytest.mark.parametrize("source,mask,later,apply", _cases())
def test_mask_taken_before_later_op(source, mask, later, apply, parquet_path):
    d = DataStore(_frame()) if source == "dataframe" else DataStore.from_file(parquet_path)
    apply_ds, apply_pd = APPLY[apply]
    result = apply_ds(LATER[later](d), MASKS[mask](d))

    p = _frame()
    with warnings.catch_warnings():
        # "Boolean Series key will be reindexed to match DataFrame index"
        warnings.simplefilter("ignore", UserWarning)
        expected = apply_pd(LATER[later](p), MASKS[mask](p))

    assert_datastore_equals_pandas(result, expected)


def test_macroscope_chain():
    """where() with a shift mask after a descending sort keeps the mask's x=3 row."""
    d = DataStore(pd.DataFrame({"x": [1, 2, 3]}))
    mask = d["x"].shift(1) > 1
    result = d.sort_values("x", ascending=False).where(mask, 0)

    p = pd.DataFrame({"x": [1, 2, 3]})
    pd_mask = p["x"].shift(1) > 1
    expected = p.sort_values("x", ascending=False).where(pd_mask, 0)

    assert_datastore_equals_pandas(result, expected)


UNKNOWN_READS = {
    "sort_filter": (lambda f, m: f.sort_values("a").filter(m), lambda f, m: f.sort_values("a")[m]),
    "sort_getitem": (lambda f, m: f.sort_values("a")[m], lambda f, m: f.sort_values("a")[m]),
    "sort_where": (lambda f, m: f.sort_values("a").where(m, 0), lambda f, m: f.sort_values("a").where(m, 0)),
    "filter_mask": (lambda f, m: f[f["b"] > 10].mask(m, 0), lambda f, m: f[f["b"] > 10].mask(m, 0)),
}


@pytest.mark.parametrize("chain", sorted(UNKNOWN_READS))
def test_window_mask_with_unknown_columns(chain):
    """A window ordered by SQL text reads unknown columns; it is aligned, not inlined."""
    pdf = pd.DataFrame({"a": [3.0, 1.0, 4.0, 2.0], "b": [10, 20, 30, 40]})
    d = DataStore(pdf.copy())
    window = F.row_number().over(order_by="a DESC NULLS LAST")
    mask = ColumnExpr(expr=window > 2, datastore=d)
    assert mask._expr.columns_read() is None
    assert not mask._expr.is_row_wise

    pd_mask = pdf["a"].rank(method="first", ascending=False) > 2
    build_ds, build_pd = UNKNOWN_READS[chain]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = build_pd(pdf, pd_mask)
    assert_datastore_equals_pandas(build_ds(d, mask), expected)
