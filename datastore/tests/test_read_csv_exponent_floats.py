"""
read_csv() and from_file() infer exponent-notation numbers such as 3.2e-05 in
CSV/TSV files as float64, like pandas.read_csv() (#770).
"""

import pandas as pd
import pytest

import datastore as ds
from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal


def _write(tmp_path, content, ext="csv"):
    sep = "\t" if ext == "tsv" else ","
    path = str(tmp_path / f"data.{ext}")
    with open(path, "w") as f:
        f.write(content.replace(",", sep))
    return path, sep


READERS = [
    pytest.param(lambda path, sep: ds.read_csv(path, sep=sep), id="read_csv"),
    pytest.param(lambda path, sep: DataStore.from_file(path), id="from_file"),
]


@pytest.mark.parametrize("ext", ["csv", "tsv"])
@pytest.mark.parametrize("read", READERS)
def test_issue_repro(read, ext, tmp_path):
    path, sep = _write(tmp_path, "a,b\n1,3.2e-05\n2,1.5e+03\n", ext)
    pd_df = pd.read_csv(path, sep=sep)
    result = read(path, sep).to_pandas()
    assert result.dtypes["b"] == "float64"
    assert_frame_equal(result, pd_df)


@pytest.mark.parametrize("read", READERS)
def test_mixed_integers_missing_and_uppercase(read, tmp_path):
    path, sep = _write(tmp_path, "a,b\n1,1\n2,\n3,-2.5E-3\n4,1E5\n")
    pd_df = pd.read_csv(path, sep=sep)
    assert_frame_equal(read(path, sep).to_pandas(), pd_df)


@pytest.mark.parametrize("read", READERS)
def test_assign_sort_chain(read, tmp_path):
    path, sep = _write(tmp_path, "a,b\n1,3.2e-05\n2,1.5e+03\n3,4e2\n4,7.5e-1\n")
    pd_df = pd.read_csv(path, sep=sep)
    ds_df = read(path, sep)

    pd_result = pd_df.assign(c=lambda d: d["b"] * 2, big=lambda d: d["b"] > 0.5)
    ds_result = ds_df.assign(c=lambda d: d["b"] * 2, big=lambda d: d["b"] > 0.5)
    assert_frame_equal(ds_result.sort_values("b").to_pandas(), pd_result.sort_values("b"))


@pytest.mark.parametrize("read", READERS)
def test_sum_of_exponent_column(read, tmp_path):
    path, sep = _write(tmp_path, "k,b\nx,1e3\ny,2.5e-1\nx,3e0\n")
    pd_df = pd.read_csv(path, sep=sep)
    ds_df = read(path, sep)
    assert_series_equal(ds_df.groupby("k")["b"].sum().to_pandas(), pd_df.groupby("k")["b"].sum())


@pytest.mark.parametrize("read", READERS)
def test_non_numeric_text_stays_string(read, tmp_path):
    path, sep = _write(tmp_path, "a,b\n1,1e5x\n2,3e2\n")
    pd_df = pd.read_csv(path, sep=sep)
    result = read(path, sep).to_pandas()
    assert result["b"].tolist() == pd_df["b"].tolist()
    assert not pd.api.types.is_numeric_dtype(result["b"])
