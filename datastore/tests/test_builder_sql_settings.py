"""
SQL built by SQLBuilder (computed columns referenced by later operations)
must carry the source's format settings regardless of the query text.

A column or literal containing the word SETTINGS must not stop the settings
from being emitted: without ``input_format_parquet_preserve_order`` rows of
a parquet file with many row groups come back out of order.
"""

import io
import logging

import numpy as np
import pandas as pd

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


def _executed_sqls(ds):
    logger = logging.getLogger("datastore")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        ds.to_pandas()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    return [
        line.split("Executing SQL:", 1)[1].strip()
        for line in stream.getvalue().splitlines()
        if "Executing SQL:" in line
    ]


def test_settings_kept_when_query_mentions_settings(tmp_path):
    n = 100_000
    pdf = pd.DataFrame(
        {
            "id": np.arange(n),
            "g": np.arange(n) % 97,
            "label": np.where(np.arange(n) % 5 == 0, "SETTINGS", "other"),
        }
    )
    path = str(tmp_path / "rows.parquet")
    pdf.to_parquet(path, row_group_size=500)

    pd_result = pdf.copy()
    pd_result["z"] = pd_result["g"] * 2
    pd_result = pd_result[(pd_result["z"] > 10) & (pd_result["label"] != "SETTINGS")]

    ds = DataStore.from_file(path)
    ds["z"] = ds["g"] * 2
    ds = ds[(ds["z"] > 10) & (ds["label"] != "SETTINGS")]

    sqls = _executed_sqls(ds)
    assert len(sqls) == 1
    assert sqls[0].endswith(
        "SETTINGS session_timezone='UTC', input_format_parquet_preserve_order=1"
    )
    assert_datastore_equals_pandas(ds, pd_result)
