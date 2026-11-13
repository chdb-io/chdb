"""A user column named ``__orig_row_num__`` must survive row-order SQL.

DataStore injects ``rowNumberInAllBlocks() AS __orig_row_num__`` before a
filter so later ``ORDER BY`` can keep the source order. When the source
already has that column, ClickHouse sees two columns of the same name and
raises AMBIGUOUS_COLUMN_NAME (Code 352). The user's column has to be kept.

Regression for https://github.com/chdb-io/chdb/issues/772.
"""

import os

import pandas as pd

from datastore import DataStore
from datastore.utils import row_order_alias
from tests.test_utils import assert_datastore_equals_pandas


def _csv_store(df, directory):
    path = os.path.join(directory, "t.csv")
    df.to_csv(path, index=False)
    return DataStore.from_file(path)


def _assign_filter_tail(frame):
    """The chain reported in issue #772: assign, filter, then tail."""
    return (lambda data: data[data["v"] > 1].tail(2))(frame.assign(w=frame["v"] * 2))


def test_row_order_alias_skips_taken_names():
    """The fallback alias still contains the row-order sentinel substring."""
    assert row_order_alias([]) == "__orig_row_num__"
    assert row_order_alias(["__orig_row_num__"]) == "__orig_row_num___"
    both = row_order_alias(["__orig_row_num__", "__orig_row_num___"])
    assert both == "__orig_row_num____"
    assert "__orig_row_num__" in both


class TestUserOrigRowNumColumn:
    """User data must not collide with the internal row-number alias."""

    def test_csv_assign_filter_tail_keeps_user_column(self, tmp_path):
        """CSV assign/filter/tail keeps ``__orig_row_num__`` and matches pandas."""
        df = pd.DataFrame({"__orig_row_num__": [5, 6, 7, 8], "v": [1, 2, 3, 4]})
        assert_datastore_equals_pandas(
            _assign_filter_tail(_csv_store(df, tmp_path)),
            _assign_filter_tail(df),
        )

    def test_csv_filter_uses_the_user_column(self, tmp_path):
        """A filter on the user's ``__orig_row_num__`` reads that column."""
        df = pd.DataFrame({"__orig_row_num__": [5, 6, 7, 8], "v": [1, 2, 3, 4]})
        ds = _csv_store(df, tmp_path)
        assert_datastore_equals_pandas(
            ds[ds["__orig_row_num__"] > 6],
            df[df["__orig_row_num__"] > 6],
        )

    def test_csv_where_keeps_user_column(self, tmp_path):
        """where() also captures a pre-filter row number and must not drop the column."""
        df = pd.DataFrame({"__orig_row_num__": [5, 6, 7, 8], "v": [1, 2, 3, 4]})
        ds = _csv_store(df, tmp_path)
        assert_datastore_equals_pandas(
            ds.where(ds["v"] > 1, 0),
            df.where(df["v"] > 1, 0),
        )

    def test_csv_stable_sort_after_filter(self, tmp_path):
        """Stable sort after a filter uses the same alias as a tie-breaker."""
        df = pd.DataFrame(
            {"__orig_row_num__": [5, 6, 7, 8, 9], "v": [1, 2, 2, 1, 2]}
        )
        ds = _csv_store(df, tmp_path)
        assert_datastore_equals_pandas(
            ds[ds["v"] > 1].sort_values("v", kind="stable"),
            df[df["v"] > 1].sort_values("v", kind="stable"),
        )

    def test_csv_both_reserved_names_kept(self, tmp_path):
        """A column that already uses the first fallback alias is kept too."""
        df = pd.DataFrame(
            {
                "__orig_row_num__": [5, 6, 7, 8],
                "__orig_row_num___": [9, 8, 7, 6],
                "v": [1, 2, 3, 4],
            }
        )
        assert_datastore_equals_pandas(
            _assign_filter_tail(_csv_store(df, tmp_path)),
            _assign_filter_tail(df),
        )

    def test_dataframe_assign_filter_tail(self):
        """The same chain on a DataFrame source keeps the user column."""
        df = pd.DataFrame({"__orig_row_num__": [5, 6, 7, 8], "v": [1, 2, 3, 4]})
        assert_datastore_equals_pandas(
            _assign_filter_tail(DataStore(df)),
            _assign_filter_tail(df),
        )
