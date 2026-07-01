"""
Test that ``chdb.query(..., output_format="DataStore")`` dispatches to the
DataFrame -> DataStore wrapper and is case-insensitive.

Covers the output_format dispatch added in chdb/__init__.py so that
``chdb.query(SQL, output_format="DataStore")`` returns a
``chdb.datastore.DataStore`` (mirrored from chdb-core). The existing suite
exercises ``output_format="DataFrame"`` extensively but never the DataStore
variant, so this guards the dispatch table and the DataFrame->DataStore wrap.
"""

import unittest
import chdb
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestQueryOutputFormatDataStore(unittest.TestCase):
    """chdb.query(..., output_format="DataStore") returns a DataStore."""

    # Two columns of differing type so a wrong dispatch / wrap surfaces clearly.
    SQL = "SELECT number AS n, toString(number) AS s FROM numbers(5)"

    def test_query_output_format_datastore_returns_datastore(self):
        """output_format="DataStore" returns a DataStore holding the query data."""
        # pandas reference via the DataFrame output format
        pd_result = chdb.query(self.SQL, output_format="DataFrame")

        # DataStore output format (mirror of the DataFrame query)
        ds_result = chdb.query(self.SQL, output_format="DataStore")

        self.assertIsInstance(ds_result, DataStore)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_output_format_datastore_is_case_insensitive(self):
        """Every casing of "DataStore" returns an equivalent DataStore."""
        pd_result = chdb.query(self.SQL, output_format="DataFrame")

        for fmt in ("datastore", "DATASTORE", "DataStore", "dataStore"):
            ds_result = chdb.query(self.SQL, output_format=fmt)
            self.assertIsInstance(
                ds_result,
                DataStore,
                msg=f"output_format={fmt!r} did not return a DataStore",
            )
            assert_datastore_equals_pandas(
                ds_result, pd_result, msg=f"output_format={fmt!r}"
            )


if __name__ == "__main__":
    unittest.main()
