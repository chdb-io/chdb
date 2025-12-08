#!/usr/bin/env python3

import unittest
import platform
import subprocess
import pyarrow as pa
import pyarrow.compute as pc
import chdb
from chdb import ChdbError
from utils import is_musl_linux


class TestUnsupportedArrowTypes(unittest.TestCase):
    """Test that chDB properly handles unsupported Arrow types"""

    def setUp(self):
        """Set up test data"""
        self.sample_data = [1, 2, 3, 4, 5]
        self.sample_strings = ["a", "b", "c", "d", "e"]

    def test_sparse_union_type(self):
        """Test SPARSE_UNION type - should fail"""
        # Create a sparse union type
        children = [
            pa.array([1, None, 3, None, 5]),
            pa.array([None, "b", None, "d", None])
        ]
        type_ids = pa.array([0, 1, 0, 1, 0], type=pa.int8())

        union_array = pa.UnionArray.from_sparse(type_ids, children)
        table = pa.table([union_array], names=["sparse_union_col"])

        with self.assertRaises(Exception) as context:
            chdb.query("SELECT * FROM Python(table)")

        exception_str = str(context.exception)
        self.assertTrue("unknown" in exception_str or "Unsupported" in exception_str)

    def test_dense_union_type(self):
        """Test DENSE_UNION type - should fail"""
        # Create a dense union type
        children = [
            pa.array([1, 3, 5]),
            pa.array(["b", "d"])
        ]
        type_ids = pa.array([0, 1, 0, 1, 0], type=pa.int8())
        offsets = pa.array([0, 0, 1, 1, 2], type=pa.int32())

        union_array = pa.UnionArray.from_dense(type_ids, offsets, children)
        table = pa.table([union_array], names=["dense_union_col"])

        with self.assertRaises(Exception) as context:
            chdb.query("SELECT * FROM Python(table)")

        exception_str = str(context.exception)
        self.assertTrue("unknown" in exception_str or "Unsupported" in exception_str)

    def test_interval_month_day_type(self):
        """Test INTERVAL_MONTH_DAY type - should fail"""
        pass

    def test_interval_day_time_type(self):
        """Test INTERVAL_DAY_TIME type - should fail"""
        pass

    def test_interval_month_day_nano_type(self):
        """Test INTERVAL_MONTH_DAY_NANO type - should fail"""
        start_timestamps = pc.strptime(
            pa.array(["2021-01-01 00:00:00", "2022-01-01 00:00:00", "2023-01-01 00:00:00"]),
            format="%Y-%m-%d %H:%M:%S",
            unit="ns"
        )

        end_timestamps = pc.strptime(
            pa.array(["2021-04-01 00:00:00", "2022-05-01 00:00:00", "2023-07-01 00:00:00"]),
            format="%Y-%m-%d %H:%M:%S",
            unit="ns"
        )

        interval_array = pc.month_day_nano_interval_between(start_timestamps, end_timestamps)
        table = pa.table([interval_array], names=["interval_month_col"])

        with self.assertRaises(Exception) as context:
            chdb.query("SELECT * FROM Python(table)")

        exception_str = str(context.exception)
        self.assertTrue("unknown" in exception_str or "Unsupported" in exception_str)

    @unittest.skipIf(is_musl_linux(), "Skip test on musl systems")
    def test_list_view_type(self):
        """Test LIST_VIEW type - should fail"""
        # Create list view array
        list_data = [[1, 2], [3, 4, 5], [6], [], [7, 8, 9]]
        list_view_array = pa.array(list_data, type=pa.list_view(pa.int64()))
        table = pa.table([list_view_array], names=["list_view_col"])

        with self.assertRaises(Exception) as context:
            chdb.query("SELECT * FROM Python(table)")

        exception_str = str(context.exception)
        self.assertTrue("unknown" in exception_str or "Unsupported" in exception_str)

    @unittest.skipIf(is_musl_linux(), "Skip test on musl systems")
    def test_large_list_view_type(self):
        """Test LARGE_LIST_VIEW type - should fail"""
        # Create large list view array (if available)
        list_data = [[1, 2], [3, 4, 5], [6], [], [7, 8, 9]]
        large_list_view_array = pa.array(list_data, type=pa.large_list_view(pa.int64()))
        table = pa.table([large_list_view_array], names=["large_list_view_col"])

        with self.assertRaises(Exception) as context:
            chdb.query("SELECT * FROM Python(table)")

        exception_str = str(context.exception)
        self.assertTrue("unknown" in exception_str or "Unsupported" in exception_str)

    @unittest.skipIf(is_musl_linux(), "Skip test on musl systems")
    def test_run_end_encoded_type(self):
        """Test RUN_END_ENCODED type - should fail"""
        # Create run-end encoded array
        values = pa.array([1, 2, 3])
        run_ends = pa.array([3, 7, 10], type=pa.int32())
        ree_array = pa.RunEndEncodedArray.from_arrays(run_ends, values)
        table = pa.table([ree_array], names=["run_end_encoded_col"])

        with self.assertRaises(Exception) as context:
            chdb.query("SELECT * FROM Python(table)")

        exception_str = str(context.exception)
        self.assertTrue("unknown" in exception_str or "Unsupported" in exception_str)

    @unittest.skipIf(is_musl_linux(), "Skip test on musl systems")
    def test_skip_unsupported_columns_setting(self):
        """Test input_format_arrow_skip_columns_with_unsupported_types_in_schema_inference=1 skips unsupported columns"""
        # Create a table with both supported and unsupported columns
        supported_col = pa.array([1, 2, 3, 4, 5])  # int64 - supported
        # Create union array (unsupported)
        union_children = [
            pa.array([10, None, 30, None, 50]),
            pa.array([None, "b", None, "d", None])
        ]
        union_type_ids = pa.array([0, 1, 0, 1, 0], type=pa.int8())
        unsupported_col = pa.UnionArray.from_sparse(union_type_ids, union_children)

        table = pa.table([
            supported_col,
            unsupported_col
        ], names=["supported_col", "unsupported_col"])

        # Without the setting, query should fail
        with self.assertRaises(Exception) as context:
            chdb.query("SELECT * FROM Python(table)")

        exception_str = str(context.exception)
        self.assertTrue("unknown" in exception_str or "Unsupported" in exception_str)

        # With the setting, query should succeed but skip unsupported column
        result = chdb.query(
            "SELECT * FROM Python(table) settings input_format_arrow_skip_columns_with_unsupported_types_in_schema_inference=1"
        )
        self.assertEqual(str(result), "1\n2\n3\n4\n5\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
