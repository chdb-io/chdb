#!/usr/bin/env python3

import unittest
import sys
from chdb import session


def check_pyarrow_available():
    """Check if pyarrow is available for import."""
    try:
        import pyarrow
        return True
    except ImportError:
        return False


def check_pandas_available():
    """Check if pandas is available for import."""
    try:
        import pandas
        return True
    except ImportError:
        return False


class TestOptionalDependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.sess = session.Session()
        return super().setUp()

    def tearDown(self) -> None:
        self.sess.close()
        return super().tearDown()

    def test_arrowtable_output_format(self):
        """Test ArrowTable output format with/without pyarrow dependency."""
        pyarrow_available = check_pyarrow_available()

        if pyarrow_available:
            # If pyarrow is available, should work normally
            try:
                ret = self.sess.query("SELECT 1 AS x, 'hello' AS y", "ArrowTable")
                self.assertIsNotNone(ret)
                # Verify it's actually an ArrowTable
                import pyarrow as pa
                self.assertIsInstance(ret, pa.Table)
                self.assertEqual(ret.column('x').to_pylist(), [1])
                self.assertEqual(ret.column('y').to_pylist(), ['hello'])
            except Exception as e:
                self.fail(f"ArrowTable format should work when pyarrow is available, but got: {e}")
        else:
            # If pyarrow is not available, should raise ImportError
            with self.assertRaises(ImportError) as context:
                self.sess.query("SELECT 1 AS x, 'hello' AS y", "ArrowTable")

            # Verify the error message mentions pyarrow
            error_msg = str(context.exception).lower()
            self.assertIn('pyarrow', error_msg)

    def test_dataframe_output_format(self):
        """Test DataFrame output format with/without pandas dependency."""
        pandas_available = check_pandas_available()

        # DataFrame format requires pandas
        if pandas_available:
            # If both are available, should work normally
            try:
                ret = self.sess.query("SELECT 1 AS x, 'hello' AS y", "DataFrame")
                self.assertIsNotNone(ret)
                # Verify it's actually a DataFrame
                import pandas as pd
                self.assertIsInstance(ret, pd.DataFrame)
                self.assertEqual(ret['x'].iloc[0], 1)
                self.assertEqual(ret['y'].iloc[0], 'hello')
            except Exception as e:
                self.fail(f"DataFrame format should work when pandas is available, but got: {e}")
        else:
            # If pandas is missing, should raise ImportError
            with self.assertRaises(ImportError) as context:
                self.sess.query("SELECT 1 AS x, 'hello' AS y", "DataFrame")

            # Verify the error message mentions the missing dependency
            error_msg = str(context.exception).lower()
            self.assertIn('pandas', error_msg)

    def test_dependency_status_logging(self):
        """Log the current dependency status for debugging."""
        pyarrow_status = "available" if check_pyarrow_available() else "not available"
        pandas_status = "available" if check_pandas_available() else "not available"

        print(f"\nDependency Status:")
        print(f"  PyArrow: {pyarrow_status}")
        print(f"  Pandas: {pandas_status}")
        print(f"  Python version: {sys.version}")

        # This test always passes, it's just for logging
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
