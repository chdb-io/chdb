#!python3

import io
import unittest
import shutil
import pyarrow as pa
from datetime import date
from chdb import session


class TestArrowDataTypes(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_date_arrow_output(self):
        test_arrow_date_types_dir = ".tmp_test_arrow_date_types_dir"
        shutil.rmtree(test_arrow_date_types_dir, ignore_errors=True)
        sess = session.Session(test_arrow_date_types_dir)

        sql = """
        SELECT
            toDate('1970-01-01') as epoch_date,
            toDate('2000-01-01') as y2k_date,
            toDate('2023-12-31') as recent_date,
            toDate32('1970-01-01') as epoch_date32,
            toDate32('2000-01-01') as y2k_date32,
            toDate32('2023-12-31') as recent_date32
        """

        arrow_table = sess.query(sql, 'ArrowTable')

        for field in arrow_table.schema:
            if 'date' in field.name:
                self.assertEqual(field.type, pa.date32(), f"{field.name} should be date32, got {field.type}")

        df = arrow_table.to_pandas()

        expected_values = {
            'epoch_date': date(1970, 1, 1),
            'y2k_date': date(2000, 1, 1),
            'recent_date': date(2023, 12, 31),
            'epoch_date32': date(1970, 1, 1),
            'y2k_date32': date(2000, 1, 1),
            'recent_date32': date(2023, 12, 31)
        }

        for col, expected in expected_values.items():
            actual = df[col].iloc[0]
            if hasattr(actual, 'date'):
                actual = actual.date()
            self.assertEqual(actual, expected, f"{col}: expected {expected}, got {actual}")

        sess.close()
        shutil.rmtree(test_arrow_date_types_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
