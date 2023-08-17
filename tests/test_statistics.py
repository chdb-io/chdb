#!python3

import os
import unittest
import chdb

N = 1000


class TestQueryStatistics(unittest.TestCase):
    def setUp(self) -> None:
        # create tmp csv file
        with open(".test.csv", "w") as f:
            f.write("a,b,c\n")
            for i in range(N):
                f.write(f"{i},{i*2},{i*3}\n")
        return super().setUp()

    def tearDown(self) -> None:
        # remove tmp csv file
        os.remove(".test.csv")
        return super().tearDown()

    def test_csv_stats(self):
        ret = chdb.query("SELECT * FROM file('.test.csv', CSV)", "CSV")
        self.assertEqual(ret.rows_read(), N)
        self.assertGreater(ret.elapsed(), 0.000001)
        self.assertEqual(ret.bytes_read(), 27000)
        print(f"SQL read {ret.rows_read()} rows, {ret.bytes_read()} bytes, elapsed {ret.elapsed()} seconds")

    def test_non_exist_stats(self):
        ret = chdb.query("SELECT * FROM file('notexist.parquet', Parquet)", "Parquet")
        self.assertEqual(ret.rows_read(), 0)
        self.assertEqual(ret.bytes_read(), 0)
        print(f"SQL read {ret.rows_read()} rows, {ret.bytes_read()} bytes, elapsed {ret.elapsed()} seconds")


if __name__ == "__main__":
    unittest.main()
