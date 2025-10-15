#!python3

import unittest
import pandas # noqa
from chdb import session


class TestStatefulDataFrame(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_query_fmt(self):
        with session.Session() as sess:
            ret = sess.query("SELECT 1 AS x", "DataFrame")
            self.assertEqual(ret.x[0], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
