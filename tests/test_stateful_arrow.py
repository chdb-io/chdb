#!python3

import unittest
import pyarrow # noqa
from chdb import session


class TestStatefulArrow(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_query_fmt(self):
        with session.Session() as sess:
            ret = sess.query("SELECT 1 AS x", "ArrowTable")
            self.assertEqual(
                str(ret),
                """pyarrow.Table
x: uint8 not null
----
x: [[1]]""",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
