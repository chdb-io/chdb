#!python3

import unittest
from chdb import session


class TestMultipleQuery(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_multiple_query(self):
        sess = session.Session()

        ret = sess.query("SELECT 1;SELECT 2;SELECT 3", "CSV")
        self.assertEqual(str(ret), "1\n2\n3\n")
        print(ret)

        ret = sess.query("SELECT '1a';SELECT '2a';SELECT '3a'", "JSON")
        self.assertIn("1a", str(ret))
        self.assertIn("2a", str(ret))
        self.assertIn("3a", str(ret))

        sess.close()


if __name__ == '__main__':
    unittest.main()
