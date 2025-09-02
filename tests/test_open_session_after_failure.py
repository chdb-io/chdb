#!python3

import unittest
import shutil
from chdb import session


test_dir1 = ".test_open_session_after_failure"
test_dir2 = "/usr/bin"


class TestStateful(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(test_dir1, ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(test_dir1, ignore_errors=True)
        return super().tearDown()

    def test_path(self):
        # Test that creating session with invalid path (read-only directory) raises exception
        with self.assertRaises(Exception):
            sess = session.Session(test_dir2)

        # Test that creating session with valid path works after failure
        sess = session.Session(test_dir1)

        ret = sess.query("select 'aaaaa'")
        self.assertEqual(str(ret), "\"aaaaa\"\n")

        sess.close()


if __name__ == '__main__':
    unittest.main()
