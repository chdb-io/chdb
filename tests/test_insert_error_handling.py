#!/usr/bin/env python3

import unittest
import shutil
import os
from chdb import session

test_dir = ".test_insert_error"

class TestInsertErrorHandling(unittest.TestCase):
    """Test cases for INSERT query error handling to ensure proper exceptions are thrown."""

    def setUp(self) -> None:
        shutil.rmtree(test_dir, ignore_errors=True)
        self.sess = session.Session(test_dir)
        return super().setUp()

    def tearDown(self) -> None:
        """Clean up test environment."""
        shutil.rmtree(test_dir, ignore_errors=True)
        return super().tearDown()

    def test_incomplete_insert_values_throws_error(self):
        """Test that incomplete INSERT VALUES query throws RuntimeError instead of hanging."""

        self.sess.query(
            "CREATE TABLE test_table(id UInt32, name String, value Float64) ENGINE = Memory"
        )

        # This should throw an error because VALUES clause is incomplete (no actual values provided)
        with self.assertRaises(RuntimeError) as context:
            self.sess.query("INSERT INTO test_table (id, name, value) VALUES")


if __name__ == "__main__":
    unittest.main()
