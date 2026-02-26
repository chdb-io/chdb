#!/usr/bin/env python3

import unittest
import chdb


class TestChdbClientName(unittest.TestCase):
    def test_default_client_name_is_empty(self):
        result = chdb.query("SELECT getSetting('chdb_client_name')", "TabSeparated")
        self.assertEqual(result.bytes().strip(), b"")

    def test_set_app_name_in_query(self):
        result = chdb.query("""
            SET chdb_client_name = 'my-service';
            SELECT getSetting('chdb_client_name')
        """, "TabSeparated")
        self.assertEqual(result.bytes().strip(), b"my-service")

    def test_set_app_name_in_session(self):
        """User can set application name in a session."""
        sess = chdb.session.Session()
        sess.query("SET chdb_client_name = 'session-app'")
        result = sess.query("SELECT getSetting('chdb_client_name')", "TabSeparated")
        self.assertEqual(result.bytes().strip(), b"session-app")
        sess.close()

    def test_set_app_name_via_settings_clause(self):
        """User can set application name via SETTINGS clause."""
        result = chdb.query("""
            SELECT getSetting('chdb_client_name')
            SETTINGS chdb_client_name = 'query-app'
        """, "TabSeparated")
        self.assertEqual(result.bytes().strip(), b"query-app")


if __name__ == '__main__':
    unittest.main()
