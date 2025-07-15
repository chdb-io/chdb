import unittest
import subprocess
import sys

class TestChdbCLI(unittest.TestCase):

    def run_chdb_command(self, args):
        cmd = [sys.executable, '-m', 'chdb'] + args
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result
        except subprocess.TimeoutExpired:
            self.fail("Command timed out")

    def test_basic_sql_query(self):
        result = self.run_chdb_command(['SELECT 1'])
        self.assertEqual(result.returncode, 0)
        self.assertIn('1', result.stdout)

    def test_sql_with_format(self):
        result = self.run_chdb_command(['SELECT 1', 'JSON'])
        self.assertEqual(result.returncode, 0)
        self.assertIn('1', result.stdout)
        self.assertIn('data', result.stdout)

    def test_no_arguments_error(self):
        result = self.run_chdb_command([])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SELECT 1', result.stderr)
        self.assertIn('the following arguments are required: sql', result.stderr)

    def test_invalid_option_error(self):
        result = self.run_chdb_command(['--invalid-option'])
        self.assertNotEqual(result.returncode, 0)

if __name__ == '__main__':
    unittest.main()