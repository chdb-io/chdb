import subprocess
import sys
import textwrap
import unittest


def _run(snippet, timeout=60):
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/tmp",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _ray_available():
    try:
        import ray  # noqa
        return True
    except ImportError:
        return False


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only")
@unittest.skipUnless(_ray_available(), "ray not installed")
class TestImportCrossStlIsolation(unittest.TestCase):

    def test_import_ray_then_chdb_does_not_crash(self):
        rc, out = _run("""
            import ray  # noqa
            import chdb  # noqa
            print('ok')
        """)
        self.assertEqual(rc, 0, f"exit {rc}, output:\n{out}")
        self.assertIn("ok", out)
        self.assertNotEqual(rc, 134, f"SIGABRT, output:\n{out}")

    def test_import_chdb_then_ray_does_not_crash(self):
        rc, out = _run("""
            import chdb  # noqa
            import ray  # noqa
            print('ok')
        """)
        self.assertEqual(rc, 0, f"exit {rc}, output:\n{out}")
        self.assertIn("ok", out)

    def test_query_after_ray_first_import_returns_correct_value(self):
        rc, out = _run("""
            import ray  # noqa
            import chdb
            r = chdb.query('SELECT 1+1 AS x', 'CSV')
            print('result:', str(r).strip())
        """)
        self.assertEqual(rc, 0, f"exit {rc}, output:\n{out}")
        self.assertIn("result:", out)
        self.assertIn("2", out, f"unexpected output:\n{out}")


if __name__ == "__main__":
    unittest.main()
