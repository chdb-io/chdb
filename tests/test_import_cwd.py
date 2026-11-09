"""`import chdb` must not touch the process working directory.

The working directory is shared by every thread, so changing it during the
import, even briefly, is visible to code running alongside. Each case runs in a
subprocess so the import starts from a clean interpreter; an import hook stands
in for a native loader that fails.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

_PRELUDE = """
import os
import sys

chdir_calls = []
_chdir = os.chdir
def _recording_chdir(path):
    chdir_calls.append(path)
    _chdir(path)
os.chdir = _recording_chdir

class _FailingLoader:
    def find_spec(self, name, path=None, target=None):
        if {exc} is not None and name == "chdb._chdb":
            raise {exc}({message!r}, **({kwargs}))

sys.meta_path.insert(0, _FailingLoader())
"""

_BODY = """
try:
    import chdb
except Exception as e:
    print("IMPORT_FAILED", type(e).__name__, str(e))
else:
    print("IMPORT_OK")
    assert os.path.realpath(chdb.__file__) == {expected_init!r}, chdb.__file__
assert not chdir_calls, chdir_calls
print("NO_CHDIR")
"""


def _run(code, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestImportLeavesCwdAlone(unittest.TestCase):
    def setUp(self):
        self.expected_init = os.path.realpath(importlib.util.find_spec("chdb").origin)

    def _check(self, expected, exc=None, message="", kwargs="{}"):
        prelude = _PRELUDE.format(exc=exc, message=message, kwargs=kwargs)
        body = _BODY.format(expected_init=self.expected_init)
        with tempfile.TemporaryDirectory() as workdir:
            proc = _run(prelude + body, workdir)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(expected, proc.stdout)
        self.assertIn("NO_CHDIR", proc.stdout)
        return proc.stdout

    def test_successful_import(self):
        self._check("IMPORT_OK", exc="None")

    def test_native_import_error(self):
        out = self._check("IMPORT_FAILED ImportError", exc="ImportError", message="simulated dlopen failure")
        self.assertIn("simulated dlopen failure", out)

    def test_native_loader_raises_other_exception(self):
        self._check("IMPORT_FAILED OSError", exc="OSError", message="simulated loader failure")

    def test_engine_not_installed(self):
        out = self._check(
            "IMPORT_FAILED ImportError",
            exc="ModuleNotFoundError",
            message="No module named 'chdb._chdb'",
            kwargs="{'name': 'chdb._chdb'}",
        )
        self.assertIn("pip install chdb-core", out)


if __name__ == "__main__":
    unittest.main()
