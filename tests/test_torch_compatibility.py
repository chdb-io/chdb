#!python3

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestTorchCompatibility(unittest.TestCase):
    """Test that chdb and torch can coexist in the same process."""

    def test_import_torch_then_chdb(self):
        import torch
        import chdb

        # torch: basic tensor operation
        t = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(t.sum().item(), 6.0)

        # chdb: basic SQL query
        res = chdb.query("SELECT 1 + 2 AS result", "CSV")
        self.assertIn("3", str(res))

    def test_import_chdb_then_torch(self):
        import chdb
        import torch

        # chdb: basic SQL query
        res = chdb.query("SELECT 'ok' AS status", "CSV")
        self.assertIn("ok", str(res))

        # torch: basic tensor operation
        t = torch.tensor([[1, 2], [3, 4]])
        self.assertEqual(t.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
