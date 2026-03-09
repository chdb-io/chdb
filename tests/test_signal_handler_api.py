#!/usr/bin/env python3
"""
Tests for the chDB C API signal handler control functions exposed via Python:
  - chdb._chdb.set_signal_handlers_enabled(int)
  - chdb._chdb.reset_signal_handlers()

These tests verify that:
1. Default behavior installs signal handlers on the first query.
2. set_signal_handlers_enabled(0) prevents handler installation AND removes existing ones.
3. reset_signal_handlers() restores SIG_DFL for all chDB-managed signals.
4. Re-enabling after disable works correctly.
"""

import ctypes
import ctypes.util
import os
import signal
import sys
import unittest


def _get_sigaction_handler(signum):
    """
    Return the sa_handler address (int) for *signum* via sigaction(2).
    Returns 0 when the disposition is SIG_DFL.
    """
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    if sys.platform == "darwin":
        class _SigAction(ctypes.Structure):
            _fields_ = [
                ("sa_handler", ctypes.c_void_p),
                ("sa_mask", ctypes.c_uint32),
                ("sa_flags", ctypes.c_int),
            ]
    else:
        class _SigAction(ctypes.Structure):
            _fields_ = [
                ("sa_handler", ctypes.c_void_p),
                ("sa_flags", ctypes.c_ulong),
                ("sa_restorer", ctypes.c_void_p),
                ("sa_mask", ctypes.c_uint64 * 16),
            ]

    act = _SigAction()
    ret = libc.sigaction(signum, None, ctypes.byref(act))
    if ret != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return act.sa_handler or 0


def _force_sig_dfl(signum):
    """Force *signum* back to SIG_DFL via sigaction."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    if sys.platform == "darwin":
        class _SigAction(ctypes.Structure):
            _fields_ = [
                ("sa_handler", ctypes.c_void_p),
                ("sa_mask", ctypes.c_uint32),
                ("sa_flags", ctypes.c_int),
            ]
    else:
        class _SigAction(ctypes.Structure):
            _fields_ = [
                ("sa_handler", ctypes.c_void_p),
                ("sa_flags", ctypes.c_ulong),
                ("sa_restorer", ctypes.c_void_p),
                ("sa_mask", ctypes.c_uint64 * 16),
            ]

    act = _SigAction()
    act.sa_handler = ctypes.c_void_p(0)
    act.sa_flags = 0
    if sys.platform == "darwin":
        act.sa_mask = 0
    libc.sigaction(signum, ctypes.byref(act), None)


_DEADLY_SIGNALS = [
    signal.SIGABRT,
    signal.SIGSEGV,
    signal.SIGILL,
    signal.SIGBUS,
    signal.SIGFPE,
]


def _has_signal_api():
    try:
        import chdb
        return (
            hasattr(chdb._chdb, "set_signal_handlers_enabled")
            and hasattr(chdb._chdb, "reset_signal_handlers")
        )
    except Exception:
        return False


@unittest.skipUnless(_has_signal_api(), "chdb signal handler API not available – build first")
class TestSignalHandlerControlAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import chdb
        cls._chdb = chdb._chdb

    def _reset_all(self):
        """Reset via API then force SIG_DFL for each deadly signal."""
        self._chdb.reset_signal_handlers()
        for sig in _DEADLY_SIGNALS:
            _force_sig_dfl(sig)

    # Test 1: reset_signal_handlers restores SIG_DFL
    def test_reset_restores_sig_dfl(self):
        """reset_signal_handlers() should restore SIG_DFL for all chDB signals."""
        self._chdb.set_signal_handlers_enabled(1)
        import chdb
        chdb.query("SELECT 1", "CSV")

        self._chdb.reset_signal_handlers()

        for sig in _DEADLY_SIGNALS:
            handler = _get_sigaction_handler(sig)
            self.assertEqual(handler, 0,
                f"reset_signal_handlers() should restore SIG_DFL for signal {sig}")

    # Test 2: set_signal_handlers_enabled(0) removes handlers and prevents future installation
    def test_disable_removes_and_prevents(self):
        """set_signal_handlers_enabled(0) must remove existing handlers and prevent future ones."""
        self._chdb.set_signal_handlers_enabled(1)
        import chdb
        chdb.query("SELECT 1", "CSV")

        self._chdb.set_signal_handlers_enabled(0)

        for sig in _DEADLY_SIGNALS:
            handler = _get_sigaction_handler(sig)
            self.assertEqual(handler, 0,
                f"set_signal_handlers_enabled(0) should reset signal {sig} to SIG_DFL")

        chdb.query("SELECT 1", "CSV")

        for sig in _DEADLY_SIGNALS:
            handler = _get_sigaction_handler(sig)
            self.assertEqual(handler, 0,
                f"After query with disabled handlers, signal {sig} should remain SIG_DFL")

        self._chdb.set_signal_handlers_enabled(1)

    # Test 3: re-enable after disable
    def test_reenable_after_disable(self):
        """set_enabled(1) after set_enabled(0) should allow handler installation."""
        import chdb
        self._reset_all()

        self._chdb.set_signal_handlers_enabled(0)
        chdb.query("SELECT 1", "CSV")
        handler_disabled = _get_sigaction_handler(signal.SIGSEGV)
        self.assertEqual(handler_disabled, 0,
            "Handler should NOT be installed when disabled")

        self._chdb.set_signal_handlers_enabled(1)
        chdb.query("SELECT 1", "CSV")
        handler_enabled = _get_sigaction_handler(signal.SIGSEGV)
        self.assertNotEqual(handler_enabled, 0,
            "Handler should be installed after re-enabling")

    # Test 4: reset is safe with no handlers
    def test_reset_when_no_handlers_no_crash(self):
        """reset_signal_handlers() should not crash when no chDB handlers are installed."""
        for sig in _DEADLY_SIGNALS:
            _force_sig_dfl(sig)

        try:
            self._chdb.reset_signal_handlers()
        except Exception as exc:
            self.fail(f"reset_signal_handlers() raised unexpectedly: {exc}")

        for sig in _DEADLY_SIGNALS:
            handler = _get_sigaction_handler(sig)
            self.assertEqual(handler, 0,
                f"Signal {sig} should remain SIG_DFL after reset with no handlers")

    # Test 5: multiple enable/disable cycles
    def test_multiple_enable_disable_cycles(self):
        """Rapid enable/disable/enable cycles should leave state consistent."""
        import chdb

        for _ in range(3):
            self._reset_all()
            self._chdb.set_signal_handlers_enabled(0)
            chdb.query("SELECT 1", "CSV")
            handler = _get_sigaction_handler(signal.SIGSEGV)
            self.assertEqual(handler, 0,
                "Handler must NOT be installed when disabled (cycle iteration)")

            self._chdb.set_signal_handlers_enabled(1)
            chdb.query("SELECT 1", "CSV")
            handler = _get_sigaction_handler(signal.SIGSEGV)
            self.assertNotEqual(handler, 0,
                "Handler must be installed after re-enabling (cycle iteration)")

    # Test 6: concurrent enable/disable calls are thread-safe
    def test_concurrent_enable_disable_no_crash(self):
        """Calling set_signal_handlers_enabled from multiple threads must not crash."""
        import threading
        errors = []

        def toggle():
            try:
                for i in range(50):
                    self._chdb.set_signal_handlers_enabled(i % 2)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=toggle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [],
            f"Concurrent set_signal_handlers_enabled raised: {errors}")

        self._chdb.set_signal_handlers_enabled(1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
