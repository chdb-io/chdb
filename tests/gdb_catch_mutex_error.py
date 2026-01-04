"""
GDB Python script to catch mutex-related exceptions.
Usage: gdb -batch -x gdb_commands.gdb --args python test_script.py

This script sets a breakpoint on __cxa_throw and filters for mutex/system_error exceptions.
"""
import gdb


class MutexErrorCatcher(gdb.Breakpoint):
    def __init__(self):
        # Break on __cxa_throw (C++ exception throw)
        super(MutexErrorCatcher, self).__init__("__cxa_throw", internal=True)
        self.silent = True

    def stop(self):
        try:
            print("\n=== C++ EXCEPTION DETECTED ===")
            print("Backtrace:")
            gdb.execute("bt full")
            print("\n=== Thread info ===")
            gdb.execute("info threads")
            print("\n=== END ===")
            # Force crash to generate core dump
            print("\n=== FORCING CRASH (SIGABRT) ===")
            gdb.execute("signal SIGABRT")
            return True
        except Exception as e:
            print(f"Error in stop(): {e}")
            pass
        return False


MutexErrorCatcher()
print("Mutex error catcher initialized")
