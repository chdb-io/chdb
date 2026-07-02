"""
Shared fixtures for the chDB clickhouse-connect backend tests.

Mirrors clickhouse-connect's own root test conftest: it pins the process timezone to UTC
and resets clickhouse-connect's global format/timezone state between tests. The backend
inherits clickhouse-connect's DateTime handling unchanged, so the same deterministic-UTC
test environment the upstream suite assumes must be in force here too (otherwise naive-UTC
DateTime assertions pick up the machine's local zone).
"""

import os
import time
from datetime import timezone

import pytest

os.environ["TZ"] = "UTC"
# time.tzset() is POSIX-only; on Windows the TZ env var still steers strftime/strptime
# behavior in the C runtime, so we set TZ and only call tzset() when it exists. The chDB
# backend is not supported on Windows anyway (cc_backend.py raises NotSupportedError on win32),
# so the chDB backend tests will skip there — but we still want the module to import cleanly.
if hasattr(time, "tzset"):
    time.tzset()


@pytest.fixture(autouse=True)
def _clean_clickhouse_connect_global_state():
    try:
        from clickhouse_connect.datatypes.format import clear_all_formats
        from clickhouse_connect.driver import tzutil
    except Exception:  # noqa: BLE001 -- clickhouse-connect not installed; module is skipped anyway
        yield
        return
    clear_all_formats()
    tzutil.local_tz = timezone.utc
    yield
