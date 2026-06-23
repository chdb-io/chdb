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
