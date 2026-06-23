"""
Injected at the root of a cloned clickhouse-connect checkout so its *own* test suite runs
against the embedded chDB backend instead of an HTTP server.

This is the chDB analogue of the chdb-node upstream harness redirecting clickhouse-js's
single client factory to ``chdb://memory``. clickhouse-connect funnels every client through
``clickhouse_connect.driver.create_client`` / ``create_async_client``; we replace those at
import time (this root conftest is imported before the integration conftest binds the
names) so each ``create_client(host=..., port=...)`` call instead builds a chDB-backed
client. HTTP-transport-only keyword arguments (host, port, auth, compression, pool, proxy,
TLS, ...) are dropped, since the embedded engine has no concept of them.

Skip handling is entirely data-driven (no pytest markers): an empirically-built nodeid
blacklist (``skip_list.txt``) skips cases for capabilities the embedded engine genuinely
cannot support, plus an optional expected-divergences list (``expected_divergences.txt``)
marks documented behavior differences strict-xfail. Both are owned in this repo and never
require a clickhouse-connect change.
"""

from __future__ import annotations

import os

import clickhouse_connect.driver as _ccd

# Keyword arguments that only make sense for the HTTP transport; the chDB backend ignores
# them, so strip before delegating (passing host=... to the chdb factory would be noise).
_HTTP_ONLY = {
    "host", "port", "interface", "secure", "dsn", "user", "username", "password",
    "access_token", "token_provider", "compress", "client_name", "pool_mgr", "http_proxy",
    "https_proxy", "verify", "ca_cert", "client_cert", "client_cert_key", "server_host_name",
    "connect_timeout", "send_receive_timeout", "headers", "connector_limit",
    "connector_limit_per_host", "keepalive_timeout",
}

# Where the redirected `create_client` opens its chdb engine. By default (and per chdb-core's
# process-singleton constraint) every redirected client in the same run shares the same
# underlying engine via cc_backend.py's connection cache -- this is intentional and matches
# how clickhouse-connect's session fixture treats the HTTP backend (one session-scoped
# database name per pytest session, used across all tests). Cross-test isolation comes from
# clickhouse-connect's existing fixture design (the session fixture creates a
# `ch_connect__<random>__<ts>` test database) rather than per-test engine instances; the few
# cases where that assumption breaks are catalogued in skip_list.txt.
#
# Override with CHDB_UPSTREAM_SUITE_PATH for a dedicated on-disk path; the runner sets this
# automatically to a temp dir per invocation so consecutive runs do not share data on disk.
_CHDB_PATH = os.getenv("CHDB_UPSTREAM_SUITE_PATH", ":memory:")

_orig_create_client = _ccd.create_client
_orig_create_async_client = _ccd.create_async_client


def _strip(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in _HTTP_ONLY}


def _chdb_create_client(**kwargs):
    return _orig_create_client(backend="chdb", path=_CHDB_PATH, **_strip(kwargs))


async def _chdb_create_async_client(**kwargs):
    return await _orig_create_async_client(backend="chdb", path=_CHDB_PATH, **_strip(kwargs))


# Replace the attributes on the driver package *before* the integration conftest does its
# `from clickhouse_connect.driver import create_client`, so it binds these wrappers.
_ccd.create_client = _chdb_create_client
_ccd.create_async_client = _chdb_create_async_client

import clickhouse_connect as _cc  # noqa: E402

_cc.create_client = _chdb_create_client
_cc.create_async_client = _chdb_create_async_client
_cc.get_client = _chdb_create_client
_cc.get_async_client = _chdb_create_async_client


def _load_patterns(env_var: str) -> list[str]:
    """Read newline-separated patterns from the file named by ``env_var`` (``#`` comments ok)."""
    path = os.getenv(env_var)
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


def pytest_collection_modifyitems(config, items):
    """Apply the empirical skip / xfail gates.

    * ``CHDB_SUITE_SKIP_FILE``  -- newline-separated nodeid substrings skipped outright,
      one per case the embedded engine genuinely cannot satisfy (capability-not-supported).
      Owned in this repo as ``skip_list.txt`` and curated *empirically* by running the full
      clickhouse-connect integration suite twice (HTTP server vs chDB) and adding only the
      cases that pass over HTTP and genuinely cannot work embedded.
    * ``CHDB_SUITE_XFAIL_FILE`` -- substrings marked ``xfail(strict=True)``: documented
      embedded-vs-server behavior differences that still run and MUST fail; a divergence
      that silently disappears (an xpass) breaks the build.
    """
    import pytest

    skip_patterns = _load_patterns("CHDB_SUITE_SKIP_FILE")
    xfail_patterns = _load_patterns("CHDB_SUITE_XFAIL_FILE")
    for item in items:
        nodeid = item.nodeid
        if any(p in nodeid for p in skip_patterns):
            item.add_marker(pytest.mark.skip(reason="chdb: capability not supported by embedded engine"))
            continue
        if any(p in nodeid for p in xfail_patterns):
            item.add_marker(pytest.mark.xfail(reason="chdb: documented embedded-vs-server divergence", strict=True))
