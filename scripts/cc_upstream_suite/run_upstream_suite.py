#!/usr/bin/env python3
"""
Run clickhouse-connect's *own* test suite against the embedded chDB backend.

This is the chDB analogue of the chdb-node upstream harness (PR #49) that ran clickhouse-js's
own integration specs against embedded chDB. The design proposal (§8.4) puts this in the
chDB repository's CI -- clickhouse-connect's CI never touches chDB; chDB's CI runs
clickhouse-connect's suite against ``ChdbBackend`` and owns the skip list.

Mechanism:

1. Resolve the clickhouse-connect checkout: ``--cc-repo PATH`` for a local checkout, else
   clone ``https://github.com/ClickHouse/clickhouse-connect`` at the tag matching the
   installed clickhouse-connect version.
2. Drop ``_force_chdb_conftest.py`` in as the checkout's root ``conftest.py`` so every
   ``create_client(...)`` builds a chDB-backed client (see that file's docstring).
3. Run the selected tests with the skip-list and expected-divergence (xfail) lists applied.

Gate: pytest exits non-zero on any *unexpected* failure -- a failure not covered by the
skip-list and not marked as a documented divergence is a real byte-compat regression. An
xpass (a documented divergence that started passing) also fails, by strict xfail.

Usage:
    python run_upstream_suite.py --cc-repo /path/to/clickhouse-connect [-- pytest args]
    python run_upstream_suite.py            # clone the installed version's tag
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP_FILE = HERE / "skip_list.txt"
XFAIL_FILE = HERE / "expected_divergences.txt"
INJECTED_CONFTEST = HERE / "_force_chdb_conftest.py"
REPO_URL = "https://github.com/ClickHouse/clickhouse-connect"


def _installed_cc_version() -> str:
    import importlib.metadata as md

    return md.version("clickhouse-connect")


def _clone_tag(version: str, dest: Path) -> None:
    for ref in (f"v{version}", version):
        rc = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, REPO_URL, str(dest)],
            capture_output=True, text=True,
        )
        if rc.returncode == 0:
            print(f"Cloned clickhouse-connect @ {ref}")
            return
    raise SystemExit(f"Could not clone clickhouse-connect at v{version}/{version}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc-repo", help="Path to a local clickhouse-connect checkout (else clone the installed tag)")
    parser.add_argument("--tests", default="tests/unit_tests",
                        help="pytest target path inside the checkout (default: tests/unit_tests). "
                             "Use tests/integration_tests for the full server-parity suite.")
    parser.add_argument("--keep", action="store_true", help="Keep the cloned/working checkout")
    parser.add_argument("pytest_args", nargs="*", help="Extra args passed through to pytest")
    args = parser.parse_args()

    tmp = None
    if args.cc_repo:
        # Work on a copy so we don't drop a conftest into the user's checkout.
        tmp = Path(tempfile.mkdtemp(prefix="cc-upstream-"))
        repo = tmp / "clickhouse-connect"
        shutil.copytree(args.cc_repo, repo, ignore=shutil.ignore_patterns(".git", "build", "*.egg-info", "__pycache__"))
    else:
        tmp = Path(tempfile.mkdtemp(prefix="cc-upstream-"))
        repo = tmp / "clickhouse-connect"
        _clone_tag(_installed_cc_version(), repo)

    # Inject the redirect conftest at the checkout root (loaded before the integration conftest).
    shutil.copyfile(INJECTED_CONFTEST, repo / "conftest.py")

    env = dict(os.environ)
    env["CHDB_SUITE_SKIP_FILE"] = str(SKIP_FILE)
    env["CHDB_SUITE_XFAIL_FILE"] = str(XFAIL_FILE)
    # Force single-process: the redirect monkeypatch must apply in the test process.
    cmd = [
        sys.executable, "-m", "pytest", *args.tests.split(),
        "-p", "no:cacheprovider", "-p", "no:xdist", "-o", "addopts=",
        "-rsxX", *args.pytest_args,
    ]
    print(f"Running: {' '.join(cmd)}\n  cwd={repo}\n  skip_list={SKIP_FILE.name} xfail={XFAIL_FILE.name}")
    rc = subprocess.run(cmd, cwd=repo, env=env)

    if not args.keep and tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return rc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
