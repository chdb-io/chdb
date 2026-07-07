# Contributing to chDB

Welcome — and thanks for considering a contribution. All contributors
are expected to be open, considerate, reasonable, and respectful; see
[`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md).

This is the contributor-facing guide. [`AGENTS.md`](./AGENTS.md) is
short and stays loaded by AI coding agents at all times; it captures
the design and testing principles of chdb-ds (DataStore). Everything
else — setup, build, the modify-then-test workflow, PR conventions,
CI, releases — lives here.

## What chDB is

chDB is an in-process OLAP SQL engine powered by ClickHouse. End
users do `pip install chdb` and embed the full ClickHouse SQL dialect
into their Python process — querying Parquet / CSV / JSON / Arrow /
ORC / S3 / HTTPFS sources with zero-copy interop with pandas and
PyArrow.

This repository (`chdb-io/chdb`) is the **Python layer** of chDB:
`chdb/` (small shim), `datastore/` (the chdb-ds project — a
pandas-compatible lazy API that compiles to ClickHouse SQL), and
`agent/skills/` (published Skills).

**The C++ engine itself does NOT live here.** It lives in the sibling
repository [`chdb-io/chdb-core`](https://github.com/chdb-io/chdb-core)
and is consumed as the `chdb-core` PyPI package. If a user-reported
bug or feature turns out to need engine-side changes, redirect the
work there.

### What makes chDB worth picking

chDB packs the full ClickHouse SQL surface — typed JSON, funnel /
cohort / percentile aggregates, ~80 data formats, dozens of source
connectors (S3 / MySQL / Postgres / Iceberg / Kafka / …), vector and
session primitives, plus `remoteSecure()` federation to a remote
ClickHouse cluster — into one in-process engine you load with
`import chdb`. A pandas-compatible DataFrame surface (`datastore`)
pushes the same code down to SQL with a one-line import swap.

### Where does my change go? (chdb vs chdb-core)

| What you want to change | Repo | Key path |
|---|---|---|
| DataStore / chdb-ds API, pandas compat | **chdb-io/chdb** (here) | `datastore/` |
| Top-level Python `chdb/` shim & re-exports | **chdb-io/chdb** (here) | `chdb/__init__.py`, `chdb/datastore.py` |
| Published agent skills (Cursor/Claude/Codex) | **chdb-io/chdb** (here) | `agent/skills/` |
| Bun / Go / Rust / Node / Zig / Ruby bindings | sibling repo | `chdb-io/chdb-{bun,go,rust,node,zig,ruby}` |

Anything C++ — the ClickHouse SQL engine, format readers, codecs,
storage engines, the public C ABI consumed by every binding, the
chdb-core wheel build — lives in
[chdb-io/chdb-core](https://github.com/chdb-io/chdb-core).

## Reporting issues

Search [open and closed issues](https://github.com/chdb-io/chdb/issues)
first — your problem may already have a workaround. A good new issue
includes:

- Operating system + Python version
- A minimal reproduction (the smallest snippet that still shows the
  bug)
- The expected behaviour and the observed behaviour

For performance issues, use the **Performance issue** template — it
asks for dataset shape, observed vs expected timing, and (optional)
profiling output, which speeds review significantly.

## Setting up

```bash
git clone https://github.com/chdb-io/chdb && cd chdb
python -m pip install --upgrade pip      # pip ≥ 21.3 needed for PEP 660 editable install
pip install -r requirements-dev.txt
pip install -e .
# For doc builds: pip install -r sphinx/requirements.txt
```

Pure-Python repository. No C++ build, no submodules. `pip install -e .`
pulls in the `chdb-core` engine wheel automatically. Supported:
Python 3.9–3.14 on macOS (arm64, x86_64) and Linux (x86_64, arm64).

`make wheel` is the **release** path — it shells out to
`tox -e build -- --wheel`, builds an isolated environment, and
produces an artefact in `dist/`. For development, `pip install -e .`
once is enough; iterate from there.

## I changed X — what to run

90% of contributions land in one of these buckets. Pick your row,
copy the verify command, iterate.

| You changed | Rebuild? | How to verify |
|---|---|---|
| **A. `datastore/*.py`** (most common) | No | **Full suite (default):** `cd datastore && python -m pytest tests/ -v --tb=short -x` (~1 min, ~10k tests, fail-fast). `make test` runs the same suite without `-x`. Targeted (`pytest tests/test_<file>.py`) is fine while debugging, but **the full suite is what you sign off on**. |
| **B. `datastore/tests/*.py` only** | No | Same as A |
| **C. `chdb/__init__.py` or `chdb/datastore.py` shim** | Effective yes | `pip uninstall -y chdb && pip install -e .`, restart Python, verify both `import chdb` and `from chdb import datastore` succeed |
| **D. `sphinx/`** | No | `make docs` (HTML on :8001) or `make docs-md` |
| **E. `examples/` / `benchmark/` / `refs/` scripts** | No | `python <path-to-script>.py` |
| **F. `agent/skills/*` content** | No | `bash install_skill.sh --project`, inspect `./.agents/skills/` |
| **G. C++ / SQL engine** | — | Wrong repo — go to [chdb-io/chdb-core](https://github.com/chdb-io/chdb-core) |

> ⚠️ **Prefer the full suite when in doubt.** chdb-ds has a lot of
> hidden coupling between lazy operators, expression rewrites, and
> the SQL / pandas execution split — a change that *looks* localised
> routinely surfaces a failure 50 tests away. If your change touches
> more than one row or you're not sure which code paths it reaches,
> **just run the full suite** (~1 min on a recent laptop).

**Before opening a PR**: run `ruff check datastore --statistics`
(the only PR gate) plus the full suite from row A. If both are green
locally, CI will almost always be green too.

## Things to avoid

### Capture the stack trace before changing code on a crash

When a test crashes with `SIGSEGV`, `SIGABRT`, `SIGFPE`, `SIGILL`,
`SIGBUS`, or `SIGSYS`, **always obtain the stack trace first**, then
analyse the root cause before attempting fixes. Specifically, do
*not*:

- remove the failing assertion or exception-throwing statement
- add a defensive `try/except` that swallows the crash
- mark the test as `xfail` / `skip` to make the suite go green

The crash is signalling a real bug — usually in the chdb-core engine
binding or in a refcounting / memoryview path. Capture the trace
(`gdb -ex run --args python -m pytest ...` on Linux, `lldb -- python
-m pytest ...` on macOS, or `faulthandler.enable()` for Python-side
traces), find the actual cause, and fix it where it lives. If the
cause is in `chdb-core`, file the bug there.

### Keep secrets out of test fixtures

S3 access keys, ClickHouse Cloud DSNs, OAuth tokens. Use environment
variables and skip patterns (`pytest.skip` if env not set). Tests
that require remote credentials should be runnable locally with
`nosign=true` or with a documented env-var setup.

## PR & commit conventions

**PR titles** — follow ClickHouse style: **start with a capitalised
verb, no Conventional-Commit prefix**. Describe user impact, not
internal mechanics.

Good:

- `Add timedelta64 type support for pandas DataFrame input`
- `Fix memory leak in session cleanup`
- `Improve filter pushdown for large DataStore frames`
- `Update documentation for remote connection`

Avoid:

- `feat: add timedelta64 type support` (no Conventional-Commit prefix)
- `[Feature] Add timedelta64 support` (no brackets)
- `feat(datastore): add timedelta support` (no scope notation)
- `Refactored _resolve_columns to take a span` (describes mechanics,
  not user impact)

**Commit messages** inside a PR can be lowercase as long
as the PR title itself follows the rule above. chdb uses merge
commits (not squash), so each commit title still lives in `git log`
forever — keep them readable.

**Branch names** — descriptive with a category prefix:
`fix/timedelta-support`, `feat/datastore-session`, `docs/update-readme`,
`refactor/cleanup-imports`. External contributors may also use
`<github-handle>/<topic>`.

**Scope** — one concern per PR. Split refactors away from fixes;
mixed PRs slow review and complicate bisection.

**Tests** — every behaviour change comes with a test. See
[`AGENTS.md`](./AGENTS.md) §4 for the DataStore mirror-code testing
pattern.

## Documentation changes

`sphinx/` (the legacy Sphinx site) is built from the `sphinx/` directory. The Sphinx-side dependencies aren't in
`requirements-dev.txt`; install them once before building:

```bash
pip install -r sphinx/requirements.txt
make docs        # builds HTML and serves on :8001
make docs-md     # builds markdown into buildlib/markdowndocs/
```

For doc-only PRs, the test suite still has to be green, but you can
skip running it locally if you only touched `sphinx/`.

## CI

The build-and-test matrix:

- **Python**: 3.9, 3.10, 3.11, 3.12, 3.13, 3.14
- **pandas**: `pandas<3.0` always; `pandas>=3.0` additionally on
  Python ≥ 3.11
- **Platforms**: 4 (Linux x86_64, Linux arm64, macOS x86_64,
  macOS arm64)

To reproduce CI locally:

```bash
ruff check datastore --statistics
cd datastore && python -m pytest tests/ -v --tb=short -x
```

## Security

- No secrets in test fixtures (see "Keep secrets out of test
  fixtures" above).
- Capture stack traces for engine crashes.
- Report security issues via GitHub Security Advisories on
  `chdb-io/chdb`, not public issues.

## Maintainer release flow

Wheels are built by GitHub Actions on tag pushes. The high-level
steps for a release:

1. Confirm `main` is green across the full CI matrix.
2. Tag the release commit: `git tag v<version>` and push
   `git push upstream v<version>`.
3. The release workflow builds and uploads to PyPI automatically.

For ad-hoc local wheels (not a release), `make wheel` produces an
artefact in `dist/`.

## Related files

- [`AGENTS.md`](./AGENTS.md) — chdb-ds design and testing principles
  (LLM-facing, kept short)
- [`dev-docs/ARCHITECTURE.md`](./dev-docs/ARCHITECTURE.md) — high-level
  DataStore architecture
- [`dev-docs/PANDAS_COMPATIBILITY.md`](./dev-docs/PANDAS_COMPATIBILITY.md)
- User docs: <https://chdb.readthedocs.io/en/latest/index.html>

When asked to *use* chDB, the readthedocs site is canonical. When
asked to *contribute to* chDB, this file is canonical.
