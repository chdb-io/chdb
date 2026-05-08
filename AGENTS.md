# AGENTS.md — chDB

This file gives AI coding agents (Codex, Cursor, Claude Code, Aider,
Zed, Jules, Devin, Copilot, Gemini CLI, ...) the project-specific context
they need to make safe, useful changes to this repository.

It follows the [agents.md](https://agents.md) open standard, stewarded by
the Linux Foundation's Agentic AI Foundation. Read it before doing
anything destructive (running full test suites, regenerating wheels,
mass-rewriting files, etc.).

> **Disambiguation — `AGENTS.md` vs `agent_skills/`**
> This repository contains a top-level directory called `agent_skills/`
> that ships **chDB's own published AI coding skills** — the artifacts
> that end users install into their Cursor / Claude Code / Codex via
> `install_skill.sh`. That directory is *content shipped outward to
> users*. **This file (`AGENTS.md`)** is *guidance inward for agents
> working on this repo's source*. Don't conflate the two.

---

## 1. What chDB is (60-second context)

chDB is an in-process OLAP SQL engine powered by ClickHouse. End users
do `pip install chdb` and embed the full ClickHouse SQL dialect into
their Python process — querying Parquet / CSV / JSON / Arrow / ORC / S3
/ HTTPFS sources with zero-copy interop with pandas and PyArrow.

This repository (`chdb-io/chdb`) is the **Python layer** of chDB:

- `chdb/` — a small shim (2 files) that bridges the `chdb` package
  namespace to the engine
- `datastore/` — the **chdb-ds** project, a pandas-compatible lazy API
  that compiles to ClickHouse SQL or falls back to pandas
- `agent_skills/` — chDB's published Skills for AI coding agents
  (chdb-datastore, chdb-sql)
- `examples/`, `benchmark/`, `docs/`, `refs/` — supporting material

**The C++ engine itself does NOT live here.** It lives in the sibling
repository [`chdb-io/chdb-core`](https://github.com/chdb-io/chdb-core)
and is consumed as the `chdb-core` PyPI package (declared as a
dependency in `pyproject.toml`). If a user-reported bug or feature
turns out to need engine-side changes, redirect the work there — do
not patch around it from this repository.

---

## 2. Things agents should NOT do (read first — this is the highest-ROI section)

These are concrete, repository-specific failure modes that will burn
real reviewer time or break installations. They are placed at the top
because they are easy to violate by default.

### 2.1 Do not casually refactor the `chdb/` shim package

The `chdb/` directory contains exactly two files, both load-bearing:

- **`chdb/__init__.py`** walks `sys.path` to add `chdb-core`'s
  `chdb/` directory to its own `__path__`. This is what lets an
  editable install (`pip install -e .`) coexist with the
  `chdb-core` engine package in `site-packages` without one
  shadowing the other.
- **`chdb/datastore.py`** is a shim: on first import, it patches
  `chdb.__version__` to the *chdb* pip-package version (rather than
  the `chdb-core` engine version), then replaces itself in
  `sys.modules` so that `from chdb import datastore` transparently
  resolves to the top-level `datastore` package.

Both files look "weird" to an agent that thinks it's tidying things
up. **Don't tidy them.** Removing the `sys.path` walk breaks
`import chdb` for editable installs. Reordering the version-patching
in the shim shows the wrong version string to users.

### 2.2 Do not edit the `chdb-core` engine from this repository

There is no C++ source tree, no `programs/`, no `src/`, no `contrib/`,
no ClickHouse git submodule in this repository. Don't try to invent
one. SQL engine, query parser, format readers, storage engines,
codecs — all of that is in [`chdb-io/chdb-core`](https://github.com/chdb-io/chdb-core).

If you find yourself wanting to change a SQL function's behavior, a
data-type cast, a Parquet reader option, or anything else that isn't
pure Python composition: stop, and redirect the change to
`chdb-io/chdb-core`. From this repository, the right move is usually
a feature request or a `pyproject.toml` minimum-version bump after
the fix lands upstream.

### 2.3 Do not run a full DataStore test suite reflexively

`datastore/tests/` contains 267 test files. A full run takes tens of
minutes and exercises the chdb-core engine extensively. **Don't run
it as a sanity check** when you've changed two lines.

Default to targeted runs:

```bash
cd datastore && python -m pytest tests/test_<file>.py -v --tb=short
cd datastore && python -m pytest tests/test_<file>.py::TestClass::test_method -v
```

The full run is what CI is for. Run it locally only when you've
finished a non-trivial change and are about to push.

### 2.4 Do not run `make wheel` for ordinary development

`make wheel` is the **release** path: it shells out to
`tox -e build -- --wheel`, which builds an isolated environment and
produces an artifact in `dist/`. For development, the right command
is `pip install -e .` once, then run tests directly.

### 2.5 Do not refactor large files for "readability"

`datastore/core.py` (~280 KB), `datastore/column_expr.py` (~248 KB),
and `datastore/function_definitions.py` (~391 KB) are large for
reasons. Some sections are auto-generated patterns; others are
performance-critical hot paths with deliberately localized control
flow. **Don't propose splitting them, renaming methods, or extracting
helpers** unless you have an issue or maintainer-approved RFC for
that specific change. Reviewer-facing changes that just shuffle
lines around will be rejected.

### 2.6 Do not bypass crash diagnosis

When a test crashes with `SIGSEGV`, `SIGABRT`, `SIGFPE`, `SIGILL`,
`SIGBUS`, or `SIGSYS`, **always obtain the stack trace first**, then
analyze the root cause before attempting fixes.

Specifically, do *not*:

- remove the failing assertion or exception-throwing statement
- add a defensive `try/except` that swallows the crash
- mark the test as `xfail` / `skip` to make the suite go green

The crash is signaling a real bug — usually in the chdb-core engine
binding or in a refcounting / memoryview path. Capture the trace
(`gdb python -c run -- -m pytest ...` on Linux,
`lldb -- python -m pytest ...` on macOS, or just enabling
`faulthandler.enable()` for Python-side traces), find the actual
cause, and fix it where it lives. If the cause is in `chdb-core`,
file the bug there.

### 2.7 Do not follow the inherited PR template literally

`.github/PULL_REQUEST_TEMPLATE.md` historically inherited
ClickHouse-upstream content (changelog categories like "Performance
Improvement", references to ASAN/TSAN/UBSAN test batches, ClickHouse
documentation links). It is being replaced — but if you happen to
encounter the old form, **don't try to populate every checkbox**.
Use the conventions in §10 of this file instead.

### 2.8 Do not follow `CONTRIBUTING.md` literally for build/docs commands

`CONTRIBUTING.md` is general-purpose contribution etiquette and was
generated from a template. Some commands in it are stale; for
example, it mentions `tox -e docs` for documentation, but the
working entry point is `make docs`. When `CONTRIBUTING.md` and this
file disagree on a *command*, this file wins. When they disagree on
*social conventions* (be respectful, file an issue first, ...),
`CONTRIBUTING.md` wins.

### 2.9 Do not add new lint / type-check gates without an issue first

`setup.cfg` deliberately limits flake8 to critical errors only
(`select = E9,F63,F7,F82`). `requirements-dev.txt` lists `black`,
`mypy`, and `pre-commit`, but **none of them are PR gates** today.
Adding a new gate (e.g. enforcing `black`, turning on `mypy --strict`,
migrating flake8 → ruff) is a project-wide decision; open an issue
and let maintainers weigh in. A drive-by PR that imposes a new
formatter on a 1.5 MB Python codebase will be rejected on process
grounds.

### 2.10 Do not commit secrets in test fixtures

S3 access keys, ClickHouse Cloud DSNs, OAuth tokens. Use environment
variables and skip patterns (`pytest.skip` if env not set). Tests that
require remote credentials should be runnable locally with `nosign=true`
or with a documented env-var setup.

### 2.11 Do not bump dependency minimums without testing both ends

CI runs the test suite against **both** `pandas<3.0` and `pandas>=3.0`
on Python 3.11+ (see §11). If you change a `requirements.txt` minimum
or a `pyproject.toml` dependency pin, run the matrix locally on at
least one Python version against both pandas constraints before
opening the PR.

---

## 3. Repository layout

```
chdb/                          ← repository root
│
├── chdb/                      thin shim package — DO NOT casually refactor (§2.1)
│   ├── __init__.py            sys.path bridge so editable installs find chdb-core
│   └── datastore.py           shim: `from chdb import datastore` → top-level datastore
│
├── datastore/                 the chdb-ds project (~1.5 MB Python, 35 modules)
│   ├── core.py                       (~280 KB) DataStore main class
│   ├── column_expr.py                (~248 KB) column-expression DSL
│   ├── function_definitions.py       (~391 KB) generated function tables
│   ├── pandas_compat.py              (~118 KB) pandas-API compatibility
│   ├── sql_executor.py               (~110 KB) SQL execution backend
│   ├── lazy_ops.py                   (~92 KB)  lazy operation system
│   ├── pandas_api.py                 (~72 KB)  pandas API surface
│   ├── connection.py, query_planner.py, expression_evaluator.py, ...
│   ├── accessors/                    .str / .dt / list / array / ... accessors
│   ├── dtype_correction/             pandas/ClickHouse dtype reconciliation
│   └── tests/                        267 test files + conftest.py + dataset/ + test_utils.py
│
├── agent_skills/              chDB's published AI coding Skills (NOT this file's audience!)
│   ├── chdb-datastore/        SKILL.md + references/ + examples/ + scripts/
│   └── chdb-sql/              SKILL.md + references/ + examples/ + scripts/
│
├── benchmark/                 microbenchmarks (clickbench.py, chdb_vs_pandas_memory.py, ...)
├── examples/                  user-facing examples (.py and .ipynb)
├── docs/                      Sphinx documentation source
│   ├── ARCHITECTURE.md, PANDAS_COMPATIBILITY.md, REMOTE_SESSION_DESIGN.md, ...
│   ├── api.rst, examples.rst, installation.rst, quickstart.rst, ...
│   └── conf.py, requirements.txt
├── refs/                      reference material (ClickHouse formats, table functions, ...)
│
├── .cursor/rules/             Cursor-specific shells that delegate to this file
├── .github/
│   ├── workflows/             CI (see §11)
│   ├── ISSUE_TEMPLATE/        bug, feature, question
│   ├── PULL_REQUEST_TEMPLATE.md   project-specific (see §2.7)
│   └── FUNDING.yml
│
├── CLAUDE.md                  chdb-ds design philosophy + testing rigor — see §9
├── CONTRIBUTING.md            social etiquette + sphinx-build hints (note §2.8)
├── CODE_OF_CONDUCT.md
├── README.md, README-zh.md
├── pyproject.toml             package metadata; runtime dep on `chdb-core>=26.1.0`
├── tox.ini, setup.cfg
├── requirements.txt           runtime: chdb-core, pandas, pyarrow
├── requirements-dev.txt       dev: pytest, pytest-cov, black, flake8, mypy, pre-commit, twine, ...
├── Makefile                   wheel | docs | docs-md | test | clean | pub
└── install_skill.sh           user-facing installer for agent_skills/
```

---

## 4. Setup (first time)

This is a pure-Python repository. There is no C++ build, no submodule
init, no CMake, no ninja, no LLVM. The full setup is three commands:

```bash
git clone https://github.com/chdb-io/chdb
cd chdb
pip install -r requirements-dev.txt
pip install -e .
```

`pip install -e .` will pull in `chdb-core` (the engine, ~30–50 MB
wheel) from PyPI as a transitive dependency. After that, `import chdb`
and `from chdb import datastore` work, and the test suite is
runnable.

A virtual environment is recommended:

```bash
python -m venv .venv && source .venv/bin/activate
# or: conda create -n chdb python=3.11 && conda activate chdb
pip install -r requirements-dev.txt
pip install -e .
```

Supported Python versions: 3.9, 3.10, 3.11, 3.12, 3.13, 3.14
(see CI matrix in §11). Supported platforms: macOS (arm64, x86_64) and
Linux (x86_64, arm64).

---

## 5. The four scenarios — which command set to reach for

90% of contributions land in one of these four buckets. The right
command set is different for each.

| You changed | Need to rebuild? | Install command | Test command |
|---|---|---|---|
| **A. Anything under `datastore/*.py`** (the common case) | No | `pip install -e .` once, then iterate | `cd datastore && python -m pytest tests/test_<file>.py -v --tb=short` |
| **B. `datastore/tests/*.py` only** | No | nothing | same as A |
| **C. `chdb/__init__.py` or `chdb/datastore.py` shim** | Effectively yes | `pip uninstall -y chdb && pip install -e .` (forces a clean re-resolve) | restart Python; verify `import chdb` and `from chdb import datastore` |
| **D. `docs/`** | No | nothing | `make docs` (renders + serves on :8001) or `make docs-md` |
| **E. `examples/` / `benchmark/` / `refs/` standalone scripts** | No | nothing | `python examples/<file>.py` |
| **F. `agent_skills/*` content** | No | nothing | `bash install_skill.sh --project` and inspect `./.agents/skills/` |
| **G. C++ / SQL engine logic** | — | this is the wrong repo | go to [chdb-io/chdb-core](https://github.com/chdb-io/chdb-core) |

If your change spans multiple buckets, run them in order top-to-bottom.

---

## 6. Build & test commands (paste-able)

### 6.1 Run a targeted test

```bash
# single file
cd datastore && python -m pytest tests/test_groupby.py -v --tb=short

# single test class
cd datastore && python -m pytest tests/test_groupby.py::TestBasicGroupby -v

# single test
cd datastore && python -m pytest tests/test_groupby.py::TestBasicGroupby::test_simple_count -v

# pattern across files
cd datastore && python -m pytest tests/ -k "groupby and not slow" -v --tb=short
```

### 6.2 Run the full DataStore suite (slow — minutes to an hour)

Use this only before pushing or after a substantial change:

```bash
cd datastore && python -m pytest tests/ -v --tb=short
```

Add `-x` for fail-fast (matches the CI behavior):

```bash
cd datastore && python -m pytest tests/ -v --tb=short -x
```

### 6.3 Lint (the only PR gate)

This is what the `pr_ci.yaml` workflow runs on every PR:

```bash
flake8 datastore --count --show-source --statistics
```

Configuration is in `setup.cfg`. Note that the rule selection is
deliberately limited to `E9,F63,F7,F82` (genuine errors only — not
style). See §2.9 for why.

### 6.4 Documentation

```bash
make docs        # build HTML + start http.server on :8001
make docs-md     # build markdown into buildlib/markdowndocs/
```

The Sphinx config lives in `docs/conf.py`; sources include
`*.rst` files and several `*.md` design documents
(`ARCHITECTURE.md`, `PANDAS_COMPATIBILITY.md`, `EXPLAIN_METHOD.md`,
`PROFILING.md`, `REMOTE_SESSION_DESIGN.md`, ...).

### 6.5 Release wheel (do NOT run for ordinary development)

```bash
make wheel       # → tox -e build -- --wheel
make clean       # remove buildlib/, dist/, *.egg-info
make pub         # → tox -e publish (uploads to PyPI; maintainer-only)
```

### 6.6 Local environment hygiene

If a stale install is misbehaving:

```bash
pip uninstall -y chdb chdb-core
pip cache purge
pip install -r requirements-dev.txt
pip install -e .
```

---

## 7. Code style

Style enforcement is intentionally light:

- **Line length**: 120 characters max (`setup.cfg [flake8]`).
- **Cyclomatic complexity**: 10 max (advisory; flake8 only flags
  critical errors today).
- **Imports**: Top-of-file. `__init__.py` is exempt from F401
  ("imported but unused") because re-exports are intentional.
- **`black` and `mypy`** are listed in `requirements-dev.txt` but are
  **not** PR gates today (see §2.9).
- **Conventional commits** are **not** used in PR titles (see §10).

Match the surrounding code's style. Don't run `black` over files
whose neighbors aren't black-formatted — you'll create noise diffs
that derail the actual review.

---

## 8. Testing instructions (the chdb-ds testing rigor)

chdb-ds has unusually strict testing principles, distilled in
`CLAUDE.md` and mirrored in `.cursor/rules/chdb-ds.mdc`. **Read
CLAUDE.md before writing any test.** This section summarizes what
agents most often get wrong.

### 8.1 Tests must mirror DataStore and pandas operations

Every chdb-ds test should write *both* the pandas form and the
DataStore form of the same operation, then compare:

```python
from tests.test_utils import assert_datastore_equals_pandas

# pandas
pd_df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
pd_result = pd_df[pd_df['age'] > 20].sort_values('name')

# DataStore (mirror)
ds_df = DataStore({'name': ['Alice', 'Bob'], 'age': [25, 30]})
ds_result = ds_df[ds_df['age'] > 20].sort_values('name')

# Comparison: columns + values + row order
assert_datastore_equals_pandas(ds_result, pd_result)
```

Available helpers in `datastore/tests/test_utils.py`:

- `assert_datastore_equals_pandas(ds, pd, check_row_order=True)`
- `assert_column_values_equal(ds, pd, 'col_name')`
- `assert_columns_match(ds, pd)`
- `assert_row_count_match(ds, pd)`

For unordered results (e.g. `groupby` without `sort=True`):

```python
assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
```

### 8.2 Forbidden test patterns

These are real anti-patterns the project has fought, not generic
advice:

- ❌ Comments describing expected behavior **without an actual
  assertion**
- ❌ `print()` or `logging` calls **without a corresponding
  assertion**
- ❌ Asserting only `len(result)` and skipping the actual values
- ❌ "Verify X" or "TODO: verify later" comments — verify, or delete
- ❌ `reset_index()` to mask a DataStore index bug — that's a
  DataStore bug, fix it instead

### 8.3 Test names must reflect what is being verified

```python
# BAD
def test_complex_pipeline(self): ...

# GOOD
def test_sql_pandas_sql_segments_exact_values_and_structure(self): ...
```

### 8.4 SQL execution verification via captured logs

For tests that care about which SQL was actually executed:

```python
log_output = log_capture.getvalue()
self.assertIn('WHERE "value" > 30', log_output)
self.assertIn('Segment 1/', log_output)
self.assertIn('[Pandas]', log_output)
```

### 8.5 Self-check before submitting any test

- Does every "expected behavior" comment have a corresponding
  assertion?
- Are the actual data **values** verified, not just lengths?
- Are the DataStore and pandas code paths mirrored (same operations,
  same style)?
- Is column **order** verified — not just column names as a set?
- Is row **order** verified for order-preserving operations?
- Is the segment structure (type, ops, `is_first_segment`) verified
  where relevant?
- Are error messages descriptive enough to debug a failure from CI
  output alone?
- Would this test catch a real bug, or would it pass trivially?

If any of these is "no", the test isn't done.

---

## 9. The chdb-ds design rules (the "what" of correct code)

`CLAUDE.md` is the source of truth for chdb-ds *design philosophy*.
This file is the source of truth for *contributing process*. They
overlap deliberately — design rules constrain how an agent should
write a fix.

The four design rules in one paragraph each (full text in
`CLAUDE.md`):

1. **Fully lazy execution.** All methods returning DataFrame/Series
   should return Lazy objects (`DataStore`, `LazySeries`, ...). Defer
   execution until results are actually needed. The execution-engine
   choice (pandas vs chdb-core SQL) is made at the final execute
   stage by the config system; API style does not determine engine.

2. **Natural execution triggers — no explicit `_execute()`.**
   Execution should be triggered by `.columns`, `len()`, `.index`,
   `repr()`, `__iter__`, `.equals()`, etc. Avoid explicit
   `to_df()` / `to_list()` / `to_pandas()` in user-facing code paths
   unless interop is the point.

3. **Unified architecture, simplicity first.** No backward
   compatibility constraint; no split class hierarchies for different
   engines; `ColumnExpr` uniformly wraps all expression types;
   `LazyOp` system uniformly manages all lazy operations.

4. **Testing rigor (see §8).** Discovered problems are opportunities
   to fix the library, not to weaken the test. Don't reach for
   `reset_index()` to make a test pass.

---

## 10. PR & commit conventions

### 10.1 PR and commit titles

chDB uses [Conventional Commits](https://www.conventionalcommits.org/)
prefixes. The recent history is consistent on this — every merged
commit in the last several months follows the format.

Common types:

- `feat:` — user-visible new feature
- `fix:` — bug fix
- `refactor:` — non-behavior-changing code restructure
- `test:` — tests-only change
- `docs:` — documentation-only change
- `ci:` — CI / build infrastructure
- `chore:` — tooling, dependencies, housekeeping

Optional scope tag for the affected component:

- `(datastore)` — chdb-ds module
- `(ci)` — CI workflows
- `(chdb)` — the shim package or chdb-core integration

Examples drawn from the chDB history:

- `feat(datastore): use streaming DataFrame execution to reduce memory peak by 80%`
- `fix(datastore): prevent GC-triggered SIGABRT during streaming and fix pandas 3.x compat`
- `ci: skip flaky groupby_agg_sum timing test on macOS x86_64`
- `refactor(datastore): deduplicate SQL helpers, remove hasattr`

The title should describe **user impact**, not internal mechanics.
"Fix groupby on nullable Int64" is useful; "Refactor
`_resolve_column` to take a list" is not.

Each commit's title should stand on its own. chDB's history uses
merge commits (not squash), so individual commit titles stay in
`git log` forever — write each one like the readers of `git blame`
five years from now will be its primary audience.

### 10.2 Branch names

Use either of these patterns — pick what makes the branch
self-explanatory:

- `<github-handle>/<topic>` — common for external contributors
  (e.g. `nklmish/fix-url-headers-positional`,
  `wudidapaopao/add_benchmark_tests`)
- `<type>/<topic>` — common for maintainers
  (e.g. `fix/timedelta-support`, `feat/datastore-session`,
  `docs/update-readme`, `refactor/cleanup-imports`)

### 10.3 Commit body

The PR title carries the user-facing summary, so commit bodies are
usually empty in chDB's history. If you do write a body, focus on
*why* not *what* — the diff already shows what.

### 10.4 Scope discipline — one feature per PR

If you're tempted to bundle a refactor with a fix, split it. Mixed
PRs slow review and complicate bisection. Sole exception: a small
test added with the fix.

### 10.5 Tests with PRs

Every behavior change requires a test. Bug fixes require a regression
test that fails before the change. Document-only PRs are exempt.

---

## 11. CI workflows

`.github/workflows/` contains five files:

| File | Trigger | What it runs |
|---|---|---|
| `pr_ci.yaml` | PR opened / synchronize | `flake8 datastore` (this is the only PR gate that blocks merge) |
| `build_linux_x86_wheels.yml` | push to main / PR / release / manual | pyenv install Py 3.9–3.14 → `make wheel` → for each Python × pandas constraint, `cd datastore && pytest tests/ -v --tb=short -x`; on tag, `gh release upload` + `twine upload` |
| `build_linux_arm64_wheels-gh.yml` | same | same matrix on Linux arm64 |
| `build_macos_x86_wheels.yml` | same | same on macOS x86_64 |
| `build_macos_arm64_wheels.yml` | same | same on macOS arm64 |

The matrix on the build-and-test workflows is:

- **Python**: 3.9, 3.10, 3.11, 3.12, 3.13, 3.14
- **pandas**: `pandas<3.0` always; `pandas>=3.0` additionally on Python ≥ 3.11
- **Platforms**: 4 (Linux x86_64, Linux arm64, macOS x86_64, macOS arm64)

That's roughly 40 wheel-and-test runs per push to main. Don't add
matrix dimensions casually.

To reproduce CI locally with the minimum command set:

```bash
flake8 datastore --count --show-source --statistics
cd datastore && python -m pytest tests/ -v --tb=short -x
```

---

## 12. Cross-references

### 12.1 Related repositories in the chDB ecosystem

| Repository | What it is |
|---|---|
| [`chdb-io/chdb-core`](https://github.com/chdb-io/chdb-core) | C++ engine (ClickHouse fork + pybind11). All SQL engine, parser, format reader, codec, storage changes go here. |
| [`chdb-io/chdb-bun`](https://github.com/chdb-io/chdb-bun) | Bun/TypeScript bindings |
| [`chdb-io/chdb-go`](https://github.com/chdb-io/chdb-go) | Go bindings |
| [`chdb-io/chdb-rust`](https://github.com/chdb-io/chdb-rust) | Rust bindings |
| [`chdb-io/chdb-node`](https://github.com/chdb-io/chdb-node) | Node.js bindings |

### 12.2 In-repo files agents should know about

- [`./CLAUDE.md`](./CLAUDE.md) — chdb-ds design philosophy &
  testing rigor (read before writing tests; see §8 and §9)
- [`./CONTRIBUTING.md`](./CONTRIBUTING.md) — social etiquette;
  ignore the build/docs commands (see §2.8)
- [`./CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md)
- [`./docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — high-level
  DataStore architecture
- [`./docs/PANDAS_COMPATIBILITY.md`](./docs/PANDAS_COMPATIBILITY.md)
- [`./docs/PANDAS_MIGRATION_GUIDE.md`](./docs/PANDAS_MIGRATION_GUIDE.md)
- [`./docs/REMOTE_SESSION_DESIGN.md`](./docs/REMOTE_SESSION_DESIGN.md)
- [`./docs/PROFILING.md`](./docs/PROFILING.md)

### 12.3 User-facing documentation (not for contributors)

- User docs: <https://chdb.readthedocs.io/en/latest/index.html>
- LLM-friendly user-facing context: `https://clickhouse.com/chdb/llms.txt`
  (planned; see project roadmap). When asked to *use* chDB,
  `llms.txt` is the canonical reference. When asked to *contribute
  to* chDB, this file is the canonical reference.

---

## 13. Security

- **No secrets in test fixtures** (see §2.10). S3 keys, ClickHouse
  Cloud DSNs, OAuth tokens — none of them belong in the test tree.
  Use environment variables, and `pytest.skip` if they're not set.
- **Stack traces matter** (see §2.6). When the engine crashes,
  capture the trace, don't paper over it.
- For broader security policy, look for `SECURITY.md` if present, or
  ask a maintainer. Vulnerability reports should go through GitHub's
  security advisory channel, not public issues.

---

## 14. Maintenance of this file

- This file is the **single source of truth** for "how to contribute
  to this repository." `.cursor/rules/*.mdc`, `CLAUDE.md`, and any
  future `GEMINI.md` / `.github/copilot-instructions.md` should
  reference it rather than duplicate its content. Cursor and Claude
  Code already read `AGENTS.md` automatically; Codex, Aider, Zed,
  Jules, Devin, and Copilot do as well per the agents.md spec.

- When you change a build / test / lint command in this repository,
  also update this file in the same PR. A stale `AGENTS.md` is worse
  than no `AGENTS.md` because agents follow it confidently.

- Subdirectory `AGENTS.md` files are allowed and encouraged where
  the rules are noticeably different from the root (for example,
  `agent_skills/AGENTS.md` constrains how Skills are authored,
  `datastore/AGENTS.md` could narrow the rules for the chdb-ds
  module specifically). The agents.md spec says the *nearest* file
  wins, so subdirectory files override the root for files inside
  them.

- If you find this file misleading or out of date, fix it in the
  same PR as the underlying change. Don't open a "docs only" PR
  later — that's how stale docs accumulate.

---

*Last updated: 2026-05-08. Standard followed: [agents.md](https://agents.md).*
