# clickhouse-connect upstream suite vs the chDB backend

This harness runs **clickhouse-connect's own test suite against the embedded chDB backend**,
so the "chDB is a drop-in clickhouse-connect backend" claim is verified against the very
tests the library maintainers wrote — not a parallel set of our own. It is the chDB analogue
of the chdb-node Layer-2 harness (PR #49) that ran clickhouse-js's own integration specs
against embedded chDB.

Per the design proposal (§8.4) this lives in **chDB's** CI, never in clickhouse-connect's:
clickhouse-connect's CI runs only against a real HTTP server and never depends on chDB; chDB
runs clickhouse-connect's suite against `ChdbBackend` here and **owns the skip list as data**.

## How it works

`create_client(...)` is the single chokepoint every clickhouse-connect client flows through
(the analogue of clickhouse-js's one client factory). `run_upstream_suite.py`:

1. Resolves a clickhouse-connect checkout (`--cc-repo PATH`, else clones the source
   currently being tracked — see *Version subscription* below).
2. Drops `_force_chdb_conftest.py` in as the checkout's root `conftest.py`. At import time —
   before the integration conftest binds the name — it replaces
   `clickhouse_connect.driver.create_client` / `create_async_client` with wrappers that build
   a **chDB-backed** client and strip HTTP-transport-only keyword arguments.
3. Runs the selected tests with the two empirical gates below applied.

```bash
# against a local checkout (no server needed — chDB is in-process)
python run_upstream_suite.py --cc-repo /path/to/clickhouse-connect --tests tests/integration_tests

# against the installed clickhouse-connect's released tag
python run_upstream_suite.py --tests tests/integration_tests
```

## Two gates — entirely data-driven, no pytest markers

* **`skip_list.txt`** — newline-separated nodeid substrings skipped outright: capabilities
  the embedded engine genuinely cannot support, established by **running the integration
  suite twice (HTTP server vs chDB) and adding only the cases that pass over HTTP and
  genuinely cannot pass embedded**. The chdb-node analogue is `skip-list.json` (whole spec
  files). This is the authoritative chDB-cannot-do-X list; clickhouse-connect knows nothing
  about it.
* **`expected_divergences.txt`** — substrings marked `xfail(strict=True)`: documented
  behavior differences (formatting, dtype representation, ...). They still run and **must**
  fail; a divergence that silently disappears (an xpass) breaks the build. The analogue is
  clickhouse-js's `expectations.patch`.

The job **gates**: a failure that is neither skipped nor a documented divergence is a real
regression and fails CI.

**Why no `pytest.mark` capability markers?** Markers would force clickhouse-connect to ship
a vocabulary of categories (`requires_keeper`, `http_only`, ...) and to tag every test, just
to support an out-of-tree backend's skip needs. The minimal-CC-change principle says no.
The empirical nodeid blacklist puts the skip decision exactly where it belongs — in this
repo, derived from observed behavior — and lets us add a single case without touching
upstream.

## Version subscription policy

The skip list is keyed to a specific clickhouse-connect source tree. We rotate the
subscription target as the chDB backend matures upstream:

| Phase | Subscribed source | When |
|---|---|---|
| **1. Pre-merge** | `ShawnChen-Sirius/clickhouse-connect@feat/pluggable-backend-registry` (the chDB-author fork branch this PR is on) | While the CC-side change is in review |
| **2. Merged on `main`** | `ClickHouse/clickhouse-connect@main` | Once the CC PR lands but before a release ships |
| **3. Released** | The latest published `clickhouse-connect` release matching what `pip` resolves | Steady state |

Only when clickhouse-connect publishes a **new release** does chDB re-curate the skip list:
run the suite against the new version, diff vs the prior baseline, add/remove only what
truly changed. Day-to-day chDB development does not touch this list.

The runner takes a `--cc-repo` flag for phase 1 / phase 2 (point it at a local clone of
whichever source is current), and falls back to cloning the installed release's tag for
phase 3.

## Documented divergence categories (empirically established)

Running the full integration suite against chDB surfaces these categories. Each entry in
the skip / xfail files is justified against one of them; the README lists categories rather
than every individual nodeid so the data file stays canonical.

1. **HTTP transport** — connection pooling, retries, proxy, TLS, keep-alive, HTTP error
   codes, form-encoded params, protocol-version negotiation. *(skip — no socket in-process)*
2. **Auth / RBAC** — JWT/access tokens, `GRANT`/`CREATE ROLE`/row policies (`ACCESS_DENIED`).
   *(skip — embedded engine has no access-control layer)*
3. **`external_data`** — tables shipped over the wire with a query. *(skip — unsupported embedded)*
4. **Transport-attribute assertions** — `client.compression == "gzip"` and similar: chDB
   reports `None` because there is no wire compression. *(xfail)*
5. **Server-only `system` tables** — e.g. `system.session_log`. *(skip)*
6. **Behavior round-trips through stricter / less-strict server modes** — set-and-observe
   round-trips that depend on server-side defaults the embedded engine doesn't carry. *(xfail)*

The complementary already-green evidence is `tests/clickhouse_connect/test_parity.py`,
which asserts chDB-vs-real-server output parity across a type and operation matrix.

## CI workflow

`.github/workflows/clickhouse-connect-backend.yml` runs three hard gates: the relocated
backend suite (embedded-only, no server needed), the parity suite (chDB vs real CH 26.5,
service-container), and clickhouse-connect's own integration suite against the chDB
backend with `skip_list.txt` + `expected_divergences.txt` applied. Any failure outside the
gates is a real regression.
