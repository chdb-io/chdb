# chDB Agent Tool — cross-language contract

One behavior, many bindings. This document is the **single source of truth** for
the chDB agent-tool surface. `chdb.agents.ChDBTool` (Python) is the reference
implementation; the TypeScript binding (`chdb-node`) and any future language
binding under `chdb-io` implement the **same** methods with the **same**
semantics and are verified against the **same** fixture (`conformance/cases.jsonl`).

The goal is explicit: stop each language re-inventing query + introspection +
safety with subtly different behavior. If a binding needs to diverge, it must add
a case to the shared fixture documenting the divergence — not silently differ.

## Methods (names are canonical; per-language casing may adapt)

| Canonical tool name | Purpose |
|---|---|
| `run_select_query` | Run read SQL with bound params; return rows + truncation metadata |
| `list_databases` | List databases |
| `list_tables` | List tables in a database (current if omitted) |
| `describe_table` | Columns/types of a table **or** a table-function expression |
| `get_sample_data` | A few sample rows from a table or table function |
| `list_functions` | List available SQL functions |
| `attach_file` | Register a local file as a queryable named table (writable tools only) |

Bindings expose these as native tools (Python methods on `ChDBTool`; TS
`chdbTools()`; MCP tools in mcp-clickhouse) but the **capability set, argument
meaning, error classification, and safety guarantees are identical**.

**Language-optional capability (not in the cross-language conformance):**
`dataframe_query` — query in-process DataFrames via the `Python()` table
function. Meaningful only in languages/deployments where the agent runtime and
the engine share a process (Python co-located; not applicable to a remote MCP or
to TS, which has no in-process pandas). Bindings may omit it.

## The four pillars (normative)

### P1 — Read-only is enforced by the engine
- Default on. Implemented as **`SET readonly=2`** on the tool's session at
  creation. `readonly=2` (NOT `readonly=1`) is required: level 1 rejects the
  `file()`/`s3()`/`url()` table functions that are chDB's core use case; level 2
  allows reads + those functions while still rejecting `INSERT`/`CREATE`/`ALTER`/`DROP`.
- Read-only is **immutable for the session** (the engine forbids lowering
  `readonly`), so it is fixed at construction, never per-call. Opt out entirely
  with the binding's write flag (`read_only=False` / `allowWrite=true`).
- Escapes are blocked by the engine: `SET readonly=0` and writes inside
  multi-statement SQL both fail.

### P2 — Values are bound, identifiers are quoted
- Query **values** pass as chDB server-side params (`{name:Type}` + a params
  map). They are **never** concatenated into SQL text — including on metadata
  filters and destructive paths.
- **Identifiers** (db/table/column names) cannot be params; they pass through the
  shared `quote_ident` rules (backtick-double-escape, reject NUL).

### P3 — Results are capped, truncation is flagged
- Every result is capped by `max_rows` (and a `max_bytes` guard). When the engine
  produced more than the cap, the result sets `truncated = true`. Truncation is
  **never silent**.

### P4 — Errors reach the model
- The **tool-dispatch** path (`call()` / the tool executor) returns a structured
  envelope `{ ok: false, error: { code, type, message } }` instead of throwing,
  so an agent can read the engine message and self-correct.
- Direct library methods raise a typed error (`ChDBError` and subclasses).
- Error classification is shared: parse `Code: N. DB::Exception: <msg>. (TYPE)`;
  map by code (`164→READONLY`, `62→SYNTAX`, `46/47/60/81/115→UNKNOWN_*`).

### P5 — Resource and source controls (normative, optional-per-deployment)
- **Query timeout** — an optional `max_execution_time` (seconds) bounds runaway
  queries at the engine (`TIMEOUT_EXCEEDED`). Off by default; set per deployment.
- **File allowlist** — an optional list of path prefixes. When set, `file()` /
  `s3()` / `url()` literal paths outside it are rejected (`ACCESS_DENIED`), and
  `attach_file` refuses out-of-allowlist paths. Best-effort on raw SQL (literal
  args only) — `readonly=2` remains the real write backstop; OS-level sandboxing
  is the real filesystem backstop.
- **Source catalog** — `attach_file(name, path[, format])` registers a file as a
  view. It is a write, so it works only on a writable tool; a read-only tool
  declares files at construction (`attachments=`), materialized **before** the
  read-only lock. The path/format are baked in via `quote_string` (a stored view
  can't carry bound params); `quote_string` is the shared literal-escaping helper
  (backslash + single-quote), used only where the engine cannot bind.

## Silent-conversion policy
No coercion beyond JSON decode. 64-bit integers are returned in their exact form
(JSON with `output_format_json_quote_64bit_integers=1`), not lossy floats. Any
future coercion must be opt-in and documented here.

## Conformance
`conformance/cases.jsonl` is a language-neutral list of cases: setup-free inputs
(they use `numbers()`, `file()` on `conformance/fixtures/`, `system.*`, params,
and write-blocking DDL) plus an expected outcome. Each binding ships a thin
runner (~30 LOC) that loads the fixture, invokes the named method, and asserts.
A binding is "contract-conformant" when every case passes. See
`conformance/README.md`.
