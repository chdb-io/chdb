# chDB Agent Tool — cross-language contract

> **Status: beta / experimental (introduced in chdb 4.2.0).** The surface may
> change in a minor release while it stabilizes across bindings; pin a version if
> you depend on it.
>
> **Contract version: `0.2.0`** — exposed as `CONTRACT_VERSION` and via
> `capabilities()` in every binding. Downstream consumers probe capabilities
> instead of guessing from a package version (see *Versioning & capabilities*).

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

## descriptors.json — the model-visible surface (single source)

`descriptors.json` (next to this file; vendored byte-identical in every
binding) is the single source of truth for the **model-visible** surface: tool
names, description text, and argument schemas. Everything the model reads is
generated from it:

- `tool_specs(dialect)` / `toolSpecs(dialect)` render it per runtime family —
  `anthropic` (`input_schema`), `openai` (`{type: "function", function}`),
  `mcp` (`inputSchema`).
- Framework adapters derive their native schema declarations from it (the TS
  Vercel AI SDK / Mastra adapters generate zod schemas mechanically; schema-
  declaring frameworks like CrewAI / smolagents generate their arg specs the
  same way). Hand-copying a description into an adapter re-creates the drift
  this file exists to kill.
- Changing `descriptors.json` **is a contract change**: bump `contract_version`
  (in the file, the code constant, and the conformance fixture header) and sync
  every vendored copy.

## Versioning & capabilities

- `CONTRACT_VERSION` (semver string) names the contract revision a binding
  implements. It changes when `descriptors.json`, `conformance/cases.jsonl`, or
  normative text here changes. It is intentionally **not** the package version.
- `capabilities()` returns `{contract_version, tools, features}`. `features`
  flags the capability-gated parts of the contract, so a consumer asks *"does
  this binding do X?"* instead of maintaining a version matrix:
  `dataframe_query` (Python-only, co-located engine), `attachments`,
  `file_allowlist`, `max_execution_time`, `async`, `streaming` (reserved,
  currently `false` everywhere).
- Conformance cases carrying `"requires": "<feature>"` run only on bindings
  whose `capabilities().features[<feature>]` is true; all other cases are core
  and every binding must pass them.

## Async

Engine calls are synchronous in-process; how a binding exposes async is part of
the contract surface:

- **Python** — `aquery()` / `acall()` run the sync call in a worker thread
  (`asyncio.to_thread`), documented as such, never faked as native async. This
  is what async-first frameworks (AutoGen, Pydantic AI) bind to, instead of
  each re-implementing a thread-pool wrapper.
- **TypeScript** — `query()` / `call()` are natively Promise-based.
- The sync path (Python) remains the canonical one the conformance runner
  drives; async forms must return identical results.

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
- **A caller-provided session is probed, never mutated**: the constructor reads
  `getSetting('readonly')` and requires it to match the declared mode (2 for
  read-only, 0 for writable); a mismatch fails construction with
  `CONFIG_MISMATCH`. Silently applying `SET readonly=2` would irreversibly lock
  the caller's shared session; silently skipping it would leave a tool that
  claims read-only but is not. Sessions the tool creates itself keep the
  set-at-construction behavior.
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
- `max_bytes` counts **UTF-8 bytes of each row's compact JSON encoding** (no
  spaces, non-ASCII kept raw). This is normative because it is model-visible:
  counting UTF-16 units or ASCII-escaped characters instead moves the truncation
  point by up to ~3× on non-ASCII data, splitting the corpus between bindings.
- A non-numeric per-call cap (`max_rows`, `limit`) is a typed
  `INVALID_ARGUMENT` error, never a silently disabled cap. On the model-facing
  `call()` path a JSON `null` for an optional argument means "omitted".

### P4 — Errors reach the model
- The **tool-dispatch** path (`call()` / the tool executor) returns a structured
  envelope `{ ok: false, error: { code, type, message } }` instead of throwing,
  so an agent can read the engine message and self-correct.
- That includes caller mistakes on the dispatch path itself: an unknown tool
  name and a non-object `arguments` payload both come back as envelopes
  (`UNKNOWN_TOOL` / `INVALID_ARGUMENT`), never as a raised exception.
- Direct library methods raise a typed error (`ChDBError` and subclasses).
- Error classification is shared: parse `Code: N. DB::Exception: <msg>. (TYPE)`;
  map by code (`164→READONLY`, `62→SYNTAX`, `46/47/60/81/115→UNKNOWN_*`).
- Binding-side validation failures (no engine round-trip) use `code: 0` with a
  shared `type`: `INVALID_ARGUMENT` (bad argument value, e.g. a non-numeric
  cap or an unknown `tool_specs` dialect), `ACCESS_DENIED` (allowlist),
  `CONFIG_MISMATCH` (a caller-provided session whose readonly state conflicts
  with the declared mode), `UNKNOWN_TOOL` (bad `call()` name), `TOOL_ERROR`
  (any other non-engine failure surfaced through the envelope).

### P5 — Resource and source controls (normative, optional-per-deployment)
- **Query timeout** — an optional `max_execution_time` (seconds) bounds runaway
  queries at the engine (`TIMEOUT_EXCEEDED`). Off by default; set per deployment.
- **File allowlist** — an optional list of path prefixes. When set, raw SQL is
  scanned for **every table function the running engine exposes** (live
  `system.table_functions` snapshot, unioned with a static fallback) that is
  not in the shared safe-by-construction set: each such call must carry a
  **literal** source argument inside the allowlist, else `ACCESS_DENIED`. The
  scan runs over masked SQL — string literals and comments blanked
  position-preserving, backtick/double-quote-wrapped function names matched —
  so a path-looking string never false-positives and quoting, comments, or a
  computed argument never bypass the gate. `attach_file` refuses
  out-of-allowlist paths. `readonly=2` remains the write backstop; OS-level
  sandboxing is the filesystem backstop. (`dataframe_query` exempts only the
  `Python()` names it itself injects.)
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
and write-blocking DDL) plus an expected outcome. Its first line is a header
record `{"fixture": ..., "contract_version": ...}` (no `id`), which runners
check against their binding's `CONTRACT_VERSION` and skip. A case with
`"requires": "<feature>"` runs only where `capabilities()` reports that feature
(everything else is core). Each binding ships a thin runner (~30 LOC) that
loads the fixture, invokes the named method, and asserts. A binding is
"contract-conformant" when every applicable case passes. See
`conformance/README.md`.
