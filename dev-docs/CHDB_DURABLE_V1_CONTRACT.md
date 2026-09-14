# chDB Durable V1 Contract

> Version: **V1**  
> Applies to: `chdb-core`, `chdb` (Python), `chdb-node`, `chdb-go`, `chdb-rust`, and any downstream adapter that `dlopen`s `libchdb` directly.  
> Status: **Normative**. V1 covers only a single database, a single writer, a statement WAL and full checkpoints. Business state and integration constraints of downstream products are out of scope for this protocol.

## 0. Normative markers

| Marker | Meaning |
| --- | --- |
| **[CORE]** | Must be decided or performed by `chdb-core`; a binding MUST NOT reimplement the equivalent logic with SQL string concatenation, prefix matching or regexes |
| **[FROZEN]** | Cross-binding data model and behavioural contract; JSON semantics are frozen, byte-level identity (whitespace, key order) is not required |
| **[BINDING]** | Implemented by each binding; the API shape may follow language conventions, but externally observable behaviour must be identical |

`MUST` / `MUST NOT` are requirements; `SHOULD` is a strong recommendation, and any deviation must be documented in that binding's own documentation.

An implementation may claim chDB Durable V1 support only after it satisfies every V1 requirement and passes the conformance suite in §7. A binding for some language shipping later than V1 does not make that implementation protocol V2.

---

## 1. The minimal capability frozen by V1

The scope of V1 is exactly:

- one durable object maps to one chDB database;
- one writer, any number of read-only openers;
- the public `execute()` accepts exactly one statement per call;
- in-database mutations use a statement WAL;
- checkpoints use a complete `BACKUP DATABASE`;
- object storage holds immutable base/WAL objects plus one CAS-updated `head.json`;
- lease, heartbeat, fencing, timeout and ambiguous-commit reconciliation;
- length and SHA-256 integrity verification of base and WAL;
- all bindings call the same core backup/restore/query-analysis ABI.

V1 explicitly does not support:

- preambles, UDFs or any other persistent state outside the database;
- multi-statement public execution;
- multi-writer merging;
- multiple databases inside one object;
- Parquet / data WAL;
- incremental checkpoints;
- GC, destroy and cross-object transactions.

Ownership of these capabilities is covered in §8.

---

## 2. Layering and responsibilities

```text
                    +-------------------------------------+
   [BINDING]        | language API, error types, teardown  |
                    +-------------------------------------+
   [FROZEN]         | object layout, head, WAL, checksums  |
                    | lease, CAS, fencing, state machine,  |
                    | error categories                     |
                    +-------------------------------------+
   [BINDING]        | provider, auth, retry, timeout,      |
                    | reconcile, local scratch, streaming  |
                    | upload/download, orchestration       |
                    +-------------------------------------+
   [CORE]           | backup, restore, query analysis      |
                    | connection and query execution       |
                    +-------------------------------------+
```

The test for where something belongs:

- questions that only the ClickHouse parser, AST, catalog or backup engine can answer reliably belong to core;
- cloud providers, distributed consistency and lifecycle policy belong to the bindings;
- formats and behaviours that decide whether an object can be restored by another language belong to the frozen contract.

Object-storage SDKs, authentication, lease, WAL, CAS, fencing and remote retries **MUST NOT** enter core.

---

## 3. Required chdb-core changes **[CORE]**

### 3.1 V1 ABI overview

`programs/local/chdb.h` must export:

1. database full backup;
2. database restore;
3. query analysis carrying a statement count and proof of the target database;
4. the existing `chdb_version()`, so bindings can record and check engine identity.

Every new symbol must land in both the Linux and macOS export allowlists and must be covered directly by C ABI tests; exposing only a Python wrapper does not count as completing the core ABI.

### 3.2 backup / restore ABI

The current RC shape may keep the optional `base_file_path` as a low-level engine capability:

```c
chdb_result * chdb_backup_database_n(
    chdb_connection conn,
    const char * database,       size_t database_len,
    const char * file_path,      size_t file_path_len,
    const char * base_file_path, size_t base_file_path_len);

chdb_result * chdb_restore_database_n(
    chdb_connection conn,
    const char * database,  size_t database_len,
    const char * file_path, size_t file_path_len);
```

But a Durable V1 binding calling backup **MUST** pass `NULL/0` and produce only full backups. `base_file_path` is not part of V1 durable conformance; an end-to-end portable incremental chain belongs to V2.

Core must guarantee that:

- the database identifier and the file path are passed separately, with core building the AST and quoting safely; a binding **MUST NOT** assemble `BACKUP` / `RESTORE` SQL;
- `file_path` is absolute, its parent directory already exists, and it is constrained by `backups.allowed_path`;
- backup does not overwrite an existing file;
- restore does not implicitly change the connection's current database;
- restoring over an existing table of the same name must not be treated by a binding as a replace. A V1 binding must restore into an empty target database inside a fresh scratch directory;
- errors surface through the standard `chdb_result` and are released with the standard result destruction ABI;
- a failed backup/restore must not leave behind a result that a binding could mistake for a successful archive or a successful restore.

### 3.3 The current classify ABI is not sufficient for V1

Returning only:

```text
(query_class, has_secrets)
```

is not enough, because a binding still cannot reliably determine:

- whether the input contains one executable statement or several;
- whether `PARALLEL WITH` expands into multiple execution arms;
- whether every write target of a mutation belongs to this durable database;
- where an unqualified object name points after being resolved against the current session settings;
- whether a statement covering multiple objects, such as `RENAME` or `EXCHANGE`, crosses database boundaries;
- whether a seemingly ordinary `INSERT` actually writes into a table function, an outfile or engine-external state;
- whether the statement creates, drops or renames the durable database itself; that is object lifecycle and must not be treated as an ordinary WAL mutation.

All of these must be decided by the parser/AST and cannot be delegated to a Python, TypeScript, Go or Rust regex.

### 3.4 The V1 query-analysis ABI

Before the first RC freezes, `chdb_classify_query_n` should be extended into an analysis ABI that takes a target database; if the existing symbol has already shipped publicly, add a versioned symbol instead of breaking the ABI in place.

The recommended minimal C shape is below; fields use fixed-width types and retain a `struct_size`:

```c
typedef enum chdb_query_class {
    CHDB_QUERY_READ_ONLY       = 0,
    CHDB_QUERY_MUTATING        = 1,
    CHDB_QUERY_MUTATING_GLOBAL = 2,
    CHDB_QUERY_CONTROL         = 3,
    CHDB_QUERY_UNKNOWN         = 4
} chdb_query_class;

typedef enum chdb_query_analysis_flag {
    CHDB_QUERY_HAS_SECRETS                 = 1u << 0,
    CHDB_QUERY_WRITES_ONLY_TARGET_DATABASE = 1u << 1,
    CHDB_QUERY_CHANGES_DATABASE_LIFECYCLE  = 1u << 2
} chdb_query_analysis_flag;

typedef struct chdb_query_analysis_v1 {
    uint32_t struct_size;       /* caller sets to sizeof(struct) */
    uint32_t statement_count;   /* executable statements; PARALLEL arms count */
    uint32_t flags;             /* chdb_query_analysis_flag */
    uint32_t query_class;       /* chdb_query_class; fixed-width ABI field */
} chdb_query_analysis_v1;

chdb_state chdb_classify_query_n(
    chdb_connection conn,
    const char * sql,             size_t sql_len,
    const char * target_database, size_t target_database_len,
    chdb_query_analysis_v1 * out_analysis);
```

The names may be adjusted in the core PR to match existing ABI conventions, but none of the following semantics may be reduced:

- the caller must set `struct_size`; core returns `CHDBError` for a struct that is too small, must not write past the boundary the caller declared, and must initialise every V1 field on success;
- `statement_count`: counts executable statements; `0` for empty input or a parse failure, greater than `1` for multiple top-level statements, and the execution arms of `PARALLEL WITH` also count as multiple statements;
- `CHDB_QUERY_HAS_SECRETS`: set when the AST contains credentials; a parse failure must not claim that secrets were safely identified;
- `CHDB_QUERY_WRITES_ONLY_TARGET_DATABASE`: set only when core can prove that every persistent write target lies inside `target_database`; it must not be set if there is any other database, `system`, table function, outfile or other engine-external write target;
- `CHDB_QUERY_CHANGES_DATABASE_LIFECYCLE`: set when the statement changes the database container itself, such as `CREATE/DROP/RENAME DATABASE`; in V1 the cold-object database creation is done internally by the adapter, and the public WAL must not change container lifecycle;
- unqualified targets must be resolved using that connection's actual parser/session semantics before being compared against `target_database`;
- in V1, `MUTATING_GLOBAL` is only an accurate diagnostic classification: the public entry point always refuses it, and it never enters any preamble;
- `CONTROL` covers session mutation, managed operations such as BACKUP/RESTORE/SYSTEM, and external writes;
- statement settings that explicitly enable async insert, disable insert wait or otherwise weaken mutation synchronisation guarantees must be classified as `CONTROL`; public SQL must not bypass the managed connection's synchronous-completion policy;
- a parse failure or a new, not-yet-covered AST returns `UNKNOWN`, and the binding fails closed;
- analysis does not execute SQL and does not change the current database, settings or query log;
- it must step correctly over the inline data of `INSERT ... VALUES`, `INSERT ... FORMAT` and `EXPLAIN INSERT`; a semicolon or blank line inside data must not be mistaken for the start of the next statement.

The core gate for V1 `execute(sql)` is fixed as:

```text
statement_count == 1
AND query_class == MUTATING
AND WRITES_ONLY_TARGET_DATABASE
AND NOT CHANGES_DATABASE_LIFECYCLE
AND NOT HAS_SECRETS
```

The core gate for V1 `query(sql)` is fixed as:

```text
statement_count == 1
AND query_class == READ_ONLY
```

`READ_ONLY` SQL may run even when it contains secrets, because it never enters the WAL; the binding must keep SQL text, errors and tracing from leaking the secrets verbatim.

Python's `_chdb.Connection` must expose every analysis field in lockstep; the old tuple that returns only `(query_class, has_secrets)` cannot serve Durable V1. Node, Go, Rust and direct FFI adapters must all bind the same struct and must not infer it from the Python wrapper or compute missing fields themselves.

### 3.5 The core acceptance matrix

The first core RC must at minimum cover:

- quoted database/table identifiers;
- resolution of unqualified targets;
- writes into the target database versus writes into another database;
- cross-database `RENAME` / `EXCHANGE` / multi-target DDL;
- the `CREATE/DROP/RENAME DATABASE` lifecycle flag;
- `INSERT INTO TABLE` versus `INSERT INTO FUNCTION`;
- `INTO OUTFILE`;
- writes into the `system` database;
- `CREATE FUNCTION`, access entities, named collections → `MUTATING_GLOBAL`;
- `SET` / `USE` / `SYSTEM` / `BACKUP` / `RESTORE` → `CONTROL`;
- explicit async insert or settings that weaken mutation synchronisation → `CONTROL`;
- ordinary SELECTs and secret-bearing read-only queries;
- one statement, several statements, trailing semicolons and comments;
- `PARALLEL WITH`;
- statement boundaries after `VALUES` / arbitrary `FORMAT` inline data;
- parse failure / new unknown AST → `UNKNOWN`;
- session state and query log unchanged before and after analysis;
- backup absolute paths, allowed paths, pre-existing targets, error result lifetime;
- restore into a fresh directory succeeds, and a failed restore cannot be mistaken for a usable one;
- Linux/macOS symbol exports and the Python surface call the same C ABI.

### 3.6 Core process constraints

V1 inherits the current engine constraints:

- one process has at most one active EmbeddedServer / data path;
- multiple connections may share the same physical path;
- opening a different path must fail while any connection to the old path is still alive;
- a new path may be bound only after all old connections are closed.

Therefore only one durable object's scratch path can be active in one process at a time. Bulk scans must open objects sequentially, or the layer above must use multi-process workers. Lifting this restriction is follow-up architectural work in core; a binding must not paper over it with a process-level registry.

---

## 4. Object protocol **[FROZEN]**

### 4.1 Object layout

```text
<namespace>/<object-id>/
  head.json
  checkpoints/<generation>-<seq>-<uuid8>.tar.gz
  wal/<generation>-<seq>-<uuid8>.jsonl
```

- `generation` and `seq` are decimal integers with no leading zeros;
- `uuid8` is the first 8 characters of a UUID4 in lowercase hexadecimal form;
- checkpoint and WAL keys must be unique on every attempt and must be published with a true conditional create;
- object references in head are keys relative to `<namespace>/<object-id>/`; they must use `/` as the separator, must not start with `/`, and must not contain empty segments, `.` or `..`;
- V1 has no `preamble/` or blob directory.

### 4.2 `head.json`

UTF-8 JSON, no BOM. Key order and whitespace are not frozen; field types, meanings and state transitions are frozen.

```json
{
  "protocol": {
    "version": 1,
    "reader_features": [],
    "writer_features": []
  },
  "engine": {
    "name": "chdb",
    "version": "26.7.2-rc.2",
    "backup_format": 1,
    "min_reader": "26.7.2-rc.2"
  },
  "lease": {
    "generation": 3,
    "owner": "worker-visible-name",
    "instance": "unique-live-instance-id",
    "expires_at": 1788230400.0
  },
  "manifest": {
    "db": "default",
    "base": {
      "key": "checkpoints/3-8-acde1234.tar.gz",
      "size": 1048576,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    "wal": [
      {
        "key": "wal/3-9-acde5678.jsonl",
        "size": 127,
        "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
      }
    ],
    "seq": 9
  }
}
```

A released lease is represented exactly as:

```json
{
  "generation": 3,
  "owner": null,
  "instance": null,
  "expires_at": null
}
```

Rules:

- `lease.generation` increments on every acquisition from an unowned state, on expiry takeover and on explicit force takeover; heartbeats and ordinary manifest commits do not increment it;
- `manifest.seq` increments every time a new WAL or checkpoint reference is published; heartbeats do not increment it;
- `engine.version` is the exact value returned by the writer's `chdb_version()`; it is used only for diagnostics and auditing, never as an exact-match gate;
- `engine.backup_format` is the generation of the chDB backup archive format; the V1 baseline is `1`;
- `engine.min_reader` is the lowest chDB engine version able to read the current object;
- starting from the first Durable V1 archive, every later chdb-core release must be able to restore a V1 full backup created via `chdb_backup_database_n` by the same release or an earlier one; only when the archive format generation is genuinely incompatible should `backup_format` be raised so that older readers fail closed;
- `manifest.db` is the one database this object holds; on cold creation it takes the name the caller supplied, and when the caller supplies none every binding uses `default`; once written, head is authoritative and any database argument passed on a later open must be ignored;
- `manifest.base` is an object reference or `null`;
- every object reference must carry `key`, the byte `size`, and a complete lowercase SHA-256;
- `manifest.wal` is ordered for replay;
- all integers must stay within the cross-language safe integer range;
- when a binding writes head back, it must preserve unrecognised fields at the top level and inside `protocol`, `engine`, `lease` and `manifest`.

### 4.3 Version and feature negotiation

- a `protocol.version` above this implementation's baseline → refuse to open;
- an unrecognised entry in `reader_features` → refuse to read;
- an unrecognised entry in `writer_features` → may open read-only once the read checks pass, but must refuse to take the writer lease;
- the V1 baseline defines no non-empty feature name; any new optional semantics must enter the registry, the scenarios and the fixtures before a binding implements them;
- adding a JSON field does not automatically increment the protocol version, but a writer must round-trip unknown fields.

Engine compatibility does not compare `engine.version` for exact equality with the running version; it uses explicit compatibility fields instead:

```text
head.engine.backup_format > reader backup-format baseline  -> engine_incompatible
running chdb_version() < head.engine.min_reader             -> engine_incompatible
otherwise                                                   -> open
```

This means a V1 object written by `26.7.2-rc.2` can be opened by any later chdb-core release that satisfies `min_reader` and supports the same `backup_format`; `engine.version` stays in head so that "who originally wrote this object" remains easy to investigate.

### 4.4 WAL

The WAL is UTF-8 JSONL, one statement record per line:

```json
{"sql":"INSERT INTO t VALUES (1)"}
```

Rules:

- exactly one JSON object per line, terminated by `\n`;
- `sql` must be a string, executed on recovery in manifest order and then line order;
- recovery execution goes through the adapter's internal replay path and must not re-enter the public `execute()`;
- `size` and SHA-256 are computed over the whole segment before upload; both must be verified after download and before parsing;
- a writer must use a unique key and a conditional create, and must not overwrite an already published segment;
- in V1, one `sql` value is at most 64 MiB of UTF-8 and an uncompressed WAL segment is at most 128 MiB; a writer must refuse locally beyond those limits and a reader must handle input up to them; checkpoints have no protocol-level size limit and rely on streaming/provider limits instead;
- `now()`, `rand()`, UUIDs, externally mutable inputs and the like inside SQL are not materialised automatically by the binding. V1 guarantees that the original statements are replayed in order; it does not guarantee that non-deterministic statements produce the same result. That is the caller's responsibility;
- secret-bearing mutations are already refused by §3.4 and must never appear in the WAL.

### 4.5 Checksums and corruption handling

- base and WAL must have both `size` and SHA-256 verified before use;
- a missing immutable object, a length mismatch or a checksum mismatch is all `corrupt`;
- a corrupt WAL must not be skipped, must not cause a fall back to an older base, and must not yield a partially recovered session;
- `head.json` must pass strict schema validation; preserving unknown fields does not mean known fields may be loosely typed;
- in V1, `head.json` is at most 1 MiB; a writer must checkpoint before crossing the limit or return `limit_exceeded`, and a reader fails closed on an oversized head;
- JSON comparison is semantic; different bindings are not required to produce the same key order or whitespace.

---

## 5. Binding state machine **[BINDING + FROZEN BEHAVIOR]**

### 5.1 Minimal backend semantics

Every backend must provide equivalent capabilities:

```text
get(key)                               -> bytes | missing
get_with_etag(key)                     -> (bytes | missing, etag | missing)
put_file_if_absent(key, local_path)    -> created | already-exists | ambiguous
put_bytes_if_absent(key, bytes)        -> created | already-exists | ambiguous
replace_if_match(key, bytes, etag)     -> new-etag | not-replaced | ambiguous
download_to_file(key, local_path)      -> found | missing
```

The concrete method names and streaming APIs may differ, but they must satisfy:

- checkpoint upload/download must not force the whole archive into memory at once; file-based or streaming transfer must be supported;
- conditional create and conditional replace must be the provider's real atomic conditional operations and must not be emulated with HEAD + PUT;
- an ETag is an opaque CAS token and must not be assumed to be an MD5;
- a local target file is first written to a unique temporary path and only atomically published to its final path in scratch after verification;
- `delete_prefix` is not part of V1, because V1 has no destroy/GC;
- every provider claimed as supported must pass conformance independently against the real backend; compatibility for one provider cannot be inferred from an in-memory mock or from another provider.

### 5.2 Writer open

A writer `open` must:

1. read and strictly validate head; when head does not exist, atomically create a cold manifest with a single conditional create and take the generation-1 lease;
2. check protocol features, `backup_format` and `min_reader`; it must not refuse merely because `engine.version` differs from the running version;
3. for an existing head, acquire the lease via CAS; a normal acquisition is allowed only for an unowned lease or one already past expiry plus the clock-skew window;
4. create a binding-private, unique, empty scratch directory;
5. download and verify base, then restore into an empty target database; when base is `null`, create a cold database;
6. download, verify and replay the WAL in order;
7. renew via CAS once more before returning the object, confirming that the lease was not lost during recovery;
8. on failure at any step, close the partial engine, release or invalidate the lease as soon as possible, and clean up the scratch it owns.

Read-only open:

- returns `not_found` when head does not exist and must not create it implicitly;
- does not take a lease;
- restores the first manifest snapshot it read; immutable references guarantee that this snapshot stays readable through the writer's later commits;
- allows only `READ_ONLY` queries.

### 5.3 Operation serialization

Every durable object must have an explicit operation queue/mutex. At minimum the following operations and the heartbeat head CAS must be coordinated into one definite order:

```text
execute / query / flush / checkpoint / close / lease renewal
```

"The current language runtime is single-threaded" or "the native call is synchronous today" must not be relied on as the serialization guarantee. State-machine invariants must also hold across asynchronous provider I/O.

A durable object must own a managed connection that is not exposed to the caller, and must pin the current database to the manifest database. The adapter must disable async insert or wait for it to complete, and must configure mutations for synchronous completion; public `SET`/`USE` and statement settings that weaken those guarantees are already refused by core analysis.

### 5.4 `query` / `execute` / `flush`

- `query()`: accepts only the query gate from §3.4; writes no WAL; secret-bearing read-only SQL may run but must not leak;
- `execute()`: accepts only the execute gate from §3.4; executes locally first, then appends to the in-memory WAL buffer on success; a failed statement must not be recorded;
- a successful `execute()` means only that local execution and the buffer append succeeded; it does not mean the write is durable remotely;
- `flush()`: publishes an immutable WAL segment and then CAS-updates head; it returns success only when the CAS is confirmed, or when reconciliation per §5.8 proves the commit landed;
- any product that needs "a successful call means cross-process durability" must call and await `flush()` before its own success response; that is a product/binding integration policy and does not change the general semantics of `execute()`.

### 5.5 `checkpoint`

A checkpoint must run exclusively inside the operation queue:

1. produce a full backup of the current local database, including every mutation executed so far;
2. upload it to a unique checkpoint key, by stream or by file;
3. record and verify its `size` / SHA-256;
4. CAS head, replacing `base` with the new reference, clearing `wal` and incrementing `manifest.seq`;
5. while the CAS has not committed, the old base/WAL manifest remains authoritative; the local uncommitted WAL buffer must not be lost;
6. only after the CAS succeeds may the local WAL buffer already contained in the checkpoint be cleared.

V1 does not use an incremental base and does not rely on a provider-native `BACKUP TO S3()`.

### 5.6 `close`

A writer close must:

1. stop accepting new operations;
2. wait for the operation queue to drain;
3. attempt a flush;
4. CAS-release the lease once the flush succeeds;
5. close the native connection and clean up local scratch whether or not the remote steps succeeded;
6. make a persistence or lease-release failure visible to the caller.

Any public operation after close returns `closed`. A destructor / `Drop` may reclaim resources on a best-effort basis, but must not masquerade as an explicit close that successfully completed a durability barrier.

### 5.7 Lease, heartbeat and fencing

- a writer must renew periodically; the heartbeat interval must not exceed one third of the lease TTL;
- every head CAS, heartbeat included, must carry the current ETag and must be coordinated inside the operation queue;
- once a writer cannot confirm renewal within the validity window it believes in, it must enter a self-fenced state and refuse new execute/flush/checkpoint calls;
- an ordinary takeover is allowed only when the lease has expired and exceeded the maximum clock-skew allowance the implementation declares;
- an unexpired lease may only be taken over by an explicit administrative `force` operation; force must never be applied automatically during ordinary retries;
- every takeover increments generation; any later head CAS by the old writer must fail and must be mapped to `lease_fenced`;
- a force operation must return or log an explicit warning: unflushed local writes of the original writer may be lost.

Default TTL, heartbeat and clock-skew allowance may be configurable through the binding API, but their defaults, units and precedence rules must be documented in that binding's documentation and must enter the same scenario tests.

### 5.8 Timeout and ambiguous-commit reconciliation

A provider request timeout cannot automatically be treated as a failure: the server may already have committed.

- an immutable PUT with an uncertain response: re-read the same unique key; if `size` and SHA-256 match, treat it as uploaded; mismatching content is `corrupt`; still undecidable is `commit_ambiguous`;
- a head CAS with an uncertain response: re-read head. If the intended immutable key is already referenced, the manifest/seq match expectations, and the lease instance/generation still belong to the current writer, treat it as a success;
- if the re-read still shows the old ETag/state, the CAS may be retried within the same deadline;
- head has changed and ownership was lost → `lease_fenced`;
- success or failure still unprovable within the deadline → `commit_ambiguous`; the caller must never be told it succeeded;
- every retry must have a deadline, backoff and a maximum attempt count, and must not keep writing across a self-fenced state.

---

## 6. Error categories **[FROZEN]**

A language may express these as exceptions, error values or a `Result`, but the following categories must be programmatically distinguishable:

| Category | Trigger |
| --- | --- |
| `not_found` | read-only open of a non-existent object, or an object missing when existing-only was explicitly requested |
| `lease_held` | another unexpired writer holds the lease |
| `lease_fenced` | this instance has lost generation/ETag ownership |
| `engine_incompatible` | the object uses a backup format the current engine cannot restore, or the current `chdb_version()` is below the object's declared `min_reader`, or the engine name is not `chdb` |
| `protocol_unsupported` | unsupported protocol version or feature |
| `corrupt` | bad head schema, missing immutable object, length/checksum mismatch, or incomplete restored content |
| `classification_refused` | statement count, class or target database does not satisfy the entry gate |
| `secret_refused` | a mutation contains secrets and cannot be written to the WAL |
| `engine` | core query/backup/restore error |
| `backend` | provider network, authentication or non-conditional conflict error |
| `timeout` | the operation definitely did not commit and exceeded its deadline |
| `commit_ambiguous` | after reconciliation it is still unprovable whether the remote committed |
| `limit_exceeded` | SQL, WAL segment, head or provider object exceeds a declared V1/implementation limit |
| `closed` | an operation on an object that has completed close; a self-fenced writer returns `lease_fenced` instead |

A provider's precondition failed / 412 must first be interpreted as CAS contention and then mapped using the lease/state context; it must not be lumped in with a plain `backend` error.

Error messages must not contain secret-bearing SQL, provider credentials or unredacted connection parameters.

---

## 7. V1 conformance

### 7.1 ABI

- [ ] backup/restore/classify go through the same core C ABI, with no SQL assembly and no table of SQL regexes
- [ ] query analysis returns statement count, the secret flag and target-database-only proof
- [ ] all platforms export the same symbols, and the Python surface is only a wrapper over the same ABI
- [ ] interop participants use the same Durable V1 contract/fixtures and cover all three engine gates: producer version differs but is compatible, reader below `min_reader`, and `backup_format` too new

### 7.2 Format fixtures

- [ ] `empty-object`
- [ ] read-only open of a missing object → `not_found`
- [ ] `checkpoint-only`
- [ ] `checkpoint-plus-wal`
- [ ] `quoted-database-name`
- [ ] `missing-base` / `missing-wal` → `corrupt`
- [ ] `bad-base-size` / `bad-base-sha256` → `corrupt`
- [ ] `bad-wal-size` / `bad-wal-sha256` → `corrupt`
- [ ] `unknown-reader-feature` → refuse to open
- [ ] `unknown-writer-feature` → read-only allowed, writer lease refused
- [ ] `future-protocol-version` → refuse to open
- [ ] `producer-version-differs-but-compatible` → open
- [ ] `engine-reader-too-old` → `engine_incompatible`
- [ ] `backup-format-too-new` → `engine_incompatible`
- [ ] unknown fields still preserved after `open → execute → flush → checkpoint → close`
- [ ] different JSON key orders and whitespace are mutually readable

### 7.3 Classification scenarios

- [ ] no query/execute method name can bypass analysis
- [ ] multiple statements and `PARALLEL WITH` are refused
- [ ] writes to another database, to system or to an external sink are refused
- [ ] database lifecycle statements are refused
- [ ] every `MUTATING_GLOBAL` is refused
- [ ] secret mutations are refused and the error does not leak the SQL
- [ ] secret-bearing `READ_ONLY` runs but writes no WAL
- [ ] `limit_exceeded` is returned when SQL/WAL/head exceeds a frozen limit
- [ ] UNKNOWN fails closed

### 7.4 Fault matrix

- [ ] in a conditional-create race on the object/head, only one writer succeeds
- [ ] when the WAL PUT succeeds and the head CAS fails, the old manifest stays valid and the local buffer is retained
- [ ] when the head CAS committed but the response was lost, it reconciles to success
- [ ] when the CAS outcome cannot be proven, `commit_ambiguous` is returned
- [ ] when the checkpoint PUT succeeds and the head CAS fails, the old base/WAL is still restorable
- [ ] when restore/replay fails, the partial engine is closed and no partial session is returned
- [ ] a heartbeat concurrent with flush/checkpoint does not break ETag/seq
- [ ] a writer self-fences when heartbeats fail through to expiry
- [ ] after a force takeover, the old writer's next commit is fenced
- [ ] when the close flush fails, the error is visible and local resources are still released

### 7.5 Cross-binding

- [ ] every fixture produced by a writer is read by at least two other bindings
- [ ] at least one real object backend passes the full conditional-write and fault suite
- [ ] every additionally claimed provider passes provider conformance independently
- [ ] Python, Node and Go use the same authoritative source for protocol/scenarios/fixtures
- [ ] Rust joins the same V1 test set once its Session/path lifecycle work lands; its later implementation is not protocol V2

---

## 8. V2 and follow-up work

The following capabilities are out of V1. Each needs its own proposal, feature name, fixtures and upgrade/downgrade policy before implementation starts.

### 8.1 chdb-durable protocol V2 candidates

1. **Preamble / global state**: SQL UDFs, WASM blobs, compression, delete semantics, missing-driver and replay-failure policy. V1 fails closed on every `MUTATING_GLOBAL`.
2. **Incremental checkpoints**: enter the protocol only after resolving cross-machine addressing of the base chain, chain integrity, compaction, merging, maximum chain length and GC; a local absolute `base_file_path` must not be persisted directly.
3. **Parquet / data WAL**: introduce only after defining the data schema, DDL-versus-data ordering, mutation representation, MV replay and the rules for mixing statement and data WAL.
4. **GC / destroy**: safe enumeration of references, orphan grace periods, protection of concurrent readers, recovery windows and explicit confirmation for dangerous operations.
5. **Multi-writer**: needs a conflict model, commit ordering, idempotency keys and merge/rebase semantics; loosening the lease alone is not enough.
6. **Multiple databases / cross-object transactions**: needs a new manifest and atomic commit model.
7. **Engine compatibility matrix and online migration**: V1 already defines the compatibility gate for later engines reading earlier full backups (`backup_format` + `min_reader`); more complex online migration, schema migration, incompatible archive-format transitions and rollback policy are future design work.

### 8.2 Release ordering is not a protocol version

- product policy for a specific downstream adapter is out of scope for this contract;
- `chdb-rust` first fixes its Session/path registry and resource lifecycle, then implements the same Durable V1 binding;
- multiple durable paths in parallel inside one process requires changing the chdb-core EmbeddedServer/path lifecycle model first; that is a follow-up core capability and does not automatically become protocol V2 because of when it is implemented;
- a provider or language binding shipping late reflects implementation progress only and does not automatically become V2;
- archives, configuration or business transactions outside a downstream application's own database should be specified by that application's own binding/integration spec, unless a genuinely cross-application protocol need is demonstrated later.

---

## Appendix A. V1 convergence relative to the earlier draft

The following was adjusted before this freeze:

1. Removed V1 `preamble/`, blobs, UDF replay, the related warnings and fixtures; every `MUTATING_GLOBAL` now fails closed.
2. Restricted both public `query()` and `execute()` to one executable statement; `PARALLEL WITH` is not accepted as a single-statement loophole.
3. Extended core classify from `(class, has_secrets)` into query analysis, adding a statement count and proof that all write targets belong to the given database only.
4. `execute()` now accepts only single-statement, secret-free `MUTATING` writes confined to the durable database; cross-database, system and external-sink writes are refused.
5. Changed JSON from "byte-identical across bindings" to semantic compatibility under a strict schema; key order and whitespace are not frozen.
6. Added length and SHA-256 to base/WAL references, with mandatory verification before restore.
7. Changed the checkpoint backend from whole-object `bytes` to a required file or streaming path, so a full backup no longer has to sit entirely in memory.
8. Made the per-object operation queue explicit, with heartbeats participating in the same CAS serialization.
9. Made heartbeats, self-fencing, expiry takeover and explicit unexpired force takeover explicit.
10. Added the `not_found`, `engine_incompatible`, `timeout`, `commit_ambiguous` and `closed` error categories.
11. Removed V1 `delete_prefix` / destroy; GC and destruction are deferred to a later safety design.
12. Moved incremental checkpoints, Parquet/data WAL, multi-writer, multi-database and cross-version migration explicitly into the V2 candidate list.
