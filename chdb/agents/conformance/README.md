# Cross-language conformance fixture

`cases.jsonl` is a language-neutral list of behaviors every chDB agent-tool
binding must satisfy. It is the executable half of `../CONTRACT.md`.

Each line is one case:

```json
{"id": "...", "pillar": "P1|P2|P3|P4|introspection", "method": "query|call|list_databases|list_tables|describe|get_sample_data|list_functions", "args": {...}, "expect": {...}}
```

`args` for `method: "call"` are `{name, arguments}` (the tool-dispatch path);
for every other method they are the method's keyword arguments.

`expect` is one of:

| key | meaning |
|---|---|
| `rows` | exact row list equality |
| `error_type` | the method must fail with this error `type` (raising path) |
| `truncated` + `row_count` | truncation flag and returned row count |
| `row_count` | returned row count |
| `contains_all` | every listed value present in the returned list |
| `min_len` | returned list length ≥ N |
| `describe_column` | describe result contains a column with this name |
| `envelope_ok` (+ `error_type`) | `call()` envelope `ok` flag (and error type) |

`{{fixtures}}` in any SQL is replaced by the runner with the absolute path to
`./fixtures`.

## Running it

- **Python** (reference): `python -m unittest tests.test_agents_conformance`
  (loads this file, runs one read-only `ChDBTool`, asserts every case).
- **TypeScript** (`chdb-node`): a ~30-line runner loads this same file, maps each
  `method` to `chdbTools()` / the `ChDBVector`-adjacent query surface, and
  asserts identically. The fixture is vendored (or git-submoduled) so both repos
  test the *same* behaviors.

A binding is **contract-conformant** when every case passes. To document an
intentional per-language divergence, add a case with the divergent expectation
and a `desc` explaining why — never diverge silently.
