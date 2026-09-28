# AGENTS.md — chDB

This repository contains chDB's pandas-compatible DataStore API. The embedded
ClickHouse engine and native bindings live in
[chdb-io/chdb-core](https://github.com/chdb-io/chdb-core).

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for setup, commands, and contributor
workflow.

## Architecture

- Keep DataFrame- and Series-producing operations lazy until their results are
  naturally required.
- Do not call `_execute()` directly or add eager conversions to work around
  planner problems.
- Keep one expression and planning model; select pandas or chDB execution at
  execution time.

## Tests

- Mirror DataStore operations in pandas and compare columns, values, and row
  order when the operation defines it.
- Regression tests should reproduce the user's full operation chain. Run the
  property-based chain tests for dispatcher, planner, or SQL-builder changes.
- Do not weaken assertions or use `reset_index()` to hide incorrect behavior.
- Run the smallest relevant tests first, then the relevant full suite. Report
  only checks that were actually run.

## Repository skills

Shared coding-agent skills live under [`.agents/skills`](./.agents/skills).
