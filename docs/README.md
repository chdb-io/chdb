# chDB docs — staging copy (not yet the source of truth)

This directory holds a snapshot of the official chDB documentation in its
published form: the `docs/products/chdb` slice of the aggregated docs tree
in [ClickHouse/ClickHouse](https://github.com/ClickHouse/ClickHouse/tree/master/docs/products/chdb)
(44 `.mdx` pages + `navigation.json`), taken on 2026-07-08.

**Do not edit documentation here yet.** Until the docs-platform cutover
(July 17, 2026) has stabilized, content changes still go through the
current live pipeline; this copy exists so that structure, tooling, and
CI checks can be prepared ahead of time:

- `.github/workflows/docs_verify.yml` validates this directory as a
  drop-in replacement for the aggregator slice (same mechanism as
  clickhouse-connect).
- `scripts/generate_llms_full.py --docs-source local` builds
  `llms-full.txt` from this directory.

After the cutover stabilizes, a final re-sync of this snapshot will be
taken, this directory becomes the single source of truth for chDB docs,
and changes flow the other way: edits here → sync PR → aggregated docs →
clickhouse.com/docs/chdb. This notice will be replaced by contributor
instructions at that point.

`_static/` is unrelated to the docs build: it holds images referenced by
frozen public URLs (org profile, PyPI release pages, notebooks) and is
excluded from any sync.
