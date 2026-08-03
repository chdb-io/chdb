# chDB documentation

This directory is the source of truth for the chDB documentation published at
[clickhouse.com/docs/chdb](https://clickhouse.com/docs/chdb). Make chDB content
changes here rather than in the mirrored
[`ClickHouse/ClickHouse/docs/chdb`](https://github.com/ClickHouse/ClickHouse/tree/master/docs/chdb)
directory.

## Contributing

1. Edit the MDX pages and `navigation.json` in this directory.
2. Run `python scripts/generate_llms_full.py` and commit the updated
   `llms-full.txt` when documentation content changes.
3. Open a pull request. `.github/workflows/docs_verify.yml` validates this
   directory against the current ClickHouse documentation site configuration.

To publish a documentation change immediately, add the `sync-docs` label to
its pull request before merging it. The merged pull request then triggers
`.github/workflows/docs_sync.yml`; a published chDB release or a manual
workflow dispatch also triggers it. The workflow opens or refreshes an
automated pull request which mirrors these pages into
`ClickHouse/ClickHouse/docs/chdb`; merging that pull request deploys the content
through the normal ClickHouse documentation pipeline.

`README.md` and `_static/` are repository-only files and are excluded from the
published mirror. `_static/` contains images referenced by frozen public URLs
such as the organization profile, PyPI release pages, and notebooks.
