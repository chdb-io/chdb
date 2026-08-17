# chDB documentation

This directory is the source of truth for the chDB documentation published at
[clickhouse.com/docs/chdb](https://clickhouse.com/docs/chdb). The ClickHouse
docs project consumes it as a Mintlify multi-repo source, so make chDB content
changes here rather than in the ClickHouse documentation repository.

## Contributing

1. Edit the MDX pages and `navigation.json` in this directory.
2. Run `python scripts/generate_llms_full.py` and commit the updated
   `llms-full.txt` when documentation content changes.
3. Open a pull request. `.github/workflows/docs_verify.yml` validates this
   directory against the current ClickHouse documentation site configuration.

Publishing is handled by the ClickHouse docs Mintlify configuration. This
repository no longer opens sync pull requests into `ClickHouse/ClickHouse`, and
documentation changes do not need a `sync-docs` label.

`README.md` and `_static/` are repository-only files and are excluded from the
published docs source. `_static/` contains images referenced by frozen public
URLs such as the organization profile, PyPI release pages, and notebooks.
