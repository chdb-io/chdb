# AGENTS.md — chdb/ shim package

This directory is a **load-bearing shim**, not the place where chDB's
main code lives. It contains exactly two files:

- `__init__.py` — walks `sys.path` so `import chdb` finds
  `chdb-core`'s `_chdb` extension when the wrapper is installed
  editably alongside the engine package in `site-packages`
- `datastore.py` — patches `chdb.__version__` to the wrapper's
  pip-package version (rather than `chdb-core`'s engine version) and
  swaps `sys.modules[__name__]` so `from chdb import datastore`
  resolves to the top-level `datastore` package

Both files look refactor-able. They are not. See
[`../AGENTS.md`](../AGENTS.md) §2.1 for the long-form reasoning.

## Where the actual code lives (not here)

- The chdb-ds project (bulk of this repository): [`../datastore/`](../datastore/)
- The C++ engine (SQL, parser, formats, codecs, ...): consumed as the
  `chdb-core` PyPI package; source is at
  [chdb-io/chdb-core](https://github.com/chdb-io/chdb-core)

## Things you can safely do here

- Add or improve docstrings and comments that *explain* the shim
  mechanics — this directory is undocumented enough that prose
  improvements are welcome
- Fix a real bug in the `sys.path` walking logic (rare; the logic
  is small and has been stable)

## Things you should not do here

- "Clean up" or shorten either file. They are minimal already, and
  what looks like dead code is the shim's correctness invariant.
- Add new public APIs to the `chdb` namespace without an issue and
  maintainer agreement first. The wrapper is intentionally minimal —
  user-facing API surface should grow in `chdb-core` (engine) or
  `datastore/` (DataStore), not here.
- Move logic *out* of the shim into the top-level `datastore/`
  package. The shim's job is to *delegate*, not to compute.
- Edit the `_arrow_format` / `_process_result_format_funs` tables
  without checking how they're consumed downstream — they are
  imported by name from other places.

## Verifying changes

After any edit here, do a clean re-install and smoke-test both
import paths:

```bash
pip uninstall -y chdb && pip install -e .
python -c "import chdb; from chdb import datastore; print(chdb.__version__, datastore.DataStore)"
```

Both must succeed:

- If `import chdb` raises, the `sys.path` bridge in `__init__.py` is
  broken
- If `from chdb import datastore` raises, the `sys.modules` swap in
  `datastore.py` is broken

If you've changed the version-patching code, also confirm the
reported version is the *wrapper* version, not the engine version:

```bash
python -c "import chdb; print('chdb=', chdb.__version__, 'core=', getattr(chdb, 'core_version', None))"
```

`chdb.__version__` should match `pyproject.toml`. `chdb.core_version`
should match the installed `chdb-core` package version.
