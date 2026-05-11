<!--
Thanks for contributing to chDB!

Before opening: skim CONTRIBUTING.md for setup, the "I changed X — what
to run" table, things to avoid, and PR conventions. AGENTS.md captures
the chdb-ds design and testing principles (kept short — stays loaded
in AI coding agents at all times).
-->


## Summary

<!-- One or two sentences describing what this PR changes and why.
     Focus on user impact, not internal mechanics. -->

## Type of change

<!-- Check the one that fits. PR titles start with a Capitalised
     verb (e.g. "Add ...", "Fix ...", "Improve ..."), no
     Conventional-Commit prefix. See CONTRIBUTING.md → "PR & commit
     conventions". -->

- [ ] New feature
- [ ] Bug fix
- [ ] Refactor (no behavior change)
- [ ] Tests only
- [ ] Documentation only
- [ ] CI / build infrastructure
- [ ] Tooling / dependencies / housekeeping

## Linked issues

<!-- "Fixes #123" closes the issue when merged. "Refs #123" links
     without closing. Delete the line if no issue exists. -->

Fixes #

## How to verify

<!-- The exact command(s) you ran locally and the result. For
     DataStore changes, include the targeted pytest command. -->

```bash
# example
cd datastore && python -m pytest tests/test_<file>.py -v --tb=short
```

## Checklist

- [ ] Tests added or updated for the behavior change (regression test for bug fixes)
- [ ] `ruff check datastore --statistics` passes locally
- [ ] If the change touches DataStore semantics: ran the relevant pytest module on **both** `pandas<3.0` and `pandas>=3.0` (Python ≥ 3.11)
- [ ] User-facing API or behavior changes have updated docs (`docs/`, docstrings, `examples/`)
- [ ] No new dependencies added to `requirements.txt` without prior discussion

## Notes for reviewers

<!-- Optional: design alternatives considered, performance
     concerns, screenshots, anything specific you want the
     reviewer to look at. Delete this section if not used. -->
