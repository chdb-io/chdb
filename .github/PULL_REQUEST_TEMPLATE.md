<!--
Thanks for contributing to chDB!

Before opening: skim AGENTS.md (repository root) for the contribution
process. The §2 "Things agents should NOT do" list and §10 "PR &
commit conventions" cover the most common review feedback.
-->

## Summary

<!-- One or two sentences describing what this PR changes and why.
     Focus on user impact, not internal mechanics. -->

## Type of change

<!-- Check the one that fits. The same prefix should appear on each
     commit in this PR (e.g. `fix(datastore): ...`, `feat: ...`,
     `ci: ...`). See AGENTS.md Section 10.1. -->

- [ ] `feat` — user-visible new feature
- [ ] `fix` — bug fix
- [ ] `refactor` — non-behavior-changing code change
- [ ] `test` — tests-only change
- [ ] `docs` — documentation-only change
- [ ] `ci` — CI / build infrastructure
- [ ] `chore` — tooling, dependencies, housekeeping

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
- [ ] `flake8 datastore --count --show-source --statistics` passes locally
- [ ] If the change touches DataStore semantics: ran the relevant pytest module on **both** `pandas<3.0` and `pandas>=3.0` (Python ≥ 3.11)
- [ ] User-facing API or behavior changes have updated docs (`docs/`, docstrings, `examples/`)
- [ ] No new dependencies added to `requirements.txt` without prior discussion

## Notes for reviewers

<!-- Optional: design alternatives considered, performance
     concerns, screenshots, anything specific you want the
     reviewer to look at. Delete this section if not used. -->
