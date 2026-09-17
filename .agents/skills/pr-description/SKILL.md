---
name: pr-description
description: Draft or update pull request titles and descriptions for chdb-io/chdb. Use when preparing, creating, or editing a PR in this repository.
---

# chDB PR Description

Inspect the branch diff against its base and the commits included in the pull
request. Describe only behavior and validation supported by that evidence. Do
not create or edit a remote pull request unless the user explicitly asks.

## Title

- Start with a capitalized imperative verb and state the user-visible outcome.
- Be specific enough to distinguish the change without reading the body.
- Do not use Conventional Commit prefixes such as `feat:`, `fix:`, or `chore:`.

## Body

Use this structure and remove placeholder text that does not apply:

### Summary

Explain the user impact in one or two sentences. For a non-trivial change, add
brief context about the problem, approach, and important tradeoffs.

### Type of change

Select exactly one:

- New feature
- Bug fix
- Refactor
- Tests only
- Documentation only
- CI / build
- Tooling / dependencies / housekeeping

### Linked issues

Use `Fixes #123` for an issue closed by the PR or `Refs #123` for a related
issue. Omit this section when there is no linked issue.

### How to verify

List only commands and checks that were actually run, with their results. For
DataStore behavior changes, identify the relevant targeted or full test suite
and note pandas-version coverage when it matters.

### Checklist

Mark an item complete only when it is true:

- Tests were added or updated for behavior changes.
- `ruff` passes.
- pandas `<3` and `>=3` were considered for pandas-semantics changes.
- User-facing API changes are documented.
- No dependency was added without prior discussion.

### Notes for reviewers

Use this optional section for design choices, alternatives, performance
considerations, known limitations, or areas that deserve close review.

## Avoid

- Vague titles, internal ticket language, and openings such as “This PR”.
- Descriptions that list implementation details without explaining impact.
- Invented tests, results, issue links, or unsupported claims.
- Boilerplate tables or overly regular prose that makes the change harder to
  review.

AI-assisted wording is acceptable, but the author must review the final text
for accuracy. When the user asks to apply the description, use the repository's
GitHub tooling; otherwise return a draft for review.
