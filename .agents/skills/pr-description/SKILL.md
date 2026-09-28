---
name: pr-description
description: Draft or update a pull request title and description for the current repository. Use when preparing, creating, or editing a pull request.
---

# Pull Request Description

Inspect the diff against the base branch and the included commits before
writing.

## Title

- Use a concise, specific, capitalized imperative.
- Describe the outcome rather than the implementation mechanism.
- Do not use Conventional Commit prefixes such as `feat:` or `fix:`.

## Body

- Explain what changed, why it changed, and its impact.
- Link related issues when they are known.
- List only validation that was actually run, including the result.
- Mention breaking changes, limitations, or important reviewer context when
  applicable.

Do not invent behavior, test results, or issue links. Create or edit a remote
pull request only when the user explicitly asks.
