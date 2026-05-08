# AGENTS.md — agent_skills/ (chDB's published Skills)

This directory ships **chDB's published AI coding skills** — content
that end users install into their Cursor / Claude Code / Codex via
[`../install_skill.sh`](../install_skill.sh). This is *content shipped
outward to users*, not contributor guidance. The contributor guide is
[`../AGENTS.md`](../AGENTS.md) at the repository root.

## Two skills, two purposes

| Skill | Audience | What it teaches the user's agent |
|---|---|---|
| `chdb-datastore/` | users who want a pandas-style API | how to use `DataStore`, `from_uri`, `connect`, lazy execution |
| `chdb-sql/` | users who want raw ClickHouse SQL | how to write `chdb.query(...)` against files / S3 / remote sources |

Don't blur the boundary. If a feature belongs in both skills, factor
the shared piece into `references/` and link from both — don't copy
prose between `SKILL.md` files.

## Skill directory layout

Every skill follows this layout:

```
<skill_name>/
├── SKILL.md                  ← skill manifest + main prompt the agent reads
├── references/               ← reference docs the agent loads on demand
│   ├── api-reference.md
│   └── ...                   ← skill-specific
├── examples/
│   └── examples.md
└── scripts/
    └── verify_install.py
```

## File names are part of the install contract

[`../install_skill.sh`](../install_skill.sh) hard-codes the list of
files to download per skill. **If you add a new file, you must also
update `install_skill.sh`** in the same change — otherwise users
won't receive it.

The current contracts (verbatim from the install script):

```
DATASTORE_FILES="SKILL.md references/connectors.md references/api-reference.md examples/examples.md scripts/verify_install.py"
SQL_FILES="SKILL.md references/api-reference.md references/table-functions.md references/sql-functions.md examples/examples.md scripts/verify_install.py"
```

If a file isn't in the list above, it isn't shipped — review pages,
drafts, and notes do not need to be (and should not be) added.

## Verifying changes

After editing a skill, install it locally and inspect the result:

```bash
bash install_skill.sh --project
ls ./.agents/skills/chdb-datastore ./.agents/skills/chdb-sql
diff -r agent_skills/chdb-datastore ./.agents/skills/chdb-datastore   # should be empty
```

Any difference means either a missing entry in `install_skill.sh` or
a path typo.

## Things to avoid

- **Don't add vendor-specific frontmatter** to `SKILL.md` (e.g.
  Cursor-only fields, Claude-Code-only fields). The same file is
  consumed across vendors — use vendor-neutral conventions.
- **Don't link to repository-internal paths** from inside `SKILL.md`
  or `references/*`. Skills are copied to the user's machine
  standalone; relative repo links break. Link to public
  documentation URLs (e.g. `https://chdb.readthedocs.io/...`) or
  inline the content.
- **Don't add scripts requiring extra dependencies** beyond what the
  user already has (`chdb`, the standard library). Skills are meant
  to be zero-config — `verify_install.py` should run on a clean
  `pip install chdb` venv with nothing else.
- **Don't add a third skill silently.** A new skill is a roadmap
  decision; raise it on an issue first so naming, scope, and
  cross-references stay coherent.
