#!/usr/bin/env bash
# Install chdb AI Skills for coding agents (Cursor, Claude Code, Codex, etc.)
# Two skills: chdb-datastore (pandas API) and chdb-sql (raw SQL)
set -e

BASE_URL="https://raw.githubusercontent.com/chdb-io/chdb/main/agent/skills"
SKILLS="chdb-datastore chdb-sql"

DATASTORE_FILES="SKILL.md references/connectors.md references/api-reference.md examples/examples.md scripts/verify_install.py"
SQL_FILES="SKILL.md references/api-reference.md references/table-functions.md references/sql-functions.md examples/examples.md scripts/verify_install.py"

install_skill() {
  local skill="$1"
  local dest="$2/$skill"
  local files="$3"
  mkdir -p "$dest/references" "$dest/examples" "$dest/scripts"
  for f in $files; do
    curl -sL "$BASE_URL/$skill/$f" -o "$dest/$f"
  done
  echo "  -> $dest"
}

install_to() {
  local dest="$1"
  echo "Installing to $dest ..."
  install_skill "chdb-datastore" "$dest" "$DATASTORE_FILES"
  install_skill "chdb-sql" "$dest" "$SQL_FILES"
  echo "  Done."
}

# --project flag: install to project-level .agents/skills/
if [ "${1:-}" = "--project" ]; then
  install_to "./.agents/skills"
  echo ""
  echo "Installed to project-level .agents/skills/ (git-committable)."
  echo "  chdb-datastore — pandas-style DataStore API"
  echo "  chdb-sql       — raw ClickHouse SQL queries"
  exit 0
fi

installed=0

# Cursor
if [ -d "$HOME/.cursor" ]; then
  install_to "$HOME/.cursor/skills"
  installed=1
fi

# Claude Code
if [ -d "$HOME/.claude" ]; then
  install_to "$HOME/.claude/skills"
  installed=1
fi

# Codex (OpenAI)
codex_home="${CODEX_HOME:-$HOME/.codex}"
if [ -d "$codex_home" ]; then
  install_to "$codex_home/skills"
  installed=1
fi

# .agents/ (cross-agent convention)
if [ -d "$HOME/.agents" ]; then
  install_to "$HOME/.agents/skills"
  installed=1
fi

# None detected — ask user
if [ "$installed" -eq 0 ]; then
  echo "No coding agent detected."
  echo ""
  echo "Where should the skills be installed?"
  echo "  1) ~/.cursor/skills     (Cursor)"
  echo "  2) ~/.claude/skills     (Claude Code)"
  echo "  3) ~/.codex/skills      (Codex)"
  echo "  4) ./.agents/skills     (project-level, git-committable)"
  echo "  5) All of the above"
  echo ""
  printf "Choose [1-5]: "
  read -r choice
  case "$choice" in
    1) install_to "$HOME/.cursor/skills" ;;
    2) install_to "$HOME/.claude/skills" ;;
    3) install_to "$HOME/.codex/skills" ;;
    4) install_to "./.agents/skills" ;;
    5)
      install_to "$HOME/.cursor/skills"
      install_to "$HOME/.claude/skills"
      install_to "$HOME/.codex/skills"
      ;;
    *) echo "Invalid choice. Exiting." && exit 1 ;;
  esac
fi

echo ""
echo "Installed chdb AI Skills:"
echo "  chdb-datastore — Drop-in pandas replacement (import chdb.datastore as pd)"
echo "  chdb-sql       — In-process ClickHouse SQL (chdb.query(), Session)"
echo ""
echo "Your AI assistant can now write correct chdb code out of the box."
