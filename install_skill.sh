#!/usr/bin/env bash
# Install chdb AI Skill for coding agents (Cursor, Claude Code, Codex, etc.)
set -e

BASE_URL="https://raw.githubusercontent.com/chdb-io/chdb/main/agent/skills/using-chdb"
FILES="SKILL.md reference.md examples.md"

install_to() {
  mkdir -p "$1"
  for f in $FILES; do curl -sL "$BASE_URL/$f" -o "$1/$f"; done
  echo "✓ Installed → $1"
}

installed=0

# Cursor
if [ -d "$HOME/.cursor" ]; then
  install_to "$HOME/.cursor/skills/using-chdb"
  installed=1
fi

# Claude Code
if [ -d "$HOME/.claude" ]; then
  install_to "$HOME/.claude/skills/using-chdb"
  installed=1
fi

# Codex (OpenAI)
codex_home="${CODEX_HOME:-$HOME/.codex}"
if [ -d "$codex_home" ]; then
  install_to "$codex_home/skills/using-chdb"
  installed=1
fi

# None detected — install to all default locations
if [ "$installed" -eq 0 ]; then
  install_to "$HOME/.cursor/skills/using-chdb"
  install_to "$HOME/.claude/skills/using-chdb"
fi
