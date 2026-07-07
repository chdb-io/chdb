#!/usr/bin/env python3
"""Generate llms-full.txt: the full chDB documentation corpus in one file.

Pulls every page under docs/chdb from the ClickHouse/clickhouse-docs repo
(the source the docs site renders at clickhouse.com/docs/chdb), appends
repo-native references (docs/ARCHITECTURE.md and the language-binding
READMEs), and writes llms-full.txt at the repo root.

Each page becomes a block of:

    # <title>

    **URL:** <canonical clickhouse.com or github.com URL>

    <page body>

so that RAG chunks keep their provenance. Run from anywhere:

    python scripts/generate_llms_full.py

Set GITHUB_TOKEN to raise the GitHub API rate limit (only one API call is
made; unauthenticated is normally fine).
"""

import json
import os
import re
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_REPO = "ClickHouse/clickhouse-docs"
DOCS_PREFIX = "docs/chdb"
DOCS_BRANCH = "main"
SITE_BASE = "https://clickhouse.com/docs"

# Pages are emitted in this prefix order so a truncated read still gets the
# essentials first. Anything unmatched sorts to the end alphabetically.
SECTION_ORDER = [
    "docs/chdb/index.md",
    "docs/chdb/getting-started",
    "docs/chdb/install",
    "docs/chdb/api",
    "docs/chdb/datastore",
    "docs/chdb/guides",
    "docs/chdb/configuration",
    "docs/chdb/debugging",
    "docs/chdb/reference",
]

# Repo-native content appended after the docs corpus.
EXTRA_SOURCES = [
    (
        "chDB architecture deep dive",
        "https://github.com/chdb-io/chdb/blob/main/docs/ARCHITECTURE.md",
        os.path.join(REPO_ROOT, "docs", "ARCHITECTURE.md"),
    ),
]
BINDING_READMES = [
    ("chdb-node — Node.js binding", "chdb-io/chdb-node"),
    ("chdb-bun — Bun binding", "chdb-io/chdb-bun"),
    ("chdb-go — Go binding", "chdb-io/chdb-go"),
    ("chdb-rust — Rust binding", "chdb-io/chdb-rust"),
]


def fetch(url, token=None):
    req = urllib.request.Request(url, headers={"User-Agent": "chdb-llms-full-generator"})
    if token and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def parse_frontmatter(text):
    """Return (frontmatter dict, body). Handles the simple 'key: value' shape
    the docs repo uses; list values (keywords) are ignored."""
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?", text, re.DOTALL)
    if not m:
        return {}, text
    fm = {}
    for line in m.group(1).splitlines():
        if ":" not in line or line.startswith((" ", "\t", "-")):
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip("'\"")
        if value:
            fm[key.strip()] = value
    return fm, text[m.end():]


def clean_body(body):
    # Upstream docs occasionally have a non-breaking space after '##', which
    # breaks both markdown rendering and the anchor-stripping below
    body = body.replace("\xa0", " ")
    # Drop MDX import lines — component tags themselves are kept, matching how
    # platform.claude.com serves its .md pages.
    body = re.sub(r"^import\s+.*?;?\s*$", "", body, flags=re.MULTILINE)
    # Strip Docusaurus explicit heading anchors: '## Setup {#setup}' -> '## Setup'
    body = re.sub(r"^(#{1,6} .*?)\s*\{#[^}]+\}\s*$", r"\1", body, flags=re.MULTILINE)
    # Collapse the blank runs the removals leave behind
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def order_key(path):
    for i, prefix in enumerate(SECTION_ORDER):
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "."):
            return (i, path)
    return (len(SECTION_ORDER), path)


def main():
    token = os.environ.get("GITHUB_TOKEN")

    tree = json.loads(
        fetch(f"https://api.github.com/repos/{DOCS_REPO}/git/trees/{DOCS_BRANCH}?recursive=1", token)
    )
    pages = sorted(
        (
            entry["path"]
            for entry in tree["tree"]
            if entry["type"] == "blob"
            and entry["path"].startswith(DOCS_PREFIX + "/")
            and entry["path"].endswith((".md", ".mdx"))
        ),
        key=order_key,
    )
    if not pages:
        sys.exit(f"no pages found under {DOCS_PREFIX} in {DOCS_REPO}")

    # Reuse the hand-written preamble (H1 + blockquote + agent notes) from llms.txt.
    with open(os.path.join(REPO_ROOT, "llms.txt"), encoding="utf-8") as f:
        llms_txt = f.read()
    preamble = llms_txt.split("\n## ", 1)[0].strip()

    blocks = [
        preamble,
        "This file contains the full chDB documentation corpus. The companion"
        " index is at https://github.com/chdb-io/chdb/blob/main/llms.txt.",
    ]

    for path in pages:
        raw = fetch(f"https://raw.githubusercontent.com/{DOCS_REPO}/{DOCS_BRANCH}/{path}", token)
        fm, body = parse_frontmatter(raw)
        slug = fm.get("slug") or "/" + path[len("docs/"):].rsplit(".", 1)[0]
        title = fm.get("title") or slug
        block = [f"# {title}", "", f"**URL:** {SITE_BASE}{slug}", ""]
        if fm.get("description"):
            block += [fm["description"], ""]
        block.append(clean_body(body))
        blocks.append("\n".join(block))
        print(f"  docs  {path} -> {SITE_BASE}{slug}", file=sys.stderr)

    for title, url, local_path in EXTRA_SOURCES:
        with open(local_path, encoding="utf-8") as f:
            content = f.read()
        blocks.append(f"# {title}\n\n**URL:** {url}\n\n{clean_body(content)}")
        print(f"  local {local_path}", file=sys.stderr)

    for title, repo in BINDING_READMES:
        try:
            content = fetch(f"https://raw.githubusercontent.com/{repo}/main/README.md")
        except Exception as err:  # noqa: BLE001 - a missing README should not sink the build
            print(f"  skip  {repo}: {err}", file=sys.stderr)
            continue
        blocks.append(
            f"# {title}\n\n**URL:** https://github.com/{repo}\n\n{clean_body(content)}"
        )
        print(f"  repo  {repo}/README.md", file=sys.stderr)

    out_path = os.path.join(REPO_ROOT, "llms-full.txt")
    output = "\n\n---\n\n".join(blocks) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(
        f"wrote {out_path}: {len(output):,} bytes, {len(pages)} docs pages"
        f" + {len(blocks) - len(pages) - 2} extra sources",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
