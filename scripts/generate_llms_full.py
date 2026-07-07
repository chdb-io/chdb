#!/usr/bin/env python3
"""Generate llms-full.txt: the full chDB documentation corpus in one file.

Pulls every page under docs/chdb from the ClickHouse/clickhouse-docs repo
(the source the docs site renders at clickhouse.com/docs/chdb), appends
repo-native references (dev-docs/ARCHITECTURE.md and the language-binding
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
import posixpath
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

# Repo-native content appended after the docs corpus:
# (title, canonical URL, local path, repo, directory the file lives in)
EXTRA_SOURCES = [
    (
        "chDB architecture deep dive",
        "https://github.com/chdb-io/chdb/blob/main/dev-docs/ARCHITECTURE.md",
        os.path.join(REPO_ROOT, "dev-docs", "ARCHITECTURE.md"),
        "chdb-io/chdb",
        "dev-docs",
    ),
]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")

# Verified-dead targets in upstream content (404 repo / NXDOMAIN domain).
# Matched by prefix; the link is unwrapped to its anchor text so agents don't
# chase it. The weekly llms-txt-check CI surfaces new candidates for this list.
DEAD_LINK_PREFIXES = (
    "https://github.com/JeffSackmann/tennis_atp",   # dataset repo removed from GitHub
    "https://jupysql.ploomber.io",                  # Ploomber docs domain no longer resolves
)
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


def clean_body(body, doc_ctx=None, gh_repo=None, gh_dir=""):
    # doc_ctx = (source dir of this docs page, {source path: published slug});
    # file-relative links resolve through the slug map because filenames and
    # published slugs diverge (e.g. guides/querying-pandas.md -> /chdb/guides/pandas)
    # Upstream docs occasionally have a non-breaking space after '##', which
    # breaks both markdown rendering and the anchor-stripping below
    body = body.replace("\xa0", " ")
    # Drop MDX import lines — most component tags are kept, matching how
    # platform.claude.com serves its .md pages...
    body = re.sub(r"^import\s+.*?;?\s*$", "", body, flags=re.MULTILINE)
    # ...except <Image img={var}/>, whose img reference dangles once the
    # import is gone — nothing useful survives, so drop the tag.
    body = re.sub(r"<Image\b[^>]*/?>", "", body)

    # Video embeds don't work in a text file; replace each iframe with a plain
    # link (YouTube embed URLs mapped back to their watch form).
    def iframe_to_link(m):
        src = re.search(r'src="([^"]+)"', m.group(0))
        if not src:
            return ""
        url = src.group(1)
        yt = re.match(r"https://www\.youtube(?:-nocookie)?\.com/embed/([\w-]+)", url)
        if yt:
            url = f"https://www.youtube.com/watch?v={yt.group(1)}"
        return f"[Watch the video]({url})"

    body = re.sub(r"<iframe\b.*?(?:</iframe>|/>)", iframe_to_link, body, flags=re.DOTALL)

    # Strip Docusaurus explicit heading anchors: '## Setup {#setup}' -> '## Setup'
    body = re.sub(r"^(#{1,6} .*?)\s*\{#[^}]+\}\s*$", r"\1", body, flags=re.MULTILINE)

    # Rewrite relative markdown link/image targets to absolute URLs — a plain
    # text file has no base URL, so anything relative is dead as served.
    def absolute_target(m):
        target = m.group(2)
        # Leave alone anything with a URI scheme (https:, mailto:, and also
        # file:/data: connection-string examples in the API docs) or in-page anchors
        if re.match(r"^([a-z][a-z0-9+.-]*:|#)", target):
            return m.group(0)
        path, _, frag = target.partition("#")
        frag = f"#{frag}" if frag else ""
        if doc_ctx is not None:
            src_dir, slug_by_path = doc_ctx
            if path.startswith("/"):
                # Site-absolute ('/interfaces/formats') is already a slug path,
                # unless it points at a source file ('/chdb/x.md') — map that
                # through the slug table like a relative link
                resolved = re.sub(r"\.mdx?$", "", path)
                resolved = slug_by_path.get("docs" + path, resolved)
            else:
                fp = posixpath.normpath(posixpath.join(src_dir, path))
                for candidate in (fp, fp + ".md", fp + ".mdx", fp + "/index.md"):
                    if candidate in slug_by_path:
                        resolved = slug_by_path[candidate]
                        break
                else:
                    resolved = "/" + re.sub(r"\.mdx?$", "", fp).removeprefix("docs/")
                    resolved = re.sub(r"/index$", "", resolved) or "/"
            return f"{m.group(1)}{SITE_BASE}{resolved}{frag})"
        if gh_repo is not None:
            p = posixpath.normpath(
                path.lstrip("/") if path.startswith("/") else posixpath.join(gh_dir, path)
            )
            view = "raw" if p.lower().endswith(IMAGE_EXTS) else "blob"
            host = "raw.githubusercontent.com" if view == "raw" else "github.com"
            mid = "main" if view == "raw" else "blob/main"
            return f"{m.group(1)}https://{host}/{gh_repo}/{mid}/{p}{frag})"
        return m.group(0)

    body = re.sub(r"(\[[^\]]*\]\()([^)\s]+)\)", absolute_target, body)

    # Unwrap links whose targets are known to be dead, keeping the anchor text
    for prefix in DEAD_LINK_PREFIXES:
        body = re.sub(
            r"!?\[([^\]]*)\]\(" + re.escape(prefix) + r"[^)]*\)", r"\1", body
        )

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

    page_data = []
    for path in pages:
        raw = fetch(f"https://raw.githubusercontent.com/{DOCS_REPO}/{DOCS_BRANCH}/{path}", token)
        fm, body = parse_frontmatter(raw)
        slug = fm.get("slug") or "/" + path[len("docs/"):].rsplit(".", 1)[0]
        page_data.append((path, fm, slug, body))
    slug_by_path = {p: s for p, _, s, _ in page_data}

    for path, fm, slug, body in page_data:
        title = fm.get("title") or slug
        block = [f"# {title}", "", f"**URL:** {SITE_BASE}{slug}", ""]
        if fm.get("description"):
            block += [fm["description"], ""]
        block.append(clean_body(body, doc_ctx=(posixpath.dirname(path), slug_by_path)))
        blocks.append("\n".join(block))
        print(f"  docs  {path} -> {SITE_BASE}{slug}", file=sys.stderr)

    for title, url, local_path, repo, repo_dir in EXTRA_SOURCES:
        with open(local_path, encoding="utf-8") as f:
            content = f.read()
        cleaned = clean_body(content, gh_repo=repo, gh_dir=repo_dir)
        blocks.append(f"# {title}\n\n**URL:** {url}\n\n{cleaned}")
        print(f"  local {local_path}", file=sys.stderr)

    for title, repo in BINDING_READMES:
        try:
            content = fetch(f"https://raw.githubusercontent.com/{repo}/main/README.md")
        except Exception as err:  # noqa: BLE001 - a missing README should not sink the build
            print(f"  skip  {repo}: {err}", file=sys.stderr)
            continue
        blocks.append(
            f"# {title}\n\n**URL:** https://github.com/{repo}\n\n"
            f"{clean_body(content, gh_repo=repo)}"
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
