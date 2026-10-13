"""Render benchmark JSON files as Markdown tables.

    python -m benchmark.perf_guide.report [results_dir] [suite ...]

Defaults to the ``results`` directory next to this file and all suites in it.
"""

import glob
import json
import os
import sys

HIDE = {"case", "args", "runs", "value", "trace", "checksum", "rss_start_mb"}
ORDER = ["wall_s", "peak_rss_mb"]


def fmt(key, value):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if key.endswith("_s"):
            return f"{value:.2f}"
        if key.endswith("_mb"):
            return f"{value:,.0f}"
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def args_text(args):
    return ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""


def render(payload):
    lines = []
    machine = payload.get("machine", {})
    lines.append(f"## {payload['suite']}")
    lines.append("")
    if machine:
        lines.append(
            f"{machine.get('cpu', '?')}, {machine.get('cpu_threads', '?')} threads, "
            f"{machine.get('mem_total_gb', '?')} GB RAM; chdb-core {machine.get('chdb_core', '?')} "
            f"(engine {machine.get('chdb_engine', '?')}), pandas {machine.get('pandas', '?')}, "
            f"pyarrow {machine.get('pyarrow', '?')}; max_threads={machine.get('max_threads')}, "
            f"repeat={machine.get('repeat')}, {machine.get('timestamp', '')}"
        )
        lines.append("")
    if payload.get("datasets"):
        lines.append("Datasets (MB): " + ", ".join(f"{k}={v}" for k, v in payload["datasets"].items()))
        lines.append("")
    cases = payload["cases"]
    extra = []
    for case in cases:
        for key in case:
            if key not in HIDE and key not in ORDER and key not in extra:
                extra.append(key)
    columns = ORDER + extra
    lines.append("| case | args | " + " | ".join(columns) + " |")
    lines.append("|" + " --- |" * (len(columns) + 2))
    for case in cases:
        if "error" in case and "wall_s" not in case:
            lines.append(f"| {case['case']} | {args_text(case['args'])} | FAIL: {case['error'][:80]} |")
            continue
        cells = [fmt(col, case.get(col)) for col in columns]
        lines.append(f"| {case['case']} | {args_text(case['args'])} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv):
    results = (
        argv[1] if len(argv) > 1 and os.path.isdir(argv[1]) else os.path.join(os.path.dirname(__file__), "results")
    )
    suites = [a for a in argv[1:] if not os.path.isdir(a)]
    paths = sorted(glob.glob(os.path.join(results, "*.json")))
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if suites and name not in suites:
            continue
        with open(path, encoding="utf-8") as handle:
            print(render(json.load(handle)))


if __name__ == "__main__":
    main(sys.argv)
