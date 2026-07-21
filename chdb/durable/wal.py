"""Write-ahead log — the incremental tier between full checkpoints.

Each write statement is appended to a buffer; `flush()` writes the buffered
statements as one segment and records it in the manifest, so RPO ≈ the flush
interval instead of the (larger) checkpoint interval. On open, the base
checkpoint is restored, then every WAL segment is replayed in order.

V1 is statement-based: each segment is newline-delimited JSON of the write SQL,
and `open()` replays those statements to reconstruct state since the base.

**Correct usage — write only deterministic statements between checkpoints.**
Because recovery *re-executes* the logged SQL, a statement must produce the same
result on replay as it did originally:

  - DO log inserts/updates of *literal* values, e.g.
    `INSERT INTO mem.beliefs VALUES ('k', 'v', '2026-01-01 00:00:00')`.
  - DON'T log statements whose result depends on evaluation time or randomness —
    `now()`, `today()`, `rand()`, `generateUUIDv4()`, `INSERT ... SELECT` from a
    volatile/streaming source, or anything reading external mutable state. On
    replay these yield *different* rows than were originally committed.
  - If you need a timestamp/id, compute it in the caller and log the literal.
  - Non-deterministic or bulk transformations belong in a `checkpoint()` (which
    snapshots actual state), not in WAL statements.

A future V2 may store row data as Parquet segments (so replay is data, not
re-execution, and `ns.scan()` can read segments via `s3()` without restoring),
which would remove this determinism requirement.
"""
from __future__ import annotations

import json
from typing import List


class WalBuffer:
    """Accumulates write statements until flushed to a segment."""

    def __init__(self):
        self._stmts: List[str] = []

    def append(self, sql: str) -> None:
        self._stmts.append(sql)

    def __len__(self) -> int:
        return len(self._stmts)

    def serialize(self) -> bytes:
        return b"".join(json.dumps({"sql": s}).encode() + b"\n" for s in self._stmts)

    def clear(self) -> None:
        self._stmts = []


def replay(segment: bytes, run) -> int:
    """Replay one segment's statements via `run(sql)`. Returns count."""
    n = 0
    for line in segment.splitlines():
        line = line.strip()
        if not line:
            continue
        run(json.loads(line)["sql"])
        n += 1
    return n
