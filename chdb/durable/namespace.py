"""Namespace — addressable objects on one backend, and cross-object query.

`ns.open(id)` mints (or reopens) the object at `<base>/<id>`. `ns.scan()`
runs the same read-only query across many objects and unions the rows.
It restores each object in turn (cost is O(number of objects)); it suits
small fan-outs, not large cross-object analytics.
"""
from __future__ import annotations

import json
import time
from typing import Iterable, List, Tuple

from .backends import make_backend
from .errors import LeaseError
from .object import validate_oid
from .object import DurableObject


class Namespace:
    def __init__(self, url: str, *, owner: str = None, db: str = "mem"):
        self.url = url
        self.owner = owner
        self.db = db

    def open(self, oid: str, *, read_only: bool = False,
             force: bool = False) -> DurableObject:
        backend = make_backend(self.url, oid)
        obj = DurableObject(oid, backend, owner=self.owner, db=self.db,
                            read_only=read_only)
        try:
            obj.open(force=force)  # force=True: crash-recovery takeover (fence-safe)
        except BaseException:
            obj.close()  # reclaim the temp dir if open() failed (e.g. lease held)
            raise
        return obj

    def destroy(self, oid: str, *, force: bool = False) -> None:
        validate_oid(oid)  # a hierarchical/empty id would delete unrelated objects
        backend = make_backend(self.url, oid)
        if not force:
            # refuse to delete out from under a live writer — that would lose
            # its head/checkpoints/WAL and fail its next commit.
            raw = backend.get("head.json")
            if raw:
                lease = json.loads(raw).get("lease", {})
                if lease.get("owner") and lease.get("expires_at", 0) >= time.time():
                    raise LeaseError(
                        f"{oid!r} has an active lease; pass force=True to destroy anyway")
        backend.delete_prefix("")

    def scan(self, sql: str, ids: Iterable[str], fmt: str = "CSV") -> List[Tuple[str, str]]:
        """Run `sql` read-only against each object; return [(id, result)]."""
        out: List[Tuple[str, str]] = []
        for oid in ids:
            obj = self.open(oid, read_only=True)
            try:
                out.append((oid, obj.query(sql, fmt).data()))
            finally:
                obj.close()
        return out
