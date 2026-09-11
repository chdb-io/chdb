"""Namespace — addressable objects on one backend, and cross-object query.

`ns.open(id)` mints (or reopens) the object at `<base>/<id>`.

`ns.scan()` runs the same read-only query across several objects and returns
the rows per object. It restores each one in turn, so the cost is linear in the
number of objects: it suits a small fan-out, not cross-object analytics. It is
also *necessarily* sequential — chdb-core allows one active data path per
process (contract §3.6), so two objects cannot be open here at once. Lifting
that is engine work; a registry in the binding would only hide it.
"""
from __future__ import annotations

import time
from typing import Iterable, List, Optional, Tuple

from .backends import make_backend
from .errors import LeaseHeld, NotFound
from .head import Head
from .object import DEFAULT_DATABASE, DurableObject, validate_oid
from .protocol import HEAD_KEY


class Namespace:
    def __init__(self, url: str, *, owner: Optional[str] = None,
                 db: str = DEFAULT_DATABASE, **object_kwargs):
        """`db` names the one database a cold object is created with, and is
        ignored for an object that already exists — its head is authoritative.

        `object_kwargs` are passed to every `DurableObject` this namespace
        opens — `lease_ttl`, `clock_skew`, `heartbeat_interval`,
        `commit_deadline`."""
        self.url = url
        self.owner = owner
        self.db = db
        self.object_kwargs = dict(object_kwargs)

    def open(self, oid: str, *, read_only: bool = False,
             force: bool = False) -> DurableObject:
        backend = make_backend(self.url, oid)
        obj = DurableObject(oid, backend, owner=self.owner, db=self.db,
                            read_only=read_only, **self.object_kwargs)
        try:
            # force=True is the administrative takeover of an unexpired lease;
            # see DurableObject.open.
            obj.open(force=force)
        except BaseException:
            obj.close()  # reclaim the scratch directory if open() failed
            raise
        return obj

    def destroy(self, oid: str, *, force: bool = False) -> None:
        """Delete an object outright.

        **Not part of Durable V1**, which has no destroy and no GC — the
        protocol deliberately leaves deletion to a later design with orphan
        grace periods and concurrent-reader protection (contract §8.1). This
        exists for development and tests; on a shared object it can pull state
        out from under a reader that is mid-restore.
        """
        validate_oid(oid)  # a hierarchical/empty id would delete unrelated objects
        backend = make_backend(self.url, oid)
        if not force:
            # Refuse to delete out from under a live writer — that would lose
            # its head/checkpoints/WAL and fail its next commit.
            raw = backend.get(HEAD_KEY)
            if raw:
                try:
                    lease = Head.parse(raw).lease
                except Exception:
                    lease = None
                if lease is not None and lease.held_at(time.time()):
                    raise LeaseHeld(
                        f"{oid!r} has an active lease; pass force=True to destroy anyway")
        backend.delete_prefix("")

    def scan(self, sql: str, ids: Iterable[str], fmt: str = "CSV",
             *, missing_ok: bool = False) -> List[Tuple[str, str]]:
        """Run `sql` read-only against each object; return [(id, result)].

        A missing object raises `not_found`, the same as opening it would.
        Pass `missing_ok=True` to skip ids that do not exist yet, which is what
        a fan-out over a tenant list usually wants.
        """
        out: List[Tuple[str, str]] = []
        for oid in ids:
            try:
                obj = self.open(oid, read_only=True)
            except NotFound:
                if missing_ok:
                    continue
                raise
            try:
                out.append((oid, obj.query(sql, fmt).data()))
            finally:
                obj.close()
        return out
