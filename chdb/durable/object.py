"""The Durable Analytical Object.

An addressable chDB engine whose authoritative state lives in object storage:
a base checkpoint (a chDB File backup) plus WAL segments. The lease and the
manifest live in a *single* `head.json` object so the cold path is few
round-trips, which is what dominates in-region open latency:

  open (warm) = 1 read (head, with etag) + [ take-lease PUT ‖ base GET ]
              + restore + 1 re-assert PUT (confirm the lease survived restore)

The lease is the head's `lease` field; taking it is a conditional replace on
the head's etag, which *is* the generation fence — a superseded writer's next
commit fails the compare-and-set. Local MergeTree is the hot working copy;
the object is a portable folder of open-format files.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from typing import List, Optional
from xml.sax.saxutils import escape as _xml_escape

from chdb import session as chs

from .backends import Backend
from .errors import LeaseError

_HEAD = "head.json"


def validate_oid(oid: str) -> str:
    """Object ids must be flat, non-overlapping keys — no path separators, no
    empty/`.`/`..`. This prevents one id's prefix from containing another's
    (e.g. destroying "tenant" wiping "tenant/user")."""
    if not oid or "/" in oid or "\\" in oid or oid in (".", ".."):
        raise ValueError(f"invalid object id {oid!r}: must be non-empty with no '/' or '\\'")
    return oid


class DurableObject:
    def __init__(self, oid: str, backend: Backend, *, owner: Optional[str] = None,
                 db: str = "mem", read_only: bool = False, lease_ttl: float = 60.0):
        validate_oid(oid)
        if not (lease_ttl > 0) or not math.isfinite(lease_ttl):
            # 0/negative/NaN/inf would set expires_at in the past (or never),
            # breaking single-writer exclusion.
            raise ValueError("lease_ttl must be a positive, finite number of seconds")
        self.oid = oid
        self.backend = backend
        self.owner = owner or uuid.uuid4().hex[:8]
        self.db = db
        self.read_only = read_only
        self.ttl = lease_ttl
        # sanitize the id for the temp-dir prefix — a "/" (e.g. "tenant/user")
        # would make mkdtemp reference a nonexistent subdir and fail.
        _safe = re.sub(r"[^A-Za-z0-9._-]", "_", oid)[:64] or "obj"
        self._work = tempfile.mkdtemp(prefix=f"dao-{_safe}-")
        self._bkp = os.path.join(self._work, "backup")
        os.makedirs(self._bkp, exist_ok=True)
        self._cfg = os.path.join(self._work, "conf.xml")
        with open(self._cfg, "w") as f:
            # XML-escape the path — a TMPDIR containing & or < would otherwise
            # produce malformed config and fail session startup.
            f.write(f"<clickhouse><backups><allowed_path>{_xml_escape(self._bkp)}"
                    f"</allowed_path></backups></clickhouse>")
        self.session: Optional[chs.Session] = None
        # manifest fields live inside head.json
        self.base: Optional[str] = None
        self.wal: List[str] = []
        self.seq = 0
        self.generation = 0
        self._head_etag: Optional[str] = None
        self._buf: List[str] = []  # unflushed write statements
        self._lease_expires = 0.0  # our local view of the lease deadline
        self._instance = uuid.uuid4().hex  # this live instance (owner string may repeat)

    def _now(self) -> float:
        return time.time()

    def _head_body(self, now: float) -> bytes:
        return json.dumps({
            "lease": {"owner": self.owner, "instance": self._instance,
                      "generation": self.generation, "expires_at": now + self.ttl},
            "manifest": {"db": self.db, "base": self.base,
                         "wal": self.wal, "seq": self.seq},
        }).encode()

    def _write_head(self) -> None:
        """Commit the head via compare-and-set on our etag. The CAS *is* the
        fence: if a newer writer took the lease, our etag is stale and this
        raises instead of clobbering their state."""
        now = self._now()
        new_etag = self.backend.replace_if_match(_HEAD, self._head_body(now), self._head_etag)
        if new_etag is None:
            raise LeaseError(f"fenced: object {self.oid} was taken by another writer")
        self._head_etag = new_etag
        # same instant as the persisted expires_at (not now+RTT), so renewal
        # isn't scheduled past the real deadline.
        self._lease_expires = now + self.ttl

    def _start_session(self) -> None:
        self.session = chs.Session(f"{self._work}?config-file={self._cfg}")

    # -- lifecycle --------------------------------------------------------
    def open(self, force: bool = False) -> bool:
        # Read the head first — no chDB session yet, so a held object raises
        # LeaseError before we hit chDB's one-session-per-process limit.
        data, etag = self.backend.get_with_etag(_HEAD)

        if data is None:                       # cold: object does not exist yet
            if not self.read_only:
                self.generation = 1
                now = self._now()
                etag = self.backend.put_if_absent(_HEAD, self._head_body(now))
                if not etag:
                    raise LeaseError("object created concurrently by another writer")
                # adopt the etag from the create itself — a separate head_etag()
                # could race a concurrent open(force=True) and pick up its etag.
                self._head_etag = etag
                self._lease_expires = now + self.ttl
            try:
                self._start_session()
                self.session.query(f"CREATE DATABASE IF NOT EXISTS {self.db}")
            except Exception:
                self._abort_open()
                raise
            return False

        head = json.loads(data)
        m = head["manifest"]
        # honor the persisted database name — the object owns it. Keeping a
        # caller-supplied self.db would rewrite the manifest and make _restore
        # build the wrong database (WAL replay then fails).
        self.db = m.get("db", self.db)
        self.base, self.wal, self.seq = m.get("base"), list(m.get("wal", [])), m.get("seq", 0)

        if self.read_only:
            base_blob = self.backend.get(self.base) if self.base else None
            try:
                self._start_session()
                self._restore(base_blob)
            except Exception:
                self._abort_open()  # free the chDB session on a failed restore
                raise
            return True

        lease = head.get("lease", {})
        owner, exp = lease.get("owner", ""), lease.get("expires_at", 0)
        inst = lease.get("instance", "")
        # Held iff a *different live instance* still owns it. An unexpired lease
        # blocks even the same owner string — it may be another live instance,
        # so same-owner is NOT a free pass (that would fence the live holder and,
        # via chDB's one-session limit, abort to owner="" and let a third writer
        # in). Only released (owner=""), expired, or our own instance is free.
        # force=True is the crash-recovery takeover: skip the held check and
        # take the lease anyway. It's safe because the head CAS fences the old
        # holder — its next commit fails — so we don't need to wait out the TTL.
        if not force and owner != "" and exp >= self._now() and inst != self._instance:
            raise LeaseError(
                f"object held by {owner!r}/{inst[:8]} (gen {lease.get('generation')})")
        self.generation = lease.get("generation", 0) + 1

        # overlap the lease-take (CAS PUT) with the base download — independent
        # once we know the base key, so wall-clock ≈ one round-trip not two.
        now = self._now()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_lease = ex.submit(self.backend.replace_if_match, _HEAD, self._head_body(now), etag)
            f_base = ex.submit(self.backend.get, self.base) if self.base else None
            new_etag = f_lease.result()
            base_blob = f_base.result() if f_base else None
        if new_etag is None:
            raise LeaseError("lease taken by another writer during open")
        self._head_etag = new_etag
        self._lease_expires = now + self.ttl
        try:
            self._start_session()
            self._restore(base_blob)
            # A slow restore can outlast lease_ttl; re-assert the lease so we
            # never return a writable session whose lease already expired and
            # was taken by another writer. The CAS fails if we were superseded.
            self._write_head()
        except Exception:
            self._abort_open()  # release the just-taken lease so we don't block others
            raise
        return True

    def _abort_open(self) -> None:
        """A failure after the lease was taken must not strand it until TTL:
        release it via CAS (owner="") and drop the partial session."""
        if not self.read_only and self._head_etag is not None:
            saved = self.owner
            try:
                self.owner = ""  # serialize an empty owner into the release write only
                self._write_head()
            except Exception:
                pass
            finally:
                self.owner = saved  # don't leave the instance owner-less for a retry
            self._head_etag = None  # released here; close() must not release again
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None

    def _restore(self, base_blob: Optional[bytes]) -> None:
        if self.base and base_blob is None:
            # the manifest references a base checkpoint but it's gone from
            # storage — fail loudly rather than silently opening empty and
            # replacing committed state (same policy as missing WAL segments).
            raise RuntimeError(f"missing base checkpoint {self.base} — durable state incomplete")
        if base_blob is not None:
            local = os.path.join(self._bkp, "base.tar.gz")
            with open(local, "wb") as f:
                f.write(base_blob)
            self.session.query(f"RESTORE DATABASE {self.db} FROM File('{local}')", "CSV")
        else:
            self.session.query(f"CREATE DATABASE IF NOT EXISTS {self.db}")
        from .wal import replay
        for seg_key in self.wal:
            seg = self.backend.get(seg_key)
            if seg is None:
                # every WAL key was committed in the head manifest — a missing
                # object means incomplete/corrupt state; fail loudly, don't
                # silently open with committed writes omitted.
                raise RuntimeError(f"missing WAL segment {seg_key} — durable state incomplete")
            replay(seg, lambda sql: self.session.query(sql, "CSV"))

    # -- read / write -----------------------------------------------------
    def query(self, sql: str, fmt: str = "CSV"):
        return self.session.query(sql, fmt)

    def execute(self, sql: str) -> None:
        if self.read_only:
            raise RuntimeError("object opened read-only")
        self.session.query(sql, "CSV")
        self._buf.append(sql)
        # Renew the lease before it lapses so a long-lived writer isn't fenced by
        # another writer while still active. No extra round-trip until near expiry.
        if self._now() >= self._lease_expires - self.ttl * 0.25:
            self._write_head()

    # -- durability -------------------------------------------------------
    def _reconcile_committed(self, *, base=None, wal_key=None) -> bool:
        """After an ambiguous `_write_head` failure (the CAS response may have
        been lost), re-read the head. Because our keys are unique, if the head
        now references our key then the CAS *did* commit — adopt its etag and
        treat the write as durable, instead of rolling back (which would make a
        retry double-publish the same boundary)."""
        data, etag = self.backend.get_with_etag(_HEAD)
        if data is None:
            return False
        head = json.loads(data)
        m = head.get("manifest", {})
        lease = head.get("lease", {})
        key_ok = (base is not None and m.get("base") == base) or \
                 (wal_key is not None and wal_key in (m.get("wal") or []))
        # A retained key proves our write is durable — but we must ALSO still
        # own the lease. If another writer took over (kept the key), adopting
        # their etag would let us CAS over them and defeat the fence.
        owns = (lease.get("instance") == self._instance
                and lease.get("generation") == self.generation)
        if key_ok and owns:
            self._head_etag = etag
            self._buf = []
            return True
        return False

    def flush(self) -> Optional[str]:
        """Cut a WAL segment and commit the head (RPO boundary). One PUT for the
        segment + one CAS for the head."""
        if self.read_only or not self._buf:
            return None
        from .wal import WalBuffer
        wb = WalBuffer()
        for s in self._buf:
            wb.append(s)
        new_seq = self.seq + 1
        # unique suffix so a retry after an ambiguous head CAS (committed but
        # the response was lost) never overwrites an already-published segment.
        key = f"wal/{self.generation}-{new_seq}-{uuid.uuid4().hex[:8]}.jsonl"
        self.backend.put(key, wb.serialize())
        # Adopt the new manifest state only if the head CAS succeeds; on failure
        # roll back (keeping _buf) so a retry publishes a fresh unique segment.
        # The uploaded-but-unreferenced segment is left as an orphan — see the
        # GC TODO on checkpoint().
        prev_seq, prev_wal = self.seq, list(self.wal)
        self.seq, self.wal = new_seq, prev_wal + [key]
        try:
            self._write_head()
        except Exception:
            if self._reconcile_committed(wal_key=key):
                return key  # CAS actually landed; the response was lost
            self.seq, self.wal = prev_seq, prev_wal
            raise
        self._buf = []
        return key

    def checkpoint(self) -> str:
        """Fold base + WAL into a fresh base; truncate the WAL. One PUT for the
        base + one CAS for the head.

        Checkpoints are *full* snapshots. Incremental checkpoints (ClickHouse
        `base_backup`) are planned once native BACKUP-to-S3 ships in a chdb
        release (chdb-core #133/#134), to cut cost for larger objects.

        TODO(gc): no garbage collection yet. A superseded base checkpoint, the
        folded WAL segments, and orphan segments/checkpoints left by failed or
        ambiguous flushes all remain in object storage indefinitely — only the
        current base + live WAL are referenced by the head. A future
        compaction/GC pass should delete blobs the head no longer references
        (guarded so an in-flight reader of an older head isn't cut off).
        """
        if self.read_only:
            raise RuntimeError("object opened read-only")
        new_seq = self.seq + 1
        stamp = uuid.uuid4().hex[:8]  # unique key: retry after an ambiguous CAS won't overwrite
        local = os.path.join(self._bkp, f"ckpt-{self.generation}-{new_seq}-{stamp}.tar.gz")
        if os.path.exists(local):
            os.remove(local)
        self.session.query(f"BACKUP DATABASE {self.db} TO File('{local}')", "CSV")
        with open(local, "rb") as f:
            data = f.read()
        key = f"checkpoints/{self.generation}-{new_seq}-{stamp}.tar.gz"
        self.backend.put(key, data)
        # Adopt the new base only after the head CAS succeeds (roll back on failure).
        prev = (self.seq, self.base, list(self.wal))
        self.seq, self.base, self.wal = new_seq, key, []
        try:
            self._write_head()
        except Exception:
            if self._reconcile_committed(base=key):
                self._buf = []
                return key  # CAS actually landed; the response was lost
            self.seq, self.base, self.wal = prev
            raise
        self._buf = []
        return key

    def suspend(self) -> None:
        self.flush()

    def close(self) -> None:
        try:
            if self.read_only:
                return
            if self.session is not None:
                # If we've been fenced, flush() raises — let it propagate so the
                # caller learns their buffered writes were NOT persisted rather
                # than close() returning normally and silently dropping them.
                # (The session + temp dir are still reclaimed by the finally.)
                self.flush()
            # release the lease (owner="") via CAS so a next writer takes it cleanly
            if self._head_etag is not None:
                self.owner = ""
                try:
                    self._write_head()
                except LeaseError:
                    pass
        finally:
            if self.session is not None:
                self.session.close()
                self.session = None
            # reclaim the per-instance temp dir (config + local backup scratch)
            shutil.rmtree(self._work, ignore_errors=True)
