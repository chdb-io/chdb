"""The durable object: a chDB database whose authoritative state is in object
storage, and the state machine that keeps the two honest.

State lives as a full checkpoint plus a statement WAL under one prefix, with
the lease and the manifest in a single `head.json`. Putting them together is
what makes the cold path cheap and the fence exact:

  open (warm) = 1 read (head + etag) + [ take-lease CAS ‖ base download ]
              + restore + 1 re-assert CAS (the lease survived the restore)

Taking the lease is a conditional replace on the head's ETag, and that CAS
*is* the generation fence — a superseded writer's next commit fails it. Local
MergeTree is the hot working copy; the object is a portable folder of
open-format files.

Three invariants are worth stating outright, because most of the code below
exists to hold them:

* **Nothing is claimed durable until a CAS says so.** `execute()` runs a
  statement locally and buffers it. `flush()` publishes it. A caller who
  promises a request is recoverable elsewhere has to wait for `flush()`.
* **A lost response is not a failure.** An indeterminate conditional write is
  reconciled by looking (§5.8), and only reported as `commit_ambiguous` when
  looking cannot settle it. Reporting success we cannot prove is the one
  outcome that is never allowed.
* **Every public statement goes through chdb-core's parser.** The entry gates
  in `_gate_query` and `_gate_execute` are built on `classify_query`, not on
  prefixes or regexes over SQL text (§3.3).

See `dev-docs/CHDB_DURABLE_V1_CONTRACT.md` for the contract these implement.
"""
from __future__ import annotations

import concurrent.futures
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import warnings
from typing import List, Optional

from .digest import bytes_digest, file_digest, verify_bytes, verify_file
from .engine import ManagedConnection, engine_version, require_engine_compatible, require_v1_abi
from .errors import (
    BackendAmbiguous,
    BackendError,
    ClassificationRefused,
    Closed,
    CommitAmbiguous,
    Corrupt,
    DurableError,
    LeaseFenced,
    LeaseHeld,
    LimitExceeded,
    NotFound,
    ProtocolUnsupported,
    SecretRefused,
)
from .head import Head, Lease, ObjectRef
from .protocol import (
    BACKUP_FORMAT,
    COMMIT_BACKOFF_BASE,
    COMMIT_MAX_ATTEMPTS,
    DEFAULT_CLOCK_SKEW,
    DEFAULT_COMMIT_DEADLINE,
    DEFAULT_DATABASE,
    DEFAULT_LEASE_TTL,
    HEAD_KEY,
    HEARTBEAT_TTL_FRACTION,
    PROTOCOL_VERSION,
    SUPPORTED_READER_FEATURES,
    SUPPORTED_WRITER_FEATURES,
)
from .wal import WalBuffer, replay

_NEW, _OPEN, _CLOSING, _CLOSED = "new", "open", "closing", "closed"


def validate_oid(oid: str) -> str:
    """Object ids must be flat, non-overlapping keys — no path separators, no
    empty/`.`/`..`. This prevents one id's prefix from containing another's
    (e.g. destroying "tenant" wiping "tenant/user")."""
    if not oid or "/" in oid or "\\" in oid or oid in (".", ".."):
        raise ValueError(f"invalid object id {oid!r}: must be non-empty with no '/' or '\\'")
    return oid


def _positive_finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number of seconds")
    value = float(value)
    if not (value > 0) or not math.isfinite(value):
        # 0/negative/NaN/inf would put expires_at in the past (or never),
        # breaking single-writer exclusion.
        raise ValueError(f"{name} must be a positive, finite number of seconds")
    return value


class DurableObject:
    """One addressable durable object. Not shared between processes.

    All public operations serialize on a single per-object lock, together with
    the heartbeat's head CAS (§5.3). The contract is explicit that "the runtime
    is single-threaded" and "the native call is synchronous" are not
    serialization guarantees, so the lock is real rather than notional.
    """

    def __init__(self, oid: str, backend, *, owner: Optional[str] = None,
                 db: str = DEFAULT_DATABASE, read_only: bool = False,
                 lease_ttl: float = DEFAULT_LEASE_TTL,
                 clock_skew: float = DEFAULT_CLOCK_SKEW,
                 heartbeat_interval: Optional[float] = None,
                 commit_deadline: float = DEFAULT_COMMIT_DEADLINE):
        validate_oid(oid)
        self.oid = oid
        self.backend = backend
        self.owner = owner or uuid.uuid4().hex[:8]
        self.db = db
        self.read_only = read_only
        self.ttl = _positive_finite(lease_ttl, "lease_ttl")
        self.commit_deadline = _positive_finite(commit_deadline, "commit_deadline")
        if not isinstance(clock_skew, (int, float)) or isinstance(clock_skew, bool) \
                or clock_skew < 0 or not math.isfinite(clock_skew):
            raise ValueError("clock_skew must be a non-negative, finite number of seconds")
        self.clock_skew = float(clock_skew)
        if heartbeat_interval is None:
            heartbeat_interval = self.ttl * HEARTBEAT_TTL_FRACTION
        self.heartbeat_interval = _positive_finite(heartbeat_interval, "heartbeat_interval")
        if self.heartbeat_interval > self.ttl * HEARTBEAT_TTL_FRACTION + 1e-9:
            # §5.7: a heartbeat slower than a third of the TTL cannot survive a
            # single lost renewal, which is the case it exists for.
            raise ValueError(
                f"heartbeat_interval {self.heartbeat_interval}s exceeds a third of "
                f"lease_ttl {self.ttl}s")

        #: This live instance. The owner string may repeat across processes; a
        #: lease is only ours if the instance matches too.
        self._instance = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._state = _NEW
        self._fenced = False
        self._head: Optional[Head] = None
        self._head_etag: Optional[str] = None
        self._lease_expires = 0.0  # our local view of the lease deadline
        self._buf = WalBuffer()
        self._engine: Optional[ManagedConnection] = None
        self._work: Optional[str] = None
        self._scratch: Optional[str] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop: Optional[threading.Event] = None

    # -- introspection ----------------------------------------------------
    @property
    def generation(self) -> int:
        return 0 if self._head is None else self._head.lease.generation

    @property
    def seq(self) -> int:
        return 0 if self._head is None else self._head.manifest.seq

    @property
    def base(self) -> Optional[ObjectRef]:
        return None if self._head is None else self._head.manifest.base

    @property
    def wal(self) -> List[ObjectRef]:
        return [] if self._head is None else list(self._head.manifest.wal)

    @property
    def pending(self) -> int:
        """Statements executed locally but not yet published by `flush()`."""
        return len(self._buf)

    @property
    def fenced(self) -> bool:
        return self._fenced

    @property
    def closed(self) -> bool:
        return self._state == _CLOSED

    def __enter__(self) -> "DurableObject":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        """Reclaim local resources only — never a stand-in for `close()`.

        §5.6 allows a destructor to tidy up but not to impersonate a
        durability barrier, so this touches no network: it will not flush, and
        it will not release the lease (which then expires on its own TTL). It
        exists so a forgotten `close()` does not leave a multi-gigabyte scratch
        directory and an open engine behind.
        """
        try:
            if self._state != _CLOSED:
                self._teardown_local()
        except Exception:
            pass

    def _now(self) -> float:
        return time.time()

    # -- head I/O ---------------------------------------------------------
    def _read_head(self):
        data, etag = self.backend.get_with_etag(HEAD_KEY)
        if data is None:
            return None, None
        return Head.parse(data), etag

    def _owns(self, lease: Lease) -> bool:
        return lease.instance == self._instance and lease.generation == self.generation

    def _fence(self) -> None:
        self._fenced = True

    def _check_protocol(self, head: Head, *, for_writer: bool) -> None:
        if head.protocol.version > PROTOCOL_VERSION:
            raise ProtocolUnsupported(
                f"object declares protocol version {head.protocol.version}; "
                f"this implementation speaks {PROTOCOL_VERSION}")
        unknown_reader = sorted(set(head.protocol.reader_features) - SUPPORTED_READER_FEATURES)
        if unknown_reader:
            raise ProtocolUnsupported(
                f"object requires reader feature(s) {unknown_reader} this implementation "
                f"does not have; refusing to read it")
        unknown_writer = sorted(set(head.protocol.writer_features) - SUPPORTED_WRITER_FEATURES)
        if unknown_writer and for_writer:
            # §4.3: the object is still readable — only writing it would risk
            # dropping semantics we do not implement.
            raise ProtocolUnsupported(
                f"object requires writer feature(s) {unknown_writer} this implementation "
                f"does not have; open it read-only instead")

    def _check_engine(self, head: Head) -> None:
        require_engine_compatible(
            engine_name=head.engine.name,
            backup_format=head.engine.backup_format,
            min_reader=head.engine.min_reader,
        )

    # -- lifecycle --------------------------------------------------------
    def open(self, force: bool = False) -> bool:
        """Acquire the object. Returns True if it already existed.

        `force=True` is the administrative takeover of §5.7: it takes a lease
        that has *not* expired, which is only correct when the holder is known
        to be gone. It is safe against a live holder in the sense that the CAS
        fences them, but their unflushed writes are lost — never reach for it
        as a retry.
        """
        with self._lock:
            if self._state != _NEW:
                raise DurableError(f"object {self.oid!r} has already been opened")
            # Before any storage round-trip: an engine without the Durable V1
            # ABI cannot serve this object at all, and saying so here beats
            # failing halfway through an open.
            require_v1_abi()
            head, etag = self._read_head()

            if self.read_only:
                if head is None:
                    # §5.2: a reader never creates the object it came to read.
                    raise NotFound(f"object {self.oid!r} does not exist")
                self._check_protocol(head, for_writer=False)
                self._check_engine(head)
                self._head, self._head_etag = head, etag
                self.db = head.manifest.db  # the object owns its database name
                try:
                    self._start_engine()
                    self._restore()
                except BaseException:
                    self._abort_open()
                    raise
                self._state = _OPEN
                return True

            existed = head is not None
            if head is None:
                self._create_cold()
            else:
                self._check_protocol(head, for_writer=True)
                self._check_engine(head)
                # The object owns its database name. Keeping a caller-supplied
                # one would rewrite the manifest and make the restore build a
                # database the WAL statements do not name.
                self.db = head.manifest.db
                self._take_lease(head, etag, force=force)

            try:
                self._start_engine()
                self._restore()
                if existed:
                    # A slow restore can outlast the TTL. Re-assert before
                    # handing back a writable object, so we never return one
                    # whose lease already expired and was taken (§5.2 step 7).
                    self._cas_head()
                self._engine.use_database(self.db)
            except BaseException:
                self._abort_open()
                raise
            self._state = _OPEN
            self._start_heartbeat()
            return existed

    def _create_cold(self) -> None:
        """Create the object with one conditional PUT, or lose the race."""
        now = self._now()
        version = engine_version()
        head = Head.cold(
            db=self.db,
            engine_version=version,
            lease=Lease(generation=1, owner=self.owner, instance=self._instance,
                        expires_at=now + self.ttl),
        )
        etag = self.backend.put_bytes_if_absent(HEAD_KEY, head.to_bytes())
        if etag is None:
            raise LeaseHeld(f"object {self.oid!r} was created concurrently by another writer")
        # Adopt the etag the create itself returned: a follow-up read could
        # pick up a *different* writer's head and let us CAS over them.
        self._head, self._head_etag = head, etag
        self._lease_expires = now + self.ttl

    def _take_lease(self, head: Head, etag: str, *, force: bool) -> None:
        now = self._now()
        lease = head.lease
        # Held iff a *different live instance* still owns it, allowing for the
        # clock skew between whoever wrote the expiry and us (§5.7). An
        # unexpired lease blocks even the same owner *string*: it may be
        # another live instance, and fencing a live holder is exactly what the
        # lease exists to prevent.
        if lease.instance != self._instance and lease.held_at(now, clock_skew=self.clock_skew):
            if not force:
                raise LeaseHeld(
                    f"object {self.oid!r} is held by {lease.owner!r}/{(lease.instance or '')[:8]} "
                    f"(generation {lease.generation})")
            # §5.7: taking a lease that has not expired is an administrative
            # act with a cost, and the cost has to be said out loud — the
            # previous writer's buffered, unflushed statements are gone.
            warnings.warn(
                f"force takeover of {self.oid!r}: the lease held by {lease.owner!r}/"
                f"{(lease.instance or '')[:8]} had not expired, and any writes it had "
                f"executed but not flushed are lost",
                RuntimeWarning, stacklevel=3)
        head.lease = Lease(
            generation=lease.generation + 1,  # every takeover moves the fence
            owner=self.owner,
            instance=self._instance,
            expires_at=now + self.ttl,
            extra=lease.extra,
        )
        head.engine.version = engine_version()  # who wrote this object last
        self._head = head

        # Overlap the lease CAS with the base download: independent once the
        # base key is known, so the wall clock is one round-trip, not two.
        base = head.manifest.base
        body = head.to_bytes()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            cas = pool.submit(self.backend.replace_if_match, HEAD_KEY, body, etag)
            fetch = pool.submit(self._prefetch_base, base) if base else None
            try:
                try:
                    new_etag = cas.result()
                except BackendAmbiguous:
                    # We do not know whether we hold the lease. Re-read: our
                    # own instance in the head is proof, anything else is a
                    # refusal.
                    new_etag = self._confirm_lease_taken()
            finally:
                # Harvest the download even when the lease attempt failed, or
                # its temporary directory — holding a whole base archive —
                # would be left behind with nobody holding its name.
                if fetch is not None:
                    try:
                        self._prefetched = fetch.result()
                    except Exception:
                        self._prefetched = None
        if new_etag is None:
            raise LeaseHeld(f"lease on {self.oid!r} was taken by another writer during open")
        self._head_etag = new_etag
        self._lease_expires = self._head.lease.expires_at or 0.0

    def _prefetch_base(self, ref: ObjectRef) -> Optional[str]:
        """Download the base while the lease CAS is in flight.

        The download goes to the object's own scratch, which does not exist
        yet at this point, so it lands in a private temporary directory and is
        moved into scratch by `_restore`.
        """
        # Keep the key's own basename: ClickHouse decides "archive or
        # directory" from the file extension, so a `.tar.gz` archive restored
        # from a path without it is reported as simply not being a backup.
        path = os.path.join(
            tempfile.mkdtemp(prefix="chdb-durable-base-"), os.path.basename(ref.key))
        if not self.backend.download_to_file(ref.key, path):
            return None
        return path

    def _confirm_lease_taken(self) -> Optional[str]:
        current, etag = self._read_head()
        if current is None:
            return None
        if current.lease.instance == self._instance \
                and current.lease.generation == self._head.lease.generation:
            self._head = current
            return etag
        return None

    def _start_engine(self) -> None:
        # A private, unique, empty scratch directory per open (§5.2 step 4):
        # the engine's data path, the archives we download, and the archives we
        # produce all live under it and are removed on close.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", self.oid)[:64] or "obj"
        self._work = tempfile.mkdtemp(prefix=f"chdb-durable-{safe}-")
        data = os.path.join(self._work, "data")
        self._scratch = os.path.join(self._work, "objects")
        os.makedirs(data, exist_ok=True)
        os.makedirs(self._scratch, exist_ok=True)
        self._engine = ManagedConnection(data, self._scratch)

    def _restore(self) -> None:
        manifest = self._head.manifest
        base = manifest.base
        if base is None:
            self._engine.create_database(self.db)
        else:
            local = os.path.join(self._scratch, os.path.basename(base.key))
            incoming = local + ".incoming"
            prefetched = getattr(self, "_prefetched", None)
            self._prefetched = None
            if prefetched:
                shutil.move(prefetched, incoming)
                shutil.rmtree(os.path.dirname(prefetched), ignore_errors=True)
            elif not self.backend.download_to_file(base.key, incoming):
                # The manifest committed this key, so its absence is not an
                # empty object — it is incomplete state. §4.5 forbids falling
                # back to an older base or opening without it.
                raise Corrupt(
                    f"missing base checkpoint {base.key} — durable state incomplete")
            verify_file(incoming, base, what="base checkpoint")
            # Only what verified is published under the name the engine will
            # read (§5.1): a half-transferred archive never occupies it.
            os.replace(incoming, local)
            # Restore into the brand-new scratch: RESTORE appends to existing
            # tables, so V1 only ever restores into an empty target (§3.2).
            self._engine.restore(self.db, local)
            os.unlink(local)
        # Replay in the object's own database. The manifest owns the name, and
        # unqualified statements were classified against it, so anything else
        # would silently replay into `default` — invisible to the caller and
        # dropped by the next `BACKUP DATABASE`.
        self._engine.use_database(self.db)
        for ref in manifest.wal:
            segment = self.backend.get(ref.key)
            if segment is None:
                raise Corrupt(f"missing WAL segment {ref.key} — durable state incomplete")
            verify_bytes(segment, ref, what="WAL segment")
            # Replay is the internal path, never public execute(): these
            # statements were classified and logged once already (§4.4).
            replay(segment, ref.key, lambda sql: self._engine.query(sql, "CSV"))

    def _abort_open(self) -> None:
        """Undo a partial open: release the lease we just took, drop the engine.

        Leaving a lease stranded until its TTL would block the next writer for
        no reason — the object was never opened.
        """
        if not self.read_only and self._head is not None and self._head_etag is not None:
            self._head.lease = Lease(generation=self._head.lease.generation,
                                     extra=self._head.lease.extra)
            try:
                self.backend.replace_if_match(HEAD_KEY, self._head.to_bytes(), self._head_etag)
            except Exception:
                pass
            self._head_etag = None  # released here; close() must not release again
        self._teardown_local()

    def _teardown_local(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        prefetched = getattr(self, "_prefetched", None)
        if prefetched:
            shutil.rmtree(os.path.dirname(prefetched), ignore_errors=True)
            self._prefetched = None
        if self._work:
            shutil.rmtree(self._work, ignore_errors=True)
            self._work = self._scratch = None

    # -- guards -----------------------------------------------------------
    def _require_open(self) -> None:
        if self._state in (_CLOSED, _CLOSING):
            raise Closed(f"object {self.oid!r} is closed")
        if self._state != _OPEN:
            raise DurableError(f"object {self.oid!r} is not open")

    def _require_writable(self) -> None:
        self._require_open()
        if self.read_only:
            raise ClassificationRefused(
                f"object {self.oid!r} was opened read-only; only read-only statements are allowed")
        if self._fenced:
            raise LeaseFenced(
                f"object {self.oid!r} lost its lease; this writer is fenced and accepts no writes")

    # -- read / write -----------------------------------------------------
    def _gate_query(self, sql: str):
        analysis = self._engine.analyze(sql, self.db)
        # UNKNOWN first: text that did not parse also counts zero statements,
        # and "could not classify" is the useful half of that answer.
        if analysis.query_class == "UNKNOWN":
            raise ClassificationRefused(
                "chdb-core could not classify this statement; refusing rather than running "
                "something whose effect is unknown")
        if analysis.statement_count != 1:
            raise ClassificationRefused(
                f"query() takes exactly one statement; chdb-core counted "
                f"{analysis.statement_count}")
        if analysis.query_class != "READ_ONLY":
            raise ClassificationRefused(
                f"query() takes a read-only statement; chdb-core classified this one as "
                f"{analysis.query_class}")
        return analysis

    def _gate_execute(self, sql: str):
        # Refusal messages never quote the statement: it may carry a
        # credential, and §6 forbids putting one in an error.
        analysis = self._engine.analyze(sql, self.db)
        # UNKNOWN first: text that did not parse also counts zero statements,
        # and "could not classify" is the useful half of that answer.
        if analysis.query_class == "UNKNOWN":
            raise ClassificationRefused(
                "chdb-core could not classify this statement; refusing rather than logging "
                "something whose effect is unknown")
        if analysis.statement_count != 1:
            raise ClassificationRefused(
                f"execute() takes exactly one statement; chdb-core counted "
                f"{analysis.statement_count} (a PARALLEL WITH arm counts as one)")
        if analysis.query_class == "READ_ONLY":
            raise ClassificationRefused(
                "this statement changes nothing — use query(), which does not write the WAL")
        if analysis.query_class == "CONTROL":
            raise ClassificationRefused(
                "control statements (USE/SET/SYSTEM/BACKUP/RESTORE, or a write outside the "
                "engine) are managed by the durable object and cannot be issued through it")
        if analysis.query_class == "MUTATING_GLOBAL":
            raise ClassificationRefused(
                "this statement changes state outside every database, so a checkpoint could "
                "not carry it; V1 has no preamble and refuses it (see contract §8.1)")
        if analysis.has_secrets:
            raise SecretRefused(
                "this mutation carries a credential, and a WAL segment is stored in plain "
                "text; compute the result and log literals instead")
        if analysis.changes_database_lifecycle:
            raise ClassificationRefused(
                "creating, dropping or renaming a database changes the object's own "
                "container; the durable object manages that, and V1 does not log it")
        if not analysis.writes_only_target_database:
            raise ClassificationRefused(
                f"chdb-core could not prove every write lands in {self.db!r}; a write to "
                f"another database, to system, to a table function or to a file is not "
                f"part of this object's state")
        return analysis

    def query(self, sql: str, fmt: str = "CSV"):
        """Run one read-only statement. Nothing is added to the WAL.

        Allowed on a fenced writer: the local database is still exactly what
        this instance restored and wrote, and §5.7 fences writes, not reads.
        Read-only SQL carrying a credential runs, because it never reaches the
        WAL — but a failure from it is reported without the statement text.
        """
        with self._lock:
            self._require_open()
            analysis = self._gate_query(sql)
            return self._engine.query(sql, fmt, redact=analysis.has_secrets)

    def execute(self, sql: str) -> None:
        """Run one mutation locally and buffer it for the next `flush()`.

        Returning does **not** mean the statement is durable — it means it ran
        here and is queued to be published. Wait for `flush()` before telling
        anyone else it happened.
        """
        with self._lock:
            self._require_writable()
            # Refuse an over-limit statement *before* running it: executing
            # something the WAL cannot hold would leave local state that
            # object storage can never be made to match (§4.4).
            if self._buf.would_exceed(sql):
                raise LimitExceeded(
                    f"WAL segment already holds {self._buf.nbytes} bytes and this statement "
                    f"would take it over the V1 limit; flush() first")
            self._gate_execute(sql)
            self._engine.query(sql, "CSV")
            self._buf.append(sql)

    # -- durability -------------------------------------------------------
    def _publish_bytes(self, ref: ObjectRef, payload: bytes) -> None:
        try:
            etag = self.backend.put_bytes_if_absent(ref.key, payload)
        except BackendAmbiguous as exc:
            self._confirm_immutable(ref, as_file=False, cause=exc)
            return
        if etag is None:
            raise BackendError(
                f"{ref.key} already exists; every published key is minted unique, so this "
                f"is a collision the protocol does not expect")

    def _publish_file(self, ref: ObjectRef, local_path: str) -> None:
        try:
            etag = self.backend.put_file_if_absent(ref.key, local_path)
        except BackendAmbiguous as exc:
            self._confirm_immutable(ref, as_file=True, cause=exc)
            return
        if etag is None:
            raise BackendError(
                f"{ref.key} already exists; every published key is minted unique, so this "
                f"is a collision the protocol does not expect")

    def _confirm_immutable(self, ref: ObjectRef, *, as_file: bool, cause: Exception) -> None:
        """§5.8: settle an indeterminate PUT by reading the unique key back.

        Matching length and digest means the upload landed. A different body
        under our own unique key is `corrupt`, not a retryable state. Not
        being able to look at all leaves the commit genuinely unresolved.
        """
        try:
            if as_file:
                probe = os.path.join(self._scratch, f"confirm-{uuid.uuid4().hex}")
                try:
                    if not self.backend.download_to_file(ref.key, probe):
                        raise CommitAmbiguous(
                            f"upload of {ref.key} was indeterminate and the key is not "
                            f"readable: {cause}")
                    verify_file(probe, ref, what="uploaded checkpoint")
                finally:
                    if os.path.exists(probe):
                        os.unlink(probe)
            else:
                data = self.backend.get(ref.key)
                if data is None:
                    raise CommitAmbiguous(
                        f"upload of {ref.key} was indeterminate and the key is not "
                        f"readable: {cause}")
                verify_bytes(data, ref, what="uploaded WAL segment")
        except BackendError as exc:
            raise CommitAmbiguous(
                f"upload of {ref.key} was indeterminate and could not be re-read: {exc}") from exc

    def _reconcile(self, expect: Optional[ObjectRef]) -> bool:
        """Did our head CAS actually commit, despite what the response said?

        Only a *unique* key can answer this, which is why every published key
        carries a UUID: if the head now references it at the sequence number we
        intended, and the lease is still ours, the CAS landed and the response
        was lost. Ownership is not optional — a new writer that kept our key
        proves durability, not that we may CAS over them.
        """
        if expect is None:
            return False
        try:
            current, etag = self._read_head()
        except DurableError:
            return False
        if current is None:
            return False
        manifest = current.manifest
        referenced = (manifest.base is not None and manifest.base.key == expect.key) \
            or any(ref.key == expect.key for ref in manifest.wal)
        if not referenced or manifest.seq != self._head.manifest.seq:
            return False
        if not (current.lease.instance == self._instance
                and current.lease.generation == self._head.lease.generation):
            return False
        self._head, self._head_etag = current, etag
        self._lease_expires = current.lease.expires_at or 0.0
        return True

    def _cas_head(self, *, expect: Optional[ObjectRef] = None) -> None:
        """Commit `self._head` by conditional replace, and mean it.

        The CAS is the fence, so a failure is never retried blindly: we look at
        what the head says, and either prove the write landed, prove we were
        superseded, or refresh our ETag and try again inside the deadline.
        """
        deadline = self._now() + self.commit_deadline
        saw_ambiguous = False
        attempt = 0
        while True:
            attempt += 1
            now = self._now()
            if self._head.lease.owner is not None:
                self._head.lease.expires_at = now + self.ttl
            body = self._head.to_bytes()
            applied = None
            try:
                applied = self.backend.replace_if_match(HEAD_KEY, body, self._head_etag)
            except BackendAmbiguous:
                saw_ambiguous = True
            if applied is not None:
                self._head_etag = applied
                self._lease_expires = self._head.lease.expires_at or 0.0
                return
            if self._reconcile(expect):
                return
            current, current_etag = self._read_head()
            if current is None or not (
                    current.lease.instance == self._instance
                    and current.lease.generation == self._head.lease.generation):
                self._fence()
                raise LeaseFenced(
                    f"object {self.oid!r} was taken by another writer; this commit was fenced")
            # Still ours: our ETag was stale (a heartbeat and a commit crossed,
            # or an ambiguous write did land and moved it). Refresh and retry.
            self._head_etag = current_etag
            if attempt >= COMMIT_MAX_ATTEMPTS or self._now() >= deadline:
                if saw_ambiguous:
                    raise CommitAmbiguous(
                        f"could not determine whether the head commit for {self.oid!r} "
                        f"landed after {attempt} attempt(s)")
                raise BackendError(
                    f"head commit for {self.oid!r} did not apply after {attempt} attempt(s)")
            time.sleep(min(COMMIT_BACKOFF_BASE * (2 ** (attempt - 1)),
                           max(deadline - self._now(), 0.0)))

    def flush(self) -> Optional[str]:
        """Publish the buffered statements as one WAL segment (§5.4).

        One conditional PUT for the segment, one CAS for the head. Returning
        the segment key means the head commit is confirmed — this is the
        recovery point another process would restore to.
        """
        with self._lock:
            self._require_writable()
            return self._flush_locked()

    def _flush_locked(self) -> Optional[str]:
        if not len(self._buf):
            return None
        payload = self._buf.serialize()
        size, sha256 = bytes_digest(payload)
        manifest = self._head.manifest
        new_seq = manifest.seq + 1
        # A unique key per attempt: a retry after an indeterminate commit must
        # never overwrite an already-published segment (§4.4).
        ref = ObjectRef(
            key=f"wal/{self.generation}-{new_seq}-{uuid.uuid4().hex[:8]}.jsonl",
            size=size, sha256=sha256)
        self._publish_bytes(ref, payload)
        saved = (list(manifest.wal), manifest.seq)
        manifest.wal = saved[0] + [ref]
        manifest.seq = new_seq
        try:
            self._cas_head(expect=ref)
        except BaseException:
            # The old manifest is still authoritative and the buffer is still
            # ours, so a retry publishes a fresh unique segment. The uploaded
            # one is left as an orphan — see the GC note on checkpoint().
            manifest.wal, manifest.seq = saved
            raise
        self._buf.clear()
        return ref.key

    def checkpoint(self) -> str:
        """Fold base + WAL into a fresh base and clear the WAL list (§5.5).

        Checkpoints are *full* snapshots. An incremental backup records its
        base by local path, which does not survive object storage, so a
        portable incremental chain is V2 work (contract §8.1).

        TODO(gc): V1 has no garbage collection, by design — it also has no
        destroy, so nothing here may delete. A superseded base, the folded WAL
        segments, and any orphans left by an indeterminate commit all stay in
        object storage; only the current base and live WAL are referenced. A
        later compaction pass has to define orphan grace periods and
        concurrent-reader protection first (§8.1).
        """
        with self._lock:
            self._require_writable()
            manifest = self._head.manifest
            new_seq = manifest.seq + 1
            stamp = uuid.uuid4().hex[:8]
            name = f"{self.generation}-{new_seq}-{stamp}.tar.gz"
            local = os.path.join(self._scratch, f"ckpt-{name}")
            # chdb-core builds the BACKUP statement and quotes both the
            # database and the path itself (§3.2) — no SQL is assembled here.
            self._engine.backup(self.db, local)
            try:
                size, sha256 = file_digest(local)
                ref = ObjectRef(key=f"checkpoints/{name}", size=size, sha256=sha256)
                self._publish_file(ref, local)
            finally:
                if os.path.exists(local):
                    os.unlink(local)

            saved = (manifest.base, list(manifest.wal), manifest.seq,
                     self._head.engine.version, self._head.engine.min_reader,
                     self._head.engine.backup_format)
            manifest.base = ref
            manifest.wal = []
            manifest.seq = new_seq
            # This engine wrote the new base, so it is now the oldest engine
            # that can restore the object (§4.3). Asked of the connection that
            # produced the archive, not of the process at large.
            version = self._engine.engine_version()
            self._head.engine.version = version
            self._head.engine.min_reader = version
            self._head.engine.backup_format = BACKUP_FORMAT
            try:
                self._cas_head(expect=ref)
            except BaseException:
                (manifest.base, manifest.wal, manifest.seq, self._head.engine.version,
                 self._head.engine.min_reader, self._head.engine.backup_format) = saved
                raise
            # The backup was taken from the live database, so it already
            # contains everything the buffer held (§5.5 step 6).
            self._buf.clear()
            return ref.key

    # -- lease upkeep -----------------------------------------------------
    def _start_heartbeat(self) -> None:
        if self.read_only:
            return
        self._hb_stop = threading.Event()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, name=f"chdb-durable-lease-{self.oid}", daemon=True)
        self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        # Signal and join *without* the lock: the thread takes it to renew, and
        # joining while holding it would deadlock close().
        if self._hb_stop is not None:
            self._hb_stop.set()
        thread = self._hb_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(self.heartbeat_interval, 1.0) + 5.0)
        self._hb_thread = None

    def _heartbeat_loop(self) -> None:
        """Renew the lease on a cadence, and stop writing if renewal fails.

        A renewal is a head CAS like any other, so it queues behind whatever
        operation is running rather than racing it for the ETag (§5.7). If
        renewals keep failing until the lease we believe in has run out, the
        writer fences *itself*: past that instant another writer may legally
        take over, and writing anyway is how two writers end up believing they
        own the same object.
        """
        while not self._hb_stop.wait(self.heartbeat_interval):
            try:
                with self._lock:
                    if self._state != _OPEN or self._fenced:
                        return
                    self._cas_head()
            except LeaseFenced:
                return  # _cas_head already fenced us
            except DurableError:
                with self._lock:
                    if self._now() >= self._lease_expires:
                        self._fence()
                        return

    # -- close ------------------------------------------------------------
    def close(self) -> None:
        """Stop accepting operations, publish what is buffered, release the lease.

        A failure to persist or to release is raised, not swallowed: a caller
        whose buffered writes did not make it needs to know that from `close()`
        rather than discover it on the next open. Local resources are released
        either way.
        """
        with self._lock:
            if self._state in (_CLOSED, _CLOSING):
                return
            was_open = self._state == _OPEN
            self._state = _CLOSING
        self._stop_heartbeat()
        with self._lock:
            try:
                if was_open and not self.read_only:
                    if self._fenced:
                        if len(self._buf):
                            raise LeaseFenced(
                                f"object {self.oid!r} lost its lease; {len(self._buf)} buffered "
                                f"statement(s) were not persisted")
                    else:
                        self._flush_locked()
                        if self._head_etag is not None:
                            self._head.lease = Lease(
                                generation=self._head.lease.generation,
                                extra=self._head.lease.extra)
                            self._cas_head()
            finally:
                self._teardown_local()
                self._state = _CLOSED
