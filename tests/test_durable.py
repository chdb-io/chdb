"""chdb.durable Durable V1 conformance tests — runnable as a plain script.

    DAO_TEST_URL=local:/tmp/dao-test python tests/test_durable.py
    DAO_TEST_URL=s3://dao/ns CHDB_DURABLE_S3_ENDPOINT=http://127.0.0.1:9000 \
        AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
        python tests/test_durable.py

The sections mirror `dev-docs/CHDB_DURABLE_V1_CONTRACT.md` §7: format
fixtures, classification scenarios, and the fault matrix. Everything runs
sequentially on purpose — chdb-core allows one active data path per process
(§3.6), so two durable objects are never open at the same time.
"""
import json
import os
import pathlib
import tempfile
import time
import warnings

from chdb import durable as cd
from chdb.durable import protocol, wal as wal_mod
from chdb.durable.protocol import DEFAULT_DATABASE
from chdb.durable.backends import make_backend
from chdb.durable.engine import engine_has_v1_abi, engine_version

# Durable V1 needs chdb-core's backup / restore / query-analysis ABI (26.7.2-rc.2
# or newer). On an older engine there is nothing here to test, so skip the module
# under pytest rather than fail 49 checks for one missing dependency.
if not engine_has_v1_abi():
    _REASON = ("chdb-core does not export the Durable V1 ABI "
               "(backup_database / restore_database / classify_query)")
    try:
        import pytest

        pytest.skip(_REASON, allow_module_level=True)
    except ImportError:
        raise SystemExit(f"skipped: {_REASON}")
from chdb.durable.errors import (
    BackendAmbiguous,
    BackendError,
    ClassificationRefused,
    Closed,
    CommitAmbiguous,
    Corrupt,
    EngineError,
    EngineIncompatible,
    LeaseFenced,
    LeaseHeld,
    LimitExceeded,
    NotFound,
    ProtocolUnsupported,
    SecretRefused,
)

URL = os.getenv("DAO_TEST_URL", "local:" + tempfile.mkdtemp(prefix="dao-test-"))
HEAD = protocol.HEAD_KEY

_PASSED = []


def _fresh_ns(**kwargs):
    return cd.Namespace(URL, owner="w1", **kwargs)


def _put_head(oid, doc, *, ns=None):
    """Replace an object with a hand-written head — the fixture mechanism."""
    ns = ns or _fresh_ns()
    ns.destroy(oid, force=True)
    backend = make_backend(URL, oid)
    assert backend.put_bytes_if_absent(HEAD, json.dumps(doc).encode()) is not None
    return backend


def _head_of(oid):
    return json.loads(make_backend(URL, oid).get(HEAD))


def _valid_head(**overrides):
    """A schema-valid head for a cold object, before overrides."""
    doc = {
        "protocol": {"version": 1, "reader_features": [], "writer_features": []},
        "engine": {"name": "chdb", "version": engine_version(),
                   "backup_format": 1, "min_reader": engine_version()},
        "lease": {"generation": 1, "owner": None, "instance": None, "expires_at": None},
        "manifest": {"db": "mem", "base": None, "wal": [], "seq": 0},
    }
    for section, patch in overrides.items():
        doc[section].update(patch)
    return doc


def _seeded(oid, rows=100, *, db=DEFAULT_DATABASE, checkpoint=True, ns=None):
    """An object holding `rows` rows in `db`.t, checkpointed by default."""
    ns = ns or _fresh_ns(db=db)
    ns.destroy(oid, force=True)
    obj = ns.open(oid)
    obj.execute("CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n")
    obj.execute(f"INSERT INTO t SELECT number FROM numbers({rows})")
    if checkpoint:
        obj.checkpoint()
    else:
        obj.flush()
    obj.close()
    return ns


def _count(ns, oid, *, db=DEFAULT_DATABASE):
    reader = ns.open(oid, read_only=True)
    try:
        return reader.query(f"SELECT count() FROM `{db}`.t", "CSV").data().strip()
    finally:
        reader.close()


def _expect(kind, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except kind as exc:
        return exc
    raise AssertionError(f"expected {kind.__name__}")


class FaultyBackend:
    """A backend that lies about the outcome of a write, on request.

    Every hook is "what the provider told us", not "what happened": that is
    the whole point of the §5.8 reconcile, and a test that only ever injects
    clean failures never exercises it.
    """

    def __init__(self, inner):
        self._inner = inner
        self.fail_replace = None      # None | "ambiguous_after" | "lost" | "error"
        self.fail_create = None       # None | "ambiguous_before" | "ambiguous_after"
        self.hide_head = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_with_etag(self, key):
        if self.hide_head and key == HEAD:
            return (None, None)
        return self._inner.get_with_etag(key)

    def _conditional_create(self, call, key, *rest):
        mode = self.fail_create
        if mode == "ambiguous_before":
            raise BackendAmbiguous(f"injected: create {key} never sent a response")
        result = call(key, *rest)
        if mode == "ambiguous_after":
            raise BackendAmbiguous(f"injected: create {key} landed but the response was lost")
        return result

    def put_bytes_if_absent(self, key, data):
        return self._conditional_create(self._inner.put_bytes_if_absent, key, data)

    def put_file_if_absent(self, key, path):
        return self._conditional_create(self._inner.put_file_if_absent, key, path)

    def replace_if_match(self, key, data, etag):
        mode = self.fail_replace
        if mode == "error":
            raise BackendError(f"injected: replace {key} failed")
        if mode == "lost":
            return None  # the CAS did not apply, and the object is still ours
        result = self._inner.replace_if_match(key, data, etag)
        if mode == "ambiguous_after":
            raise BackendAmbiguous(f"injected: replace {key} landed but the response was lost")
        return result


# =====================================================================
# §7.2 format fixtures
# =====================================================================

def test_empty_object():
    ns = _fresh_ns()
    ns.destroy("f-empty", force=True)
    obj = ns.open("f-empty")
    assert obj.base is None and obj.wal == [] and obj.seq == 0
    assert obj.generation == 1
    obj.close()
    doc = _head_of("f-empty")
    assert doc["protocol"] == {"version": 1, "reader_features": [], "writer_features": []}
    assert doc["engine"]["name"] == "chdb" and doc["engine"]["backup_format"] == 1
    # a released lease is one fixed shape, not "owner set to empty string"
    assert doc["lease"] == {"generation": 1, "owner": None, "instance": None,
                            "expires_at": None}
    assert doc["manifest"] == {"db": DEFAULT_DATABASE, "base": None, "wal": [], "seq": 0}
    _PASSED.append("empty-object: cold create publishes a V1 head, lease released on close")


def test_readonly_missing_object():
    ns = _fresh_ns()
    ns.destroy("f-absent", force=True)
    _expect(NotFound, ns.open, "f-absent", read_only=True)
    # and it must not have created anything on the way past
    assert make_backend(URL, "f-absent").get(HEAD) is None
    _PASSED.append("read-only open of a missing object: not_found, nothing created")


def test_checkpoint_only():
    ns = _seeded("f-ckpt", 1000)
    doc = _head_of("f-ckpt")
    assert doc["manifest"]["base"]["key"].startswith("checkpoints/")
    assert doc["manifest"]["wal"] == []
    assert _count(ns, "f-ckpt") == "1000"
    _PASSED.append("checkpoint-only: 1000 rows restored from the base alone")


def test_checkpoint_plus_wal():
    ns = _seeded("f-both", 1000)
    obj = ns.open("f-both")
    obj.execute("INSERT INTO t SELECT number FROM numbers(1000, 500)")
    key = obj.flush()
    assert key.startswith("wal/")
    obj.close()
    assert _count(ns, "f-both") == "1500"
    # a checkpoint folds the WAL away without changing what you can read
    obj = ns.open("f-both")
    folded = obj.checkpoint()
    assert obj.base.key == folded and obj.wal == []
    obj.close()
    assert _count(ns, "f-both") == "1500"
    _PASSED.append("checkpoint-plus-wal: base 1000 + WAL 500 replayed, then folded")


def test_quoted_database_name():
    # 'my-mem-db' is only valid backtick-quoted; unquoted it parses as
    # subtraction, so every create/backup/restore would fail. chdb-core quotes
    # the name for us — this proves it does, through the whole round trip.
    ns = _seeded("f-quoted", 42, db="my-mem-db", checkpoint=False)
    assert _count(ns, "f-quoted", db="my-mem-db") == "42"
    _PASSED.append("quoted-database-name: my-mem-db round-trips through backup and replay")


def test_missing_base_and_wal():
    ref = {"key": "checkpoints/nope.tar.gz", "size": 10, "sha256": "0" * 64}
    _put_head("f-nobase", _valid_head(manifest={"base": ref, "seq": 1}))
    exc = _expect(Corrupt, _fresh_ns().open, "f-nobase")
    assert "missing base" in str(exc), exc

    seg = {"key": "wal/nope.jsonl", "size": 10, "sha256": "0" * 64}
    _put_head("f-nowal", _valid_head(manifest={"wal": [seg], "seq": 1}))
    exc = _expect(Corrupt, _fresh_ns().open, "f-nowal")
    assert "missing WAL" in str(exc), exc
    _PASSED.append("missing-base / missing-wal: corrupt, never a partial open")


def test_bad_base_size_and_digest():
    ns = _seeded("f-badbase", 7)
    good = _head_of("f-badbase")["manifest"]["base"]

    _put_head("f-badsize", _valid_head(manifest={"base": dict(good, size=good["size"] + 1),
                                                 "seq": 1}))
    # the fixture's base key belongs to another object's prefix, so copy the blob across
    make_backend(URL, "f-badsize").put_bytes_if_absent(
        good["key"], make_backend(URL, "f-badbase").get(good["key"]))
    exc = _expect(Corrupt, _fresh_ns().open, "f-badsize")
    assert "bytes" in str(exc), exc

    _put_head("f-baddigest", _valid_head(manifest={"base": dict(good, sha256="0" * 64),
                                                   "seq": 1}))
    make_backend(URL, "f-baddigest").put_bytes_if_absent(
        good["key"], make_backend(URL, "f-badbase").get(good["key"]))
    exc = _expect(Corrupt, _fresh_ns().open, "f-baddigest")
    assert "SHA-256" in str(exc), exc
    _PASSED.append("bad-base-size / bad-base-sha256: corrupt before the archive is restored")


def test_bad_wal_size_and_digest():
    payload = b'{"sql":"CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n"}\n'
    import hashlib
    digest = hashlib.sha256(payload).hexdigest()
    key = "wal/1-1-deadbeef.jsonl"

    backend = _put_head("f-badwalsize", _valid_head(
        manifest={"wal": [{"key": key, "size": len(payload) + 5, "sha256": digest}], "seq": 1}))
    backend.put_bytes_if_absent(key, payload)
    exc = _expect(Corrupt, _fresh_ns().open, "f-badwalsize")
    assert "bytes" in str(exc), exc

    backend = _put_head("f-badwaldigest", _valid_head(
        manifest={"wal": [{"key": key, "size": len(payload), "sha256": "1" * 64}], "seq": 1}))
    backend.put_bytes_if_absent(key, payload)
    exc = _expect(Corrupt, _fresh_ns().open, "f-badwaldigest")
    assert "SHA-256" in str(exc), exc
    _PASSED.append("bad-wal-size / bad-wal-sha256: corrupt before any statement is replayed")


def test_unknown_reader_feature():
    _put_head("f-rfeat", _valid_head(protocol={"reader_features": ["compressed-wal"]}))
    ns = _fresh_ns()
    _expect(ProtocolUnsupported, ns.open, "f-rfeat")
    _expect(ProtocolUnsupported, ns.open, "f-rfeat", read_only=True)
    _PASSED.append("unknown-reader-feature: refused for reading and for writing")


def test_unknown_writer_feature():
    ns = _seeded("f-wfeat", 5)
    doc = _head_of("f-wfeat")
    doc["protocol"]["writer_features"] = ["multi-writer"]
    backend = make_backend(URL, "f-wfeat")
    _, etag = backend.get_with_etag(HEAD)
    assert backend.replace_if_match(HEAD, json.dumps(doc).encode(), etag)
    # readable, because nothing about reading it is in doubt...
    reader = ns.open("f-wfeat", read_only=True)
    assert reader.query("SELECT count() FROM t", "CSV").data().strip() == "5"
    reader.close()
    # ...but writing it would drop semantics we do not implement
    _expect(ProtocolUnsupported, ns.open, "f-wfeat")
    _PASSED.append("unknown-writer-feature: read-only allowed, writer lease refused")


def test_future_protocol_version():
    _put_head("f-future", _valid_head(protocol={"version": 2}))
    ns = _fresh_ns()
    _expect(ProtocolUnsupported, ns.open, "f-future", read_only=True)
    _expect(ProtocolUnsupported, ns.open, "f-future")
    _PASSED.append("future-protocol-version: refused rather than partially understood")


def test_producer_version_differs_but_compatible():
    ns = _seeded("f-prod", 9)
    doc = _head_of("f-prod")
    # written by some later core, but declaring a min_reader we satisfy
    doc["engine"]["version"] = "99.9.9"
    doc["engine"]["min_reader"] = "1.0.0"
    backend = make_backend(URL, "f-prod")
    _, etag = backend.get_with_etag(HEAD)
    backend.replace_if_match(HEAD, json.dumps(doc).encode(), etag)
    assert _count(ns, "f-prod") == "9"
    obj = ns.open("f-prod")
    obj.close()
    # the writer records itself, and leaves min_reader where the base put it
    after = _head_of("f-prod")
    assert after["engine"]["version"] == engine_version()
    assert after["engine"]["min_reader"] == "1.0.0"
    _PASSED.append("producer-version-differs-but-compatible: opens; min_reader is the gate")


def test_engine_reader_too_old():
    ns = _seeded("f-tooold", 3)
    doc = _head_of("f-tooold")
    doc["engine"]["min_reader"] = "999.0.0"
    backend = make_backend(URL, "f-tooold")
    _, etag = backend.get_with_etag(HEAD)
    backend.replace_if_match(HEAD, json.dumps(doc).encode(), etag)
    _expect(EngineIncompatible, ns.open, "f-tooold", read_only=True)
    _expect(EngineIncompatible, ns.open, "f-tooold")
    _PASSED.append("engine-reader-too-old: engine_incompatible")


def test_backup_format_too_new():
    ns = _seeded("f-fmt", 3)
    doc = _head_of("f-fmt")
    doc["engine"]["backup_format"] = protocol.BACKUP_FORMAT + 1
    backend = make_backend(URL, "f-fmt")
    _, etag = backend.get_with_etag(HEAD)
    backend.replace_if_match(HEAD, json.dumps(doc).encode(), etag)
    _expect(EngineIncompatible, ns.open, "f-fmt", read_only=True)
    _PASSED.append("backup-format-too-new: engine_incompatible, fails closed")


def test_unknown_fields_round_trip():
    ns = _fresh_ns()
    ns.destroy("f-unknown", force=True)
    obj = ns.open("f-unknown")
    obj.close()
    backend = make_backend(URL, "f-unknown")
    doc, etag = backend.get_with_etag(HEAD)
    doc = json.loads(doc)
    doc["future_top"] = {"a": 1}
    doc["protocol"]["future_protocol"] = 2
    doc["engine"]["future_engine"] = 3
    doc["lease"]["future_lease"] = 4
    doc["manifest"]["future_manifest"] = 5
    backend.replace_if_match(HEAD, json.dumps(doc).encode(), etag)

    obj = ns.open("f-unknown")
    obj.execute("CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n")
    obj.flush()
    obj.checkpoint()
    obj.close()
    after = _head_of("f-unknown")
    assert after["future_top"] == {"a": 1}, after
    assert after["protocol"]["future_protocol"] == 2
    assert after["engine"]["future_engine"] == 3
    assert after["lease"]["future_lease"] == 4
    assert after["manifest"]["future_manifest"] == 5
    _PASSED.append("unknown fields survive open -> execute -> flush -> checkpoint -> close")


def test_json_shape_is_not_frozen():
    # a head written with different key order and indentation must read the same
    ns = _seeded("f-shape", 11)
    backend = make_backend(URL, "f-shape")
    doc, etag = backend.get_with_etag(HEAD)
    reshaped = json.dumps(json.loads(doc), sort_keys=True, indent=4).encode()
    assert reshaped != doc
    backend.replace_if_match(HEAD, reshaped, etag)
    assert _count(ns, "f-shape") == "11"
    _PASSED.append("JSON key order and whitespace are not part of the contract")


# =====================================================================
# §7.3 classification scenarios
# =====================================================================

def test_gates_cannot_be_bypassed_by_method_choice():
    ns = _fresh_ns()
    ns.destroy("c-gate", force=True)
    obj = ns.open("c-gate")
    obj.execute("CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n")
    # a mutation through query() is refused, so nothing can write without logging
    _expect(ClassificationRefused, obj.query, "INSERT INTO t VALUES (1)")
    # a read through execute() is refused, so the WAL never fills with SELECTs
    _expect(ClassificationRefused, obj.execute, "SELECT 1")
    assert obj.pending == 1  # only the CREATE
    obj.close()
    _PASSED.append("neither method name bypasses the analysis")


def test_multi_statement_and_parallel_with_refused():
    ns = _fresh_ns()
    ns.destroy("c-multi", force=True)
    obj = ns.open("c-multi")
    exc = _expect(ClassificationRefused, obj.execute,
                  "CREATE TABLE a (n Int64) ENGINE=Memory; CREATE TABLE b (n Int64) ENGINE=Memory")
    assert "one statement" in str(exc)
    _expect(ClassificationRefused, obj.execute,
            "CREATE TABLE a (n Int64) ENGINE=Memory "
            "PARALLEL WITH CREATE TABLE b (n Int64) ENGINE=Memory")
    _expect(ClassificationRefused, obj.query, "SELECT 1; SELECT 2")
    assert obj.pending == 0
    obj.close()
    _PASSED.append("multi-statement and PARALLEL WITH refused (both arms execute)")


def test_writes_outside_the_object_refused():
    ns = _fresh_ns()
    ns.destroy("c-outside", force=True)
    obj = ns.open("c-outside")
    obj.execute("CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n")
    for sql in ("INSERT INTO other_db.t VALUES (1)",
                "RENAME TABLE t TO other_db.t2",
                "INSERT INTO FUNCTION file('/tmp/chdb-durable-should-not-exist.csv') SELECT 1",
                "SELECT 1 INTO OUTFILE '/tmp/chdb-durable-should-not-exist.csv'"):
        _expect(ClassificationRefused, obj.execute, sql)
    _expect(ClassificationRefused, obj.execute, "INSERT INTO system.numbers VALUES (1)")
    assert obj.pending == 1
    assert not os.path.exists("/tmp/chdb-durable-should-not-exist.csv")
    obj.close()
    _PASSED.append("writes to another database, system, a table function or a file refused")


def test_database_lifecycle_refused():
    ns = _fresh_ns()
    ns.destroy("c-life", force=True)
    obj = ns.open("c-life")
    for sql in ("CREATE DATABASE something", "DROP DATABASE mem",
                "RENAME DATABASE mem TO mem2"):
        exc = _expect(ClassificationRefused, obj.execute, sql)
        assert "container" in str(exc), exc
    obj.close()
    _PASSED.append("CREATE/DROP/RENAME DATABASE refused: the object owns its container")


def test_mutating_global_refused():
    ns = _fresh_ns()
    ns.destroy("c-global", force=True)
    obj = ns.open("c-global")
    for sql in ("CREATE FUNCTION addone AS x -> x + 1",
                "CREATE USER someone",
                "CREATE NAMED COLLECTION creds AS key = 'value'"):
        exc = _expect(ClassificationRefused, obj.execute, sql)
        assert "outside every database" in str(exc), exc
    obj.close()
    _PASSED.append("every MUTATING_GLOBAL refused: V1 has no preamble to replay it from")


def test_control_statements_refused():
    # Deliberately not the default database: "USE default" has to be refused
    # rather than merely be a no-op, and only an object living somewhere else
    # can tell those apart.
    ns = _fresh_ns(db="mem")
    ns.destroy("c-control", force=True)
    obj = ns.open("c-control")
    obj.execute("CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n")
    for sql in ("USE default", "SET max_threads = 1", "SYSTEM FLUSH LOGS",
                # a statement setting that would let an insert complete after
                # execute() returned: chdb-core classifies it CONTROL, so the
                # durability promise cannot be relaxed from the SQL
                "INSERT INTO t SETTINGS async_insert = 1 VALUES (1)"):
        _expect(ClassificationRefused, obj.execute, sql)
    # and the object's own database is still current afterwards
    assert obj.query("SELECT currentDatabase()", "CSV").data().strip() == '"mem"'
    obj.close()
    _PASSED.append("control statements refused, including async-insert relaxation")


def test_secret_mutation_refused_without_leaking():
    ns = _fresh_ns()
    ns.destroy("c-secret", force=True)
    obj = ns.open("c-secret")
    obj.execute("CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n")
    secret = "SUPERSECRETKEY123"
    exc = _expect(SecretRefused, obj.execute,
                  f"INSERT INTO t SELECT 1 FROM s3('https://x/y.csv', 'AKIAEXAMPLE', '{secret}')")
    assert secret not in str(exc) and "s3(" not in str(exc), exc
    assert obj.pending == 1
    obj.close()
    _PASSED.append("secret-bearing mutation refused, and the error quotes no SQL")


def test_secret_read_only_runs_and_is_not_logged():
    ns = _fresh_ns()
    ns.destroy("c-secretread", force=True)
    obj = ns.open("c-secretread")
    secret = "SUPERSECRETKEY123"
    # A malformed URL fails inside the engine immediately, with no network
    # wait: what is under test is the gate and the redaction, not S3.
    sql = f"SELECT * FROM s3('not a url', 'AKIAEXAMPLE', '{secret}')"
    # It passes the gate — a read never reaches the WAL. It then fails, and
    # that failure must not carry the credential.
    exc = _expect(EngineError, obj.query, sql)
    assert secret not in str(exc), exc
    assert obj.pending == 0
    assert obj.flush() is None  # nothing to publish
    obj.close()
    _PASSED.append("secret-bearing read-only SQL runs, writes no WAL, and redacts failures")


def test_unknown_statement_fails_closed():
    ns = _fresh_ns()
    ns.destroy("c-unknown", force=True)
    obj = ns.open("c-unknown")
    exc = _expect(ClassificationRefused, obj.execute, "this is not sql at all")
    assert "could not classify" in str(exc), exc
    _expect(ClassificationRefused, obj.query, "this is not sql at all")
    obj.close()
    _PASSED.append("UNKNOWN fails closed for both entry points")


def test_limits_report_limit_exceeded():
    # The frozen limits are 64 MiB of SQL and 128 MiB per segment. Allocating
    # those to prove an arithmetic comparison is not worth a minute of test
    # time, so the limits are shrunk and the same code paths exercised.
    sql_limit, seg_limit = wal_mod.MAX_SQL_BYTES, wal_mod.MAX_WAL_SEGMENT_BYTES
    head_limit = protocol.MAX_HEAD_BYTES
    ns = _fresh_ns()
    ns.destroy("c-limits", force=True)
    obj = ns.open("c-limits")
    obj.execute("CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n")
    try:
        wal_mod.MAX_SQL_BYTES = 64
        exc = _expect(LimitExceeded, obj.execute,
                      "INSERT INTO t VALUES " + ", ".join("(1)" for _ in range(40)))
        assert "over the V1 limit" in str(exc)
        wal_mod.MAX_SQL_BYTES = sql_limit
        wal_mod.MAX_WAL_SEGMENT_BYTES = 40
        exc = _expect(LimitExceeded, obj.execute, "INSERT INTO t VALUES (2)")
        assert "flush()" in str(exc)
    finally:
        wal_mod.MAX_SQL_BYTES, wal_mod.MAX_WAL_SEGMENT_BYTES = sql_limit, seg_limit
    # an over-limit head is the writer's problem to notice, while it can still
    # checkpoint the WAL list away
    import chdb.durable.head as head_mod
    try:
        head_mod.MAX_HEAD_BYTES = 100
        _expect(LimitExceeded, obj.flush)
    finally:
        head_mod.MAX_HEAD_BYTES = head_limit
    obj.close()
    _PASSED.append("SQL, WAL segment and head limits all report limit_exceeded")


# =====================================================================
# §7.4 fault matrix
# =====================================================================

def test_conditional_create_race():
    ns = _fresh_ns()
    ns.destroy("x-race", force=True)
    obj = ns.open("x-race")            # writer A creates the object
    obj.close()
    backend = FaultyBackend(make_backend(URL, "x-race"))
    backend.hide_head = True           # writer B saw no head, so it will create one
    loser = cd.DurableObject("x-race", backend, owner="w2", db="mem")
    _expect(LeaseHeld, loser.open)
    loser.close()
    _PASSED.append("conditional create race: only one writer creates the object")


def test_wal_put_lands_but_head_cas_fails():
    ns = _seeded("x-walcas", 10)
    backend = FaultyBackend(make_backend(URL, "x-walcas"))
    obj = cd.DurableObject("x-walcas", backend, owner="w1", db="mem", commit_deadline=0.3)
    obj.open()
    obj.execute("INSERT INTO t VALUES (999)")
    before = _head_of("x-walcas")["manifest"]
    backend.fail_replace = "lost"
    _expect(BackendError, obj.flush)
    # the old manifest is still authoritative, and the statement is still ours
    assert _head_of("x-walcas")["manifest"] == before
    assert obj.pending == 1
    backend.fail_replace = None
    key = obj.flush()                   # a retry publishes a fresh unique segment
    assert key and obj.pending == 0
    obj.close()
    assert _count(ns, "x-walcas") == "11"
    _PASSED.append("WAL PUT then failed head CAS: old manifest stands, buffer retained")


def test_head_cas_committed_but_response_lost():
    ns = _seeded("x-lostcas", 10)
    backend = FaultyBackend(make_backend(URL, "x-lostcas"))
    obj = cd.DurableObject("x-lostcas", backend, owner="w1", db="mem")
    obj.open()
    obj.execute("INSERT INTO t VALUES (777)")
    backend.fail_replace = "ambiguous_after"   # it landed; we just never heard
    key = obj.flush()
    backend.fail_replace = None
    assert key and obj.pending == 0
    # exactly one segment: reconcile adopted the commit instead of retrying,
    # which would have published the same boundary twice
    assert [ref["key"] for ref in _head_of("x-lostcas")["manifest"]["wal"]] == [key]
    obj.close()
    assert _count(ns, "x-lostcas") == "11"
    _PASSED.append("head CAS committed with a lost response: reconciled as success")


def test_unprovable_commit_is_ambiguous():
    ns = _seeded("x-ambig", 10)
    backend = FaultyBackend(make_backend(URL, "x-ambig"))
    obj = cd.DurableObject("x-ambig", backend, owner="w1", db="mem")
    obj.open()
    obj.execute("INSERT INTO t VALUES (555)")
    backend.fail_create = "ambiguous_before"   # never sent, but we cannot know
    exc = _expect(CommitAmbiguous, obj.flush)
    assert "indeterminate" in str(exc)
    backend.fail_create = None
    assert obj.pending == 1                     # nothing was claimed durable
    obj.close()
    assert _count(ns, "x-ambig") == "11"
    _PASSED.append("commit that cannot be proved either way: commit_ambiguous, never success")


def test_wal_upload_landed_but_response_lost():
    ns = _seeded("x-walput", 10)
    backend = FaultyBackend(make_backend(URL, "x-walput"))
    obj = cd.DurableObject("x-walput", backend, owner="w1", db="mem")
    obj.open()
    obj.execute("INSERT INTO t VALUES (444)")
    backend.fail_create = "ambiguous_after"    # the segment is there
    key = obj.flush()                          # re-read proves it by size + digest
    backend.fail_create = None
    assert key and obj.pending == 0
    obj.close()
    assert _count(ns, "x-walput") == "11"
    _PASSED.append("WAL upload with a lost response: confirmed by re-reading the unique key")


def test_checkpoint_put_lands_but_head_cas_fails():
    ns = _seeded("x-ckptcas", 10)
    backend = FaultyBackend(make_backend(URL, "x-ckptcas"))
    obj = cd.DurableObject("x-ckptcas", backend, owner="w1", db="mem", commit_deadline=0.3)
    obj.open()
    # after the open, so the lease CAS the open itself performs is not the
    # difference we would then be measuring
    before = _head_of("x-ckptcas")["manifest"]
    obj.execute("INSERT INTO t VALUES (222)")
    backend.fail_replace = "lost"
    _expect(BackendError, obj.checkpoint)
    assert _head_of("x-ckptcas")["manifest"] == before   # the old base is still the base
    assert obj.pending == 1                     # and the buffer was not cleared
    backend.fail_replace = None
    obj.close()
    assert _count(ns, "x-ckptcas") == "11"
    _PASSED.append("checkpoint PUT then failed head CAS: old base/WAL still restore")


def test_restore_failure_frees_the_engine():
    # chdb-core allows one active data path per process, so a failed restore
    # that leaked its engine would block every later open in the process.
    ref = {"key": "checkpoints/nope.tar.gz", "size": 5, "sha256": "0" * 64}
    _put_head("x-partial", _valid_head(manifest={"base": ref, "seq": 1}))
    ns = _fresh_ns()
    _expect(Corrupt, ns.open, "x-partial")
    ns.destroy("x-after", force=True)
    obj = ns.open("x-after")
    assert obj.query("SELECT 1", "CSV").data().strip() == "1"
    obj.close()
    # the lease the failed open took was released, not stranded until its TTL
    assert _head_of("x-partial")["lease"]["owner"] is None
    _PASSED.append("failed restore closes the partial engine and releases the lease")


def test_heartbeat_renews_without_moving_generation_or_seq():
    ns = _seeded("x-hb", 5)
    backend = make_backend(URL, "x-hb")
    obj = cd.DurableObject("x-hb", backend, owner="w1", db="mem",
                           lease_ttl=1.2, heartbeat_interval=0.4)
    obj.open()
    first = _head_of("x-hb")["lease"]
    time.sleep(1.0)                      # several heartbeats, no operations
    renewed = _head_of("x-hb")["lease"]
    assert renewed["expires_at"] > first["expires_at"], (first, renewed)
    # §4.2: a heartbeat renews the expiry and nothing else
    assert renewed["generation"] == first["generation"]
    assert _head_of("x-hb")["manifest"]["seq"] == obj.seq

    # ...and its head CAS shares the ETag with flush() rather than racing it
    for i in range(3):
        obj.execute(f"INSERT INTO t VALUES ({1000 + i})")
        obj.flush()
    doc = _head_of("x-hb")
    assert doc["manifest"]["seq"] == obj.seq
    assert len(doc["manifest"]["wal"]) == 3
    obj.close()
    assert _count(ns, "x-hb") == "8"
    _PASSED.append("heartbeat renews the expiry only, and shares the head with flush")


def test_writer_self_fences_when_renewal_fails():
    ns = _seeded("x-selffence", 4)
    backend = FaultyBackend(make_backend(URL, "x-selffence"))
    obj = cd.DurableObject("x-selffence", backend, owner="w1", db="mem",
                           lease_ttl=0.6, heartbeat_interval=0.15, commit_deadline=0.1)
    obj.open()
    backend.fail_replace = "error"          # every renewal from here on fails
    deadline = time.time() + 5.0
    while not obj.fenced and time.time() < deadline:
        time.sleep(0.05)
    assert obj.fenced, "writer should have fenced itself once its lease ran out"
    _expect(LeaseFenced, obj.execute, "INSERT INTO t VALUES (1)")
    _expect(LeaseFenced, obj.flush)
    _expect(LeaseFenced, obj.checkpoint)
    # reads still work: the local database is what this instance restored
    assert obj.query("SELECT count() FROM t", "CSV").data().strip() == "4"
    backend.fail_replace = None
    obj.close()
    _PASSED.append("renewal failing past the lease deadline self-fences the writer")


def test_takeover_fences_the_previous_writer():
    ns = _seeded("x-fence", 6)
    obj = ns.open("x-fence")
    obj.execute("INSERT INTO t VALUES (1)")
    # an external takeover: a new generation, written straight to the head
    backend = make_backend(URL, "x-fence")
    doc, etag = backend.get_with_etag(HEAD)
    doc = json.loads(doc)
    doc["lease"] = {"owner": "w2", "instance": "other-instance",
                    "generation": doc["lease"]["generation"] + 5,
                    "expires_at": time.time() + 600}
    assert backend.replace_if_match(HEAD, json.dumps(doc).encode(), etag)
    _expect(LeaseFenced, obj.flush)
    # close() must surface the fence rather than drop the buffered write quietly
    _expect(LeaseFenced, obj.close)
    assert obj.closed  # ...and still release local resources
    _PASSED.append("takeover fences the old writer, and close() says so")


def test_close_releases_and_then_refuses():
    ns = _seeded("x-closed", 3)
    obj = ns.open("x-closed")
    obj.execute("INSERT INTO t VALUES (42)")
    obj.close()
    assert _head_of("x-closed")["lease"]["owner"] is None
    for call in (lambda: obj.query("SELECT 1"),
                 lambda: obj.execute("INSERT INTO t VALUES (1)"),
                 obj.flush, obj.checkpoint):
        _expect(Closed, call)
    obj.close()  # idempotent
    assert _count(ns, "x-closed") == "4"
    _PASSED.append("close() flushes, releases the lease, and then refuses everything")


# =====================================================================
# lease exclusion, ids, and namespace behaviour
# =====================================================================

def test_lease_exclusion_and_same_owner_instance():
    ns = _seeded("x-excl", 2)
    obj = ns.open("x-excl")
    other = cd.Namespace(URL, owner="w2")
    _expect(LeaseHeld, other.open, "x-excl")
    # the same owner *string* is not a free pass: it may be another live
    # instance, and fencing it is what the lease exists to prevent
    same_owner = cd.DurableObject("x-excl", make_backend(URL, "x-excl"),
                                  owner="w1", db="mem")
    _expect(LeaseHeld, same_owner.open)
    same_owner.close()
    obj.close()
    _PASSED.append("an unexpired lease blocks another writer, same owner string or not")


def test_expired_lease_is_taken_without_force():
    ns = _seeded("x-expired", 2)
    backend = make_backend(URL, "x-expired")
    doc, etag = backend.get_with_etag(HEAD)
    doc = json.loads(doc)
    generation = doc["lease"]["generation"]
    doc["lease"] = {"owner": "dead", "instance": "dead-instance",
                    "generation": generation, "expires_at": time.time() - 3600}
    backend.replace_if_match(HEAD, json.dumps(doc).encode(), etag)
    obj = ns.open("x-expired")          # expired well past any clock skew
    assert obj.generation == generation + 1
    obj.close()
    _PASSED.append("an expired lease is taken normally, and moves the generation")


def test_unexpired_lease_needs_force():
    ns = _seeded("x-force", 1)
    backend = make_backend(URL, "x-force")
    doc, etag = backend.get_with_etag(HEAD)
    doc = json.loads(doc)
    doc["lease"] = {"owner": "w1", "instance": "dead-instance",
                    "generation": doc["lease"]["generation"] + 1,
                    "expires_at": time.time() + 3600}
    backend.replace_if_match(HEAD, json.dumps(doc).encode(), etag)
    _expect(LeaseHeld, ns.open, "x-force")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = ns.open("x-force", force=True)   # the administrative takeover
    # the cost of a force takeover has to be stated, not just documented
    assert any("not expired" in str(w.message) for w in caught), [str(w.message) for w in caught]
    assert obj.query("SELECT count() FROM t", "CSV").data().strip() == "1"
    obj.close()
    _PASSED.append("an unexpired lease is only taken by a force takeover, and it warns")


def test_object_id_validation_and_path_traversal():
    for bad in ("team/u1", "", "..", "a\\b"):
        _expect(ValueError, _fresh_ns().open, bad)
    for bad in ("../escape", "../../etc", "/abs/escape", "a/../../escape"):
        _expect(ValueError, make_backend,
                "local:" + tempfile.mkdtemp(prefix="dao-root-"), bad)
    root = tempfile.mkdtemp(prefix="dao-root-")
    os.symlink(tempfile.mkdtemp(prefix="dao-outside-"), os.path.join(root, "link"))
    _expect(ValueError, make_backend, "local:" + root, "link")
    make_backend("local:" + tempfile.mkdtemp(prefix="dao-root-"), "user-123")
    _PASSED.append("object ids stay flat keys, and no id escapes the backend root")


def test_sibling_isolation():
    ns = _fresh_ns()
    _seeded("foobar", 3, ns=ns)
    _seeded("foo", 7, ns=ns)
    ns.destroy("foo")
    assert _count(ns, "foobar") == "3"
    _PASSED.append("destroy('foo') leaves the sibling 'foobar' intact")


def test_destroy_refuses_an_active_lease():
    ns = _seeded("x-destroy", 2)
    obj = ns.open("x-destroy")
    _expect(LeaseHeld, ns.destroy, "x-destroy")
    ns.destroy("x-destroy", force=True)
    try:
        obj.close()          # the head is gone; being fenced here is correct
    except LeaseFenced:
        pass
    _PASSED.append("destroy refuses a leased object unless forced (not a V1 operation)")


def test_scan_across_objects():
    ns = _fresh_ns()
    _seeded("s-a", 1500, ns=ns)
    _seeded("s-b", 7, ns=ns)
    rows = dict(ns.scan("SELECT count() FROM t", ids=["s-a", "s-b"]))
    assert rows["s-a"].strip() == "1500" and rows["s-b"].strip() == "7", rows
    _expect(NotFound, ns.scan, "SELECT 1", ["s-a", "s-missing"])
    assert len(ns.scan("SELECT count() FROM t", ["s-a", "s-missing"], missing_ok=True)) == 1
    _PASSED.append("ns.scan unions read-only results, and says so when an object is absent")


def test_reopen_honors_persisted_db():
    # the object owns its database name; reopening with a different db argument
    # must not rewrite it, or the restore builds the wrong database
    ns = _seeded("x-db", 5, db="mem")
    other = cd.Namespace(URL, owner="w1", db="somethingelse")
    reader = other.open("x-db", read_only=True)
    assert reader.db == "mem"
    assert reader.query("SELECT count() FROM mem.t", "CSV").data().strip() == "5"
    reader.close()
    # a *writer* has to honour it too: it is the one that would rewrite the
    # manifest and checkpoint the wrong database
    writer = other.open("x-db")
    assert writer.db == "mem"
    writer.execute("INSERT INTO t VALUES (6)")   # unqualified, into mem
    writer.checkpoint()
    writer.close()
    assert _head_of("x-db")["manifest"]["db"] == "mem"
    assert _count(ns, "x-db", db="mem") == "6"
    _PASSED.append("reopen honors the manifest's database for readers and writers alike")


def test_cold_open_lands_in_the_objects_database():
    # a fresh object must already be in its own database, so unqualified writes
    # land where `BACKUP DATABASE` will find them
    ns = cd.Namespace(URL, owner="w1", db="analyst")
    ns.destroy("x-cold", force=True)
    obj = ns.open("x-cold")
    assert obj.query("SELECT currentDatabase()", "CSV").data().strip() == '"analyst"'
    obj.execute("CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n")
    obj.execute("INSERT INTO t VALUES (7)")
    obj.checkpoint()
    obj.close()
    assert _count(ns, "x-cold", db="analyst") == "1"
    _PASSED.append("a cold open starts in the object's own database")


def test_wal_replay_uses_the_objects_database():
    ns = cd.Namespace(URL, owner="w1", db="mem")
    ns.destroy("x-replay", force=True)
    obj = ns.open("x-replay")
    obj.execute("CREATE TABLE t (n Int64) ENGINE=MergeTree ORDER BY n")  # unqualified
    obj.execute("INSERT INTO t VALUES (1), (2), (3)")
    obj.flush()   # no checkpoint: force a replay on reopen
    obj.close()
    assert _count(ns, "x-replay", db="mem") == "3"
    _PASSED.append("WAL replay restores the object's database context")


def test_wal_keys_are_unique():
    ns = _seeded("x-keys", 1)
    obj = ns.open("x-keys")
    obj.execute("INSERT INTO t VALUES (1)")
    first = obj.flush()
    obj.execute("INSERT INTO t VALUES (2)")
    second = obj.flush()
    obj.close()
    assert first and second and first != second, (first, second)
    _PASSED.append("every flush mints a unique WAL key")


def test_local_and_file_urls_name_the_same_directory():
    # A namespace URL travels between bindings, so the local backend answers to
    # both spellings: this one wrote `local:`, Node wrote `file:`. A file: URL
    # is percent-encoded and a local: path is not, which is the only difference.
    root = tempfile.mkdtemp(prefix="dao-scheme-")
    nested = os.path.join(root, "a b")
    os.makedirs(nested, exist_ok=True)
    assert make_backend("local:" + nested, "o").root == make_backend(
        pathlib.Path(nested).as_uri(), "o").root
    _expect(ValueError, make_backend, "carrierpigeon://bucket/prefix")
    _PASSED.append("local: and file:// reach one backend; an unknown scheme is refused")


def test_constructor_validation():
    backend = make_backend("local:" + tempfile.mkdtemp(prefix="dao-ttl-"))
    for bad in (0, -1, float("inf"), float("nan")):
        _expect(ValueError, cd.DurableObject, "x", backend, lease_ttl=bad)
    _expect(ValueError, cd.DurableObject, "x", backend, clock_skew=-1)
    # a heartbeat slower than a third of the TTL cannot survive one lost renewal
    _expect(ValueError, cd.DurableObject, "x", backend, lease_ttl=30, heartbeat_interval=20)
    cd.DurableObject("x", backend, lease_ttl=30, heartbeat_interval=10)
    _PASSED.append("lease_ttl, clock_skew and heartbeat_interval are validated")


TESTS = [
    # §7.2 format fixtures
    test_empty_object,
    test_readonly_missing_object,
    test_checkpoint_only,
    test_checkpoint_plus_wal,
    test_quoted_database_name,
    test_missing_base_and_wal,
    test_bad_base_size_and_digest,
    test_bad_wal_size_and_digest,
    test_unknown_reader_feature,
    test_unknown_writer_feature,
    test_future_protocol_version,
    test_producer_version_differs_but_compatible,
    test_engine_reader_too_old,
    test_backup_format_too_new,
    test_unknown_fields_round_trip,
    test_json_shape_is_not_frozen,
    # §7.3 classification
    test_gates_cannot_be_bypassed_by_method_choice,
    test_multi_statement_and_parallel_with_refused,
    test_writes_outside_the_object_refused,
    test_database_lifecycle_refused,
    test_mutating_global_refused,
    test_control_statements_refused,
    test_secret_mutation_refused_without_leaking,
    test_secret_read_only_runs_and_is_not_logged,
    test_unknown_statement_fails_closed,
    test_limits_report_limit_exceeded,
    # §7.4 fault matrix
    test_conditional_create_race,
    test_wal_put_lands_but_head_cas_fails,
    test_head_cas_committed_but_response_lost,
    test_unprovable_commit_is_ambiguous,
    test_wal_upload_landed_but_response_lost,
    test_checkpoint_put_lands_but_head_cas_fails,
    test_restore_failure_frees_the_engine,
    test_heartbeat_renews_without_moving_generation_or_seq,
    test_writer_self_fences_when_renewal_fails,
    test_takeover_fences_the_previous_writer,
    test_close_releases_and_then_refuses,
    # leases, ids, namespace
    test_lease_exclusion_and_same_owner_instance,
    test_expired_lease_is_taken_without_force,
    test_unexpired_lease_needs_force,
    test_object_id_validation_and_path_traversal,
    test_sibling_isolation,
    test_destroy_refuses_an_active_lease,
    test_scan_across_objects,
    test_reopen_honors_persisted_db,
    test_cold_open_lands_in_the_objects_database,
    test_wal_replay_uses_the_objects_database,
    test_wal_keys_are_unique,
    test_local_and_file_urls_name_the_same_directory,
    test_constructor_validation,
]


def main():
    print(f"backend URL: {URL}")
    print(f"engine: {engine_version()}  protocol: V{cd.PROTOCOL_VERSION}")
    for test in TESTS:
        before = len(_PASSED)
        test()
        for line in _PASSED[before:]:
            print(f"  {line} ✓")
    print(f"ALL PASS ({len(_PASSED)} checks)")


if __name__ == "__main__":
    main()
