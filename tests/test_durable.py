"""chdb.durable V1 tests — runnable as a plain script.

    DAO_TEST_URL=local:/tmp/dao-test python tests/test_durable.py
    DAO_TEST_URL=s3://dao/ns CHDB_DURABLE_S3_ENDPOINT=http://127.0.0.1:9000 \
        AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
        python tests/test_durable.py
"""
import os
import tempfile
import time

import json

from chdb import durable as cd
from chdb.durable.backends import make_backend
from chdb.durable.errors import LeaseError

URL = os.getenv("DAO_TEST_URL", "local:" + tempfile.mkdtemp(prefix="dao-test-"))


def _fresh_ns():
    ns = cd.Namespace(URL, owner="w1")
    for oid in ("obj-a", "obj-b", "obj-f"):
        ns.destroy(oid)
    return ns


def test_checkpoint_roundtrip(ns):
    o = ns.open("obj-a")
    assert o.query("SELECT 1", "CSV")  # engine live
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    o.execute("INSERT INTO mem.t SELECT number FROM numbers(1000)")
    o.checkpoint()
    o.close()

    r = ns.open("obj-a")
    got = r.query("SELECT count() FROM mem.t", "CSV").data().strip()
    r.close()
    assert got == "1000", f"checkpoint restore: {got}"
    print("  checkpoint round-trip: 1000 rows restored from base ✓")


def test_wal_incremental(ns):
    # append after checkpoint, flush only (no new checkpoint) -> survives via WAL
    o = ns.open("obj-a")
    o.execute("INSERT INTO mem.t SELECT number FROM numbers(1000, 500)")  # +500
    seg = o.flush()
    assert seg and seg.startswith("wal/"), seg
    o.close()

    r = ns.open("obj-a")
    got = r.query("SELECT count() FROM mem.t", "CSV").data().strip()
    r.close()
    assert got == "1500", f"WAL replay: {got}"
    print("  WAL incremental: 1500 rows (base 1000 + WAL 500 replayed) ✓")


def test_checkpoint_folds_wal(ns):
    o = ns.open("obj-a")
    key = o.checkpoint()  # fold base+WAL into a new base
    assert o.base == key and o.wal == []
    o.close()
    r = ns.open("obj-a")
    got = r.query("SELECT count() FROM mem.t", "CSV").data().strip()
    r.close()
    assert got == "1500", got
    print("  checkpoint folds WAL: base=1500, wal=[] ✓")


def test_lease_exclusion(ns):
    o = ns.open("obj-b")
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    other = cd.Namespace(URL, owner="w2")  # a *different* writer
    try:
        other.open("obj-b")  # must be refused before any second session opens
        raise AssertionError("second writer should have raised LeaseError")
    except LeaseError:
        pass
    o.close()
    print("  lease exclusion: different writer refused ✓")


def test_fencing(ns):
    o = ns.open("obj-f")
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    # an external writer takes over: overwrite head at a higher generation,
    # invalidating o's cached etag
    b = make_backend(ns.url, "obj-f")
    data, etag = b.get_with_etag("head.json")
    h = json.loads(data)
    h["lease"] = {"owner": "w2", "generation": h["lease"]["generation"] + 5,
                  "expires_at": 9e12}
    assert b.replace_if_match("head.json", json.dumps(h).encode(), etag) is not None
    try:
        o.flush()  # o's commit must fail the compare-and-set
        raise AssertionError("o should be fenced after takeover")
    except LeaseError:
        pass
    # close() must also surface the fence (buffered writes can't be persisted)
    # rather than swallow it and return normally.
    try:
        o.close()
        raise AssertionError("close() should raise when fenced with buffered writes")
    except LeaseError:
        pass
    print("  fencing: superseded writer's commit rejected via head CAS ✓")


def test_scan(ns):
    for oid, k in (("obj-a", 1500), ("obj-b", 0)):
        pass
    # obj-a has 1500; give obj-b a table with 7 rows
    o = ns.open("obj-b")
    o.execute("INSERT INTO mem.t SELECT number FROM numbers(7)")
    o.checkpoint()
    o.close()
    rows = dict(ns.scan("SELECT count() FROM mem.t", ids=["obj-a", "obj-b"]))
    a = rows["obj-a"].strip()
    b = rows["obj-b"].strip()
    assert a == "1500" and b == "7", (a, b)
    print(f"  ns.scan across objects: obj-a={a}, obj-b={b} ✓")


def test_local_path_traversal():
    # a caller-controlled object id must not escape the backend root
    for bad in ("../escape", "../../etc", "/abs/escape", "a/../../escape"):
        try:
            make_backend("local:" + tempfile.mkdtemp(prefix="dao-root-"), bad)
            raise AssertionError(f"expected rejection for id {bad!r}")
        except ValueError:
            pass
    # a symlink id pointing outside the root must also be rejected (realpath)
    root = tempfile.mkdtemp(prefix="dao-root-")
    outside = tempfile.mkdtemp(prefix="dao-outside-")
    os.symlink(outside, os.path.join(root, "link"))
    try:
        make_backend("local:" + root, "link")
        raise AssertionError("symlink escaping root should be rejected")
    except ValueError:
        pass
    # a normal id is fine
    make_backend("local:" + tempfile.mkdtemp(prefix="dao-root-"), "user-123")
    print("  local path traversal: escaping object ids (incl. symlink) rejected ✓")


def test_sibling_isolation(ns):
    # destroy("foo") must not delete a sibling object whose id shares the prefix
    for oid in ("foo", "foobar"):
        ns.destroy(oid)
    for oid, k in (("foobar", 3), ("foo", 7)):
        o = ns.open(oid)
        o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
        o.execute(f"INSERT INTO mem.t SELECT number FROM numbers({k})")
        o.checkpoint()
        o.close()
    ns.destroy("foo")
    r = ns.open("foobar", read_only=True)
    got = r.query("SELECT count() FROM mem.t", "CSV").data().strip()
    r.close()
    assert got == "3", f"sibling clobbered: {got}"
    print("  sibling isolation: destroy('foo') left 'foobar' intact ✓")


def test_object_id_rejects_slash(ns):
    # ids must be flat keys — no '/', empty, '.', '..', '\\' (else a short id's
    # prefix could contain another id's, and destroy would delete both)
    for bad in ("team/u1", "", "..", "a\\b"):
        try:
            ns.open(bad)
            raise AssertionError(f"object id {bad!r} should be rejected")
        except ValueError:
            pass
    print("  object id validation: rejects '/', empty, '..', '\\\\' ✓")


def test_missing_reference_errors(ns):
    # a head referencing a base/WAL blob that's missing from storage must fail
    # loudly, not silently open with committed state omitted
    ns.destroy("obj-mb")
    b = make_backend(ns.url, "obj-mb")
    b.put_if_absent("head.json", json.dumps({
        "lease": {"owner": "", "generation": 1, "expires_at": 0},
        "manifest": {"db": "mem", "base": "checkpoints/nope.tar.gz", "wal": [], "seq": 1},
    }).encode())
    try:
        ns.open("obj-mb")
        raise AssertionError("expected RuntimeError for missing base")
    except RuntimeError as e:
        assert "missing base" in str(e), str(e)

    ns.destroy("obj-mb2")
    b2 = make_backend(ns.url, "obj-mb2")
    b2.put_if_absent("head.json", json.dumps({
        "lease": {"owner": "", "generation": 1, "expires_at": 0},
        "manifest": {"db": "mem", "base": None, "wal": ["wal/nope.jsonl"], "seq": 1},
    }).encode())
    try:
        ns.open("obj-mb2")
        raise AssertionError("expected RuntimeError for missing WAL segment")
    except RuntimeError as e:
        assert "missing WAL" in str(e), str(e)
    print("  missing base/WAL reference: fails loudly ✓")


def test_wal_keys_unique(ns):
    # unique segment keys so a retry can't overwrite an already-published segment
    ns.destroy("obj-uk")
    o = ns.open("obj-uk")
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    o.execute("INSERT INTO mem.t VALUES (1)")
    k1 = o.flush()
    o.execute("INSERT INTO mem.t VALUES (2)")
    k2 = o.flush()
    o.close()
    assert k1 and k2 and k1 != k2, (k1, k2)
    print("  WAL keys unique across flushes ✓")


def test_lease_renewal(ns):
    # a long-lived writer renews its lease before expiry (not fenced while active)
    ns.destroy("obj-lr")
    b = make_backend(ns.url, "obj-lr")
    o = cd.DurableObject("obj-lr", b, owner="w1", db="mem", lease_ttl=0.4)
    o.open()
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    exp1 = json.loads(b.get("head.json"))["lease"]["expires_at"]
    time.sleep(0.35)  # cross the renew threshold (ttl*0.75 = 0.3s before expiry)
    o.execute("INSERT INTO mem.t VALUES (1)")  # should renew the lease
    exp2 = json.loads(b.get("head.json"))["lease"]["expires_at"]
    o.close()
    assert exp2 > exp1, (exp1, exp2)
    print("  lease renewal: expires_at advanced on execute near expiry ✓")


def test_lease_ttl_validation():
    b = make_backend("local:" + tempfile.mkdtemp(prefix="dao-ttl-"))
    for bad in (0, -1, float("inf"), float("nan")):
        try:
            cd.DurableObject("x", b, lease_ttl=bad)
            raise AssertionError(f"lease_ttl {bad} should be rejected")
        except ValueError:
            pass
    print("  lease_ttl validation: rejects 0 / negative / inf / nan ✓")


def test_readonly_restore_failure_frees_session(ns):
    # a failed read-only restore must free the chDB session (one-per-process),
    # so a subsequent open isn't blocked
    ns.destroy("obj-ro")
    b = make_backend(ns.url, "obj-ro")
    b.put_if_absent("head.json", json.dumps({
        "lease": {"owner": "", "instance": "", "generation": 1, "expires_at": 0},
        "manifest": {"db": "mem", "base": "checkpoints/nope.tar.gz", "wal": [], "seq": 1},
    }).encode())
    o = cd.DurableObject("obj-ro", b, read_only=True, db="mem")
    try:
        o.open()
        raise AssertionError("expected RuntimeError for missing base")
    except RuntimeError:
        pass
    o2 = ns.open("obj-ro2")  # would fail with chDB one-session error if not freed
    o2.query("SELECT 1")
    o2.close()
    print("  read-only restore failure frees the session ✓")


def test_same_owner_second_instance_blocked(ns):
    # a second *live* instance with the same owner string must still be refused
    # (identity is per-instance, not per-owner) — else it would fence the first
    ns.destroy("obj-so")
    o = ns.open("obj-so")  # instance A, owner w1
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    b = make_backend(ns.url, "obj-so")
    o2 = cd.DurableObject("obj-so", b, owner=ns.owner, db="mem")  # same owner, instance B
    try:
        o2.open()
        raise AssertionError("second same-owner live instance should be refused")
    except LeaseError:
        pass
    o.close()
    print("  same-owner second live instance refused ✓")


def test_put_if_absent_returns_etag(ns):
    # cold-create adopts this etag directly (no racy second fetch)
    ns.destroy("obj-pia")
    b = make_backend(ns.url, "obj-pia")
    e = b.put_if_absent("head.json", b'{"x":1}')
    assert e, "put_if_absent should return an etag on create"
    assert b.put_if_absent("head.json", b'{"x":2}') is None, "should return None if it exists"
    print("  put_if_absent returns etag on create / None if exists ✓")


def test_reconcile_detects_committed(ns):
    # ambiguous CAS: if the head already references our unique key, reconcile
    # must report "committed" (so a lost-response retry doesn't double-publish)
    ns.destroy("obj-rec")
    o = ns.open("obj-rec")
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    o.execute("INSERT INTO mem.t VALUES (1)")
    k = o.flush()
    assert o._reconcile_committed(wal_key=k) is True
    assert o._reconcile_committed(wal_key="wal/nonexistent.jsonl") is False
    ck = o.checkpoint()
    assert o._reconcile_committed(base=ck) is True
    # ownership check: if another writer took over (kept our key), reconcile
    # must NOT adopt — a retained key proves durability, not ownership
    b = make_backend(ns.url, "obj-rec")
    h = json.loads(b.get("head.json"))
    h["lease"]["instance"] = "other-instance"
    b.put("head.json", json.dumps(h).encode())
    assert o._reconcile_committed(base=ck) is False
    o.close()
    print("  reconcile detects committed key + still-owned; rejects after takeover ✓")


def test_destroy_refuses_active_lease(ns):
    ns.destroy("obj-de")
    o = ns.open("obj-de")  # holds an active lease
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    o.flush()
    try:
        ns.destroy("obj-de")
        raise AssertionError("destroy should refuse an actively-leased object")
    except LeaseError:
        pass
    ns.destroy("obj-de", force=True)  # force overrides
    try:
        o.close()  # fenced (head gone); tolerate
    except LeaseError:
        pass
    print("  destroy refuses an active lease (force overrides) ✓")


def test_force_take(ns):
    # crash recovery: a new writer force-takes an unexpired foreign lease
    # (held by a now-dead instance) without waiting out the TTL
    ns.destroy("obj-ft")
    o = ns.open("obj-ft")
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    o.execute("INSERT INTO mem.t VALUES (9)")
    o.checkpoint()
    o.close()
    b = make_backend(ns.url, "obj-ft")
    head = json.loads(b.get("head.json"))  # forge a live-looking foreign lease
    head["lease"] = {"owner": "w1", "instance": "dead-instance",
                     "generation": head["lease"]["generation"] + 1, "expires_at": 9e12}
    b.put("head.json", json.dumps(head).encode())
    try:
        ns.open("obj-ft")  # normal open must be blocked by the unexpired lease
        raise AssertionError("held lease should block a normal open")
    except LeaseError:
        pass
    r = ns.open("obj-ft", force=True)  # force-take + restore
    got = r.query("SELECT n FROM mem.t", "CSV").data().strip()
    r.close()
    assert got == "9", got
    print("  force-take: bypasses an unexpired foreign lease (crash recovery) ✓")


def test_reopen_honors_persisted_db(ns):
    # the object owns its database name; reopening with a different db arg must
    # not rewrite it (otherwise restore builds the wrong database)
    ns.destroy("obj-db")
    o = ns.open("obj-db")  # ns default db = "mem"
    o.execute("CREATE TABLE mem.t (n Int64) ENGINE=MergeTree ORDER BY n")
    o.execute("INSERT INTO mem.t VALUES (5)")
    o.checkpoint()
    o.close()
    ns2 = cd.Namespace(ns.url, owner="w1", db="other")  # different db arg
    r = ns2.open("obj-db", read_only=True)
    got = r.query("SELECT n FROM mem.t", "CSV").data().strip()  # persisted "mem" wins
    r.close()
    assert got == "5", got
    print("  reopen honors persisted manifest db ✓")


if __name__ == "__main__":
    print(f"backend URL: {URL}")
    ns = _fresh_ns()
    test_checkpoint_roundtrip(ns)
    test_wal_incremental(ns)
    test_checkpoint_folds_wal(ns)
    test_lease_exclusion(ns)
    test_fencing(ns)
    test_scan(ns)
    test_local_path_traversal()
    test_sibling_isolation(ns)
    test_object_id_rejects_slash(ns)
    test_missing_reference_errors(ns)
    test_wal_keys_unique(ns)
    test_lease_renewal(ns)
    test_same_owner_second_instance_blocked(ns)
    test_put_if_absent_returns_etag(ns)
    test_reconcile_detects_committed(ns)
    test_destroy_refuses_active_lease(ns)
    test_force_take(ns)
    test_reopen_honors_persisted_db(ns)
    test_lease_ttl_validation()
    test_readonly_restore_failure_frees_session(ns)
    print("ALL PASS")
