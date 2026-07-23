"""chdb.durable — Durable Analytical Object.

An addressable, single-writer chDB engine whose authoritative state lives in
object storage you own (S3 / GCS / Azure Blob / any S3-compatible / a local
folder). Restore is fast, the lease makes single-writer a guarantee not an
assumption, and the on-disk format is portable — your object is a folder you
can move between clouds.

    from chdb import durable as cd
    ns = cd.Namespace("s3://bucket/prefix", owner="worker-1")
    obj = ns.open("user-123")            # lease + restore (base + WAL replay)
    obj.execute("INSERT INTO mem.beliefs VALUES (...)")
    obj.flush()                          # cut a WAL segment (RPO boundary)
    obj.checkpoint()                     # fold into a fresh base
    obj.close()                          # flush + release lease
    ns.scan("SELECT count() FROM mem.beliefs", ids=["user-1", "user-2"])

Security scope: chdb.durable provides single-writer *coordination* (the lease
+ compare-and-set fence), not security. Access control is entirely your object
store's IAM — anyone who can write the object's prefix can read, modify, or
take its lock. There is no application-level auth, no client-side encryption,
and no tamper protection: head/WAL/checkpoints are stored as-is (rely on the
bucket's server-side encryption if you need at-rest encryption). For
multi-tenant use, give each tenant credentials scoped to its own prefix.
"""
from .errors import DurableError, LeaseError, MissingDependency
from .namespace import Namespace
from .object import DurableObject
from .backends import Backend, make_backend

__all__ = [
    "Namespace", "DurableObject", "Backend", "make_backend",
    "DurableError", "LeaseError", "MissingDependency",
]
__version__ = "0.1.0"
