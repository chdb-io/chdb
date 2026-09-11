"""chdb.durable — a chDB database that lives in object storage you own.

An addressable, single-writer chDB engine whose authoritative state is in S3 /
GCS / Azure Blob / any S3-compatible store / a local folder. Restore is fast,
the lease makes single-writer a guarantee rather than an assumption, and the
layout is open format — an object is a folder you can move between clouds.

    from chdb import durable as cd
    ns = cd.Namespace("s3://bucket/prefix", owner="worker-1")
    obj = ns.open("user-123")            # lease + restore (base + WAL replay)
    obj.execute("INSERT INTO beliefs VALUES (...)")   # local; buffered
    obj.flush()                          # publish a WAL segment (the RPO boundary)
    obj.checkpoint()                     # fold into a fresh base
    obj.close()                          # flush + release the lease
    ns.scan("SELECT count() FROM beliefs", ids=["user-1", "user-2"])

This implements **Durable V1** as frozen in `dev-docs/CHDB_DURABLE_V1_CONTRACT.md`:
one database per object, one writer plus any number of read-only openers, one
statement per public call, a statement WAL, full checkpoints, and the lease /
CAS / fencing rules every binding shares. Objects it writes are readable by the
Node, Go and Rust bindings of the same protocol version, and vice versa.

`execute()` is *not* a durability barrier. It runs the statement locally and
buffers it; `flush()` is what makes it recoverable elsewhere. A service that
promises a completed request survives a crash has to await `flush()` before it
answers.

Requires chdb-core 26.7.2-rc.2 or newer (`pip install -U chdb-core`): backup,
restore and statement classification are the engine's job, and this package
builds no `BACKUP`/`RESTORE` SQL and runs no regex over a caller's statement.

Security scope: chdb.durable provides single-writer *coordination* (the lease
plus the compare-and-set fence), not security. Access control is entirely your
object store's IAM — anyone who can write the object's prefix can read,
modify, or take its lease. There is no application-level auth, no client-side
encryption, and no tamper protection: head, WAL and checkpoints are stored as
written (use the bucket's server-side encryption if you need encryption at
rest). For multi-tenant use, scope each tenant's credentials to its own prefix.
"""
from .backends import Backend, make_backend
from .errors import (
    BackendError,
    ClassificationRefused,
    Closed,
    CommitAmbiguous,
    Corrupt,
    DurableError,
    EngineError,
    EngineIncompatible,
    LeaseError,
    LeaseFenced,
    LeaseHeld,
    LimitExceeded,
    MissingDependency,
    NotFound,
    ProtocolUnsupported,
    SecretRefused,
    Timeout,
    V1_CATEGORIES,
)
from .head import EngineInfo, Head, Lease, Manifest, ObjectRef, ProtocolInfo
from .namespace import Namespace
from .object import DurableObject
from .protocol import BACKUP_FORMAT, PROTOCOL_VERSION

__all__ = [
    "Namespace", "DurableObject", "Backend", "make_backend",
    "Head", "Lease", "Manifest", "ObjectRef", "EngineInfo", "ProtocolInfo",
    "PROTOCOL_VERSION", "BACKUP_FORMAT", "V1_CATEGORIES",
    "DurableError", "NotFound", "LeaseError", "LeaseHeld", "LeaseFenced",
    "EngineIncompatible", "ProtocolUnsupported", "Corrupt",
    "ClassificationRefused", "SecretRefused", "EngineError", "BackendError",
    "Timeout", "CommitAmbiguous", "LimitExceeded", "Closed", "MissingDependency",
]

#: The protocol this package implements, not the package's own release number.
__version__ = "1.0"
