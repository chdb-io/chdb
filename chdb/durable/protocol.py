"""What V1 freezes: the baseline this implementation claims, the limits it
enforces, and the two comparisons that decide whether an object can be opened.

Everything here is contract text turned into code, so a reader can check one
file against `dev-docs/CHDB_DURABLE_V1_CONTRACT.md` §4 rather than hunting
constants through the state machine.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from .errors import Corrupt

#: Protocol baseline. A head declaring a higher version is refused (§4.3).
PROTOCOL_VERSION = 1

#: V1 defines no feature names, so any name in a head is one we do not know.
#: An unknown *reader* feature refuses the open outright; an unknown *writer*
#: feature allows a read-only open but refuses the writer lease.
SUPPORTED_READER_FEATURES = frozenset()
SUPPORTED_WRITER_FEATURES = frozenset()

#: `head.engine.name`. An object written by something else is not ours to open.
ENGINE_NAME = "chdb"

#: chDB backup archive format generation. A head above this cannot be restored
#: by this engine, so an older reader fails closed instead of guessing (§4.3).
BACKUP_FORMAT = 1

#: Frozen size limits (§4.4, §4.5). A writer refuses locally before publishing
#: anything; a reader must cope with objects up to exactly these sizes.
MAX_SQL_BYTES = 64 * 1024 * 1024
MAX_WAL_SEGMENT_BYTES = 128 * 1024 * 1024
MAX_HEAD_BYTES = 1024 * 1024

#: JSON's safe integer range, which every binding has to agree on (§4.2).
MAX_SAFE_INT = 2 ** 53 - 1

#: Lease defaults. `heartbeat_interval` must not exceed a third of the TTL
#: (§5.7); `clock_skew` is the allowance a normal (non-force) takeover adds to
#: an expiry before it believes the lease is really gone.
DEFAULT_LEASE_TTL = 60.0
DEFAULT_CLOCK_SKEW = 5.0
HEARTBEAT_TTL_FRACTION = 1.0 / 3.0

#: How long a commit keeps trying to prove itself before it reports
#: `commit_ambiguous`, and how the retries back off (§5.8).
DEFAULT_COMMIT_DEADLINE = 30.0
COMMIT_MAX_ATTEMPTS = 5
COMMIT_BACKOFF_BASE = 0.2

HEAD_KEY = "head.json"

#: The database a cold object is created with when the caller names none. The
#: head is authoritative once an object exists, so this only decides what a new
#: one is stamped with -- but it decides it identically in Python, Node and Go,
#: so the same code ported between them addresses the same table.
DEFAULT_DATABASE = "default"

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def validate_sha256(value: object, where: str) -> str:
    """A digest is a lowercase 64-character hex string, or the head is corrupt."""
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise Corrupt(f"{where}: expected a lowercase hex SHA-256, got {value!r}")
    return value


def validate_object_key(key: object, where: str) -> str:
    """An object reference is a relative key under the object's own prefix.

    §4.1 spells out what that excludes: an absolute key, a backslash, an empty
    segment, `.` or `..`. Each of those would let a head point outside the
    object it belongs to, which is how one tenant's manifest ends up naming
    another tenant's blob.
    """
    if not isinstance(key, str) or not key:
        raise Corrupt(f"{where}: expected a non-empty object key, got {key!r}")
    if key.startswith("/") or "\\" in key:
        raise Corrupt(f"{where}: object key must be relative with '/' separators: {key!r}")
    for segment in key.split("/"):
        if segment in ("", ".", ".."):
            raise Corrupt(f"{where}: object key has an empty or dot segment: {key!r}")
    return key


def validate_safe_int(value: object, where: str, *, minimum: int = 0) -> int:
    """An integer both a 53-bit and a 64-bit binding can round-trip."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise Corrupt(f"{where}: expected an integer, got {value!r}")
    if value < minimum or value > MAX_SAFE_INT:
        raise Corrupt(f"{where}: integer {value} outside the cross-language safe range")
    return value


# -- engine version comparison --------------------------------------------
#
# chDB versions come in two shapes: a release tag ("26.7.2", "26.7.2.59") and
# a pre-release of one ("26.7.2-rc.2"). The `min_reader` gate has to order
# both, and get the one case that matters right: a release is newer than its
# own pre-releases, so 26.7.2 can read what 26.7.2-rc.2 wrote, and rc.2 cannot
# read what 26.7.2 wrote.

_PRE_RANK = {"alpha": 0, "a": 0, "beta": 1, "b": 1, "pre": 2, "rc": 3}
_VERSION_RE = re.compile(
    r"\A(?P<release>[0-9]+(?:\.[0-9]+)*)"
    r"(?:[-.](?P<pre>[A-Za-z]+)\.?(?P<pre_num>[0-9]+)?)?"
    r"(?:[-.+].*)?\Z"
)


def parse_version(text: object) -> Optional[Tuple[Tuple[int, ...], int, int]]:
    """Order-comparable form of a chDB version, or None if it is not one.

    A suffix this function does not recognise as a pre-release marker (say
    `-stable`) is a plain release: 26.7.2.59-stable orders as 26.7.2.59.
    """
    if not isinstance(text, str):
        return None
    m = _VERSION_RE.match(text.strip())
    if not m:
        return None
    release = tuple(int(p) for p in m.group("release").split("."))
    pre = m.group("pre")
    if pre is None:
        return (release, len(_PRE_RANK), 0)  # a release outranks every pre-release
    rank = _PRE_RANK.get(pre.lower())
    if rank is None:
        return (release, len(_PRE_RANK), 0)
    return (release, rank, int(m.group("pre_num") or 0))


def version_lt(left: str, right: str) -> Optional[bool]:
    """Is `left` older than `right`? None when either side is unorderable.

    None is not "no": a caller gating an open on this has to fail closed,
    because an unreadable version string is not proof of compatibility.
    """
    a, b = parse_version(left), parse_version(right)
    if a is None or b is None:
        return None
    width = max(len(a[0]), len(b[0]))
    a_release = a[0] + (0,) * (width - len(a[0]))
    b_release = b[0] + (0,) * (width - len(b[0]))
    return (a_release, a[1], a[2]) < (b_release, b[1], b[2])
