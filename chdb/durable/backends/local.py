"""Local filesystem backend — the most vendor-neutral home of all: a folder.

**Scope: development, tests, and single-writer/single-host use.** The strong
concurrency guarantees a Durable Analytical Object relies on — atomic
conditional writes for the lease/head CAS — belong to the object-store
backends (S3 `IfMatch`/`IfNoneMatch`, GCS generation-match, Azure ETag), which
provide them natively. This backend approximates them on a POSIX filesystem
(`O_CREAT|O_EXCL` for create, `flock` for compare-and-set, path containment in
`_p`) as a best effort so tests are realistic, but it is not intended for
multi-host or production concurrent access. Point `durable:` at a cloud (or
S3-compatible) backend for that.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional


class LocalFSBackend:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def _p(self, key: str) -> str:
        # Constrain every key under root: an absolute key or ".." would
        # otherwise let put/delete_prefix write or rm outside the backend.
        full = os.path.realpath(os.path.join(self.root, key))
        root = os.path.realpath(self.root)
        if full != root and not full.startswith(root + os.sep):
            raise ValueError(f"key escapes backend root: {key!r}")
        return full

    def get(self, key: str) -> Optional[bytes]:
        try:
            with open(self._p(key), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def get_with_etag(self, key: str):
        # Read body and etag from the SAME file descriptor so they describe the
        # same version — a separate stat could pair an old body with a new etag
        # if the file is replaced between the two calls.
        try:
            fd = os.open(self._p(key), os.O_RDONLY)
        except FileNotFoundError:
            return (None, None)
        try:
            st = os.fstat(fd)
            f = os.fdopen(fd, "rb")
            fd = -1  # fdopen owns the descriptor now
            with f:
                data = f.read()
        except BaseException:
            if fd >= 0:  # only close if fdopen never took ownership
                os.close(fd)
            raise
        return (data, f"{st.st_mtime_ns}-{st.st_size}")

    def put(self, key: str, data: bytes) -> None:
        p = self._p(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)  # atomic

    def put_if_absent(self, key: str, data: bytes) -> Optional[str]:
        p = self._p(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            st = os.fstat(f.fileno())  # etag of exactly what we created
        return f"{st.st_mtime_ns}-{st.st_size}"

    def head_etag(self, key: str) -> Optional[str]:
        try:
            st = os.stat(self._p(key))
        except FileNotFoundError:
            return None
        return f"{st.st_mtime_ns}-{st.st_size}"

    def replace_if_match(self, key: str, data: bytes, etag: str) -> Optional[str]:
        import fcntl
        # serialize check-then-act across processes so two writers can't both
        # observe the same etag and both write (which would defeat the CAS).
        lockpath = self._p(key) + ".lock"
        os.makedirs(os.path.dirname(lockpath), exist_ok=True)
        with open(lockpath, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            if self.head_etag(key) != etag:
                return None
            self.put(key, data)
            return self.head_etag(key)

    def delete_prefix(self, prefix: str = "") -> None:
        target = self._p(prefix) if prefix else self.root
        shutil.rmtree(target, ignore_errors=True)
