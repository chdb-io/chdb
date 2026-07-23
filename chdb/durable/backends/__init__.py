"""Storage-backend abstraction — the seam that makes a Durable Analytical
Object's home vendor-neutral.

A backend is a tiny key/value store scoped to one object's prefix, with a
*conditional create* (`put_if_absent`) and *conditional replace*
(`replace_if_match`) — the two primitives the single-writer lease is built on.
Every provider implements the same `Backend` protocol; the on-disk *format*
(a chDB File backup + WAL segments + JSON manifest) is identical everywhere,
so moving an object between clouds is a byte copy.

`make_backend(url, sub)` maps a URL to a backend scoped to `<base>/<sub>`:
    local:/path/to/root
    s3://bucket/prefix          (endpoint via CHDB_DURABLE_S3_ENDPOINT for MinIO/R2)
    gcs://bucket/prefix
    azure://container/prefix
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable
from urllib.parse import urlparse


@runtime_checkable
class Backend(Protocol):
    def get(self, key: str) -> Optional[bytes]: ...
    def get_with_etag(self, key: str): ...  # -> (Optional[bytes], Optional[str])
    def put(self, key: str, data: bytes) -> None: ...
    def put_if_absent(self, key: str, data: bytes): ...  # -> Optional[str] etag, or None if it exists
    def head_etag(self, key: str) -> Optional[str]: ...
    def replace_if_match(self, key: str, data: bytes, etag: str) -> Optional[str]: ...
    def delete_prefix(self, prefix: str = "") -> None: ...


def make_backend(url: str, sub: str = "") -> Backend:
    """Build a backend scoped to `<base_prefix>/<sub>` from a URL."""
    u = urlparse(url)
    scheme = u.scheme or "local"

    if scheme == "local":
        from .local import LocalFSBackend
        root = (u.netloc + u.path) if u.netloc else u.path
        base = os.path.realpath(root)
        # Constrain the object id beneath root — a caller-controlled sub like
        # "../../x", an absolute path, or a symlink to an outside dir would
        # otherwise escape the namespace and let Namespace.destroy() rm -rf an
        # arbitrary directory. realpath resolves symlinks before the check.
        full = os.path.realpath(os.path.join(base, sub)) if sub else base
        # os.path.join(base, "") appends exactly one separator, and yields the
        # bare separator when base is the filesystem root — so `local:/` works.
        if full != base and not full.startswith(os.path.join(base, "")):
            raise ValueError(f"object id escapes backend root: {sub!r}")
        return LocalFSBackend(full)

    container = u.netloc
    base = u.path.strip("/")
    prefix = f"{base}/{sub}".strip("/") if sub else base

    if scheme == "s3":
        from .s3 import S3Backend
        region = os.getenv("AWS_REGION", "us-east-1")
        endpoint = os.getenv("CHDB_DURABLE_S3_ENDPOINT")
        if endpoint:
            # custom endpoint (MinIO / R2 / …) with static keys
            return S3Backend(container, prefix, endpoint_url=endpoint,
                             access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                             secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                             region=region)
        # real AWS: let boto3's default chain resolve creds (env incl. session
        # token, SSO, or instance profile) — don't pin static keys
        return S3Backend(container, prefix, region=region)
    if scheme == "gcs":
        from .gcs import GCSBackend
        return GCSBackend(container, prefix,
                          token=os.getenv("CHDB_DURABLE_GCS_TOKEN"),  # optional; else ADC
                          project=os.getenv("CHDB_DURABLE_GCP_PROJECT"))
    if scheme == "azure":
        from .azure import AzureBlobBackend
        return AzureBlobBackend(container, prefix,
                                conn_str=os.environ["CHDB_DURABLE_AZURE_CONN"])
    raise ValueError(f"unsupported durable backend scheme: {scheme!r}")
