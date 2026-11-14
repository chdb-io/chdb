"""The storage seam — what a durable object needs from a provider, and no more.

A backend is a small key/value store scoped to one object's prefix. Two of its
six operations carry the whole protocol (§5.1):

* **conditional create** (`put_bytes_if_absent` / `put_file_if_absent`), which
  publishes an immutable WAL segment or checkpoint under a key nobody else can
  take;
* **conditional replace** (`replace_if_match`), which is both the head commit
  and the lease fence — a superseded writer's next commit fails the
  compare-and-set.

Both must be the provider's own atomic conditional operation. A HEAD followed
by a PUT is not a substitute: two writers can pass the same HEAD.

Three more rules the implementations here follow:

* an ETag is an opaque CAS token, never assumed to be an MD5 of the content;
* a checkpoint is uploaded from a file and downloaded to a file, so a full
  backup never has to be held in memory in one piece;
* a download lands on a unique temporary path and is only renamed into place
  once it verifies, so a partial transfer is never mistaken for the object.

**Indeterminate responses.** A timeout is not a failure — the server may have
committed. A backend raises `BackendAmbiguous` when it cannot tell, and the
state machine reconciles it (§5.8) into success, `lease_fenced`, or
`commit_ambiguous`. It never guesses on the backend's behalf.

`delete_prefix` is not part of V1: V1 has no destroy and no GC. It is here for
`Namespace.destroy`, which is a development convenience outside the protocol.

`make_backend(url, sub)` maps a URL to a backend scoped to `<base>/<sub>`:
    local:/path/to/root         (`file:///path/to/root` and a bare path too)
    s3://bucket/prefix          (endpoint via CHDB_DURABLE_S3_ENDPOINT for MinIO/R2)
    gcs://bucket/prefix
    azure://container/prefix

Credentials never go in the URL, which gets logged, committed and pasted into
issues. A URL carrying them (`s3://key:secret@bucket/...`) is refused rather
than stripped; they come from the environment or the provider's own chain.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse


@runtime_checkable
class Backend(Protocol):
    def get(self, key: str) -> Optional[bytes]: ...

    def get_with_etag(self, key: str): ...  # -> (Optional[bytes], Optional[str])

    def put_bytes_if_absent(self, key: str, data: bytes) -> Optional[str]: ...

    def put_file_if_absent(self, key: str, local_path: str) -> Optional[str]: ...

    def replace_if_match(self, key: str, data: bytes, etag: str) -> Optional[str]: ...

    def download_to_file(self, key: str, local_path: str) -> bool: ...

    def delete_prefix(self, prefix: str = "") -> None: ...  # not V1


def refuse_credentials(url: str) -> None:
    """Raise if `url` carries a `user:secret@` part in its authority.

    Otherwise `s3://key:secret@bucket/p` would make `key:secret@bucket` the
    bucket name, and the provider SDK quotes the bucket in its own errors.
    `:` as well as `@`, because a secret may contain `/`, which ends the
    authority at `key:sec` before the `@` is reached; neither character is
    legal in a bucket or container name. An `@` below the authority
    (`s3://bucket/a@b`) is an ordinary key character and stays allowed.
    Refused rather than stripped, because a caller who wrote credentials here
    believes they are being used. The message must quote no part of the URL.
    """
    u = urlparse(url)
    if (u.username is not None or u.password is not None
            or "@" in u.netloc or ":" in u.netloc):
        raise ValueError(
            "a durable namespace URL must not carry credentials; the URL is "
            "logged, committed and pasted into issues. Remove the `user:secret@` "
            "part and supply credentials through the provider's usual chain: "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or ~/.aws/credentials for "
            "s3, Application Default Credentials or CHDB_DURABLE_GCS_TOKEN for "
            "gcs, CHDB_DURABLE_AZURE_CONN for azure")


def make_backend(url: str, sub: str = "") -> Backend:
    """Build a backend scoped to `<base_prefix>/<sub>` from a URL."""
    refuse_credentials(url)
    u = urlparse(url)
    scheme = u.scheme or "local"

    # `file:` and `local:` are one backend under two names. Each binding grew
    # its own spelling -- Node writes `file:`, this one wrote `local:`, Go
    # takes either -- and a namespace URL is the sort of thing that gets shared
    # between a Python service and a Node one, so all three answer to both.
    if scheme in ("local", "file"):
        from .local import LocalFSBackend
        root = (u.netloc + u.path) if u.netloc else u.path
        # A file: URL is percent-encoded; local: and a bare path are literal.
        if scheme == "file":
            root = unquote(root)
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

    # refuse_credentials left the authority a bare name. Not `u.hostname`,
    # which lowercases it.
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
