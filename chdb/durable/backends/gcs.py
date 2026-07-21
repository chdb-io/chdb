"""Native GCS backend. Lease primitive = generation-match conditional write
(`if_generation_match=0` to create), GCS's own single-writer primitive."""
from __future__ import annotations

from typing import Optional

from ..errors import MissingDependency


class GCSBackend:
    def __init__(self, bucket: str, prefix: str, *, token: Optional[str] = None, project=None):
        try:
            from google.cloud import storage
        except ImportError as e:
            raise MissingDependency(
                "gcs backend needs google-cloud-storage: pip install 'chdb[durable-gcs]'") from e
        self.prefix = prefix.strip("/")
        if token:
            # Explicit bare token: for constrained envs (e.g. a metadata-server
            # access token). NOTE: it cannot refresh — it stops working once the
            # short-lived token expires. Prefer leaving token unset to use
            # Application Default Credentials, which refresh automatically.
            from google.oauth2.credentials import Credentials
            client = storage.Client(project=project, credentials=Credentials(token=token))
        else:
            client = storage.Client(project=project)  # ADC (auto-refreshing)
        self.bucket = client.bucket(bucket)

    def _b(self, key: str):
        return self.bucket.blob(f"{self.prefix}/{key}" if self.prefix else key)

    def get(self, key: str) -> Optional[bytes]:
        return self.get_with_etag(key)[0]

    def get_with_etag(self, key: str):
        from google.cloud.exceptions import NotFound
        b = self._b(key)
        try:
            data = b.download_as_bytes()  # populates b.generation
        except NotFound:
            return (None, None)
        gen = b.generation
        if gen is None:
            b.reload()
            gen = b.generation
        return (data, str(gen))

    def put(self, key: str, data: bytes) -> None:
        self._b(key).upload_from_string(data)

    def put_if_absent(self, key: str, data: bytes) -> Optional[str]:
        from google.api_core.exceptions import PreconditionFailed
        b = self._b(key)
        try:
            b.upload_from_string(data, if_generation_match=0)
            return str(b.generation)  # generation of the object we just created
        except PreconditionFailed:
            return None

    def head_etag(self, key: str) -> Optional[str]:
        from google.cloud.exceptions import NotFound
        b = self._b(key)
        try:
            b.reload()
        except NotFound:
            return None
        return str(b.generation)

    def replace_if_match(self, key: str, data: bytes, etag: str) -> Optional[str]:
        from google.api_core.exceptions import PreconditionFailed
        b = self._b(key)
        try:
            b.upload_from_string(data, if_generation_match=int(etag))
            return str(b.generation)  # updated by the upload; no extra round-trip
        except PreconditionFailed:
            return None

    def delete_prefix(self, prefix: str = "") -> None:
        # trailing "/" bounds the match to this object (not sibling "foobar/...");
        # empty scope (both empty) = the whole bucket.
        stripped = f"{self.prefix}/{prefix}".strip("/")
        p = stripped + "/" if stripped else ""
        for blob in self.bucket.list_blobs(prefix=p):
            blob.delete()
