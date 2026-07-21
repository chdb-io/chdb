"""Native Azure Blob backend. Lease primitive = `overwrite=False` create
(ResourceExists), Azure's single-writer primitive; no S3 API needed."""
from __future__ import annotations

from typing import Optional

from ..errors import MissingDependency


class AzureBlobBackend:
    def __init__(self, container: str, prefix: str, *, conn_str: str):
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as e:
            raise MissingDependency(
                "azure backend needs azure-storage-blob: pip install 'chdb[durable-azure]'") from e
        self.prefix = prefix.strip("/")
        self.cc = ContainerClient.from_connection_string(conn_str, container)
        try:
            self.cc.create_container()
        except Exception:
            pass

    def _n(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key: str) -> Optional[bytes]:
        return self.get_with_etag(key)[0]

    def get_with_etag(self, key: str):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            d = self.cc.download_blob(self._n(key))
            return (d.readall(), d.properties.etag)  # bytes + etag in one round-trip
        except ResourceNotFoundError:
            return (None, None)

    def put(self, key: str, data: bytes) -> None:
        self.cc.upload_blob(self._n(key), data, overwrite=True)

    def put_if_absent(self, key: str, data: bytes) -> Optional[str]:
        from azure.core.exceptions import ResourceExistsError
        try:
            r = self.cc.upload_blob(self._n(key), data, overwrite=False)
            return r.get("etag")  # etag of the blob we just created
        except ResourceExistsError:
            return None

    def head_etag(self, key: str) -> Optional[str]:
        from azure.core.exceptions import ResourceNotFoundError
        try:
            return self.cc.get_blob_client(self._n(key)).get_blob_properties().etag
        except ResourceNotFoundError:
            return None

    def replace_if_match(self, key: str, data: bytes, etag: str) -> Optional[str]:
        from azure.core import MatchConditions
        from azure.core.exceptions import ResourceModifiedError, ResourceNotFoundError
        try:
            r = self.cc.upload_blob(self._n(key), data, overwrite=True,
                                    etag=etag, match_condition=MatchConditions.IfNotModified)
            return r.get("etag")  # returned by the upload; no extra round-trip
        except (ResourceModifiedError, ResourceNotFoundError):
            return None  # etag mismatch or target gone — CAS did not match

    def delete_prefix(self, prefix: str = "") -> None:
        # trailing "/" bounds the match to this object (not sibling "foobar/...");
        # empty scope (both empty) = the whole container.
        stripped = f"{self.prefix}/{prefix}".strip("/")
        p = stripped + "/" if stripped else ""
        for b in self.cc.list_blobs(name_starts_with=p):
            self.cc.delete_blob(b.name)
