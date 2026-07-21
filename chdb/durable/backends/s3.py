"""S3-compatible backend: AWS S3, MinIO, Cloudflare R2, Backblaze, Tigris,
and GCS via its S3-interoperability endpoint — one code path, `endpoint_url`
picks the provider. Lease primitive = conditional PUT (`IfNoneMatch` / `IfMatch`)."""
from __future__ import annotations

from typing import Optional

from ..errors import MissingDependency


class S3Backend:
    def __init__(self, bucket: str, prefix: str, *, endpoint_url=None,
                 access_key=None, secret_key=None, region="us-east-1"):
        try:
            import boto3
        except ImportError as e:
            raise MissingDependency("s3 backend needs boto3: pip install 'chdb[durable]'") from e
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client(
            "s3", endpoint_url=endpoint_url, region_name=region,
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
        )

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key: str) -> Optional[bytes]:
        return self.get_with_etag(key)[0]

    def get_with_etag(self, key: str):
        import botocore
        try:
            r = self.s3.get_object(Bucket=self.bucket, Key=self._k(key))
            return (r["Body"].read(), r["ETag"])  # bytes + etag in one round-trip
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NoSuchBucket"):
                return (None, None)
            raise

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)

    def put_if_absent(self, key: str, data: bytes) -> Optional[str]:
        import botocore
        try:
            r = self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data,
                                   IfNoneMatch="*")
            return r["ETag"]  # etag of the object we just created
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("PreconditionFailed", "412"):
                return None
            raise

    def head_etag(self, key: str) -> Optional[str]:
        import botocore
        try:
            return self.s3.head_object(Bucket=self.bucket, Key=self._k(key))["ETag"]
        except botocore.exceptions.ClientError as e:
            # only "absent" maps to None — transient errors (throttling, 5xx,
            # access denied) must propagate, not masquerade as a missing key.
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NoSuchBucket"):
                return None
            raise

    def replace_if_match(self, key: str, data: bytes, etag: str) -> Optional[str]:
        import botocore
        try:
            r = self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data,
                                   IfMatch=etag)
            return r["ETag"]
        except botocore.exceptions.ClientError as e:
            # precondition failed, or the target is gone (deleted) — either way
            # the compare-and-set did not match.
            if e.response["Error"]["Code"] in ("PreconditionFailed", "412",
                                               "NoSuchKey", "404"):
                return None
            raise

    def delete_prefix(self, prefix: str = "") -> None:
        # trailing "/" bounds the match to this object — without it, destroying
        # "foo" would also delete sibling keys under "foobar/...". An empty
        # scope (prefix and self.prefix both empty) means the whole bucket.
        stripped = self._k(prefix).strip("/")
        p = stripped + "/" if stripped else ""
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=p):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                resp = self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})
                errs = resp.get("Errors") or []
                if errs:  # S3 reports per-key failures here even on a 200
                    raise RuntimeError(
                        f"delete_prefix: {len(errs)} object(s) not deleted, e.g. {errs[0]}")
