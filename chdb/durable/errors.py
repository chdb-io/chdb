"""Exceptions for chdb.durable."""
from __future__ import annotations


class DurableError(RuntimeError):
    """Base class for all chdb.durable errors."""


class LeaseError(DurableError):
    """The single-writer lease could not be acquired or was lost."""


class MissingDependency(DurableError):
    """A backend extra (boto3 / google-cloud-storage / azure-storage-blob) is not installed."""
