"""Python adapter for loading chDB through ADBC."""

from . import dbapi
from .dbapi import connect, driver_path

__all__ = ["connect", "dbapi", "driver_path"]
