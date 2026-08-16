"""DB-API wrapper for the chDB ADBC driver.

The native ADBC driver is exported by the ``chdb-core`` extension module.
This module locates that library and delegates connection creation to the
standard ADBC Python driver manager.
"""

from __future__ import annotations

import os
from typing import Any, Optional

ENTRYPOINT = "chdb_adbc_init"


def driver_path() -> str:
    """Return the native chDB library path used by the ADBC driver manager."""

    override = os.environ.get("CHDB_LIB_PATH")
    if override:
        return override

    try:
        import chdb
    except ImportError as exc:
        raise ImportError(
            'chDB ADBC support requires chdb-core. Install it with `pip install "chdb[adbc]"`.'
        ) from exc

    path = getattr(getattr(chdb, "_chdb", None), "__file__", None)
    if not path:
        raise ImportError(
            "Could not locate the chdb-core native library for ADBC. "
            'Reinstall with `pip install --upgrade "chdb[adbc]"`.'
        )
    return path


def connect(uri: Optional[str] = "chdb://", **kwargs: Any):
    """Connect to chDB through the ADBC DB-API interface.

    ``uri`` defaults to an in-memory database. Keyword arguments are forwarded
    to ``adbc_driver_manager.dbapi.connect``.
    """

    try:
        from adbc_driver_manager import dbapi as manager_dbapi
    except ImportError as exc:
        raise ImportError(
            'chDB ADBC support requires adbc-driver-manager. Install it with `pip install "chdb[adbc]"`.'
        ) from exc

    db_kwargs = dict(kwargs.pop("db_kwargs", {}) or {})
    if uri is not None:
        db_kwargs.setdefault("uri", uri)

    if "driver" not in kwargs:
        kwargs["driver"] = driver_path()
        kwargs.setdefault("entrypoint", ENTRYPOINT)

    kwargs.setdefault("autocommit", True)
    kwargs["db_kwargs"] = db_kwargs

    return manager_dbapi.connect(**kwargs)
