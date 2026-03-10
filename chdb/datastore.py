"""Shim module: makes `from chdb import datastore` work.

When installed, this file lives at site-packages/chdb/datastore.py alongside
chdb-core's chdb/__init__.py.  On first import it patches chdb.__version__
to the *chdb* pip-package version (instead of the chdb-core engine version)
and then replaces itself in sys.modules with the real top-level `datastore`
package so that all public API (DataStore, DataFrame, read_csv, …) is
available transparently.
"""

import sys
import importlib.metadata

import datastore as _datastore_pkg  # the real top-level package

# --- patch chdb.__version__ to the chdb pip-package version ---------------
_chdb_mod = sys.modules.get("chdb")
if _chdb_mod is not None and not hasattr(_chdb_mod, "core_version"):
    try:
        _chdb_mod.core_version = importlib.metadata.version("chdb-core")
        _chdb_mod.__version__ = importlib.metadata.version("chdb")
        _chdb_mod.chdb_version = tuple(_chdb_mod.__version__.split("."))
    except Exception:
        pass

# --- replace this shim with the real datastore package --------------------
sys.modules[__name__] = _datastore_pkg
