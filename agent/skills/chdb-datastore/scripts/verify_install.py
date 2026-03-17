#!/usr/bin/env python3
"""Verify chdb DataStore installation and basic functionality."""

import sys

PASS = "OK"
FAIL = "FAIL"
results = []


def check(name, fn):
    try:
        fn()
        results.append((name, PASS, ""))
        print(f"  [{PASS}] {name}")
    except Exception as e:
        results.append((name, FAIL, str(e)))
        print(f"  [{FAIL}] {name}: {e}")


def check_python_version():
    assert sys.version_info >= (3, 9), f"Python 3.9+ required, got {sys.version}"


def check_chdb_import():
    import chdb
    assert hasattr(chdb, "__version__"), "chdb imported but missing __version__"
    print(f"         chdb version: {chdb.__version__}")


def check_datastore_import_from_datastore():
    from datastore import DataStore
    assert DataStore is not None


def check_datastore_import_from_chdb():
    from chdb.datastore import DataStore
    assert DataStore is not None


def check_datastore_as_pd():
    import chdb.datastore as pd
    assert hasattr(pd, "DataStore")


def check_basic_operations():
    from datastore import DataStore

    ds = DataStore({"name": ["Alice", "Bob", "Carol"], "age": [25, 30, 35]})
    filtered = ds[ds["age"] > 25]
    assert len(filtered) == 2, f"Expected 2 rows, got {len(filtered)}"


def check_sort():
    from datastore import DataStore

    ds = DataStore({"name": ["Charlie", "Alice", "Bob"], "value": [3, 1, 2]})
    sorted_ds = ds.sort_values("value")
    cols = sorted_ds.columns
    assert "name" in cols and "value" in cols, f"Unexpected columns: {cols}"


def check_groupby():
    from datastore import DataStore

    ds = DataStore({
        "dept": ["Eng", "Sales", "Eng", "Sales"],
        "salary": [100, 80, 120, 90],
    })
    result = ds.groupby("dept")["salary"].mean()
    assert len(result) == 2, f"Expected 2 groups, got {len(result)}"


if __name__ == "__main__":
    print("chdb DataStore Installation Verification")
    print("=" * 45)

    check("Python version >= 3.9", check_python_version)
    check("import chdb", check_chdb_import)
    check("from datastore import DataStore", check_datastore_import_from_datastore)
    check("from chdb.datastore import DataStore", check_datastore_import_from_chdb)
    check("import chdb.datastore as pd", check_datastore_as_pd)
    check("Basic filter operation", check_basic_operations)
    check("Sort operation", check_sort)
    check("GroupBy aggregation", check_groupby)

    print()
    print("=" * 45)
    passed = sum(1 for _, s, _ in results if s == PASS)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("\nFailed checks:")
        for name, status, err in results:
            if status == FAIL:
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("All checks passed!")
