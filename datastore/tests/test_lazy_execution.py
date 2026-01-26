"""
Test lazy execution functionality.
"""

import pytest
import os
from datastore import DataStore as ds


# Helper to get dataset path
def get_dataset_path(filename):
    """Get path to dataset file."""
    return os.path.join(os.path.dirname(__file__), 'dataset', filename)


def test_lazy_column_assignment():
    """Test that column assignment is lazy."""
    # Use local file instead of URL
    users = ds.from_file(get_dataset_path('users.csv'))

    # Record operation (should be fast, no execution yet)
    users["age_minus_1"] = users["age"] - 1

    # Check that operation was recorded
    assert len(users._lazy_ops) == 1
    assert users._lazy_ops[0].__class__.__name__ == "LazyColumnAssignment"

    # Now trigger execution
    df = users.to_df()

    # Check results
    assert "age_minus_1" in df.columns
    assert (df["age_minus_1"] == df["age"] - 1).all()


def test_multiple_lazy_operations():
    """Test multiple lazy operations."""
    users = ds.from_file(get_dataset_path('users.csv'))

    # Multiple operations
    users["age_plus_1"] = users["age"] + 1
    users["age_squared"] = users["age"] ** 2
    users["age_doubled"] = users["age"] * 2

    # Check operations recorded
    assert len(users._lazy_ops) == 3

    # Execute
    df = users.to_df()

    # Verify
    assert "age_plus_1" in df.columns
    assert "age_squared" in df.columns
    assert "age_doubled" in df.columns


def test_lazy_with_sql_operations():
    """Test mixing SQL and lazy operations."""
    users = ds.from_file(get_dataset_path('users.csv'))

    # SQL filter and select FIRST (before creating lazy columns)
    users = users.filter(users["age"] < 50)
    users = users.select("name", "age")

    # Lazy column assignment AFTER SQL operations
    users["computed"] = users["age"] * 2

    # Execute
    df = users.to_df()

    # Verify
    assert len(df.columns) == 3  # name, age, computed
    assert "computed" in df.columns
    assert "name" in df.columns
    assert "age" in df.columns
    assert all(df["age"] < 50)
    assert (df["computed"] == df["age"] * 2).all()


def test_getitem_returns_field():
    """Test that ds['col'] returns Field for expression building."""
    users = ds.from_file(get_dataset_path('users.csv'))

    # This should return a ColumnExpr that wraps a Field
    field = users["age"]

    from datastore.expressions import Field
    from datastore.column_expr import ColumnExpr

    assert isinstance(field, ColumnExpr)
    assert isinstance(field._expr, Field)
    assert field._expr.name == "age"


def test_expression_building():
    """Test building complex expressions."""
    users = ds.from_file(get_dataset_path('users.csv'))

    # Build expression (should not execute)
    expr = users["age"] * 2 + 10

    # Assign to column (should not execute)
    users["complex"] = expr

    # Execute
    df = users.to_df()

    # Verify
    assert "complex" in df.columns
    assert (df["complex"] == df["age"] * 2 + 10).all()


def test_column_selection_lazy():
    """Test that column selection is lazy."""
    users = ds.from_file(get_dataset_path('users.csv'))

    # Add some columns
    users["computed1"] = users["age"] * 2
    users["computed2"] = users["age"] * 3

    # Select columns (lazy) - this should work correctly
    # Column selection operates on the result after lazy assignments
    users = users[["name", "computed1"]]

    # Check operations recorded
    assert len(users._lazy_ops) == 3  # 2 assignments + 1 selection

    # Execute
    df = users.to_df()

    # Verify - the column selection should have been applied
    assert set(df.columns) == {"name", "computed1"}
    assert "computed2" not in df.columns  # Should be excluded


def test_explain():
    """Test explain shows operations without executing."""
    users = ds.from_file(get_dataset_path('users.csv'))

    # Add operations
    users["computed"] = users["age"] * 2
    users = users.filter(users["age"] < 50)

    # Explain should not execute
    output = users.explain()

    # Check that explain worked
    assert "Execution Plan" in output
    assert "Assign column" in output or "SQL" in output


def test_repr_triggers_execution():
    """Test that repr triggers execution."""
    users = ds.from_file(get_dataset_path('users.csv'))
    users["computed"] = users["age"] * 2

    # repr should trigger execution
    result_str = repr(users)

    # Check result contains data
    assert "name" in result_str or "DataFrame" in result_str or "user_id" in result_str


def test_str_triggers_execution():
    """Test that str/print triggers execution."""
    users = ds.from_file(get_dataset_path('users.csv'))
    users["computed"] = users["age"] * 2

    # str should trigger execution
    result_str = str(users)

    # Check result contains data
    assert len(result_str) > 0


def test_lazy_operations_are_recorded():
    """Test that lazy operations are recorded correctly."""
    users = ds.from_file(get_dataset_path('users.csv'))

    # Execute once
    df1 = users.to_df()

    # Add operation - creates new lazy op
    users["new_col"] = users["age"] * 2

    # Execute again
    df2 = users.to_df()

    # Should have new column
    assert "new_col" in df2.columns
    assert "new_col" not in df1.columns


def test_complex_workflow():
    """Test a complex workflow with multiple operations."""
    users = ds.from_file(get_dataset_path('users.csv'))

    # SQL operations
    users = users.filter(users["age"] < 50)
    users = users.select("name", "age", "country")

    # Lazy operations
    users["doubled"] = users["age"] * 2
    users["squared"] = users["age"] ** 2
    users["sum"] = users["doubled"] + users["squared"]

    # More SQL
    users = users.sort("age", ascending=False)
    users = users.limit(5)

    # Execute
    df = users.to_df()

    # Verify
    assert len(df) <= 5
    assert "doubled" in df.columns
    assert "squared" in df.columns
    assert "sum" in df.columns
    assert (df["sum"] == df["doubled"] + df["squared"]).all()


if __name__ == "__main__":
    # Run tests manually
    print("Running lazy execution tests...")

    tests = [
        test_lazy_column_assignment,
        test_multiple_lazy_operations,
        test_lazy_with_sql_operations,
        test_getitem_returns_field,
        test_expression_building,
        test_column_selection_lazy,
        test_explain,
        test_repr_triggers_execution,
        test_str_triggers_execution,
        test_lazy_operations_are_recorded,
        test_complex_workflow,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
