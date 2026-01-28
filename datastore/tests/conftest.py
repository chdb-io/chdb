"""
Pytest configuration for datastore tests.

This conftest.py ensures proper test isolation by resetting shared state
between tests.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_global_executor():
    """
    Reset the global executor before and after each test.

    This ensures test isolation because chdb's :memory: database
    shares state across connections in the same process.
    """
    from datastore.executor import reset_executor

    # Reset before test
    reset_executor()

    yield

    # Reset after test
    reset_executor()
