"""
Tests for profiling coverage across all execution paths.

Verifies that profiling data is recorded for:
- Local DataFrame execution (_execute)
- Remote SQL queries (_remote_sql)
- Metadata queries (_execute_metadata_query): databases(), tables(), describe
"""

import unittest
from unittest import mock

import pandas as pd

from datastore import DataStore
from datastore.config import (
    enable_profiling,
    disable_profiling,
    get_profiler,
    reset_profiler,
)


class TestLocalExecutionProfiling(unittest.TestCase):
    """Test profiling for local DataFrame execution path."""

    def setUp(self):
        reset_profiler()
        enable_profiling()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    def test_local_dataframe_execution_records_profiling_steps(self):
        """Local execution via _execute() should record Total Execution, Cache Check, etc."""
        ds = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds = ds[ds["a"] > 1]
        _ = len(ds)

        profiler = get_profiler()
        summary = profiler.summary()

        assert len(profiler.steps) > 0, "Profiler should have recorded steps"
        assert any(
            "Total Execution" in k for k in summary
        ), f"Expected 'Total Execution' step, got: {list(summary.keys())}"

    def test_local_sql_query_records_profiling_steps(self):
        """Local SQL via LazySQLQuery should record chDB query profiling."""
        ds = DataStore({"x": [10, 20, 30]})
        ds = ds.sql("SELECT * FROM __df__ WHERE x > 15")
        _ = len(ds)

        profiler = get_profiler()
        summary = profiler.summary()

        assert len(profiler.steps) > 0, "Profiler should have recorded steps"
        assert any(
            "LazySQLQuery" in k or "chDB" in k for k in summary
        ), f"Expected LazySQLQuery or chDB step, got: {list(summary.keys())}"

    def test_profiling_report_not_empty_after_execution(self):
        """profiler.report() should return actual data, not 'No profiling data recorded.'"""
        ds = DataStore({"val": [1, 2, 3]})
        _ = len(ds)

        profiler = get_profiler()
        report = profiler.report()

        assert report != "No profiling data recorded.", (
            "Report should contain data after execution"
        )
        assert "EXECUTION PROFILE" in report


class TestRemoteSQLProfiling(unittest.TestCase):
    """Test profiling for remote SQL execution path (_remote_sql)."""

    def setUp(self):
        reset_profiler()
        enable_profiling()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    @mock.patch("chdb.query")
    def test_remote_sql_records_profiling_step(self, mock_query):
        """_remote_sql() should record a 'Remote SQL Query' profiling step."""
        mock_query.return_value = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="testdb",
            user="default",
            password="",
        )
        ds.use("testdb")
        result = ds.sql("SELECT * FROM users")

        # .sql() on remote triggers _remote_sql() which calls chdb.query()
        mock_query.assert_called_once()

        profiler = get_profiler()
        summary = profiler.summary()

        assert len(profiler.steps) > 0, (
            "Profiler should have recorded steps for remote SQL"
        )
        assert any("Remote SQL Query" in k for k in summary), (
            f"Expected 'Remote SQL Query' step, got: {list(summary.keys())}"
        )

    @mock.patch("chdb.query")
    def test_remote_sql_profiling_has_sql_metadata(self, mock_query):
        """Remote SQL profiling step should include sql and time_ms metadata."""
        mock_query.return_value = pd.DataFrame({"x": [1]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="db",
            user="default",
            password="",
        )
        ds.use("db")
        _ = ds.sql("SELECT * FROM t")

        profiler = get_profiler()
        remote_steps = [
            s for s in profiler.steps if s.name == "Remote SQL Query"
        ]
        assert len(remote_steps) == 1, (
            f"Expected exactly 1 Remote SQL Query step, got {len(remote_steps)}"
        )

        step = remote_steps[0]
        assert "sql" in step.metadata, "Step should have 'sql' metadata"
        assert "time_ms" in step.metadata, "Step should have 'time_ms' metadata"


class TestMetadataQueryProfiling(unittest.TestCase):
    """Test profiling for metadata queries: databases(), tables(), _remote_describe()."""

    def setUp(self):
        reset_profiler()
        enable_profiling()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    @mock.patch("chdb.query")
    def test_databases_records_metadata_query_step(self, mock_query):
        """databases() should record a 'Metadata Query' profiling step."""
        mock_query.return_value = pd.DataFrame({"name": ["default", "system"]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            user="default",
            password="",
        )
        result = ds.databases()

        assert result == ["default", "system"]
        mock_query.assert_called_once()

        profiler = get_profiler()
        summary = profiler.summary()

        assert any("Metadata Query" in k for k in summary), (
            f"Expected 'Metadata Query' step, got: {list(summary.keys())}"
        )

    @mock.patch("chdb.query")
    def test_tables_records_metadata_query_step(self, mock_query):
        """tables() should record a 'Metadata Query' profiling step."""
        mock_query.return_value = pd.DataFrame({"name": ["users", "orders"]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="mydb",
            user="default",
            password="",
        )
        ds.use("mydb")
        result = ds.tables()

        assert result == ["users", "orders"]

        profiler = get_profiler()
        summary = profiler.summary()

        assert any("Metadata Query" in k for k in summary), (
            f"Expected 'Metadata Query' step, got: {list(summary.keys())}"
        )

    @mock.patch("chdb.query")
    def test_metadata_query_profiling_has_sql_metadata(self, mock_query):
        """Metadata query profiling step should include sql and time_ms metadata."""
        mock_query.return_value = pd.DataFrame({"name": ["db1"]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            user="default",
            password="",
        )
        ds.databases()

        profiler = get_profiler()
        meta_steps = [s for s in profiler.steps if s.name == "Metadata Query"]
        assert len(meta_steps) == 1

        step = meta_steps[0]
        assert "sql" in step.metadata, "Step should have 'sql' metadata"
        assert "time_ms" in step.metadata, "Step should have 'time_ms' metadata"


class TestProfilingDisabled(unittest.TestCase):
    """Test that profiling correctly records nothing when disabled."""

    def setUp(self):
        reset_profiler()
        disable_profiling()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    def test_no_profiling_data_when_disabled(self):
        """When profiling is disabled, report should say 'No profiling data recorded.'"""
        ds = DataStore({"a": [1, 2, 3]})
        _ = len(ds)

        profiler = get_profiler()
        assert profiler.report() == "No profiling data recorded."

    @mock.patch("chdb.query")
    def test_remote_sql_no_profiling_when_disabled(self, mock_query):
        """Remote SQL should not record profiling when disabled."""
        mock_query.return_value = pd.DataFrame({"x": [1]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="db",
            user="default",
            password="",
        )
        ds.use("db")
        _ = ds.sql("SELECT 1")

        profiler = get_profiler()
        assert profiler.report() == "No profiling data recorded."


class TestProfilingEnableAfterExecution(unittest.TestCase):
    """Test the scenario from the user's bug report: enable_profiling() called but no data."""

    def setUp(self):
        reset_profiler()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    def test_enable_before_execution_records_data(self):
        """Calling enable_profiling() before execution should record profiling data."""
        enable_profiling()

        ds = DataStore({"a": [1, 2, 3]})
        _ = len(ds)

        profiler = get_profiler()
        assert profiler.report() != "No profiling data recorded.", (
            "Profiling should record data when enabled before execution"
        )

    @mock.patch("chdb.query")
    def test_remote_sql_enable_before_records_data(self, mock_query):
        """enable_profiling() before remote SQL should record 'Remote SQL Query'."""
        mock_query.return_value = pd.DataFrame({"id": [1]})

        enable_profiling()

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="db",
            user="default",
            password="",
        )
        ds.use("db")
        _ = ds.sql("SELECT * FROM t")

        profiler = get_profiler()
        summary = profiler.summary()
        assert any("Remote SQL Query" in k for k in summary), (
            f"Expected 'Remote SQL Query' in summary, got: {list(summary.keys())}"
        )


if __name__ == "__main__":
    unittest.main()
