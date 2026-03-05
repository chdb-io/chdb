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


class TestLocalExecutionProfilingSteps(unittest.TestCase):
    """Test profiling records steps for local DataFrame execution path."""

    def setUp(self):
        reset_profiler()
        enable_profiling()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    def test_filter_execution_records_total_execution_and_cache_check(self):
        """Filter + len() should produce Total Execution, Cache Check, Query Planning steps."""
        ds = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds = ds[ds["a"] > 1]
        len(ds)

        profiler = get_profiler()
        summary = profiler.summary()

        assert len(profiler.steps) > 0, "Profiler should have recorded steps"
        assert "Total Execution" in summary, (
            f"Expected 'Total Execution' as top-level step, got: {list(summary.keys())}"
        )
        assert any("Cache Check" in k for k in summary), (
            f"Expected 'Cache Check' step, got: {list(summary.keys())}"
        )
        assert any("Query Planning" in k for k in summary), (
            f"Expected 'Query Planning' step, got: {list(summary.keys())}"
        )

    def test_sql_query_records_lazy_sql_and_chdb_steps(self):
        """Local SQL via LazySQLQuery should produce LazySQLQuery and chDB DataFrame Query steps."""
        ds = DataStore({"x": [10, 20, 30]})
        ds = ds.sql("SELECT * FROM __df__ WHERE x > 15")
        len(ds)

        profiler = get_profiler()
        summary = profiler.summary()

        assert any("LazySQLQuery" in k for k in summary), (
            f"Expected 'LazySQLQuery' step, got: {list(summary.keys())}"
        )
        assert any("chDB" in k for k in summary), (
            f"Expected a chDB execution step, got: {list(summary.keys())}"
        )

    def test_report_contains_execution_profile_header_after_execution(self):
        """After execution, profiler.report() should contain EXECUTION PROFILE header and timing."""
        ds = DataStore({"val": [1, 2, 3]})
        len(ds)

        profiler = get_profiler()
        report = profiler.report()

        assert "EXECUTION PROFILE" in report, (
            f"Report should contain 'EXECUTION PROFILE' header, got:\n{report}"
        )
        assert "Total Execution" in report, (
            f"Report should contain 'Total Execution' step, got:\n{report}"
        )
        assert "ms" in report, (
            f"Report should contain timing in ms, got:\n{report}"
        )


class TestRemoteSQLProfilingSteps(unittest.TestCase):
    """Test profiling records 'Remote SQL Query' step with metadata for _remote_sql path."""

    def setUp(self):
        reset_profiler()
        enable_profiling()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    @mock.patch("chdb.query")
    def test_session_sql_records_remote_sql_query_step(self, mock_query):
        """session.sql() on remote connection should record exactly one 'Remote SQL Query' step."""
        mock_query.return_value = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="testdb",
            user="default",
            password="",
        )
        ds.use("testdb")
        ds.sql("SELECT * FROM users")

        mock_query.assert_called_once()

        profiler = get_profiler()
        remote_steps = [s for s in profiler.steps if s.name == "Remote SQL Query"]
        assert len(remote_steps) == 1, (
            f"Expected exactly 1 'Remote SQL Query' step, got {len(remote_steps)}: "
            f"{[s.name for s in profiler.steps]}"
        )

    @mock.patch("chdb.query")
    def test_remote_sql_step_has_sql_preview_and_valid_timing(self, mock_query):
        """Remote SQL Query step metadata should contain sql preview and non-negative time_ms."""
        mock_query.return_value = pd.DataFrame({"x": [1]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="db",
            user="default",
            password="",
        )
        ds.use("db")
        ds.sql("SELECT * FROM t")

        profiler = get_profiler()
        step = next(s for s in profiler.steps if s.name == "Remote SQL Query")

        assert "sql" in step.metadata, (
            f"Step metadata should have 'sql' key, got: {step.metadata}"
        )
        assert "time_ms" in step.metadata, (
            f"Step metadata should have 'time_ms' key, got: {step.metadata}"
        )
        assert float(step.metadata["time_ms"]) >= 0, (
            f"time_ms should be non-negative, got: {step.metadata['time_ms']}"
        )
        assert len(step.metadata["sql"]) > 0, (
            "sql preview should not be empty"
        )


class TestMetadataQueryProfilingSteps(unittest.TestCase):
    """Test profiling records 'Metadata Query' step for databases(), tables(), describe()."""

    def setUp(self):
        reset_profiler()
        enable_profiling()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    @mock.patch("chdb.query")
    def test_databases_records_metadata_query_step_and_returns_names(self, mock_query):
        """databases() should record 'Metadata Query' step and return correct database list."""
        mock_query.return_value = pd.DataFrame({"name": ["default", "system"]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            user="default",
            password="",
        )
        result = ds.databases()

        assert result == ["default", "system"], (
            f"Expected ['default', 'system'], got: {result}"
        )
        mock_query.assert_called_once()

        profiler = get_profiler()
        meta_steps = [s for s in profiler.steps if s.name == "Metadata Query"]
        assert len(meta_steps) == 1, (
            f"Expected exactly 1 'Metadata Query' step, got {len(meta_steps)}: "
            f"{[s.name for s in profiler.steps]}"
        )

    @mock.patch("chdb.query")
    def test_tables_records_metadata_query_step_and_returns_names(self, mock_query):
        """tables() should record 'Metadata Query' step and return correct table list."""
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

        assert result == ["users", "orders"], (
            f"Expected ['users', 'orders'], got: {result}"
        )

        profiler = get_profiler()
        meta_steps = [s for s in profiler.steps if s.name == "Metadata Query"]
        assert len(meta_steps) == 1, (
            f"Expected exactly 1 'Metadata Query' step, got {len(meta_steps)}"
        )

    @mock.patch("chdb.query")
    def test_describe_records_metadata_query_step(self, mock_query):
        """_remote_describe() should record 'Metadata Query' step."""
        mock_query.return_value = pd.DataFrame({
            "name": ["id", "value"],
            "type": ["UInt64", "String"],
        })

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="mydb",
            user="default",
            password="",
        )
        ds.use("mydb")
        ds._remote_describe(database="mydb", table="test_table")

        profiler = get_profiler()
        meta_steps = [s for s in profiler.steps if s.name == "Metadata Query"]
        assert len(meta_steps) == 1, (
            f"Expected exactly 1 'Metadata Query' step for describe, got {len(meta_steps)}: "
            f"{[s.name for s in profiler.steps]}"
        )

    @mock.patch("chdb.query")
    def test_metadata_step_has_sql_preview_and_valid_timing(self, mock_query):
        """Metadata Query step metadata should contain sql preview and non-negative time_ms."""
        mock_query.return_value = pd.DataFrame({"name": ["db1"]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            user="default",
            password="",
        )
        ds.databases()

        profiler = get_profiler()
        step = next(s for s in profiler.steps if s.name == "Metadata Query")

        assert "sql" in step.metadata, (
            f"Step metadata should have 'sql' key, got: {step.metadata}"
        )
        assert "time_ms" in step.metadata, (
            f"Step metadata should have 'time_ms' key, got: {step.metadata}"
        )
        assert float(step.metadata["time_ms"]) >= 0, (
            f"time_ms should be non-negative, got: {step.metadata['time_ms']}"
        )
        assert "SELECT" in step.metadata["sql"].upper(), (
            f"sql preview should contain a SELECT statement, got: {step.metadata['sql']}"
        )


class TestProfilingDisabledRecordsNothing(unittest.TestCase):
    """Test that profiling correctly records nothing when disabled."""

    def setUp(self):
        reset_profiler()
        disable_profiling()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    def test_local_execution_no_profiling_when_disabled(self):
        """When profiling is disabled, local execution should produce empty report."""
        ds = DataStore({"a": [1, 2, 3]})
        len(ds)

        profiler = get_profiler()
        assert profiler.report() == "No profiling data recorded.", (
            f"Expected empty report when profiling disabled, got:\n{profiler.report()}"
        )
        assert len(profiler.steps) == 0, (
            f"Expected 0 steps when profiling disabled, got {len(profiler.steps)}"
        )

    @mock.patch("chdb.query")
    def test_remote_sql_no_profiling_when_disabled(self, mock_query):
        """When profiling is disabled, remote SQL should produce empty report."""
        mock_query.return_value = pd.DataFrame({"x": [1]})

        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="db",
            user="default",
            password="",
        )
        ds.use("db")
        ds.sql("SELECT 1")

        profiler = get_profiler()
        assert profiler.report() == "No profiling data recorded.", (
            f"Expected empty report when profiling disabled, got:\n{profiler.report()}"
        )
        assert len(profiler.steps) == 0, (
            f"Expected 0 steps when profiling disabled, got {len(profiler.steps)}"
        )


class TestProfilingEnableBeforeRemoteSQLRecordsData(unittest.TestCase):
    """Reproduce the user's bug: enable_profiling() before session.sql() should record data."""

    def setUp(self):
        reset_profiler()

    def tearDown(self):
        disable_profiling()
        reset_profiler()

    def test_local_enable_then_execute_records_total_execution(self):
        """enable_profiling() then local execution should record Total Execution step."""
        enable_profiling()

        ds = DataStore({"a": [1, 2, 3]})
        len(ds)

        profiler = get_profiler()
        assert "Total Execution" in profiler.summary(), (
            f"Expected 'Total Execution' after enable + execute, got: "
            f"{list(profiler.summary().keys())}"
        )

    @mock.patch("chdb.query")
    def test_remote_enable_then_sql_records_remote_sql_query(self, mock_query):
        """enable_profiling() then session.sql() should record 'Remote SQL Query' step."""
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
        ds.sql("SELECT * FROM t")

        profiler = get_profiler()
        remote_steps = [s for s in profiler.steps if s.name == "Remote SQL Query"]
        assert len(remote_steps) == 1, (
            f"Expected exactly 1 'Remote SQL Query' step after enable + remote sql, "
            f"got {len(remote_steps)}: {[s.name for s in profiler.steps]}"
        )


if __name__ == "__main__":
    unittest.main()
