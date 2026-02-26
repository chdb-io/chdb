import logging
import os
import importlib

import pytest

from datastore import DataStore, config


DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def dataset_path(filename: str) -> str:
    return os.path.join(DATASET_DIR, filename)


def test_explain_reports_lazy_ops_in_order():
    users = DataStore.from_file(dataset_path("users.csv"))

    users = users.select("name", "age").filter(users.age > 30)
    users["age_plus_1"] = users["age"] + 1
    users = users[["name", "age_plus_1"]]

    output = users.explain()

    assert "SELECT:" in output
    assert "WHERE:" in output
    assert "Assign column 'age_plus_1'" in output
    assert "Select columns: name, age_plus_1" in output

    select_idx = output.index("SELECT:")
    filter_idx = output.index("WHERE:")
    assign_idx = output.index("Assign column 'age_plus_1'")
    selection_idx = output.index("Select columns: name, age_plus_1")

    assert select_idx < filter_idx < assign_idx < selection_idx


def test_execution_logs_follow_lazy_ops(caplog):
    old_level = config.log_level
    old_format = config.log_format

    try:
        config.enable_debug()
        config.set_log_format("simple")

        users = DataStore.from_file(dataset_path("users.csv"))
        users = users.select("name", "age").filter(users.age > 30)
        users["age_plus_1"] = users["age"] + 1

        with caplog.at_level(logging.DEBUG, logger="datastore"):
            df = users.to_df()

        assert not df.empty

        log_text = caplog.text

        assert "Starting execution" in log_text
        assert "Lazy operations chain (3 operations)" in log_text
        # Segmented execution logs show segment info instead of "Executing initial SQL query"
        assert "Segment 1/" in log_text  # First SQL segment
        assert "chDB (from source)" in log_text  # SQL segment from source
        # ColumnAssignment for simple arithmetic (age + 1) is pushed to SQL now
        # So we check that it's included in the SQL query instead of executed separately
        assert "age_plus_1" in log_text  # Column appears in execution output
        assert "[chDB]" in log_text  # Unified logging prefix
        assert "SELECT" in log_text
        assert "[chDB] Result:" in log_text  # Unified result logging
    finally:
        config.set_log_format(old_format)
        config.set_log_level(old_level)


def test_execution_logs_mixed_sql_and_pandas(caplog):
    old_level = config.log_level
    old_format = config.log_format

    try:
        config.enable_debug()
        config.set_log_format("simple")

        users = DataStore.from_file(dataset_path("users.csv"))
        users = users.select("name", "age").filter(users.age > 25).sort("age", ascending=False)
        users["age_plus_1"] = users["age"] + 1
        users = users.add_prefix("p_")
        users = users.filter(users["p_age_plus_1"] > 30)
        users = users.limit(3)

        with caplog.at_level(logging.DEBUG, logger="datastore"):
            df = users.to_df()

        assert len(df) <= 3
        assert "p_age_plus_1" in df.columns

        log_text = caplog.text
        assert "Starting execution" in log_text
        # Segmented execution logs show segment info instead of "Executing initial SQL query"
        assert "Segment 1/" in log_text  # First SQL segment
        assert "chDB (from source)" in log_text  # SQL segment from source
        # ColumnAssignment for simple arithmetic can be pushed to SQL
        # AddPrefix is pandas-only, so it should be executed in pandas segment
        assert "[Pandas] Executing AddPrefix" in log_text
        # ORDER BY and LIMIT are now in SQL segments (can be from source or on DataFrame)
        assert "ORDER BY: age DESC" in log_text or 'ORDER BY "age" DESC' in log_text or "df.sort_values" in log_text
        assert "LIMIT: 3" in log_text or "LIMIT 3" in log_text or "df.head(3)" in log_text
        assert "Execution complete" in log_text
    finally:
        config.set_log_format(old_format)
        config.set_log_level(old_level)


def test_verbose_log_format_outputs_timestamp(caplog):
    old_level = config.log_level
    old_format = config.log_format

    try:
        config.enable_debug()
        config.set_log_format("verbose")

        users = DataStore.from_file(dataset_path("users.csv")).select("name").limit(1)

        with caplog.at_level(logging.DEBUG, logger="datastore"):
            users.to_df()

        config_module = importlib.import_module("datastore.config")
        logger = config_module.get_logger()
        format_strings = [h.formatter._fmt for h in logger.handlers if h.formatter]
        assert any("%(asctime)s" in fmt for fmt in format_strings), "verbose formatter not applied"
        assert any("%(name)s" in fmt for fmt in format_strings), "logger name missing in formatter"
    finally:
        config.set_log_format(old_format)
        config.set_log_level(old_level)


def test_execution_logs_sql_join_and_pandas(caplog):
    old_level = config.log_level
    old_format = config.log_format

    try:
        config.enable_debug()
        config.set_log_format("simple")

        # Don't select before join - select after join to include columns from both tables
        users = DataStore.from_file(dataset_path("users.csv"))
        orders = DataStore.from_file(dataset_path("orders.csv"))

        joined = users.join(orders, on=users.user_id == orders.user_id)
        joined = joined.select("user_id", "age", "amount")  # select after join
        joined = joined.filter(joined.age > 20)
        joined["amount_plus_1"] = joined["amount"] + 1
        joined = joined.add_suffix("_s")

        with caplog.at_level(logging.DEBUG, logger="datastore"):
            df = joined.to_df()

        assert not df.empty
        assert "amount_plus_1_s" in df.columns

        log_text = caplog.text
        assert "JOIN" in log_text
        assert "[chDB]" in log_text  # Unified logging prefix
        # ColumnAssignment for simple arithmetic can be pushed to SQL
        # AddSuffix is pandas-only, so it should be executed in pandas segment
        assert "[Pandas] Executing AddSuffix" in log_text
        assert "Execution complete" in log_text
    finally:
        config.set_log_format(old_format)
        config.set_log_level(old_level)


def test_explain_includes_join_and_pandas_ops():
    users = DataStore.from_file(dataset_path("users.csv"))
    orders = DataStore.from_file(dataset_path("orders.csv"))

    ds = users.join(orders, on=users.user_id == orders.user_id)
    ds = ds.select("user_id", "age", "amount")  # select after join
    ds = ds.filter(ds.age > 25).add_prefix("u_")

    output = ds.explain()

    assert "JOIN" in output  # SQL JOIN in the generated query
    assert "Add prefix" in output
