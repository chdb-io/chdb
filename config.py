"""
Configuration module for DataStore.

This module provides a centralized configuration mechanism for DataStore,
including logging level configuration.
"""

import logging
from typing import Optional

# Module-level logger for DataStore
_logger: Optional[logging.Logger] = None
_log_level: int = logging.WARNING
_log_format: str = "simple"  # "simple" or "verbose"

# DataStore's own logger name
LOGGER_NAME = "datastore"

# Log format templates
LOG_FORMATS = {
    "simple": "%(levelname).1s %(message)s",  # e.g., "D [6/7] Executing: ..."
    "verbose": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}


def _get_formatter() -> logging.Formatter:
    """Get formatter based on current format setting."""
    fmt = LOG_FORMATS.get(_log_format, LOG_FORMATS["simple"])
    if _log_format == "verbose":
        return logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    return logging.Formatter(fmt)


def get_logger() -> logging.Logger:
    """
    Get the DataStore logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger(LOGGER_NAME)
        _logger.setLevel(_log_level)

        # Add handler if none exists
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(_log_level)
            handler.setFormatter(_get_formatter())
            _logger.addHandler(handler)

    return _logger


def set_log_level(level: int) -> None:
    """
    Set the logging level for DataStore.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)

    Example:
        >>> import logging
        >>> from datastore import config
        >>> config.set_log_level(logging.DEBUG)  # Enable debug logging
        >>> config.set_log_level(logging.WARNING)  # Only warnings and errors
    """
    global _log_level

    _log_level = level

    if _logger is not None:
        _logger.setLevel(level)
        for handler in _logger.handlers:
            handler.setLevel(level)


def enable_debug() -> None:
    """
    Enable debug logging (shortcut for set_log_level(logging.DEBUG)).

    Example:
        >>> from datastore import config
        >>> config.enable_debug()
    """
    set_log_level(logging.DEBUG)


def disable_debug() -> None:
    """
    Disable debug logging (set to WARNING level).

    Example:
        >>> from datastore import config
        >>> config.disable_debug()
    """
    set_log_level(logging.WARNING)


def set_log_format(format_name: str) -> None:
    """
    Set the log output format.

    Args:
        format_name: "simple" (default) for minimal output, "verbose" for full timestamp/level

    Example:
        >>> from datastore import config
        >>> config.set_log_format("simple")   # "D [6/7] Executing: ..."
        >>> config.set_log_format("verbose")  # "2025-12-08 12:51:27 - datastore - DEBUG - ..."
    """
    global _log_format

    if format_name not in LOG_FORMATS:
        raise ValueError(f"Unknown format: {format_name}. Use 'simple' or 'verbose'")

    _log_format = format_name

    if _logger is not None:
        for handler in _logger.handlers:
            handler.setFormatter(_get_formatter())


# =============================================================================
# EXECUTION ENGINE CONFIGURATION
# =============================================================================


class ExecutionEngine:
    """Execution engine options."""

    AUTO = "auto"  # Auto-select best engine
    CLICKHOUSE = "clickhouse"  # Force ClickHouse/chDB
    PANDAS = "pandas"  # Force Pandas


_execution_engine: str = ExecutionEngine.AUTO
_prefer_pandas_for_simple: bool = True  # Prefer pandas for simple ops when AUTO


def get_execution_engine() -> str:
    """Get current execution engine setting."""
    return _execution_engine


def set_execution_engine(engine: str) -> None:
    """
    Set the execution engine.

    Args:
        engine: One of 'auto', 'clickhouse', 'pandas'

    Example:
        >>> from datastore import config
        >>> config.set_execution_engine('pandas')  # Force pandas
        >>> config.set_execution_engine('clickhouse')  # Force ClickHouse
        >>> config.set_execution_engine('auto')  # Auto-select
    """
    global _execution_engine
    valid = {ExecutionEngine.AUTO, ExecutionEngine.CLICKHOUSE, ExecutionEngine.PANDAS}
    if engine not in valid:
        raise ValueError(f"Invalid engine: {engine}. Use one of {valid}")
    _execution_engine = engine

    # Sync with function_config
    from .function_executor import function_config

    if engine == ExecutionEngine.PANDAS:
        function_config.prefer_pandas()
    elif engine == ExecutionEngine.CLICKHOUSE:
        function_config.prefer_chdb()
    else:
        function_config.reset()  # Auto mode uses default (chDB)


def use_pandas() -> None:
    """Force Pandas execution engine."""
    set_execution_engine(ExecutionEngine.PANDAS)


def use_clickhouse() -> None:
    """Force ClickHouse execution engine."""
    set_execution_engine(ExecutionEngine.CLICKHOUSE)


def use_auto() -> None:
    """Use auto-selection for execution engine."""
    set_execution_engine(ExecutionEngine.AUTO)


class DataStoreConfig:
    """
    Configuration class for DataStore.

    This class holds global configuration settings for DataStore instances.

    Example:
        >>> from datastore import DataStore
        >>> import logging
        >>>
        >>> # Enable debug logging
        >>> DataStore.config.log_level = logging.DEBUG
        >>>
        >>> # Or use the convenience method
        >>> DataStore.config.enable_debug()
        >>>
        >>> # Set execution engine
        >>> DataStore.config.execution_engine = 'pandas'
    """

    @property
    def log_level(self) -> int:
        """Get current log level."""
        return _log_level

    @log_level.setter
    def log_level(self, level: int) -> None:
        """Set log level."""
        set_log_level(level)

    def enable_debug(self) -> None:
        """Enable debug logging."""
        enable_debug()

    def disable_debug(self) -> None:
        """Disable debug logging."""
        disable_debug()

    def set_log_level(self, level: int) -> None:
        """Set log level."""
        set_log_level(level)

    @property
    def log_format(self) -> str:
        """Get current log format."""
        return _log_format

    @log_format.setter
    def log_format(self, format_name: str) -> None:
        """Set log format."""
        set_log_format(format_name)

    def set_log_format(self, format_name: str) -> None:
        """Set log format ('simple' or 'verbose')."""
        set_log_format(format_name)

    # ========== Execution Engine ==========

    @property
    def execution_engine(self) -> str:
        """Get current execution engine."""
        return get_execution_engine()

    @execution_engine.setter
    def execution_engine(self, engine: str) -> None:
        """Set execution engine ('auto', 'clickhouse', 'pandas')."""
        set_execution_engine(engine)

    def use_pandas(self) -> None:
        """Force Pandas execution."""
        use_pandas()

    def use_clickhouse(self) -> None:
        """Force ClickHouse execution."""
        use_clickhouse()

    def use_auto(self) -> None:
        """Use auto-selection."""
        use_auto()


# Global config instance
config = DataStoreConfig()
