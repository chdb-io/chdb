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
# CACHE CONFIGURATION
# =============================================================================

# Default cache settings
_cache_enabled: bool = True  # Caching enabled by default
_cache_ttl: float = 0.0  # TTL in seconds, 0 means no expiration


def is_cache_enabled() -> bool:
    """Check if caching is enabled."""
    return _cache_enabled


def get_cache_ttl() -> float:
    """Get cache TTL in seconds (0 = no expiration)."""
    return _cache_ttl


def enable_cache() -> None:
    """
    Enable automatic caching of executed results.

    When enabled, repeated calls to repr/str/display will use cached results
    instead of re-executing the entire pipeline.

    Example:
        >>> from datastore import config
        >>> config.enable_cache()
    """
    global _cache_enabled
    _cache_enabled = True


def disable_cache() -> None:
    """
    Disable automatic caching.

    When disabled, every access triggers fresh execution.

    Example:
        >>> from datastore import config
        >>> config.disable_cache()
    """
    global _cache_enabled
    _cache_enabled = False


def set_cache_ttl(ttl_seconds: float) -> None:
    """
    Set cache Time-To-Live in seconds.

    Args:
        ttl_seconds: TTL in seconds. Use 0 for no expiration (default).
                     The cache will be invalidated after this duration.

    Example:
        >>> from datastore import config
        >>> config.set_cache_ttl(60)  # Cache expires after 60 seconds
        >>> config.set_cache_ttl(0)   # No TTL (cache never expires based on time)
    """
    global _cache_ttl
    if ttl_seconds < 0:
        raise ValueError("TTL must be non-negative")
    _cache_ttl = ttl_seconds


# =============================================================================
# EXECUTION ENGINE CONFIGURATION
# =============================================================================


class ExecutionEngine:
    """Execution engine options."""

    AUTO = "auto"  # Auto-select best engine
    CHDB = "chdb"  # Force chDB
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
        engine: One of 'auto', 'chdb', 'pandas'

    Example:
        >>> from datastore import config
        >>> config.set_execution_engine('pandas')  # Force pandas
        >>> config.set_execution_engine('chdb')  # Force chDB
        >>> config.set_execution_engine('auto')  # Auto-select
    """
    global _execution_engine
    valid = {ExecutionEngine.AUTO, ExecutionEngine.CHDB, ExecutionEngine.PANDAS}
    if engine not in valid:
        raise ValueError(f"Invalid engine: {engine}. Use one of {valid}")
    _execution_engine = engine

    # Sync with function_config
    from .function_executor import function_config

    if engine == ExecutionEngine.PANDAS:
        function_config.prefer_pandas()
    elif engine == ExecutionEngine.CHDB:
        function_config.prefer_chdb()
    else:
        function_config.reset()  # Auto mode uses default (chDB)


def use_pandas() -> None:
    """Force Pandas execution engine."""
    set_execution_engine(ExecutionEngine.PANDAS)


def use_chdb() -> None:
    """Force chDB execution engine."""
    set_execution_engine(ExecutionEngine.CHDB)


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
        """Set execution engine ('auto', 'chdb', 'pandas')."""
        set_execution_engine(engine)

    def use_pandas(self) -> None:
        """Force Pandas execution."""
        use_pandas()

    def use_chdb(self) -> None:
        """Force chDB execution."""
        use_chdb()

    def use_auto(self) -> None:
        """Use auto-selection."""
        use_auto()

    # ========== Cache Configuration ==========

    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return is_cache_enabled()

    @cache_enabled.setter
    def cache_enabled(self, enabled: bool) -> None:
        """Enable or disable caching."""
        if enabled:
            enable_cache()
        else:
            disable_cache()

    @property
    def cache_ttl(self) -> float:
        """Get cache TTL in seconds."""
        return get_cache_ttl()

    @cache_ttl.setter
    def cache_ttl(self, ttl_seconds: float) -> None:
        """Set cache TTL in seconds."""
        set_cache_ttl(ttl_seconds)

    def enable_cache(self) -> None:
        """Enable automatic caching."""
        enable_cache()

    def disable_cache(self) -> None:
        """Disable automatic caching."""
        disable_cache()

    def set_cache_ttl(self, ttl_seconds: float) -> None:
        """Set cache TTL in seconds."""
        set_cache_ttl(ttl_seconds)


# Global config instance
config = DataStoreConfig()
