"""
Configuration module for DataStore.

This module provides a centralized configuration mechanism for DataStore,
including logging level configuration.
"""

import logging
import time
from typing import Optional, List, Dict
from contextlib import contextmanager

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
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
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
_cross_datastore_engine: str = ExecutionEngine.AUTO  # Engine for cross-DataStore ops


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


# =============================================================================
# COMPATIBILITY MODE CONFIGURATION
# =============================================================================


class CompatMode:
    """Compatibility mode options.

    Controls whether DataStore reshapes output and execution for pandas
    compatibility, or optimizes purely for SQL performance.

    - PANDAS (default): Full pandas behavior compatibility (row order preservation,
      MultiIndex, set_index, dtype corrections, skipna wrappers, etc.)
    - PERFORMANCE: SQL-first execution with no pandas compatibility overhead.
      Disables row-order preservation, stable sort tiebreakers, _row_id injection,
      __orig_row_num__ subqueries, groupby auto-sort/dropna/set_index,
      MultiIndex columns, dtype corrections, and -If/isNaN wrappers.
      Operations like first()/last() use any()/anyLast() (non-deterministic).
      Results may have different row order than pandas equivalents.
    """

    PANDAS = "pandas"
    PERFORMANCE = "performance"


_compat_mode: str = CompatMode.PANDAS


def get_compat_mode() -> str:
    """Get current compatibility mode setting."""
    return _compat_mode


def set_compat_mode(mode: str) -> None:
    """
    Set the compatibility mode.

    Args:
        mode: One of 'pandas', 'performance'

    Example:
        >>> from datastore import config
        >>> config.set_compat_mode('performance')  # SQL-first, max throughput
        >>> config.set_compat_mode('pandas')  # Full pandas compat (default)
    """
    global _compat_mode
    valid = {CompatMode.PANDAS, CompatMode.PERFORMANCE}
    if mode not in valid:
        raise ValueError(f"Invalid compat mode: {mode}. Use one of {valid}")
    _compat_mode = mode
    # Performance mode implies chDB execution engine
    if mode == CompatMode.PERFORMANCE:
        set_execution_engine(ExecutionEngine.CHDB)


def is_performance_mode() -> bool:
    """Check if performance mode is active (SQL-first, no pandas compat overhead)."""
    return _compat_mode == CompatMode.PERFORMANCE


def use_performance_mode() -> None:
    """Enable performance mode (SQL-first, no pandas compat overhead)."""
    set_compat_mode(CompatMode.PERFORMANCE)


def use_pandas_compat() -> None:
    """Enable pandas compatibility mode (default)."""
    set_compat_mode(CompatMode.PANDAS)


# =============================================================================
# CROSS-DATASTORE OPERATION CONFIGURATION
# =============================================================================


def get_cross_datastore_engine() -> str:
    """
    Get current execution engine for cross-DataStore operations.

    Returns:
        str: One of 'auto', 'chdb', 'pandas'
    """
    return _cross_datastore_engine


def set_cross_datastore_engine(engine: str) -> None:
    """
    Set the execution engine for cross-DataStore operations.

    Cross-DataStore operations occur when columns from different DataStores
    are combined (e.g., ds1['a'] + ds2['b']).

    Args:
        engine: One of 'auto', 'chdb', 'pandas'
            - 'auto': Execute each side with its optimal engine, then combine with pandas
            - 'chdb': Force chDB for each side before combining (slower for simple ops)
            - 'pandas': Force pandas for entire operation (fastest for simple arithmetic)

    Example:
        >>> from datastore import config
        >>> config.set_cross_datastore_engine('pandas')  # Fastest for simple ops
        >>> ds1['a'] + ds2['b']  # Will use pure pandas
    """
    global _cross_datastore_engine
    valid = {ExecutionEngine.AUTO, ExecutionEngine.CHDB, ExecutionEngine.PANDAS}
    if engine not in valid:
        raise ValueError(f"Invalid engine: {engine}. Use one of {valid}")
    _cross_datastore_engine = engine


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

    # ========== Compatibility Mode ==========

    @property
    def compat_mode(self) -> str:
        """Get current compatibility mode ('pandas' or 'performance')."""
        return get_compat_mode()

    @compat_mode.setter
    def compat_mode(self, mode: str) -> None:
        """Set compatibility mode ('pandas' or 'performance')."""
        set_compat_mode(mode)

    def set_compat_mode(self, mode: str) -> None:
        """Set compatibility mode."""
        set_compat_mode(mode)

    def use_performance_mode(self) -> None:
        """Enable performance mode (SQL-first, no pandas compat overhead)."""
        use_performance_mode()

    def use_pandas_compat(self) -> None:
        """Enable pandas compatibility mode (default)."""
        use_pandas_compat()

    # ========== Cross-DataStore Engine ==========

    @property
    def cross_datastore_engine(self) -> str:
        """Get current cross-DataStore execution engine."""
        return get_cross_datastore_engine()

    @cross_datastore_engine.setter
    def cross_datastore_engine(self, engine: str) -> None:
        """Set cross-DataStore execution engine ('auto', 'chdb', 'pandas')."""
        set_cross_datastore_engine(engine)

    def set_cross_datastore_engine(self, engine: str) -> None:
        """Set cross-DataStore execution engine."""
        set_cross_datastore_engine(engine)

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

    # ========== Profiling Configuration ==========

    @property
    def profiling_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return is_profiling_enabled()

    @profiling_enabled.setter
    def profiling_enabled(self, enabled: bool) -> None:
        """Enable or disable profiling."""
        if enabled:
            enable_profiling()
        else:
            disable_profiling()

    def enable_profiling(self) -> None:
        """Enable execution profiling."""
        enable_profiling()

    def disable_profiling(self) -> None:
        """Disable execution profiling."""
        disable_profiling()


# Global config instance
config = DataStoreConfig()


# =============================================================================
# PROFILER CONFIGURATION
# =============================================================================


class ProfileStep:
    """A single profiled step with timing information."""

    def __init__(self, name: str, parent: Optional["ProfileStep"] = None):
        self.name = name
        self.parent = parent
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.children: List["ProfileStep"] = []
        self.metadata: Dict[str, any] = {}

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return f"ProfileStep({self.name}, {self.duration_ms:.2f}ms)"


class Profiler:
    """
    A profiler for tracking execution timing of various steps.

    Usage:
        profiler = Profiler()
        with profiler.step("SQL Execution"):
            # do SQL stuff
        with profiler.step("DataFrame Ops"):
            # do DataFrame stuff
        profiler.report()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.steps: List[ProfileStep] = []
        self._stack: List[ProfileStep] = []
        self._logger = get_logger()

    @contextmanager
    def step(self, name: str, **metadata):
        """
        Context manager to profile a step.

        Args:
            name: Name of the step
            **metadata: Additional metadata to attach to the step
        """
        if not self.enabled:
            yield None
            return

        parent = self._stack[-1] if self._stack else None
        step = ProfileStep(name, parent)
        step.metadata = metadata
        step.start_time = time.perf_counter()

        if parent:
            parent.children.append(step)
        else:
            self.steps.append(step)

        self._stack.append(step)
        try:
            yield step
        finally:
            step.end_time = time.perf_counter()
            self._stack.pop()

    def clear(self):
        """Clear all recorded steps."""
        self.steps = []
        self._stack = []

    @property
    def total_duration_ms(self) -> float:
        """Total duration of all top-level steps in milliseconds."""
        return sum(s.duration_ms for s in self.steps)

    def report(self, min_duration_ms: float = 0.1) -> str:
        """
        Generate a human-readable report of all profiled steps.

        Args:
            min_duration_ms: Minimum duration to include in report (default: 0.1ms)

        Returns:
            Formatted string report
        """
        if not self.steps:
            return "No profiling data recorded."

        lines = []
        lines.append("=" * 70)
        lines.append("EXECUTION PROFILE")
        lines.append("=" * 70)

        def format_step(step: ProfileStep, indent: int = 0):
            if step.duration_ms < min_duration_ms:
                return
            prefix = "  " * indent
            duration_str = f"{step.duration_ms:>8.2f}ms"

            # Calculate percentage of parent/total
            if step.parent:
                pct = (
                    (step.duration_ms / step.parent.duration_ms * 100)
                    if step.parent.duration_ms > 0
                    else 0
                )
                pct_str = f"({pct:>5.1f}%)"
            else:
                pct = (
                    (step.duration_ms / self.total_duration_ms * 100)
                    if self.total_duration_ms > 0
                    else 0
                )
                pct_str = f"({pct:>5.1f}%)"

            # Add metadata if present
            meta_str = ""
            if step.metadata:
                meta_parts = [f"{k}={v}" for k, v in step.metadata.items()]
                meta_str = f" [{', '.join(meta_parts)}]"

            lines.append(f"{prefix}{duration_str} {pct_str} {step.name}{meta_str}")

            for child in step.children:
                format_step(child, indent + 1)

        for step in self.steps:
            format_step(step)

        lines.append("-" * 70)
        lines.append(f"{'TOTAL:':>12} {self.total_duration_ms:>8.2f}ms")
        lines.append("=" * 70)

        return "\n".join(lines)

    def log_report(self, min_duration_ms: float = 0.1):
        """Log the profiling report at INFO level."""
        if self.enabled and self.steps:
            report = self.report(min_duration_ms)
            for line in report.split("\n"):
                self._logger.info(line)

    def summary(self) -> Dict[str, float]:
        """
        Get a summary dict of step names to durations (ms).

        Useful for programmatic access to timing data.
        """
        result = {}

        def collect(step: ProfileStep, prefix: str = ""):
            name = f"{prefix}{step.name}" if prefix else step.name
            result[name] = step.duration_ms
            for child in step.children:
                collect(child, f"{name}.")

        for step in self.steps:
            collect(step)

        return result


# Global profiler settings
_profiling_enabled: bool = False
_current_profiler: Optional[Profiler] = None


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled."""
    return _profiling_enabled


def enable_profiling() -> None:
    """
    Enable execution profiling.

    When enabled, execution timing information will be collected and reported.

    Example:
        >>> from datastore import config
        >>> config.enable_profiling()
        >>> ds = DataStore.from_file('data.csv')
        >>> result = ds.filter(ds['x'] > 10).to_df()  # Will show timing
    """
    global _profiling_enabled
    _profiling_enabled = True


def disable_profiling() -> None:
    """
    Disable execution profiling.

    Example:
        >>> from datastore import config
        >>> config.disable_profiling()
    """
    global _profiling_enabled
    _profiling_enabled = False


def get_profiler() -> Profiler:
    """
    Get or create a profiler instance.

    When profiling is enabled, returns the shared profiler that accumulates
    all execution data. When disabled, returns a no-op profiler that won't
    record any data but allows the same API to be used.
    """
    global _current_profiler
    if _profiling_enabled:
        if _current_profiler is None:
            _current_profiler = Profiler(enabled=True)
        return _current_profiler
    # Return a disabled profiler for no-op usage
    return Profiler(enabled=False)


def reset_profiler() -> None:
    """Reset the current profiler (clear all recorded data)."""
    global _current_profiler
    _current_profiler = None
