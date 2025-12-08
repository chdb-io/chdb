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

# DataStore's own logger name
LOGGER_NAME = "datastore"


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
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
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
    global _log_level, _logger
    
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
    """
    
    def __init__(self):
        self._log_level = logging.WARNING
    
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


# Global config instance
config = DataStoreConfig()



