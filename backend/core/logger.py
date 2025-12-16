"""
Logging configuration for Jericho Chatbot.

Implements structured logging with JSON formatting for production
and readable format for development. Handles file rotation and
multiple log levels.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger
import sys


class LoggerConfig:
    """Centralized logging configuration."""

    def __init__(
        self,
        log_dir: Path = Path("logs"),
        log_file: str = "app.log",
        level: str = "INFO",
        json_format: bool = False,
    ):
        """
        Initialize logger configuration.

        Args:
            log_dir: Directory for log files
            log_file: Log filename
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            json_format: Use JSON format for logs (production) or readable (dev)
        """
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.level = level
        self.json_format = json_format

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with configured handlers.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Prevent duplicate handlers if called multiple times
        if logger.handlers:
            return logger

        # Set logger level
        logger.setLevel(getattr(logging, self.level))

        # Create formatters
        if self.json_format:
            formatter = jsonlogger.JsonFormatter(
                fmt="%(timestamp)s %(level)s %(name)s %(message)s",
                timestamp=True,
            )
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # File handler with rotation (10MB files, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / self.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(getattr(logging, self.level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (always readable format)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.level))
        
        console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger


# Global logger instance (initialized in main app)
_logger_config: Optional[LoggerConfig] = None


def init_logger(
    log_dir: Path = Path("logs"),
    log_file: str = "app.log",
    level: str = "INFO",
    json_format: bool = False,
) -> None:
    """
    Initialize global logger configuration.

    Should be called once at app startup.

    Args:
        log_dir: Directory for log files
        log_file: Log filename
        level: Logging level
        json_format: Use JSON format for logs
    """
    global _logger_config
    _logger_config = LoggerConfig(
        log_dir=log_dir,
        log_file=log_file,
        level=level,
        json_format=json_format,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Raises:
        RuntimeError: If logger not initialized via init_logger()
    """
    if _logger_config is None:
        raise RuntimeError(
            "Logger not initialized. Call init_logger() at app startup."
        )
    return _logger_config.get_logger(name)
