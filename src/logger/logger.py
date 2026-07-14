"""
Logger Factory — Base Module
=============================
Creates and configures named loggers with rotating file handlers
and console output. All 7 domain loggers are built using this factory.

Usage:
    from src.logger.logger import get_logger

    logger = get_logger("api", "logs/api.log")
    logger.info("Request received: GET /health")
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Module-level registry to avoid creating duplicate handlers
_logger_registry: dict[str, logging.Logger] = {}

# Default format shared by all loggers
_DEFAULT_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    log_format: str = _DEFAULT_FORMAT,
    date_format: str = _DEFAULT_DATE_FORMAT,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
    console_output: bool = True,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with rotating file and console handlers.

    If a logger with the same name already exists in the registry,
    it is returned immediately to prevent duplicate handlers.

    Args:
        name: Logger name (e.g., 'api', 'training'). Also used as the
              Python logging hierarchy name.
        log_file: Path to the rotating log file. Parent directories
                  are created automatically.
        level: Logging level (default: logging.INFO).
        log_format: Log record format string.
        date_format: Datetime format for log timestamps.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of rotated files to retain.
        console_output: Whether to also write to stdout.

    Returns:
        Configured logging.Logger instance.
    """
    # Return cached logger if already configured
    if name in _logger_registry:
        return _logger_registry[name]

    # Ensure the log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    # --- Rotating File Handler ---
    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- Console (Stream) Handler ---
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Cache and return
    _logger_registry[name] = logger
    return logger


def get_logger_from_config() -> dict[str, logging.Logger]:
    """
    Build all 7 domain loggers from config.yaml settings.

    Returns:
        Dictionary mapping domain name → configured logger.

    Note:
        This function lazy-imports ConfigurationManager to avoid
        circular imports at module load time.
    """
    try:
        from src.config.configuration import ConfigurationManager
        cfg = ConfigurationManager().get_logging_config()
        level = getattr(logging, cfg.level.upper(), logging.INFO)

        return {
            "api": get_logger(
                "api", cfg.files.api, level=level,
                log_format=cfg.format, date_format=cfg.date_format,
                max_bytes=cfg.max_bytes, backup_count=cfg.backup_count,
            ),
            "training": get_logger(
                "training", cfg.files.training, level=level,
                log_format=cfg.format, date_format=cfg.date_format,
                max_bytes=cfg.max_bytes, backup_count=cfg.backup_count,
            ),
            "prediction": get_logger(
                "prediction", cfg.files.prediction, level=level,
                log_format=cfg.format, date_format=cfg.date_format,
                max_bytes=cfg.max_bytes, backup_count=cfg.backup_count,
            ),
            "experiment": get_logger(
                "experiment", cfg.files.experiment, level=level,
                log_format=cfg.format, date_format=cfg.date_format,
                max_bytes=cfg.max_bytes, backup_count=cfg.backup_count,
            ),
            "deployment": get_logger(
                "deployment", cfg.files.deployment, level=level,
                log_format=cfg.format, date_format=cfg.date_format,
                max_bytes=cfg.max_bytes, backup_count=cfg.backup_count,
            ),
            "database": get_logger(
                "database", cfg.files.database, level=level,
                log_format=cfg.format, date_format=cfg.date_format,
                max_bytes=cfg.max_bytes, backup_count=cfg.backup_count,
            ),
            "security": get_logger(
                "security", cfg.files.security, level=level,
                log_format=cfg.format, date_format=cfg.date_format,
                max_bytes=cfg.max_bytes, backup_count=cfg.backup_count,
            ),
        }
    except Exception:
        # Fallback: return basic loggers if config is not yet available
        return {
            domain: get_logger(domain, f"logs/{domain}.log")
            for domain in [
                "api", "training", "prediction", "experiment",
                "deployment", "database", "security",
            ]
        }
