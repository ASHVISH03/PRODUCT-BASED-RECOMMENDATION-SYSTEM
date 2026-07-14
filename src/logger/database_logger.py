"""
Database Logger
================
Domain logger for all SQLAlchemy and database events.
Logs: connection events, query execution (summarized), session lifecycle,
      CRUD operation counts, transaction commits and rollbacks.
"""

from src.logger.logger import get_logger

_logger = get_logger("database", "logs/database.log")


def get_database_logger():
    """Return the shared Database domain logger."""
    return _logger
