"""
API Logger
===========
Domain logger for all HTTP request/response events.
Logs: endpoint, method, status code, client IP, response time, errors.
"""

from src.logger.logger import get_logger

_logger = get_logger("api", "logs/api.log")


def get_api_logger():
    """Return the shared API domain logger."""
    return _logger
