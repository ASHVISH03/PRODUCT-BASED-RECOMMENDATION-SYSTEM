"""
Security Logger
================
Domain logger for security-relevant events.
Logs: rate limit violations, suspicious request patterns,
      invalid API key attempts (future auth), input sanitization
      events, and any access control decisions.

Note: This logger intentionally avoids logging full request bodies
      to prevent accidental PII exposure in log files.
"""

from src.logger.logger import get_logger

_logger = get_logger("security", "logs/security.log")


def get_security_logger():
    """Return the shared Security domain logger."""
    return _logger
