"""
Deployment Logger
==================
Domain logger for application lifecycle events.
Logs: app startup, shutdown, configuration loading, model loading,
      service initialization, health check results.
"""

from src.logger.logger import get_logger

_logger = get_logger("deployment", "logs/deployment.log")


def get_deployment_logger():
    """Return the shared Deployment domain logger."""
    return _logger
