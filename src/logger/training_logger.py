"""
Training Logger
================
Domain logger for all ML pipeline events.
Logs: pipeline stage start/end, data statistics, training metrics,
      artifact paths, duration, errors during training.
"""

from src.logger.logger import get_logger

_logger = get_logger("training", "logs/training.log")


def get_training_logger():
    """Return the shared Training domain logger."""
    return _logger
