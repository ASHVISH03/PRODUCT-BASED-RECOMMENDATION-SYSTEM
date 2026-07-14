"""
Experiment Logger
==================
Domain logger for MLflow experiment tracking events.
Logs: run creation, parameter logging, metric logging, artifact upload,
      model registration, stage promotion, run comparison.
"""

from src.logger.logger import get_logger

_logger = get_logger("experiment", "logs/experiment.log")


def get_experiment_logger():
    """Return the shared Experiment domain logger."""
    return _logger
