"""
Prediction Logger
==================
Domain logger for all recommendation generation events.
Logs: query product_id, session_id, engine used, result count,
      confidence scores, cache hit/miss, prediction latency.
"""

from src.logger.logger import get_logger

_logger = get_logger("prediction", "logs/prediction.log")


def get_prediction_logger():
    """Return the shared Prediction domain logger."""
    return _logger
