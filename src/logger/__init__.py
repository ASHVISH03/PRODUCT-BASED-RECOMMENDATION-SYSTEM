"""
Logger package.

Provides 7 domain loggers:
    api, training, prediction, experiment, deployment, database, security
"""

from src.logger.logger import get_logger, get_logger_from_config
from src.logger.api_logger import get_api_logger
from src.logger.training_logger import get_training_logger
from src.logger.prediction_logger import get_prediction_logger
from src.logger.experiment_logger import get_experiment_logger
from src.logger.deployment_logger import get_deployment_logger
from src.logger.database_logger import get_database_logger
from src.logger.security_logger import get_security_logger

__all__ = [
    "get_logger",
    "get_logger_from_config",
    "get_api_logger",
    "get_training_logger",
    "get_prediction_logger",
    "get_experiment_logger",
    "get_deployment_logger",
    "get_database_logger",
    "get_security_logger",
]
