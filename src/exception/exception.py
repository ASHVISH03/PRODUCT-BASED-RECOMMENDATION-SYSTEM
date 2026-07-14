"""
Custom Exception Hierarchy
===========================
All exceptions in the recommendation system inherit from
RecommendationSystemException, which captures the file name
and line number where the error was raised.

Usage:
    from src.exception.exception import DatasetNotFoundError

    try:
        ...
    except FileNotFoundError as e:
        raise DatasetNotFoundError(
            "Dataset CSV not found. Run: python scripts/setup_dataset.py"
        ) from e
"""

import sys
import traceback
from typing import Optional


def _extract_error_context(error_detail: Optional[tuple] = None) -> str:
    """
    Extract file name and line number from sys.exc_info() tuple.

    Args:
        error_detail: The result of sys.exc_info() — a 3-tuple
                      (exc_type, exc_value, traceback).

    Returns:
        A human-readable string with file and line info, or empty string.
    """
    if error_detail is None:
        return ""
    _, _, exc_tb = error_detail
    if exc_tb is None:
        return ""
    # Walk to the innermost frame for the most specific location
    tb = exc_tb
    while tb.tb_next is not None:
        tb = tb.tb_next
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    return f" | File: [{file_name}] | Line: [{line_number}]"


# ---------------------------------------------------------------
# Base Exception
# ---------------------------------------------------------------

class RecommendationSystemException(Exception):
    """
    Base exception for the entire recommendation system.

    Automatically enriches the message with file/line context
    when raised inside an except block.

    Args:
        message: Human-readable error description.
        error_detail: Pass sys.exc_info() to capture traceback context.
                      If omitted, no file/line info is added.

    Example:
        try:
            risky_operation()
        except Exception as e:
            raise RecommendationSystemException(
                "Something went wrong", sys.exc_info()
            ) from e
    """

    def __init__(
        self,
        message: str,
        error_detail: Optional[tuple] = None,
    ) -> None:
        context = _extract_error_context(error_detail)
        self.error_message = f"{message}{context}"
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.error_message!r})"


# ---------------------------------------------------------------
# Data Exceptions
# ---------------------------------------------------------------

class DatasetNotFoundError(RecommendationSystemException):
    """
    Raised when neither the primary dataset nor the fallback is available.

    Resolution:
        1. Run: python scripts/setup_dataset.py (with Kaggle credentials in .env)
        2. OR: python scripts/generate_fallback.py
              then set use_fallback: true in src/config/config.yaml
    """


class DataValidationError(RecommendationSystemException):
    """
    Raised when the dataset fails validation checks.
    This could mean missing required columns, too many null values,
    data type mismatches, or schema violations.
    """


class DataTransformationError(RecommendationSystemException):
    """
    Raised during data cleaning or text preprocessing failures.
    """


class FeatureEngineeringError(RecommendationSystemException):
    """
    Raised during feature construction, TF-IDF preparation,
    or combined-text field creation.
    """


# ---------------------------------------------------------------
# Model Exceptions
# ---------------------------------------------------------------

class ModelNotLoadedError(RecommendationSystemException):
    """
    Raised when a prediction is attempted but no trained model
    artifacts exist in the models/ directory.

    Resolution:
        Run: python -m src.pipelines.training_pipeline
    """


class ModelTrainingError(RecommendationSystemException):
    """
    Raised when model training fails — e.g., TF-IDF fitting error
    or similarity matrix computation failure.
    """


class ModelEvaluationError(RecommendationSystemException):
    """
    Raised when model evaluation cannot be completed.
    """


class ModelRegistryError(RecommendationSystemException):
    """
    Raised when MLflow model registration or promotion fails.
    """


class ModelVersionError(RecommendationSystemException):
    """
    Raised when a requested model version does not exist
    in the MLflow registry.
    """


# ---------------------------------------------------------------
# Prediction Exceptions
# ---------------------------------------------------------------

class PredictionError(RecommendationSystemException):
    """
    Raised during recommendation generation, confidence scoring,
    or XAI breakdown computation.
    """


class ProductNotFoundError(RecommendationSystemException):
    """
    Raised when a requested product_id does not exist
    in the database or model index.
    """


class InvalidRecommendationStrategyError(RecommendationSystemException):
    """
    Raised when an unknown or unsupported recommendation strategy
    is requested.
    """


# ---------------------------------------------------------------
# Configuration Exceptions
# ---------------------------------------------------------------

class ConfigurationError(RecommendationSystemException):
    """
    Raised when config.yaml is missing, malformed, or contains
    invalid values.
    """


# ---------------------------------------------------------------
# API Exceptions
# ---------------------------------------------------------------

class APIError(RecommendationSystemException):
    """
    General API-layer error. Used for unexpected server-side failures
    that should return HTTP 500.
    """


class ValidationError(RecommendationSystemException):
    """
    Raised when request payload fails business-logic validation
    (beyond Pydantic schema validation). Returns HTTP 422.
    """


class RateLimitExceededError(RecommendationSystemException):
    """
    Raised when a client exceeds the configured rate limit.
    Returns HTTP 429.
    """


# ---------------------------------------------------------------
# Database Exceptions
# ---------------------------------------------------------------

class DatabaseError(RecommendationSystemException):
    """
    Raised on SQLAlchemy session errors, connection failures,
    or constraint violations.
    """


class RecordNotFoundError(RecommendationSystemException):
    """
    Raised when a database lookup returns no results for
    a required record.
    """


# ---------------------------------------------------------------
# Cache Exceptions
# ---------------------------------------------------------------

class CacheError(RecommendationSystemException):
    """
    Raised on cache read/write failures. Non-fatal — callers
    should fall through to the primary data source.
    """


# ---------------------------------------------------------------
# Session Exceptions
# ---------------------------------------------------------------

class SessionError(RecommendationSystemException):
    """
    Raised on anonymous session creation or lookup failures.
    """


# ---------------------------------------------------------------
# Search Exceptions
# ---------------------------------------------------------------

class SearchError(RecommendationSystemException):
    """
    Raised when the search service fails to process a query.
    """


# ---------------------------------------------------------------
# MLflow Exceptions
# ---------------------------------------------------------------

class MLflowError(RecommendationSystemException):
    """
    Raised on MLflow tracking or registry errors.
    """


# ---------------------------------------------------------------
# Monitoring Exceptions
# ---------------------------------------------------------------

class MonitoringError(RecommendationSystemException):
    """
    Raised when monitoring data collection or drift detection fails.
    Non-fatal — monitoring failures should never crash the main app.
    """


# ---------------------------------------------------------------
# Background Job Exceptions
# ---------------------------------------------------------------

class BackgroundJobError(RecommendationSystemException):
    """
    Raised when a background training or recompute job fails.
    """


class JobNotFoundError(RecommendationSystemException):
    """
    Raised when a background job ID does not exist.
    """


# ---------------------------------------------------------------
# Convenience Alias
# ---------------------------------------------------------------

#: Short alias for RecommendationSystemException.
#: Use this when you want a single conventional import name
#: consistent with standard ML project patterns:
#:
#:     from src.exception.exception import CustomException
#:
#: Both names refer to the same class and are fully interchangeable.
CustomException = RecommendationSystemException
