"""
Training Pipeline
==================
Orchestrates all Phase 2 data pipeline components in sequence:

    DataIngestion → DataValidation → DataTransformation → FeatureEngineering

Design principles:
    • Configuration-driven: no hardcoded paths or values
    • Structured artifacts: each component returns a typed artifact
    • Comprehensive logging: every step is timed and logged
    • Graceful failure: all exceptions are caught, logged, and re-raised
    • Idempotent: safe to re-run (overwrites previous outputs)

Usage:
    # As a module:
    python -m src.pipelines.training_pipeline

    # In code:
    from src.pipelines.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()
    artifact = pipeline.run()
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.components.data_ingestion.data_ingestion import DataIngestion
from src.components.data_transformation.data_transformation import DataTransformation
from src.components.data_validation.data_validation import DataValidation
from src.components.feature_engineering.feature_engineering import FeatureEngineering
from src.config.configuration import ConfigurationManager
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    FeatureEngineeringArtifact,
    TrainingPipelineArtifact,
)
from src.exception.exception import CustomException, DatasetNotFoundError, DataValidationError
from src.logger.training_logger import get_training_logger
from src.utils.common import create_directories, get_iso_timestamp, save_json

logger = get_training_logger()


class TrainingPipeline:
    """
    Orchestrates the complete data preparation and model training pipeline.

    Each phase returns a typed artifact that is passed as input to the
    next component. All artifacts are collected into a composite
    TrainingPipelineArtifact at the end.

    Args:
        config_path: Optional path to config.yaml. Defaults to standard location.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_manager = ConfigurationManager(config_path)
        self._start_time: Optional[float] = None

        # Pre-load all config sections
        self._data_config = self._config_manager.get_data_config()
        self._feature_config = self._config_manager.get_feature_config()
        self._model_config = self._config_manager.get_model_config()
        self._logging_config = self._config_manager.get_logging_config()

        # Ensure all required directories exist
        create_directories([
            self._data_config.raw_dir,
            self._data_config.processed_dir,
            self._data_config.interim_dir,
            self._data_config.external_dir,
            self._model_config.artifacts_dir,
            "artifacts",
            "logs",
        ])

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _run_data_ingestion(self) -> DataIngestionArtifact:
        """Step 1: Data Ingestion."""
        step_start = time.perf_counter()
        try:
            component = DataIngestion(config=self._data_config)
            artifact = component.initiate_data_ingestion()
            elapsed = time.perf_counter() - step_start
            logger.info(f"Data Ingestion duration: {elapsed:.2f}s")
            return artifact
        except DatasetNotFoundError:
            raise
        except Exception as e:
            raise CustomException(
                f"Data Ingestion step failed: {e}", sys.exc_info()
            ) from e

    def _run_data_validation(
        self, ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """Step 2: Data Validation."""
        step_start = time.perf_counter()
        try:
            component = DataValidation(
                config=self._data_config,
                ingestion_artifact=ingestion_artifact,
            )
            artifact = component.initiate_data_validation()
            elapsed = time.perf_counter() - step_start
            logger.info(f"Data Validation duration: {elapsed:.2f}s")
            return artifact
        except DataValidationError:
            raise
        except Exception as e:
            raise CustomException(
                f"Data Validation step failed: {e}", sys.exc_info()
            ) from e

    def _run_data_transformation(
        self, ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        """Step 3: Data Transformation."""
        step_start = time.perf_counter()
        try:
            component = DataTransformation(
                config=self._data_config,
                feature_config=self._feature_config,
                ingestion_artifact=ingestion_artifact,
            )
            artifact = component.initiate_data_transformation()
            elapsed = time.perf_counter() - step_start
            logger.info(f"Data Transformation duration: {elapsed:.2f}s")
            return artifact
        except Exception as e:
            if isinstance(e, CustomException):
                raise
            raise CustomException(
                f"Data Transformation step failed: {e}", sys.exc_info()
            ) from e

    def _run_feature_engineering(
        self, transformation_artifact: DataTransformationArtifact
    ) -> FeatureEngineeringArtifact:
        """Step 4: Feature Engineering."""
        step_start = time.perf_counter()
        try:
            component = FeatureEngineering(
                feature_config=self._feature_config,
                transformation_artifact=transformation_artifact,
            )
            artifact = component.initiate_feature_engineering()
            elapsed = time.perf_counter() - step_start
            logger.info(f"Feature Engineering duration: {elapsed:.2f}s")
            return artifact
        except Exception as e:
            if isinstance(e, CustomException):
                raise
            raise CustomException(
                f"Feature Engineering step failed: {e}", sys.exc_info()
            ) from e

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def _save_pipeline_summary(
        self,
        artifact: TrainingPipelineArtifact,
        duration: float,
    ) -> None:
        """Persist a human-readable JSON summary of the entire pipeline run."""
        summary = {
            "pipeline_status": "success" if artifact.success else "failed",
            "completed_at": get_iso_timestamp(),
            "pipeline_duration_seconds": round(duration, 2),
            "error_message": artifact.error_message,
            "ingestion": {
                "source": artifact.ingestion.source,
                "is_fallback": artifact.ingestion.is_fallback,
                "record_count": artifact.ingestion.record_count,
                "file_size_mb": artifact.ingestion.file_size_mb,
                "dataset_version": artifact.ingestion.dataset_version,
                "raw_filepath": artifact.ingestion.raw_filepath,
            },
            "validation": {
                "status": artifact.validation.validation_status,
                "schema_valid": artifact.validation.schema_valid,
                "missing_values": artifact.validation.missing_values_count,
                "duplicates": artifact.validation.duplicate_count,
                "issues": len(artifact.validation.issues),
                "validated_records": artifact.validation.validated_record_count,
                "report": artifact.validation.validation_report_path,
            },
            "transformation": {
                "input_records": artifact.transformation.input_record_count,
                "output_records": artifact.transformation.output_record_count,
                "removed_duplicates": artifact.transformation.removed_duplicates,
                "filled_nulls": artifact.transformation.filled_nulls,
                "dropped_nulls": artifact.transformation.dropped_nulls,
                "processed_file": artifact.transformation.processed_filepath,
            },
            "feature_engineering": {
                "feature_count": len(artifact.feature_engineering.feature_columns),
                "features": artifact.feature_engineering.feature_columns,
                "vocabulary_estimate": artifact.feature_engineering.vocabulary_estimate,
                "price_buckets": artifact.feature_engineering.price_bucket_distribution,
                "popularity_stats": artifact.feature_engineering.popularity_score_stats,
                "feature_file": artifact.feature_engineering.feature_filepath,
            },
        }

        save_json(Path("artifacts") / "pipeline_summary.json", summary)
        logger.info("Pipeline summary saved → artifacts/pipeline_summary.json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> TrainingPipelineArtifact:
        """
        Execute the complete training pipeline.

        Steps run in sequence:
            1. Data Ingestion
            2. Data Validation
            3. Data Transformation
            4. Feature Engineering

        Returns:
            TrainingPipelineArtifact (composite of all stage artifacts).

        Raises:
            DatasetNotFoundError: If the dataset is missing.
            DataValidationError: If the data fails error-level checks.
            CustomException: On unexpected failures in any step.
        """
        self._start_time = time.perf_counter()

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║         PRODUCT RECOMMENDATION SYSTEM PIPELINE           ║")
        logger.info("║                  PHASE 2: DATA PIPELINE                  ║")
        logger.info("╚══════════════════════════════════════════════════════════╝")
        logger.info(f"Started at: {get_iso_timestamp()}")
        logger.info("")

        ingestion_artifact: Optional[DataIngestionArtifact] = None
        validation_artifact: Optional[DataValidationArtifact] = None
        transformation_artifact: Optional[DataTransformationArtifact] = None
        feature_artifact: Optional[FeatureEngineeringArtifact] = None
        error_message: Optional[str] = None

        try:
            # Step 1
            ingestion_artifact = self._run_data_ingestion()

            # Step 2
            validation_artifact = self._run_data_validation(ingestion_artifact)

            # Step 3 (always runs after validation, even with warnings)
            transformation_artifact = self._run_data_transformation(ingestion_artifact)

            # Step 4
            feature_artifact = self._run_feature_engineering(transformation_artifact)

            pipeline_duration = time.perf_counter() - self._start_time

            logger.info("")
            logger.info("╔══════════════════════════════════════════════════════════╗")
            logger.info("║              PIPELINE COMPLETED SUCCESSFULLY             ║")
            logger.info(f"║  Duration: {pipeline_duration:.1f}s" + " " * (46 - len(f"{pipeline_duration:.1f}s")) + "║")
            logger.info("╚══════════════════════════════════════════════════════════╝")

            composite = TrainingPipelineArtifact(
                ingestion=ingestion_artifact,
                validation=validation_artifact,
                transformation=transformation_artifact,
                feature_engineering=feature_artifact,
                # These will be filled in Phase 3 (model training)
                model_training=None,      # type: ignore[arg-type]
                model_evaluation=None,    # type: ignore[arg-type]
                model_registry=None,      # type: ignore[arg-type]
                pipeline_duration_seconds=pipeline_duration,
                success=True,
                error_message=None,
            )

            self._save_pipeline_summary(composite, pipeline_duration)
            return composite

        except (DatasetNotFoundError, DataValidationError) as e:
            error_message = str(e)
            logger.error(f"Pipeline aborted: {error_message}")
            raise

        except Exception as e:
            pipeline_duration = time.perf_counter() - self._start_time
            error_message = str(e)
            logger.error(f"Pipeline failed after {pipeline_duration:.1f}s: {error_message}")

            # Build partial artifact for diagnosis
            if ingestion_artifact:
                partial = TrainingPipelineArtifact(
                    ingestion=ingestion_artifact,
                    validation=validation_artifact,           # type: ignore[arg-type]
                    transformation=transformation_artifact,   # type: ignore[arg-type]
                    feature_engineering=feature_artifact,     # type: ignore[arg-type]
                    model_training=None,                      # type: ignore[arg-type]
                    model_evaluation=None,                    # type: ignore[arg-type]
                    model_registry=None,                      # type: ignore[arg-type]
                    pipeline_duration_seconds=pipeline_duration,
                    success=False,
                    error_message=error_message,
                )
                try:
                    self._save_pipeline_summary(partial, pipeline_duration)
                except Exception:
                    pass

            raise CustomException(
                f"Training pipeline failed: {e}", sys.exc_info()
            ) from e


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    """Run the training pipeline from the command line."""
    try:
        pipeline = TrainingPipeline()
        artifact = pipeline.run()

        print("\n" + "=" * 60)
        print("  PHASE 2 PIPELINE SUMMARY")
        print("=" * 60)
        print(f"  Status      : SUCCESS ✓")
        print(f"  Source      : {artifact.ingestion.source}")
        print(f"  Fallback    : {artifact.ingestion.is_fallback}")
        print(f"  Raw records : {artifact.ingestion.record_count:,}")
        print(f"  Clean recs  : {artifact.transformation.output_record_count:,}")
        print(f"  Features    : {len(artifact.feature_engineering.feature_columns)}")
        print(f"  Vocab est.  : {artifact.feature_engineering.vocabulary_estimate:,}")
        print(f"  Duration    : {artifact.pipeline_duration_seconds:.1f}s")
        print(f"  Features    : {artifact.feature_engineering.feature_filepath}")
        print("=" * 60)
        print("\nPhase 2 complete. Artifacts saved to:")
        print("  artifacts/ingestion_metadata.json")
        print("  artifacts/validation_report.json")
        print("  artifacts/pipeline_summary.json")
        print("  data/processed/cleaned.csv")
        print("  data/processed/features.csv")
        print("\nReady for Phase 3: Model Training")

    except DatasetNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)
    except DataValidationError as e:
        print(f"\nValidation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
