"""
Training Pipeline
==================
Orchestrates all pipeline components in sequence:

    DataIngestion → DataValidation → DataTransformation → FeatureEngineering
    → ModelTraining → ModelEvaluation → ModelRegistry

Design principles:
    Configuration-driven: no hardcoded paths or values
    Structured artifacts: each component returns a typed artifact
    Comprehensive logging: every step is timed and logged
    Graceful failure: all exceptions are caught, logged, and re-raised
    Idempotent: safe to re-run (overwrites previous outputs)

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
from src.components.model_evaluation.model_evaluator import ModelEvaluator
from src.components.model_registry.model_registry import ModelRegistry
from src.components.model_training.model_trainer import ModelTrainer
from src.config.configuration import ConfigurationManager
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    FeatureEngineeringArtifact,
    ModelEvaluationArtifact,
    ModelRegistryArtifact,
    ModelTrainingArtifact,
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
        skip_model_training: If True, only run Phases 2 (data pipeline).
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        skip_model_training: bool = False,
    ) -> None:
        self.cfg = ConfigurationManager(config_path)
        self.skip_model_training = skip_model_training

    # ------------------------------------------------------------------
    # Phase 2: Data pipeline steps
    # ------------------------------------------------------------------

    def _run_ingestion(self) -> DataIngestionArtifact:
        logger.info("STEP 1/7: Data Ingestion")
        start = time.perf_counter()
        data_config = self.cfg.get_data_config()
        ingestion = DataIngestion(config=data_config)
        artifact = ingestion.initiate_data_ingestion()
        logger.info(
            f"Ingestion done in {time.perf_counter()-start:.2f}s — "
            f"{artifact.record_count} records from '{artifact.source}'"
        )
        return artifact

    def _run_validation(
        self, ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        logger.info("STEP 2/7: Data Validation")
        start = time.perf_counter()
        data_config = self.cfg.get_data_config()
        validation = DataValidation(
            config=data_config,
            ingestion_artifact=ingestion_artifact,
        )
        artifact = validation.initiate_data_validation()
        logger.info(
            f"Validation done in {time.perf_counter()-start:.2f}s — "
            f"status={artifact.validation_status}, "
            f"issues={len(artifact.issues)}"
        )
        return artifact

    def _run_transformation(
        self, ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        logger.info("STEP 3/7: Data Transformation")
        start = time.perf_counter()
        data_config = self.cfg.get_data_config()
        feature_config = self.cfg.get_feature_config()
        transformation = DataTransformation(
            config=data_config,
            feature_config=feature_config,
            ingestion_artifact=ingestion_artifact,
        )
        artifact = transformation.initiate_data_transformation()
        logger.info(
            f"Transformation done in {time.perf_counter()-start:.2f}s — "
            f"{artifact.input_record_count} → {artifact.output_record_count} records"
        )
        return artifact

    def _run_feature_engineering(
        self, transformation_artifact: DataTransformationArtifact
    ) -> FeatureEngineeringArtifact:
        logger.info("STEP 4/7: Feature Engineering")
        start = time.perf_counter()
        feature_config = self.cfg.get_feature_config()
        feature_eng = FeatureEngineering(
            feature_config=feature_config,
            transformation_artifact=transformation_artifact,
        )
        artifact = feature_eng.initiate_feature_engineering()
        logger.info(
            f"Feature Engineering done in {time.perf_counter()-start:.2f}s — "
            f"{len(artifact.feature_columns)} features, "
            f"vocab_est={artifact.vocabulary_estimate}"
        )
        return artifact

    # ------------------------------------------------------------------
    # Phase 3: Model pipeline steps
    # ------------------------------------------------------------------

    def _run_model_training(
        self, feature_artifact: FeatureEngineeringArtifact
    ) -> ModelTrainingArtifact:
        logger.info("STEP 5/7: Model Training")
        start = time.perf_counter()
        model_config = self.cfg.get_model_config()
        mlflow_config = self.cfg.get_mlflow_config()

        trainer = ModelTrainer(
            config=model_config,
            feature_artifact=feature_artifact,
            mlflow_config={
                "tracking_uri": mlflow_config.tracking_uri,
                "experiment_name": mlflow_config.experiment_name,
            },
        )
        artifact = trainer.initiate_model_training()
        logger.info(
            f"Model Training done in {time.perf_counter()-start:.2f}s — "
            f"vocab={artifact.vocabulary_size}, shape={artifact.matrix_shape}"
        )
        return artifact

    def _run_model_evaluation(
        self, training_artifact: ModelTrainingArtifact
    ) -> ModelEvaluationArtifact:
        logger.info("STEP 6/7: Model Evaluation")
        start = time.perf_counter()
        eval_config = self.cfg.get_evaluation_config()
        evaluator = ModelEvaluator(
            config=eval_config,
            training_artifact=training_artifact,
        )
        artifact = evaluator.initiate_model_evaluation()
        logger.info(
            f"Model Evaluation done in {time.perf_counter()-start:.2f}s — "
            f"coverage={artifact.coverage:.3f}, diversity={artifact.diversity:.3f}, "
            f"threshold_met={artifact.meets_minimum_threshold}"
        )
        return artifact

    def _run_model_registry(
        self,
        training_artifact: ModelTrainingArtifact,
        evaluation_artifact: ModelEvaluationArtifact,
    ) -> ModelRegistryArtifact:
        logger.info("STEP 7/7: Model Registry")
        start = time.perf_counter()
        mlflow_config = self.cfg.get_mlflow_config()
        registry = ModelRegistry(
            mlflow_config=mlflow_config,
            training_artifact=training_artifact,
            evaluation_artifact=evaluation_artifact,
        )
        artifact = registry.initiate_model_registry()
        logger.info(
            f"Model Registry done in {time.perf_counter()-start:.2f}s — "
            f"{artifact.model_name} v{artifact.model_version} [{artifact.stage}]"
        )
        return artifact

    # ------------------------------------------------------------------
    # Pipeline summary
    # ------------------------------------------------------------------

    def _save_pipeline_summary(
        self,
        ingestion: DataIngestionArtifact,
        validation: DataValidationArtifact,
        transformation: DataTransformationArtifact,
        feature_engineering: FeatureEngineeringArtifact,
        model_training: Optional[ModelTrainingArtifact],
        model_evaluation: Optional[ModelEvaluationArtifact],
        model_registry: Optional[ModelRegistryArtifact],
        pipeline_duration: float,
        success: bool,
        error_message: Optional[str],
    ) -> None:
        """Save a structured JSON summary of the pipeline run."""
        create_directories([Path("artifacts")])

        summary = {
            "pipeline_status": "success" if success else "failed",
            "pipeline_timestamp": get_iso_timestamp(),
            "pipeline_duration_seconds": round(pipeline_duration, 2),
            "error_message": error_message,
            "ingestion": {
                "source": ingestion.source,
                "raw_filepath": ingestion.raw_filepath,
                "record_count": ingestion.record_count,
                "column_count": ingestion.column_count,
                "file_size_mb": ingestion.file_size_mb,
                "dataset_version": ingestion.dataset_version,
                "is_fallback": ingestion.is_fallback,
                "timestamp": ingestion.ingestion_timestamp,
            },
            "validation": {
                "status": validation.validation_status,
                "issues_count": len(validation.issues),
                "record_count": validation.validated_record_count,
                "missing_values": validation.missing_values_count,
                "duplicates": validation.duplicate_count,
                "schema_valid": validation.schema_valid,
                "report_path": validation.validation_report_path,
            },
            "transformation": {
                "input_records": transformation.input_record_count,
                "output_records": transformation.output_record_count,
                "removed_duplicates": transformation.removed_duplicates,
                "filled_nulls": transformation.filled_nulls,
                "dropped_nulls": transformation.dropped_nulls,
                "text_columns_cleaned": transformation.text_columns_cleaned,
                "processed_path": transformation.processed_filepath,
            },
            "feature_engineering": {
                "feature_count": len(feature_engineering.feature_columns),
                "feature_columns": feature_engineering.feature_columns,
                "combined_text_column": feature_engineering.combined_text_column,
                "vocabulary_estimate": feature_engineering.vocabulary_estimate,
                "price_bucket_distribution": feature_engineering.price_bucket_distribution,
                "feature_path": feature_engineering.feature_filepath,
            },
        }

        if model_training:
            summary["model_training"] = {
                "vocabulary_size": model_training.vocabulary_size,
                "matrix_shape": list(model_training.matrix_shape),
                "training_duration_s": model_training.training_duration_seconds,
                "dataset_version": model_training.dataset_version,
                "mlflow_run_id": model_training.mlflow_run_id,
                "artifacts": {
                    "vectorizer": model_training.tfidf_model_path,
                    "similarity_matrix": model_training.similarity_matrix_path,
                    "product_index": model_training.product_index_path,
                    "metadata": model_training.metadata_path,
                },
            }

        if model_evaluation:
            summary["model_evaluation"] = {
                "coverage": model_evaluation.coverage,
                "diversity": model_evaluation.diversity,
                "novelty": model_evaluation.novelty,
                "intra_list_similarity": model_evaluation.intra_list_similarity,
                "meets_threshold": model_evaluation.meets_minimum_threshold,
                "report_path": model_evaluation.evaluation_report_path,
                "precision_recall": [
                    {"k": pr.k, "precision": pr.precision, "recall": pr.recall}
                    for pr in model_evaluation.precision_recall
                ],
            }

        if model_registry:
            summary["model_registry"] = {
                "registered": model_registry.registered,
                "model_name": model_registry.model_name,
                "model_version": model_registry.model_version,
                "stage": model_registry.stage,
                "registry_uri": model_registry.registry_uri,
                "promoted": model_registry.promoted,
            }

        save_json(Path("artifacts/pipeline_summary.json"), summary)
        logger.info("Pipeline summary saved: artifacts/pipeline_summary.json")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> TrainingPipelineArtifact:
        """
        Execute the full training pipeline.

        Phase 2: Data Ingestion → Validation → Transformation → Feature Engineering
        Phase 3: Model Training → Evaluation → Registry

        Returns:
            TrainingPipelineArtifact with all component results.

        Raises:
            DatasetNotFoundError: If Kaggle dataset is missing and fallback is disabled.
            DataValidationError: If data has unacceptable quality issues.
            CustomException: For any other pipeline failure.
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("  TRAINING PIPELINE — Starting")
        logger.info(f"  Timestamp: {get_iso_timestamp()}")
        logger.info("=" * 70)

        pipeline_start = time.perf_counter()
        error_message: Optional[str] = None

        # These will be populated as we progress
        ingestion_artifact: Optional[DataIngestionArtifact] = None
        validation_artifact: Optional[DataValidationArtifact] = None
        transformation_artifact: Optional[DataTransformationArtifact] = None
        feature_artifact: Optional[FeatureEngineeringArtifact] = None
        training_artifact: Optional[ModelTrainingArtifact] = None
        evaluation_artifact: Optional[ModelEvaluationArtifact] = None
        registry_artifact: Optional[ModelRegistryArtifact] = None

        try:
            # ── Phase 2: Data Pipeline ────────────────────────────────────
            ingestion_artifact = self._run_ingestion()
            validation_artifact = self._run_validation(ingestion_artifact)
            transformation_artifact = self._run_transformation(ingestion_artifact)
            feature_artifact = self._run_feature_engineering(transformation_artifact)

            # ── Phase 3: Model Pipeline ───────────────────────────────────
            if not self.skip_model_training:
                training_artifact = self._run_model_training(feature_artifact)
                evaluation_artifact = self._run_model_evaluation(training_artifact)
                registry_artifact = self._run_model_registry(
                    training_artifact, evaluation_artifact
                )

            pipeline_duration = time.perf_counter() - pipeline_start
            success = True

        except (DatasetNotFoundError, DataValidationError) as e:
            pipeline_duration = time.perf_counter() - pipeline_start
            error_message = str(e)
            success = False
            logger.error(f"Pipeline halted: {type(e).__name__}: {e}")

            if ingestion_artifact and validation_artifact and transformation_artifact and feature_artifact:
                self._save_pipeline_summary(
                    ingestion=ingestion_artifact,
                    validation=validation_artifact,
                    transformation=transformation_artifact,
                    feature_engineering=feature_artifact,
                    model_training=training_artifact,
                    model_evaluation=evaluation_artifact,
                    model_registry=registry_artifact,
                    pipeline_duration=pipeline_duration,
                    success=False,
                    error_message=error_message,
                )
            raise

        except Exception as e:
            pipeline_duration = time.perf_counter() - pipeline_start
            error_message = str(e)
            success = False
            logger.error(f"Pipeline failed with unexpected error: {e}")
            raise CustomException(str(e), sys.exc_info()) from e

        # ── Save summary ─────────────────────────────────────────────────
        self._save_pipeline_summary(
            ingestion=ingestion_artifact,
            validation=validation_artifact,
            transformation=transformation_artifact,
            feature_engineering=feature_artifact,
            model_training=training_artifact,
            model_evaluation=evaluation_artifact,
            model_registry=registry_artifact,
            pipeline_duration=pipeline_duration,
            success=success,
            error_message=error_message,
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  TRAINING PIPELINE — COMPLETE ({pipeline_duration:.2f}s)")
        logger.info("=" * 70)

        return TrainingPipelineArtifact(
            ingestion=ingestion_artifact,
            validation=validation_artifact,
            transformation=transformation_artifact,
            feature_engineering=feature_artifact,
            model_training=training_artifact,
            model_evaluation=evaluation_artifact,
            model_registry=registry_artifact,
            pipeline_duration_seconds=round(pipeline_duration, 2),
            success=success,
            error_message=error_message,
        )


# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    result = pipeline.run()
    if result.success:
        print("\nPipeline completed successfully.")
        if result.model_training:
            print(f"  Model artifacts saved to: models/")
            print(f"  Vocabulary size: {result.model_training.vocabulary_size:,}")
            print(f"  Matrix shape: {result.model_training.matrix_shape}")
        if result.model_evaluation:
            for pr in result.model_evaluation.precision_recall:
                print(f"  Precision@{pr.k}: {pr.precision:.4f}")
                print(f"  Recall@{pr.k}: {pr.recall:.4f}")
            print(f"  Coverage: {result.model_evaluation.coverage:.4f}")
            print(f"  Diversity: {result.model_evaluation.diversity:.4f}")
        if result.model_registry:
            print(
                f"  Registered: {result.model_registry.model_name} "
                f"v{result.model_registry.model_version} [{result.model_registry.stage}]"
            )
    else:
        print(f"\nPipeline FAILED: {result.error_message}")
        sys.exit(1)
