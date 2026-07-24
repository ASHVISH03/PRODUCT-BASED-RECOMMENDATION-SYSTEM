import logging

import pandas as pd

from src.monitoring.drift_detector import DriftDetector
from src.pipelines.training_pipeline import TrainingPipeline
from src.entity.artifact_entity import TrainingPipelineArtifact

logger = logging.getLogger(__name__)


def run_training_pipeline() -> None:
    """
    Run the training pipeline once.

    Shared by the manual POST /train endpoint and the background retrain
    scheduler — never raises, so a failed run never crashes its caller
    (a background task or a scheduled job).
    """
    try:
        pipeline = TrainingPipeline()
        artifact = pipeline.run()
        _record_training_history(artifact)
        _run_drift_check(artifact)
    except Exception as e:
        logger.error(f"Training pipeline run failed: {e}")


def _record_training_history(artifact: TrainingPipelineArtifact) -> None:
    """Persist this run's key metrics to the training_history table so
    model performance over time is queryable, not just log-scraped."""
    if not artifact.success or not artifact.model_training:
        return

    from app.database import SessionLocal
    from app.models import TrainingHistory

    precision_at_5 = None
    if artifact.model_evaluation:
        precision_at_5 = next(
            (pr.precision for pr in artifact.model_evaluation.precision_recall if pr.k == 5),
            None,
        )

    db = SessionLocal()
    try:
        db.add(
            TrainingHistory(
                run_id=artifact.model_training.mlflow_run_id,
                duration_seconds=artifact.pipeline_duration_seconds,
                vocab_size=artifact.model_training.vocabulary_size,
                dataset_rows=artifact.model_training.matrix_shape[0]
                if artifact.model_training.matrix_shape
                else None,
                metrics_precision_at_5=precision_at_5,
                metrics_coverage=artifact.model_evaluation.coverage
                if artifact.model_evaluation
                else None,
                model_version=artifact.model_registry.model_version
                if artifact.model_registry
                else None,
            )
        )
        db.commit()
    except Exception as e:
        logger.error(f"Failed to record training history: {e}")
        db.rollback()
    finally:
        db.close()


def _run_drift_check(artifact: TrainingPipelineArtifact) -> None:
    """Compare this run's dataset against the live-Production baseline.
    The baseline only advances when a model is actually promoted."""
    if not artifact.success or not artifact.feature_engineering:
        return

    try:
        df = pd.read_csv(artifact.feature_engineering.feature_filepath)
        detector = DriftDetector()
        report = detector.run(df)
        logger.info(f"Drift check: overall_grade={report.get('overall_grade')}")

        if artifact.model_registry and artifact.model_registry.promoted:
            detector.update_baseline(df)
            logger.info("Drift baseline advanced to newly-promoted model's dataset.")
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
