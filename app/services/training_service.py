import logging

from src.pipelines.training_pipeline import TrainingPipeline

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
        pipeline.run()
    except Exception as e:
        logger.error(f"Training pipeline run failed: {e}")
