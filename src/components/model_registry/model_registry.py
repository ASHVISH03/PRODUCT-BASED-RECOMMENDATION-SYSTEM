"""
Model Registry
===============
Registers the trained model with MLflow Model Registry and manages
version lifecycle (Staging → Production auto-promotion).

If MLflow is not available, falls back to a local JSON registry file
at artifacts/model_registry.json for basic version tracking.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelRegistryArtifact,
    ModelTrainingArtifact,
)
from src.entity.config_entity import MLflowConfig
from src.exception.exception import CustomException
from src.logger.training_logger import get_training_logger
from src.utils.common import get_iso_timestamp, save_json

logger = get_training_logger()


class ModelRegistry:
    """
    Handles model registration and version management.

    Args:
        mlflow_config: MLflow configuration entity.
        training_artifact: Output of ModelTrainer.
        evaluation_artifact: Output of ModelEvaluator.
    """

    def __init__(
        self,
        mlflow_config: MLflowConfig,
        training_artifact: ModelTrainingArtifact,
        evaluation_artifact: ModelEvaluationArtifact,
    ) -> None:
        self.mlflow_config = mlflow_config
        self.training_artifact = training_artifact
        self.evaluation_artifact = evaluation_artifact

    # ------------------------------------------------------------------
    # MLflow Registry helpers
    # ------------------------------------------------------------------

    def _register_with_mlflow(self) -> Optional[Dict[str, Any]]:
        """
        Register model in MLflow Model Registry.

        Returns:
            Dict with model_name, model_version, stage, registry_uri
            or None if registration fails.
        """
        if not self.training_artifact.mlflow_run_id:
            logger.warning("No MLflow run_id available — skipping registry registration")
            return None

        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            model_name = self.mlflow_config.model_name
            run_id = self.training_artifact.mlflow_run_id
            tracking_uri = self.mlflow_config.tracking_uri

            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()

            # Ensure registered model exists
            try:
                client.get_registered_model(model_name)
            except Exception:
                client.create_registered_model(
                    model_name,
                    description="TF-IDF Content-Based Product Recommendation Model",
                )
                logger.info(f"Created registered model: {model_name}")

            # Register a new version pointing to the artifacts
            source = f"runs:/{run_id}/model_artifacts"
            model_version = mlflow.register_model(source, model_name)

            version_number = str(model_version.version)
            logger.info(f"Registered model version: {model_name} v{version_number}")

            # Auto-promote to Staging
            client.transition_model_version_stage(
                name=model_name,
                version=version_number,
                stage=self.mlflow_config.stages.staging,
                archive_existing_versions=False,
            )
            current_stage = self.mlflow_config.stages.staging
            logger.info(f"Model transitioned to: {current_stage}")

            # Check if auto-promotion to Production is warranted
            promoted = False
            if self.evaluation_artifact.meets_minimum_threshold:
                # Check if there's a production model to compare against
                production_versions = client.get_latest_versions(
                    model_name, stages=[self.mlflow_config.stages.production]
                )
                if not production_versions:
                    # No production model yet — promote directly
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version_number,
                        stage=self.mlflow_config.stages.production,
                        archive_existing_versions=True,
                    )
                    current_stage = self.mlflow_config.stages.production
                    promoted = True
                    logger.info("Auto-promoted to Production (first production model)")
                else:
                    logger.info(
                        "Existing production model found. Manual promotion required."
                    )

            registry_uri = f"models:/{model_name}/{version_number}"
            return {
                "model_name": model_name,
                "model_version": version_number,
                "stage": current_stage,
                "registry_uri": registry_uri,
                "promoted": promoted,
            }

        except Exception as e:
            logger.warning(f"MLflow registration failed: {e}")
            return None

    def _register_locally(self) -> Dict[str, Any]:
        """
        Fallback local registry (JSON file) when MLflow is unavailable.
        """
        registry_path = Path("artifacts/model_registry.json")
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing registry
        if registry_path.exists():
            try:
                with open(registry_path, encoding="utf-8") as f:
                    registry = json.load(f)
            except Exception:
                registry = {"versions": []}
        else:
            registry = {"versions": []}

        # Determine next version number
        existing_versions = [int(v.get("version", 0)) for v in registry["versions"]]
        next_version = str(max(existing_versions, default=0) + 1)

        # Create version entry
        entry: Dict[str, Any] = {
            "version": next_version,
            "registered_at": get_iso_timestamp(),
            "dataset_version": self.training_artifact.dataset_version,
            "vocabulary_size": self.training_artifact.vocabulary_size,
            "matrix_shape": list(self.training_artifact.matrix_shape),
            "training_duration_s": self.training_artifact.training_duration_seconds,
            "evaluation": {
                "coverage": self.evaluation_artifact.coverage,
                "diversity": self.evaluation_artifact.diversity,
                "novelty": self.evaluation_artifact.novelty,
                "meets_threshold": self.evaluation_artifact.meets_minimum_threshold,
            },
            "stage": "Staging",
            "artifact_paths": {
                "vectorizer": self.training_artifact.tfidf_model_path,
                "similarity_matrix": self.training_artifact.similarity_matrix_path,
                "product_index": self.training_artifact.product_index_path,
                "metadata": self.training_artifact.metadata_path,
            },
        }

        registry["versions"].append(entry)
        registry["latest_version"] = next_version
        registry["latest_stage"] = "Staging"

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Local registry updated: {registry_path} (version {next_version})")

        return {
            "model_name": "recommendation-model",
            "model_version": next_version,
            "stage": "Staging",
            "registry_uri": f"local://artifacts/model_registry.json@{next_version}",
            "promoted": False,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initiate_model_registry(self) -> ModelRegistryArtifact:
        """
        Register the trained model and manage version lifecycle.

        Returns:
            ModelRegistryArtifact with registration details.
        """
        logger.info("=" * 60)
        logger.info("  MODEL REGISTRY — Phase 3")
        logger.info("=" * 60)

        try:
            # Try MLflow first
            mlflow_result = self._register_with_mlflow()

            if mlflow_result:
                registered = True
                model_name = mlflow_result["model_name"]
                version = mlflow_result["model_version"]
                stage = mlflow_result["stage"]
                registry_uri = mlflow_result["registry_uri"]
                promoted = mlflow_result["promoted"]
                logger.info(f"MLflow registry: {model_name} v{version} ({stage})")
            else:
                # Fallback to local registry
                local_result = self._register_locally()
                registered = True
                model_name = local_result["model_name"]
                version = local_result["model_version"]
                stage = local_result["stage"]
                registry_uri = local_result["registry_uri"]
                promoted = local_result["promoted"]
                logger.info(f"Local registry: {model_name} v{version} ({stage})")

            logger.info("=" * 60)
            logger.info(f"  REGISTRY COMPLETE: {model_name} v{version} [{stage}]")
            logger.info("=" * 60)

            return ModelRegistryArtifact(
                registered=registered,
                model_name=model_name,
                model_version=version,
                stage=stage,
                registry_uri=registry_uri,
                promoted=promoted,
            )

        except Exception as e:
            raise CustomException(
                f"ModelRegistry.initiate_model_registry failed: {e}",
                sys.exc_info(),
            ) from e
