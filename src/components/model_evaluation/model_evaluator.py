"""
Model Evaluator
================
Computes recommendation quality metrics without ground-truth labels
by using category-based relevance as a proxy.

Metrics computed:
    Precision@K  — fraction of top-K recommendations in same category as query
    Recall@K     — fraction of same-category products that appear in top-K
    Coverage     — % of catalog products that get recommended at least once
    Diversity    — average pairwise cosine distance within a recommendation list
    Novelty      — inverse-popularity-weighted novelty score
    Intra-List Similarity — avg cosine sim within recommendation set
    Inference Latency — median inference time in milliseconds

Output:
    artifacts/evaluation_report.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainingArtifact,
    PrecisionRecallAtK,
)
from src.entity.config_entity import EvaluationConfig
from src.exception.exception import CustomException
from src.logger.training_logger import get_training_logger
from src.utils.common import get_iso_timestamp, save_json

logger = get_training_logger()


class ModelEvaluator:
    """
    Evaluates recommendation quality of the trained TF-IDF model.

    Uses category-based relevance: a recommendation is "relevant" if the
    recommended product shares the same category_l1 as the query product.

    Args:
        config: EvaluationConfig (k_values, sample sizes, thresholds).
        training_artifact: ModelTrainingArtifact from the model trainer.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        training_artifact: ModelTrainingArtifact,
    ) -> None:
        self.config = config
        self.training_artifact = training_artifact
        self._df: Optional[pd.DataFrame] = None
        self._sim_matrix: Optional[np.ndarray] = None
        self._id_to_idx: Optional[Dict[str, int]] = None
        self._idx_to_id: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        """Load all model artifacts into memory for evaluation."""
        import pickle

        logger.info("Loading model artifacts for evaluation...")

        # Load products
        products_path = (
            Path(self.training_artifact.tfidf_model_path).parent / "processed_products.csv"
        )
        self._df = pd.read_csv(products_path, encoding="utf-8", low_memory=False)
        logger.info(f"Loaded {len(self._df)} products for evaluation")

        # Load similarity matrix
        with open(self.training_artifact.similarity_matrix_path, "rb") as f:
            self._sim_matrix = pickle.load(f)
        logger.info(f"Loaded similarity matrix: {self._sim_matrix.shape}")

        # Load product index
        with open(self.training_artifact.product_index_path, "rb") as f:
            product_index = pickle.load(f)

        self._id_to_idx = product_index["id_to_idx"]
        self._idx_to_id = {int(k): v for k, v in product_index["idx_to_id"].items()}
        logger.info(f"Loaded product index: {len(self._id_to_idx)} entries")

    # ------------------------------------------------------------------
    # Individual metric computations
    # ------------------------------------------------------------------

    def _get_top_k_recommendations(
        self, product_id: str, k: int
    ) -> List[str]:
        """Return top-K recommended product IDs for a given product."""
        if product_id not in self._id_to_idx:
            return []

        idx = self._id_to_idx[product_id]
        sim_scores = self._sim_matrix[idx].copy()
        sim_scores[idx] = 0.0  # exclude self

        top_indices = np.argsort(sim_scores)[::-1][:k]
        return [
            self._idx_to_id[int(i)]
            for i in top_indices
            if sim_scores[i] > 0 and int(i) in self._idx_to_id
        ]

    def _compute_precision_recall_at_k(
        self, sample_size: int, k: int
    ) -> Tuple[float, float]:
        """
        Compute Precision@K and Recall@K over a random sample.

        Relevance criterion: recommended product shares category_l1 with query.

        Args:
            sample_size: Number of query products to sample.
            k: Recommendation list length.

        Returns:
            (precision_at_k, recall_at_k)
        """
        # Build category lookup
        cat_lookup: Dict[str, str] = {}
        cat_members: Dict[str, List[str]] = {}
        for _, row in self._df.iterrows():
            pid = str(row.get("product_id", ""))
            cat1 = str(row.get("category_l1", "Unknown"))
            cat_lookup[pid] = cat1
            cat_members.setdefault(cat1, []).append(pid)

        all_ids = list(self._id_to_idx.keys())
        sample_ids = random.sample(all_ids, min(sample_size, len(all_ids)))

        precisions, recalls = [], []
        for query_id in sample_ids:
            query_cat = cat_lookup.get(query_id, "Unknown")
            relevant_ids = set(cat_members.get(query_cat, [])) - {query_id}

            if not relevant_ids:
                continue

            recs = set(self._get_top_k_recommendations(query_id, k))
            if not recs:
                continue

            hits = len(recs & relevant_ids)
            precisions.append(hits / k)
            recalls.append(hits / min(len(relevant_ids), k))

        precision = float(np.mean(precisions)) if precisions else 0.0
        recall = float(np.mean(recalls)) if recalls else 0.0
        return round(precision, 4), round(recall, 4)

    def _compute_coverage(self, sample_size: int, k: int) -> float:
        """
        Compute catalog coverage: fraction of products recommended at least once.

        Args:
            sample_size: Number of queries to run.
            k: Recommendation list length.

        Returns:
            Coverage in [0, 1].
        """
        all_ids = list(self._id_to_idx.keys())
        sample_ids = random.sample(all_ids, min(sample_size, len(all_ids)))

        recommended: Set[str] = set()
        for query_id in sample_ids:
            recs = self._get_top_k_recommendations(query_id, k)
            recommended.update(recs)

        catalog_size = len(all_ids)
        coverage = len(recommended) / catalog_size if catalog_size > 0 else 0.0
        return round(coverage, 4)

    def _compute_diversity(self, sample_size: int, k: int) -> float:
        """
        Compute average intra-list diversity (1 - avg pairwise similarity).

        Args:
            sample_size: Number of queries to sample.
            k: Recommendation list length.

        Returns:
            Diversity in [0, 1] (1 = maximally diverse).
        """
        all_ids = list(self._id_to_idx.keys())
        sample_ids = random.sample(all_ids, min(sample_size, len(all_ids)))

        diversity_scores = []
        for query_id in sample_ids:
            rec_ids = self._get_top_k_recommendations(query_id, k)
            if len(rec_ids) < 2:
                continue

            rec_indices = [
                self._id_to_idx[pid] for pid in rec_ids if pid in self._id_to_idx
            ]
            if len(rec_indices) < 2:
                continue

            # Pairwise similarities within the recommendation set
            pairwise_sims = []
            for i in range(len(rec_indices)):
                for j in range(i + 1, len(rec_indices)):
                    pairwise_sims.append(
                        float(self._sim_matrix[rec_indices[i], rec_indices[j]])
                    )

            avg_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0
            diversity_scores.append(1.0 - avg_sim)

        return round(float(np.mean(diversity_scores)) if diversity_scores else 0.0, 4)

    def _compute_novelty(self, sample_size: int, k: int) -> float:
        """
        Compute novelty as inverse-popularity of recommended items.

        Less popular recommendations = higher novelty.
        """
        if "popularity_score" not in self._df.columns:
            return 0.5

        pop_lookup = dict(zip(
            self._df["product_id"].astype(str),
            self._df["popularity_score"].fillna(0)
        ))

        all_ids = list(self._id_to_idx.keys())
        sample_ids = random.sample(all_ids, min(sample_size, len(all_ids)))

        novelty_scores = []
        for query_id in sample_ids:
            recs = self._get_top_k_recommendations(query_id, k)
            if not recs:
                continue
            avg_pop = float(np.mean([pop_lookup.get(pid, 0) for pid in recs]))
            novelty_scores.append(1.0 - avg_pop)

        return round(float(np.mean(novelty_scores)) if novelty_scores else 0.5, 4)

    def _compute_avg_similarity(self, sample_size: int, k: int) -> float:
        """Compute average cosine similarity of top-K recommendations."""
        all_ids = list(self._id_to_idx.keys())
        sample_ids = random.sample(all_ids, min(sample_size, len(all_ids)))

        avg_sims = []
        for query_id in sample_ids:
            idx = self._id_to_idx.get(query_id)
            if idx is None:
                continue
            sim_scores = self._sim_matrix[idx].copy()
            sim_scores[idx] = 0.0
            top_scores = np.sort(sim_scores)[::-1][:k]
            top_scores = top_scores[top_scores > 0]
            if len(top_scores) > 0:
                avg_sims.append(float(np.mean(top_scores)))

        return round(float(np.mean(avg_sims)) if avg_sims else 0.0, 4)

    def _compute_intra_list_similarity(self, sample_size: int, k: int) -> float:
        """Average cosine similarity within each recommendation list."""
        all_ids = list(self._id_to_idx.keys())
        sample_ids = random.sample(all_ids, min(sample_size, len(all_ids)))

        ils_scores = []
        for query_id in sample_ids:
            rec_ids = self._get_top_k_recommendations(query_id, k)
            rec_indices = [
                self._id_to_idx[pid] for pid in rec_ids if pid in self._id_to_idx
            ]
            if len(rec_indices) < 2:
                continue

            sims = []
            for i in range(len(rec_indices)):
                for j in range(i + 1, len(rec_indices)):
                    sims.append(float(self._sim_matrix[rec_indices[i], rec_indices[j]]))

            if sims:
                ils_scores.append(float(np.mean(sims)))

        return round(float(np.mean(ils_scores)) if ils_scores else 0.0, 4)

    def _measure_inference_latency(
        self, n_trials: int = 50, k: int = 10
    ) -> float:
        """
        Measure median inference latency in milliseconds.

        Args:
            n_trials: Number of random queries to time.
            k: Recommendation list length.

        Returns:
            Median inference latency in ms.
        """
        all_ids = list(self._id_to_idx.keys())
        trial_ids = random.sample(all_ids, min(n_trials, len(all_ids)))

        latencies = []
        for pid in trial_ids:
            t0 = time.perf_counter()
            self._get_top_k_recommendations(pid, k)
            latencies.append((time.perf_counter() - t0) * 1000)

        return round(float(np.median(latencies)), 3)

    def _log_evaluation_to_mlflow(
        self,
        metrics: Dict[str, Any],
        run_id: Optional[str],
    ) -> None:
        """Log evaluation metrics to the existing MLflow run."""
        if run_id is None:
            return
        try:
            import mlflow

            with mlflow.start_run(run_id=run_id):
                flat_metrics = {}
                for k_val in self.config.k_values:
                    flat_metrics[f"precision_at_{k_val}"] = metrics.get(
                        f"precision_at_{k_val}", 0.0
                    )
                    flat_metrics[f"recall_at_{k_val}"] = metrics.get(
                        f"recall_at_{k_val}", 0.0
                    )
                flat_metrics["coverage"] = metrics.get("coverage", 0.0)
                flat_metrics["diversity"] = metrics.get("diversity", 0.0)
                flat_metrics["novelty"] = metrics.get("novelty", 0.0)
                flat_metrics["intra_list_similarity"] = metrics.get(
                    "intra_list_similarity", 0.0
                )
                flat_metrics["inference_latency_ms"] = metrics.get(
                    "inference_latency_ms", 0.0
                )
                mlflow.log_metrics(flat_metrics)
                logger.info(f"Evaluation metrics logged to MLflow run: {run_id}")
        except Exception as e:
            logger.warning(f"MLflow evaluation logging failed: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Run the complete evaluation suite and save the report.

        Returns:
            ModelEvaluationArtifact with all metric results.
        """
        logger.info("=" * 60)
        logger.info("  MODEL EVALUATION — Phase 3")
        logger.info("=" * 60)

        try:
            self._load_artifacts()

            k_values = self.config.k_values or [5, 10]
            sample_size = min(self.config.diversity_sample_size, len(self._df))
            pr_results = []
            all_metrics: Dict[str, Any] = {}

            logger.info(f"Evaluating with k_values={k_values}, sample_size={sample_size}")

            for k in k_values:
                logger.info(f"Computing Precision@{k} and Recall@{k}...")
                prec, rec = self._compute_precision_recall_at_k(sample_size, k)
                pr_results.append(PrecisionRecallAtK(k=k, precision=prec, recall=rec))
                all_metrics[f"precision_at_{k}"] = prec
                all_metrics[f"recall_at_{k}"] = rec
                logger.info(f"  Precision@{k}={prec:.4f}, Recall@{k}={rec:.4f}")

            default_k = k_values[0] if k_values else 10

            logger.info("Computing Coverage...")
            coverage = self._compute_coverage(sample_size, default_k)
            logger.info(f"  Coverage={coverage:.4f}")

            logger.info("Computing Diversity...")
            diversity = self._compute_diversity(sample_size, default_k)
            logger.info(f"  Diversity={diversity:.4f}")

            logger.info("Computing Novelty...")
            novelty = self._compute_novelty(sample_size, default_k)
            logger.info(f"  Novelty={novelty:.4f}")

            logger.info("Computing Intra-List Similarity...")
            ils = self._compute_intra_list_similarity(sample_size, default_k)
            logger.info(f"  ILS={ils:.4f}")

            logger.info("Measuring Inference Latency...")
            latency_ms = self._measure_inference_latency(n_trials=100, k=default_k)
            logger.info(f"  Median Latency={latency_ms:.3f}ms")

            all_metrics.update({
                "coverage": coverage,
                "diversity": diversity,
                "novelty": novelty,
                "intra_list_similarity": ils,
                "inference_latency_ms": latency_ms,
            })

            # Check minimum threshold (Precision@5 or first k_value)
            threshold_prec = all_metrics.get(f"precision_at_{k_values[0]}", 0.0)
            min_required = self.config.min_precision_at_5
            meets_threshold = threshold_prec >= min_required

            if meets_threshold:
                logger.info(
                    f"Threshold check PASSED: Precision@{k_values[0]}={threshold_prec:.4f} >= {min_required}"
                )
            else:
                logger.warning(
                    f"Threshold check FAILED: Precision@{k_values[0]}={threshold_prec:.4f} < {min_required}"
                )

            # Build evaluation report
            report = {
                "evaluation_timestamp": get_iso_timestamp(),
                "dataset_version": self.training_artifact.dataset_version,
                "mlflow_run_id": self.training_artifact.mlflow_run_id,
                "model_params": {
                    "vocabulary_size": self.training_artifact.vocabulary_size,
                    "matrix_shape": list(self.training_artifact.matrix_shape),
                    "training_duration_s": self.training_artifact.training_duration_seconds,
                },
                "metrics": all_metrics,
                "precision_recall_by_k": [
                    {"k": pr.k, "precision": pr.precision, "recall": pr.recall}
                    for pr in pr_results
                ],
                "thresholds": {
                    "min_precision_required": min_required,
                    "precision_achieved": threshold_prec,
                    "meets_threshold": meets_threshold,
                },
                "sample_config": {
                    "sample_size": sample_size,
                    "k_values": k_values,
                },
            }

            # Save report
            Path("artifacts").mkdir(parents=True, exist_ok=True)
            report_path = "artifacts/evaluation_report.json"
            save_json(Path(report_path), report)
            logger.info(f"Evaluation report saved: {report_path}")

            # Log to MLflow
            self._log_evaluation_to_mlflow(
                all_metrics, self.training_artifact.mlflow_run_id
            )

            logger.info("=" * 60)
            logger.info("  MODEL EVALUATION COMPLETE")
            for k, v in all_metrics.items():
                logger.info(f"  {k:35s}: {v}")
            logger.info("=" * 60)

            return ModelEvaluationArtifact(
                precision_recall=pr_results,
                coverage=coverage,
                diversity=diversity,
                novelty=novelty,
                intra_list_similarity=ils,
                evaluation_report_path=report_path,
                meets_minimum_threshold=meets_threshold,
                mlflow_run_id=self.training_artifact.mlflow_run_id or "",
            )

        except Exception as e:
            raise CustomException(
                f"ModelEvaluator.initiate_model_evaluation failed: {e}",
                sys.exc_info(),
            ) from e
