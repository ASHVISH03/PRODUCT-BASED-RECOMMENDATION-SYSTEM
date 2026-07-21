"""
Recommendation Evaluator
=========================
Offline evaluation framework for recommendation quality.

Metrics implemented:
    - Precision@K
    - Recall@K
    - NDCG@K
    - Hit Rate@K
    - Catalog Coverage
    - Intra-List Diversity (mean pairwise distance)

Design principles:
    - Only computes metrics that are valid for the available data.
    - Does NOT fabricate ground-truth labels.
    - Intra-list diversity uses the precomputed similarity matrix — no fake labels needed.
    - All metrics are MLflow-compatible (returns flat Dict[str, float]).

MLflow integration:
    Designed to be called from RecommendationService or a test pipeline.
    Results should be logged via mlflow.log_metrics(evaluator.evaluate(...))
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """
    Offline recommendation evaluation.

    Args:
        catalog_size: Total number of products in the catalog.
        sim_matrix:   Optional precomputed cosine similarity matrix for diversity.
        product_index: Dict mapping product_id → row index (for diversity metric).
    """

    def __init__(
        self,
        catalog_size: int,
        sim_matrix: Optional[np.ndarray] = None,
        product_index: Optional[Dict[str, int]] = None,
    ) -> None:
        self.catalog_size = catalog_size
        self._sim_matrix = sim_matrix
        self._product_index = product_index or {}

    # -----------------------------------------------------------------------
    # Individual metrics
    # -----------------------------------------------------------------------

    def precision_at_k(self, recommended: List[str], relevant: List[str]) -> float:
        """
        Precision@K: fraction of recommended items that are relevant.
        P@K = |recommended ∩ relevant| / |recommended|

        Args:
            recommended: Ordered list of recommended product IDs (already top-K).
            relevant:    Ground-truth relevant product IDs.

        Returns:
            Float in [0, 1].
        """
        if not recommended:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for r in recommended if r in relevant_set)
        return hits / len(recommended)

    def recall_at_k(self, recommended: List[str], relevant: List[str]) -> float:
        """
        Recall@K: fraction of all relevant items found in top-K.
        R@K = |recommended ∩ relevant| / |relevant|

        Args:
            recommended: Ordered list of recommended product IDs.
            relevant:    Ground-truth relevant product IDs.

        Returns:
            Float in [0, 1].
        """
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for r in recommended if r in relevant_set)
        return hits / len(relevant)

    def ndcg_at_k(self, recommended: List[str], relevant: List[str]) -> float:
        """
        Normalised Discounted Cumulative Gain @K.
        Binary relevance: 1 if in relevant, 0 otherwise.

        Args:
            recommended: Ordered list of recommended product IDs.
            relevant:    Ground-truth relevant product IDs.

        Returns:
            Float in [0, 1].
        """
        if not recommended or not relevant:
            return 0.0

        relevant_set = set(relevant)
        k = len(recommended)

        # DCG
        dcg = 0.0
        for i, pid in enumerate(recommended, start=1):
            if pid in relevant_set:
                dcg += 1.0 / math.log2(i + 1)

        # Ideal DCG
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

        return dcg / idcg if idcg > 0 else 0.0

    def hit_rate(self, recommended: List[str], relevant: List[str]) -> float:
        """
        Hit Rate@K: 1 if any recommended item is relevant, 0 otherwise.
        Useful as a binary signal per session.

        Args:
            recommended: List of recommended product IDs.
            relevant:    Ground-truth relevant product IDs.

        Returns:
            1.0 or 0.0.
        """
        relevant_set = set(relevant)
        return 1.0 if any(r in relevant_set for r in recommended) else 0.0

    def coverage(self, recommended: List[str]) -> float:
        """
        Catalog coverage: fraction of the catalog appearing in recommended.
        High coverage indicates recommendation diversity across the catalog.

        Args:
            recommended: Flat list of all recommended product IDs across sessions.

        Returns:
            Float in [0, 1].
        """
        if self.catalog_size == 0:
            return 0.0
        unique_recommended = len(set(recommended))
        return unique_recommended / self.catalog_size

    def intra_list_diversity(self, recommended_ids: List[str]) -> float:
        """
        Intra-List Diversity: mean pairwise dissimilarity within the recommendation list.
        Dissimilarity = 1 - cosine_similarity.

        Does NOT require ground-truth labels — uses the precomputed similarity matrix.
        Higher values indicate more diverse recommendations.

        Args:
            recommended_ids: List of recommended product IDs.

        Returns:
            Float in [0, 1]. Returns 1.0 (max diversity) if sim_matrix unavailable.
        """
        if self._sim_matrix is None or len(recommended_ids) < 2:
            return 1.0  # Cannot compute, assume max diversity

        idxs = [self._product_index.get(pid) for pid in recommended_ids]
        idxs = [i for i in idxs if i is not None]

        if len(idxs) < 2:
            return 1.0

        pairwise_sim: List[float] = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                sim = float(self._sim_matrix[idxs[i], idxs[j]])
                pairwise_sim.append(sim)

        mean_sim = float(np.mean(pairwise_sim)) if pairwise_sim else 0.0
        return 1.0 - mean_sim  # convert similarity → diversity

    # -----------------------------------------------------------------------
    # Session-level evaluation
    # -----------------------------------------------------------------------

    def evaluate_session(
        self,
        history_ids: List[str],
        recommended_ids: List[str],
        relevant_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute all available metrics for a single recommendation session.

        When relevant_ids is None (no ground-truth), only label-free metrics
        are computed (intra_list_diversity, coverage@session).

        Args:
            history_ids:      Products the user viewed (interaction history).
            recommended_ids:  Products returned by the recommender.
            relevant_ids:     Optional ground-truth relevant product IDs.

        Returns:
            Dict of metric_name → float, suitable for mlflow.log_metrics().
        """
        metrics: Dict[str, float] = {}
        k = len(recommended_ids)

        # Label-free metrics (always computable)
        metrics["diversity_intra_list"] = self.intra_list_diversity(recommended_ids)
        metrics["coverage_session"] = self.coverage(recommended_ids)
        metrics["k"] = float(k)

        # Check overlap: recommendations should NOT be in history (exclusion sanity)
        history_set = set(history_ids)
        overlap = sum(1 for pid in recommended_ids if pid in history_set)
        metrics["history_leakage"] = float(overlap)  # should be 0

        # Supervised metrics (only when ground truth is available)
        if relevant_ids:
            metrics[f"precision_at_{k}"] = self.precision_at_k(recommended_ids, relevant_ids)
            metrics[f"recall_at_{k}"]    = self.recall_at_k(recommended_ids, relevant_ids)
            metrics[f"ndcg_at_{k}"]      = self.ndcg_at_k(recommended_ids, relevant_ids)
            metrics[f"hit_rate_at_{k}"]  = self.hit_rate(recommended_ids, relevant_ids)

        logger.info(f"[RecommendationEvaluator] Session metrics: {metrics}")
        return metrics

    # -----------------------------------------------------------------------
    # Scenario comparison helper
    # -----------------------------------------------------------------------

    def compare_scenarios(
        self,
        scenarios: Dict[str, Dict],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple recommendation scenarios side by side.
        Useful for Scenario A/B/C verification.

        Args:
            scenarios: Dict of scenario_name → {
                "history": [...], "recommended": [...], "relevant": [...] (optional)
            }

        Returns:
            Dict of scenario_name → metrics dict.
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, data in scenarios.items():
            history = data.get("history", [])
            recommended = data.get("recommended", [])
            relevant = data.get("relevant", None)
            metrics = self.evaluate_session(history, recommended, relevant)
            results[name] = metrics
            logger.info(f"[Evaluator] Scenario '{name}': {metrics}")
        return results
