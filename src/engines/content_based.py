"""
Content-Based Recommendation Engine
=====================================
Uses TF-IDF cosine similarity to recommend products based on
textual similarity (product_name + category + about_product).

This is the primary engine used in:
  - Similar Products section
  - Because You Viewed section
  - Recommended For You (primary signal)
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.engines.base_engine import BaseEngine, RecommendationResult, _enrich_result
from src.exception.exception import CustomException


class ContentBasedEngine(BaseEngine):
    """
    Recommends products using pre-computed TF-IDF cosine similarity.

    Requires a pre-computed similarity matrix and product index.
    Both are loaded from pickle files by the RecommendationEngine orchestrator
    and injected at construction time.

    Args:
        products_df: Feature-engineered DataFrame (features.csv).
        similarity_matrix: numpy ndarray of shape (n, n) with cosine scores.
        product_index: Dict mapping product_id → row index in the matrix.
    """

    def __init__(
        self,
        products_df: pd.DataFrame,
        similarity_matrix: np.ndarray,
        product_index: Dict[str, int],
    ) -> None:
        super().__init__(products_df)
        self._sim_matrix = similarity_matrix
        self._product_index = product_index

    @property
    def strategy_name(self) -> str:
        return "content_based"

    def _get_similarity_score(self, query_idx: int, candidate_idx: int) -> float:
        """Return cosine similarity between two matrix row indices."""
        try:
            return float(self._sim_matrix[query_idx, candidate_idx])
        except IndexError:
            return 0.0

    def _build_reason(
        self,
        query_row: pd.Series,
        candidate_row: pd.Series,
        score: float,
    ) -> Tuple[str, List[str]]:
        """
        Build a human-readable recommendation reason and tags.

        Args:
            query_row: DataFrame row for the query product.
            candidate_row: DataFrame row for the recommended product.
            score: Cosine similarity score.

        Returns:
            (reason_sentence, reason_tags)
        """
        tags: List[str] = []
        parts: List[str] = []

        q_brand = str(query_row.get("brand", ""))
        c_brand = str(candidate_row.get("brand", ""))
        q_cat1 = str(query_row.get("category_l1", ""))
        c_cat1 = str(candidate_row.get("category_l1", ""))
        q_cat2 = str(query_row.get("category_l2", ""))
        c_cat2 = str(candidate_row.get("category_l2", ""))

        if q_brand == c_brand and q_brand not in ("Unknown", "", "nan"):
            tags.append("Same Brand")
            parts.append(f"from {c_brand}")

        if q_cat2 == c_cat2 and q_cat2:
            tags.append("Same Category")
            parts.append(f"in {c_cat2}")
        elif q_cat1 == c_cat1 and q_cat1:
            tags.append("Similar Category")
            parts.append(f"in {c_cat1}")

        if score >= 0.85:
            tags.append("Highly Similar")
            parts.append("with very similar features")
        elif score >= 0.60:
            tags.append("Similar Specifications")
            parts.append("with similar specifications")

        c_rating = float(candidate_row.get("rating", 0) or 0)
        if c_rating >= 4.0:
            tags.append("Highly Rated")

        if not parts:
            reason = "Similar product based on content analysis"
        else:
            reason = "Recommended because it is " + ", ".join(parts)

        return reason, tags

    def recommend(
        self,
        product_id: str,
        top_k: int = 10,
        min_score: float = 0.0,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Return top-K most similar products by cosine similarity.

        Args:
            product_id: Query product identifier.
            top_k: Maximum results to return.
            min_score: Minimum cosine similarity threshold (0.0 = no filter).

        Returns:
            Sorted list of RecommendationResult (rank 1 = most similar).
        """
        try:
            if product_id not in self._product_index:
                return []

            query_idx = self._product_index[product_id]
            query_row = self._get_product_row(product_id)
            if query_row is None:
                return []

            # Get similarity scores for the query product
            sim_scores = self._sim_matrix[query_idx].copy()

            # Zero out self-similarity
            sim_scores[query_idx] = 0.0

            # Apply minimum threshold
            if min_score > 0.0:
                sim_scores[sim_scores < min_score] = 0.0

            # Get top-K indices (descending)
            top_indices = np.argsort(sim_scores)[::-1][:top_k * 2]  # oversample then filter

            results: List[RecommendationResult] = []
            for rank, idx in enumerate(top_indices, start=1):
                if len(results) >= top_k:
                    break

                score = float(sim_scores[idx])
                if score <= 0.0:
                    break

                candidate_id = self._df.iloc[idx]["product_id"]
                if candidate_id == product_id:
                    continue

                candidate_row = self._df.iloc[idx]
                reason, tags = self._build_reason(query_row, candidate_row, score)

                result = RecommendationResult(
                    product_id=str(candidate_id),
                    product_name=str(candidate_row.get("product_name", candidate_id)),
                    score=score,
                    confidence_score=min(1.0, score),
                    rank=len(results) + 1,
                    strategy=self.strategy_name,
                    recommendation_reason=reason,
                    reason_tags=tags,
                )
                _enrich_result(result, candidate_row)
                results.append(result)

            return results

        except Exception as e:
            raise CustomException(
                f"ContentBasedEngine.recommend failed: {e}", sys.exc_info()
            ) from e
