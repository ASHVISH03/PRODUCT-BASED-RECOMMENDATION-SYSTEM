"""
Hybrid Recommendation Engine
==============================
Combines scores from multiple engines using configurable weights.

Weight configuration (from config.yaml):
    content_based:       0.35
    category_similarity: 0.25
    brand_similarity:    0.15
    price_similarity:    0.10
    popularity:          0.10
    rating:              0.05

Algorithm:
    1. Run each sub-engine independently.
    2. Normalise each engine's scores to [0, 1].
    3. Compute weighted sum per product.
    4. Rank by combined score.
    5. Build enriched reason from all contributing signals.

Used as the DEFAULT strategy for:
  - Recommended For You
  - Because You Viewed
  - Customers Also Bought
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.engines.base_engine import BaseEngine, RecommendationResult, _enrich_result
from src.engines.brand_similarity import BrandSimilarityEngine
from src.engines.category_similarity import CategorySimilarityEngine
from src.engines.content_based import ContentBasedEngine
from src.engines.popularity import PopularityEngine
from src.engines.price_similarity import PriceSimilarityEngine
from src.exception.exception import CustomException


class HybridEngine(BaseEngine):
    """
    Weighted combination of all 6 available engines.

    Args:
        products_df: Feature-engineered DataFrame.
        similarity_matrix: Cosine similarity matrix (for ContentBasedEngine).
        product_index: product_id → matrix row index.
        weights: Dict of engine_name → weight (must sum to ~1.0).
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "content_based": 0.35,
        "category_similarity": 0.25,
        "brand_similarity": 0.15,
        "price_similarity": 0.10,
        "popularity": 0.10,
        "rating": 0.05,
    }

    def __init__(
        self,
        products_df: pd.DataFrame,
        similarity_matrix: Optional[np.ndarray] = None,
        product_index: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(products_df)
        self._weights = weights or self._DEFAULT_WEIGHTS

        # Initialise sub-engines
        self._engines: Dict[str, BaseEngine] = {
            "category_similarity": CategorySimilarityEngine(products_df),
            "brand_similarity": BrandSimilarityEngine(products_df),
            "price_similarity": PriceSimilarityEngine(products_df),
            "popularity": PopularityEngine(products_df),
        }

        if similarity_matrix is not None and product_index is not None:
            self._engines["content_based"] = ContentBasedEngine(
                products_df, similarity_matrix, product_index
            )

    @property
    def strategy_name(self) -> str:
        return "hybrid"

    def _run_sub_engines(
        self,
        product_id: str,
        top_k: int,
    ) -> Dict[str, List[RecommendationResult]]:
        """Run all available sub-engines and return their results."""
        results: Dict[str, List[RecommendationResult]] = {}
        fetch_k = top_k * 3  # oversample to ensure good coverage after merging

        for name, engine in self._engines.items():
            if name not in self._weights or self._weights[name] <= 0:
                continue
            try:
                results[name] = engine.recommend(product_id, top_k=fetch_k)
            except Exception:
                results[name] = []

        return results

    def _aggregate(
        self,
        engine_results: Dict[str, List[RecommendationResult]],
        query_product_id: str,
    ) -> Dict[str, float]:
        """
        Aggregate normalised weighted scores across all engines.

        Returns:
            Dict of product_id → combined_score.
        """
        # Collect raw scores per engine per product
        engine_scores: Dict[str, Dict[str, float]] = {}

        for engine_name, results in engine_results.items():
            scores = {r.product_id: r.score for r in results}
            if not scores:
                continue

            # Normalise to [0, 1]
            max_s = max(scores.values())
            min_s = min(scores.values())
            if max_s == min_s:
                norm = {pid: 1.0 for pid in scores}
            else:
                norm = {
                    pid: (s - min_s) / (max_s - min_s)
                    for pid, s in scores.items()
                }
            engine_scores[engine_name] = norm

        # Add rating as a standalone signal
        rating_scores: Dict[str, float] = {}
        for _, row in self._df.iterrows():
            pid = str(row.get("product_id", ""))
            if pid and pid != query_product_id:
                r = float(row.get("rating", 0) or 0)
                rating_scores[pid] = r / 5.0
        if rating_scores:
            engine_scores["rating"] = rating_scores

        # Weighted sum
        combined: Dict[str, float] = defaultdict(float)
        for engine_name, scores in engine_scores.items():
            weight = self._weights.get(engine_name, 0.0)
            for pid, norm_score in scores.items():
                if pid != query_product_id:
                    combined[pid] += weight * norm_score

        return combined

    def _build_hybrid_reason(
        self,
        query_row: pd.Series,
        candidate_row: pd.Series,
        engine_results: Dict[str, List[RecommendationResult]],
        product_id: str,
    ) -> Tuple[str, List[str]]:
        """
        Build a hybrid reason by collecting signals from each contributing engine.
        """
        tags: List[str] = []
        signals: List[str] = []

        q_brand = str(query_row.get("brand", ""))
        c_brand = str(candidate_row.get("brand", ""))
        q_cat1 = str(query_row.get("category_l1", ""))
        c_cat1 = str(candidate_row.get("category_l1", ""))
        q_cat2 = str(query_row.get("category_l2", ""))
        c_cat2 = str(candidate_row.get("category_l2", ""))
        q_bucket = str(query_row.get("price_bucket", ""))
        c_bucket = str(candidate_row.get("price_bucket", ""))
        c_rating = float(candidate_row.get("rating", 0) or 0)
        c_pop = float(candidate_row.get("popularity_score", 0) or 0)

        if q_brand == c_brand and q_brand not in ("Unknown", "", "nan"):
            tags.append("Same Brand")
            signals.append(f"from {c_brand}")

        if q_cat2 == c_cat2 and q_cat2:
            tags.append("Same Category")
            signals.append(f"in {c_cat2}")
        elif q_cat1 == c_cat1 and q_cat1:
            tags.append("Similar Category")
            signals.append(f"in {c_cat1}")

        if q_bucket == c_bucket and q_bucket:
            tags.append("Similar Price")
            signals.append(f"{c_bucket} price range")

        if c_rating >= 4.5:
            tags.append("Highly Rated")
            signals.append(f"rated {c_rating:.1f}/5")
        elif c_rating >= 4.0:
            signals.append(f"rated {c_rating:.1f}/5")

        if c_pop >= 0.8:
            tags.append("Best Seller")

        # Check if content engine contributed
        if "content_based" in engine_results:
            cb_pids = {r.product_id for r in engine_results["content_based"]}
            if product_id in cb_pids:
                tags.append("Similar Specifications")
                signals.append("similar product specifications")

        if not signals:
            reason = "Recommended based on multiple similarity signals"
        else:
            reason = "Recommended because: " + ", ".join(signals[:3])

        return reason, tags

    def recommend(
        self,
        product_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Generate hybrid recommendations combining all engine signals.

        Args:
            product_id: Query product identifier.
            top_k: Maximum results.

        Returns:
            Products ranked by weighted combined score.
        """
        try:
            query_row = self._get_product_row(product_id)
            if query_row is None:
                # Fallback to popularity
                return PopularityEngine(self._df).recommend(
                    product_id, top_k=top_k
                )

            # 1. Run all sub-engines
            engine_results = self._run_sub_engines(product_id, top_k)

            # 2. Aggregate
            combined_scores = self._aggregate(engine_results, product_id)

            if not combined_scores:
                return []

            # 3. Rank
            ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # 4. Normalise combined scores to [0, 1]
            max_combined = ranked[0][1] if ranked else 1.0

            results: List[RecommendationResult] = []
            for rank, (candidate_id, raw_score) in enumerate(ranked, start=1):
                candidate_row = self._get_product_row(candidate_id)
                if candidate_row is None:
                    continue

                confidence = min(1.0, raw_score / max_combined) if max_combined > 0 else 0.0
                reason, tags = self._build_hybrid_reason(
                    query_row, candidate_row, engine_results, candidate_id
                )

                result = RecommendationResult(
                    product_id=candidate_id,
                    product_name=str(candidate_row.get("product_name", candidate_id)),
                    score=raw_score,
                    confidence_score=confidence,
                    rank=rank,
                    strategy=self.strategy_name,
                    recommendation_reason=reason,
                    reason_tags=tags,
                )
                _enrich_result(result, candidate_row)
                results.append(result)

            return results

        except Exception as e:
            raise CustomException(
                f"HybridEngine.recommend failed: {e}", sys.exc_info()
            ) from e
