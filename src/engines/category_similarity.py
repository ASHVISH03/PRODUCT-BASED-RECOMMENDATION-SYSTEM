"""
Category Similarity Engine
============================
Recommends products from the same or closely related category.

Scoring:
  - category_l2 exact match → score 1.0
  - category_l1 match only  → score 0.6
  - No match                → score 0.0 (excluded)

Within each tier, products are ranked by popularity_score.

Used in:
  - Related Categories section
  - Browse Similar Items
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.engines.base_engine import BaseEngine, RecommendationResult, _enrich_result


class CategorySimilarityEngine(BaseEngine):
    """
    Recommends products within the same or closely related category hierarchy.

    Args:
        products_df: Feature-engineered DataFrame with category_l1, category_l2.
    """

    @property
    def strategy_name(self) -> str:
        return "category_similarity"

    def _category_score(
        self,
        query_cat1: str,
        query_cat2: str,
        candidate_cat1: str,
        candidate_cat2: str,
    ) -> float:
        """Compute category similarity score (0.0 – 1.0)."""
        if query_cat2 and candidate_cat2 and query_cat2 == candidate_cat2:
            return 1.0
        if query_cat1 and candidate_cat1 and query_cat1 == candidate_cat1:
            return 0.6
        return 0.0

    def _build_reason(
        self,
        query_cat1: str,
        query_cat2: str,
        row: pd.Series,
        score: float,
    ) -> tuple:
        """Build reason string and tags for a category match."""
        tags: List[str] = []
        c_cat2 = str(row.get("category_l2", ""))
        c_cat1 = str(row.get("category_l1", ""))

        if score >= 1.0:
            tags.append("Same Category")
            reason = f"From the same category: {c_cat2 or c_cat1}"
        else:
            tags.append("Similar Category")
            reason = f"From a related category: {c_cat1}"

        rating = float(row.get("rating", 0) or 0)
        if rating >= 4.0:
            tags.append("Highly Rated")
            reason += f" — rated {rating:.1f}/5"

        return reason, tags

    def recommend(
        self,
        product_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Return top-K products in the same or similar category.

        Args:
            product_id: Query product identifier.
            top_k: Maximum results.

        Returns:
            Products ranked by (category_score DESC, popularity_score DESC).
        """
        query_row = self._get_product_row(product_id)
        if query_row is None:
            return []

        q_cat1 = str(query_row.get("category_l1", ""))
        q_cat2 = str(query_row.get("category_l2", ""))

        df = self._df.copy()
        df = df[df["product_id"] != product_id]

        # Compute category score for all candidates
        scores = []
        for _, row in df.iterrows():
            c_cat1 = str(row.get("category_l1", ""))
            c_cat2 = str(row.get("category_l2", ""))
            scores.append(self._category_score(q_cat1, q_cat2, c_cat1, c_cat2))

        df = df.copy()
        df["_cat_score"] = scores
        df = df[df["_cat_score"] > 0]

        if df.empty:
            return []

        # Sort: category score first, then popularity
        sort_cols = ["_cat_score"]
        if "popularity_score" in df.columns:
            sort_cols.append("popularity_score")
        df = df.sort_values(sort_cols, ascending=False)
        df = df.head(top_k)

        results: List[RecommendationResult] = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            pid = str(row.get("product_id", ""))
            if not pid:
                continue

            cat_score = float(row.get("_cat_score", 0))
            pop_score = float(row.get("popularity_score", 0) or 0)
            combined = cat_score * 0.7 + pop_score * 0.3

            reason, tags = self._build_reason(q_cat1, q_cat2, row, cat_score)

            result = RecommendationResult(
                product_id=pid,
                product_name=str(row.get("product_name", pid)),
                score=combined,
                confidence_score=min(1.0, combined),
                rank=rank,
                strategy=self.strategy_name,
                recommendation_reason=reason,
                reason_tags=tags,
            )
            _enrich_result(result, row)
            results.append(result)

        return results
