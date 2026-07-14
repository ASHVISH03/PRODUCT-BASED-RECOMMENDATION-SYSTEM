"""
Brand Similarity Engine
========================
Recommends products from the same brand.

Scoring:
  - Exact brand match → base score from popularity_score
  - Brand not available ("Unknown") → falls back to category_l1 match

Used in:
  - "More from [Brand]" section
  - Brand store pages
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.engines.base_engine import BaseEngine, RecommendationResult, _enrich_result


class BrandSimilarityEngine(BaseEngine):
    """
    Recommends products from the same brand as the query product.

    Args:
        products_df: Feature-engineered DataFrame with brand column.
    """

    @property
    def strategy_name(self) -> str:
        return "brand_similarity"

    def _build_reason(self, brand: str, row: pd.Series) -> tuple:
        """Build reason string and tags for a brand match."""
        tags: List[str] = ["Same Brand"]
        rating = float(row.get("rating", 0) or 0)
        rating_count = int(row.get("rating_count", 0) or 0)

        reason_parts = [f"Also from {brand}"]
        if rating >= 4.0:
            tags.append("Highly Rated")
            reason_parts.append(f"rated {rating:.1f}/5")
        if rating_count >= 1000:
            reason_parts.append(f"{rating_count:,} reviews")

        return " — ".join(reason_parts), tags

    def recommend(
        self,
        product_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Return top-K products from the same brand.

        Args:
            product_id: Query product identifier.
            top_k: Maximum results.

        Returns:
            Products ranked by popularity_score within the same brand.
        """
        query_row = self._get_product_row(product_id)
        if query_row is None:
            return []

        brand = str(query_row.get("brand", "Unknown"))

        df = self._df.copy()
        df = df[df["product_id"] != product_id]

        if brand and brand not in ("Unknown", "nan", ""):
            brand_df = df[df["brand"] == brand]
        else:
            # Fallback to category_l1 when brand is Unknown
            cat1 = str(query_row.get("category_l1", ""))
            brand_df = df[df["category_l1"] == cat1] if cat1 else pd.DataFrame()

        if brand_df.empty:
            return []

        # Sort by popularity
        sort_col = "popularity_score" if "popularity_score" in brand_df.columns else "rating"
        brand_df = brand_df.nlargest(top_k, sort_col)

        results: List[RecommendationResult] = []
        for rank, (_, row) in enumerate(brand_df.iterrows(), start=1):
            pid = str(row.get("product_id", ""))
            if not pid:
                continue

            pop_score = float(row.get("popularity_score", 0.5) or 0.5)
            reason, tags = self._build_reason(brand, row)

            result = RecommendationResult(
                product_id=pid,
                product_name=str(row.get("product_name", pid)),
                score=pop_score,
                confidence_score=min(1.0, pop_score + 0.2),  # brand match boost
                rank=rank,
                strategy=self.strategy_name,
                recommendation_reason=reason,
                reason_tags=tags,
            )
            _enrich_result(result, row)
            results.append(result)

        return results
