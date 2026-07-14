"""
Price Similarity Engine
========================
Recommends products within the same price bucket or with a similar
normalised price (within a configurable tolerance band).

Scoring:
  - Same price_bucket AND price within ±20% → score 1.0
  - Same price_bucket only                  → score 0.7
  - Adjacent price bucket                   → score 0.4

Within each tier, products are ranked by popularity_score.

Used in:
  - Similar Price Range section
  - Deal comparisons
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.engines.base_engine import BaseEngine, RecommendationResult, _enrich_result

# Adjacent bucket mapping (bidirectional)
_ADJACENT_BUCKETS: Dict[str, str] = {
    "Budget": "Mid-Range",
    "Mid-Range": "Budget",      # lower neighbour
    "Premium": "Mid-Range",
    "Luxury": "Premium",
}


class PriceSimilarityEngine(BaseEngine):
    """
    Recommends products in a similar price range.

    Args:
        products_df: Feature-engineered DataFrame.
        price_tolerance: Relative price tolerance for exact-price matching (default 0.20 = 20%).
    """

    def __init__(
        self,
        products_df: pd.DataFrame,
        price_tolerance: float = 0.20,
    ) -> None:
        super().__init__(products_df)
        self._price_tolerance = price_tolerance

    @property
    def strategy_name(self) -> str:
        return "price_similarity"

    def _price_score(
        self,
        query_price: float,
        query_bucket: str,
        candidate_price: float,
        candidate_bucket: str,
    ) -> float:
        """Compute price similarity score (0.0 – 1.0)."""
        # Both within bucket AND within price tolerance
        if query_bucket == candidate_bucket and query_price > 0 and candidate_price > 0:
            ratio = abs(candidate_price - query_price) / query_price
            if ratio <= self._price_tolerance:
                return 1.0
            return 0.7  # same bucket but price differs more

        # Adjacent bucket
        adjacent = _ADJACENT_BUCKETS.get(query_bucket, "")
        if candidate_bucket == adjacent:
            return 0.4

        return 0.0

    def _build_reason(
        self,
        query_price: float,
        query_bucket: str,
        row: pd.Series,
        score: float,
    ) -> tuple:
        """Build reason string and tags."""
        tags: List[str] = []
        c_price = float(row.get("discounted_price", 0) or 0)
        c_bucket = str(row.get("price_bucket", ""))

        if score >= 1.0:
            tags.append("Similar Price")
            if c_price > 0:
                reason = f"Similar price range: Rs.{c_price:,.0f}"
            else:
                reason = f"Similar price range ({c_bucket})"
        elif score >= 0.7:
            tags.append("Same Price Tier")
            reason = f"In the same {c_bucket} price tier"
        else:
            tags.append("Nearby Price Range")
            reason = f"In a nearby {c_bucket} price range"

        return reason, tags

    def recommend(
        self,
        product_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Return top-K products with similar pricing.

        Args:
            product_id: Query product identifier.
            top_k: Maximum results.

        Returns:
            Products ranked by (price_score DESC, popularity_score DESC).
        """
        query_row = self._get_product_row(product_id)
        if query_row is None:
            return []

        q_price = float(query_row.get("discounted_price", 0) or 0)
        q_bucket = str(query_row.get("price_bucket", ""))

        df = self._df.copy()
        df = df[df["product_id"] != product_id]

        scores = []
        for _, row in df.iterrows():
            c_price = float(row.get("discounted_price", 0) or 0)
            c_bucket = str(row.get("price_bucket", ""))
            scores.append(self._price_score(q_price, q_bucket, c_price, c_bucket))

        df = df.copy()
        df["_price_score"] = scores
        df = df[df["_price_score"] > 0]

        if df.empty:
            return []

        sort_cols = ["_price_score"]
        if "popularity_score" in df.columns:
            sort_cols.append("popularity_score")
        df = df.sort_values(sort_cols, ascending=False).head(top_k)

        results: List[RecommendationResult] = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            pid = str(row.get("product_id", ""))
            if not pid:
                continue

            price_score = float(row.get("_price_score", 0))
            pop_score = float(row.get("popularity_score", 0) or 0)
            combined = price_score * 0.65 + pop_score * 0.35

            reason, tags = self._build_reason(q_price, q_bucket, row, price_score)

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
