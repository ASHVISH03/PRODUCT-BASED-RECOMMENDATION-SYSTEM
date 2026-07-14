"""
Trending Products Engine
=========================
Surfaces products that are "trending" — defined as a combination of
high popularity_score AND high rating_count (used as a recency proxy
since the Amazon dataset has no timestamps).

The trending score applies a recency-weighted formula:
    trending_score = popularity_score * (1 + log1p(rating_count) * decay_factor)

Used in:
  - Trending Today section
  - Trending in Category section
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd

from src.engines.base_engine import BaseEngine, RecommendationResult, _enrich_result


class TrendingEngine(BaseEngine):
    """
    Recommends currently trending products.

    Args:
        products_df: Feature-engineered DataFrame.
        decay_factor: Controls how much rating_count amplifies the trend score.
        min_rating_count: Exclude products with fewer than this many ratings.
    """

    def __init__(
        self,
        products_df: pd.DataFrame,
        decay_factor: float = 0.95,
        min_rating_count: int = 5,
    ) -> None:
        super().__init__(products_df)
        self._decay_factor = decay_factor
        self._min_rating_count = min_rating_count
        self._df = self._compute_trending_scores(self._df)

    def _compute_trending_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and add trending_score column to the DataFrame.

        Formula:
            trending_score = popularity_score * (1 + log1p(rating_count) * decay_factor)

        Scores are normalised to [0, 1].
        """
        if "popularity_score" not in df.columns:
            df["trending_score"] = 0.0
            return df

        pop = df["popularity_score"].fillna(0).clip(0, 1)
        count = df.get("rating_count", pd.Series(0, index=df.index)).fillna(0).clip(0)

        raw = pop * (1 + np.log1p(count) * self._decay_factor)
        max_score = raw.max()

        df["trending_score"] = (raw / max_score).clip(0, 1) if max_score > 0 else 0.0
        return df

    @property
    def strategy_name(self) -> str:
        return "trending"

    def _build_reason(self, row: pd.Series, rank: int) -> tuple:
        """Build reason string for a trending result."""
        tags: List[str] = ["Trending"]
        rating = float(row.get("rating", 0) or 0)
        rating_count = int(row.get("rating_count", 0) or 0)
        cat1 = str(row.get("category_l1", ""))

        parts = []
        if cat1:
            tags.append(f"Trending in {cat1}")
            parts.append(f"in {cat1}")
        if rating >= 4.0:
            parts.append(f"rated {rating:.1f}/5")
        if rating_count >= 500:
            parts.append(f"{rating_count:,} reviews")

        if not parts:
            reason = f"Trending product (#{rank} this week)"
        else:
            reason = f"Trending #{rank} — {', '.join(parts)}"

        return reason, tags

    def recommend(
        self,
        product_id: str,
        top_k: int = 12,
        category_filter: Optional[str] = None,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Return top-K trending products.

        Args:
            product_id: Query product (provides category context).
            top_k: Maximum results.
            category_filter: Explicit category_l1 filter.

        Returns:
            Trending products ranked by trending_score.
        """
        df = self._df.copy()

        # Apply minimum rating count filter
        if "rating_count" in df.columns:
            df = df[df["rating_count"].fillna(0) >= self._min_rating_count]

        # Category context
        effective_category = category_filter
        if not effective_category and product_id in self._id_to_idx:
            query_row = self._get_product_row(product_id)
            if query_row is not None:
                effective_category = str(query_row.get("category_l1", ""))

        if effective_category and "category_l1" in df.columns:
            cat_df = df[df["category_l1"] == effective_category]
            if len(cat_df) >= max(3, top_k):
                df = cat_df

        # Exclude query product
        if product_id and "product_id" in df.columns:
            df = df[df["product_id"] != product_id]

        if "trending_score" not in df.columns:
            return []

        df = df.nlargest(top_k, "trending_score")

        results: List[RecommendationResult] = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            pid = str(row.get("product_id", ""))
            if not pid:
                continue

            trend_score = float(row.get("trending_score", 0) or 0)
            reason, tags = self._build_reason(row, rank)

            result = RecommendationResult(
                product_id=pid,
                product_name=str(row.get("product_name", pid)),
                score=trend_score,
                confidence_score=trend_score,
                rank=rank,
                strategy=self.strategy_name,
                recommendation_reason=reason,
                reason_tags=tags,
            )
            _enrich_result(result, row)
            results.append(result)

        return results
