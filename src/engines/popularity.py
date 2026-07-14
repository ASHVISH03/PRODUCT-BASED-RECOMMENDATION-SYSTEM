"""
Popularity Engine
==================
Recommends products ranked by their normalised popularity score.

popularity_score = rating^rating_weight * log1p(rating_count)^count_weight
(pre-computed and normalised to [0,1] by Feature Engineering)

Used in:
  - Best Sellers section
  - Top Rated section
  - Fallback when no query product is available
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.engines.base_engine import BaseEngine, RecommendationResult, _enrich_result


class PopularityEngine(BaseEngine):
    """
    Returns the most popular products in the catalog or within a category.

    Does NOT require a similarity matrix — purely feature-based.

    Args:
        products_df: Feature-engineered DataFrame with popularity_score column.
    """

    @property
    def strategy_name(self) -> str:
        return "popularity"

    def _build_reason(self, row: pd.Series, rank: int) -> tuple:
        """Build reason string and tags for a popularity result."""
        tags: List[str] = ["Best Seller"]
        score = float(row.get("popularity_score", 0) or 0)
        rating = float(row.get("rating", 0) or 0)
        rating_count = int(row.get("rating_count", 0) or 0)
        cat1 = str(row.get("category_l1", ""))

        parts = []
        if rating >= 4.5:
            tags.append("Top Rated")
            parts.append(f"rated {rating:.1f}/5")
        elif rating >= 4.0:
            parts.append(f"rated {rating:.1f}/5")

        if rating_count >= 1000:
            parts.append(f"{rating_count:,} reviews")

        if cat1:
            parts.append(f"in {cat1}")

        if not parts:
            reason = f"One of the most popular products (rank #{rank})"
        else:
            reason = f"Best seller — {', '.join(parts)}"

        return reason, tags

    def recommend(
        self,
        product_id: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        exclude_id: Optional[str] = None,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Return top-K most popular products.

        Args:
            product_id: Query product (used to determine category context if needed).
            top_k: Maximum results.
            category_filter: Optional category_l1 to restrict results.
            exclude_id: Optional product_id to exclude from results.

        Returns:
            Ranked list of popular products.
        """
        df = self._df.copy()

        # Optionally filter by category
        if category_filter and "category_l1" in df.columns:
            cat_df = df[df["category_l1"] == category_filter]
            if len(cat_df) >= max(3, top_k):
                df = cat_df

        # If no explicit filter, try to get context from query product
        elif product_id and product_id in self._id_to_idx:
            query_row = self._get_product_row(product_id)
            if query_row is not None:
                cat1 = str(query_row.get("category_l1", ""))
                if cat1:
                    cat_df = df[df["category_l1"] == cat1]
                    if len(cat_df) >= max(3, top_k):
                        df = cat_df

        # Exclude query product
        exclude = {product_id, exclude_id} - {None, ""}
        if exclude and "product_id" in df.columns:
            df = df[~df["product_id"].isin(exclude)]

        # Sort by popularity_score
        if "popularity_score" not in df.columns:
            return []

        df = df.nlargest(top_k * 2, "popularity_score")

        results: List[RecommendationResult] = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            if len(results) >= top_k:
                break

            pid = str(row.get("product_id", ""))
            if not pid:
                continue

            pop_score = float(row.get("popularity_score", 0) or 0)
            reason, tags = self._build_reason(row, rank)

            result = RecommendationResult(
                product_id=pid,
                product_name=str(row.get("product_name", pid)),
                score=pop_score,
                confidence_score=pop_score,
                rank=rank,
                strategy=self.strategy_name,
                recommendation_reason=reason,
                reason_tags=tags,
            )
            _enrich_result(result, row)
            results.append(result)

        return results
