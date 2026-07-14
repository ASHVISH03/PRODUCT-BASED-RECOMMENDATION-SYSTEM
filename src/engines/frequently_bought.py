"""
Frequently Bought Together Engine
===================================
Recommends products that complement the query product and are likely
to be purchased together.

Strategy (without purchase history):
  For a query product P in category A|B:
    1. Primary FBT: Products in the SAME category_l1 but DIFFERENT category_l2
       → these are complementary items (e.g., phone + case + charger)
    2. Secondary FBT: Products in a different but RELATED category_l1
       → cross-category bundles (e.g., laptop + bag)
    3. Score = complement_score * 0.6 + popularity_score * 0.4

Used in:
  - Frequently Bought Together section on product pages
  - Cart recommendations
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.engines.base_engine import BaseEngine, RecommendationResult, _enrich_result

# Predefined complementary category mappings
# Maps category_l1 → list of complementary category_l1s
_COMPLEMENTARY_CATEGORIES: Dict[str, List[str]] = {
    "Electronics": ["Computers & Accessories", "Clothing", "Books"],
    "Computers & Accessories": ["Electronics", "Books"],
    "Clothing": ["Beauty & Personal Care", "Sports & Outdoors"],
    "Beauty & Personal Care": ["Clothing", "Sports & Outdoors"],
    "Sports & Outdoors": ["Clothing", "Health & Nutrition"],
    "Home & Kitchen": ["Electronics", "Appliances"],
    "Appliances": ["Home & Kitchen", "Electronics"],
    "Books": ["Electronics", "Computers & Accessories"],
    "Gaming": ["Electronics", "Computers & Accessories"],
}


class FrequentlyBoughtTogetherEngine(BaseEngine):
    """
    Recommends products frequently bought together with the query product.

    Uses category-based complementarity as a proxy for purchase co-occurrence.

    Args:
        products_df: Feature-engineered DataFrame.
    """

    @property
    def strategy_name(self) -> str:
        return "frequently_bought_together"

    def _complement_score(
        self,
        query_cat1: str,
        query_cat2: str,
        candidate_cat1: str,
        candidate_cat2: str,
    ) -> float:
        """
        Score how complementary a candidate is to the query.

        Returns:
            float in [0, 1]. 0 = not complementary, 1 = highly complementary.
        """
        # Same L1, different L2 (complementary items in same department)
        if query_cat1 == candidate_cat1 and query_cat2 != candidate_cat2:
            return 0.85

        # Complementary L1 category
        complements = _COMPLEMENTARY_CATEGORIES.get(query_cat1, [])
        if candidate_cat1 in complements:
            return 0.65

        return 0.0

    def _build_reason(
        self,
        candidate_cat2: str,
        score: float,
        row: pd.Series,
    ) -> Tuple[str, List[str]]:
        """Build reason string for an FBT result."""
        tags: List[str] = ["Frequently Bought Together"]
        rating = float(row.get("rating", 0) or 0)

        if score >= 0.8:
            reason = f"Customers often buy this with {candidate_cat2} items"
        else:
            reason = "Customers who bought this also bought this product"

        if rating >= 4.0:
            tags.append("Highly Rated")

        return reason, tags

    def recommend(
        self,
        product_id: str,
        top_k: int = 4,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Return top-K frequently-bought-together products.

        Args:
            product_id: Query product identifier.
            top_k: Maximum results (typically 4 for FBT sections).

        Returns:
            Complementary products ranked by complement_score + popularity.
        """
        query_row = self._get_product_row(product_id)
        if query_row is None:
            return []

        q_cat1 = str(query_row.get("category_l1", ""))
        q_cat2 = str(query_row.get("category_l2", ""))

        df = self._df.copy()
        df = df[df["product_id"] != product_id]

        complement_scores = []
        for _, row in df.iterrows():
            c_cat1 = str(row.get("category_l1", ""))
            c_cat2 = str(row.get("category_l2", ""))
            complement_scores.append(
                self._complement_score(q_cat1, q_cat2, c_cat1, c_cat2)
            )

        df = df.copy()
        df["_fbt_score"] = complement_scores
        df = df[df["_fbt_score"] > 0]

        if df.empty:
            return []

        # Combined score = complement * 0.6 + popularity * 0.4
        pop_col = "popularity_score" if "popularity_score" in df.columns else None
        if pop_col:
            df["_combined"] = (
                df["_fbt_score"] * 0.6 + df[pop_col].fillna(0) * 0.4
            )
        else:
            df["_combined"] = df["_fbt_score"]

        df = df.nlargest(top_k, "_combined")

        results: List[RecommendationResult] = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            pid = str(row.get("product_id", ""))
            if not pid:
                continue

            fbt_score = float(row.get("_fbt_score", 0))
            combined = float(row.get("_combined", 0))
            c_cat2 = str(row.get("category_l2", ""))

            reason, tags = self._build_reason(c_cat2, fbt_score, row)

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
