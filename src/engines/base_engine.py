"""
Base Recommendation Engine
===========================
Abstract base class for all recommendation engines.
Defines the RecommendationResult dataclass and the BaseEngine interface
that all 7 engines implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------
# Confidence grade
# ---------------------------------------------------------------

def get_confidence_grade(score: float) -> str:
    """
    Map a normalised confidence score [0, 1] to a human-readable grade.

    Thresholds:
        >= 0.75  → "High"
        >= 0.50  → "Medium"
        < 0.50   → "Low"

    Args:
        score: Confidence score in [0, 1].

    Returns:
        'High' | 'Medium' | 'Low'
    """
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "for", "in", "on", "with", "to",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "of", "by", "at", "from", "as", "into", "this",
    "that", "it", "its", "also", "each", "inch", "size", "color",
    "design", "product", "features", "compatible", "available",
}


def extract_matched_keywords(
    text1: str,
    text2: str,
    max_keywords: int = 5,
    min_length: int = 4,
) -> List[str]:
    """
    Extract common significant terms between two text strings.

    Algorithm:
        1. Tokenise both texts.
        2. Remove short words and stop-words.
        3. Return the intersection, sorted by length (longer = more specific).

    Args:
        text1: First product's text (combined_text or product_name).
        text2: Second product's text.
        max_keywords: Maximum keywords to return.
        min_length: Minimum character length for a keyword.

    Returns:
        List of matched keyword strings.
    """
    if not text1 or not text2:
        return []

    def tokenise(text: str):
        import re
        tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9+\-]*', text.lower())
        return {
            t for t in tokens
            if len(t) >= min_length and t not in _STOP_WORDS
        }

    common = tokenise(text1) & tokenise(text2)
    # Title-case and sort longest first (more specific terms first)
    keywords = sorted(common, key=lambda x: -len(x))
    return [k.title() for k in keywords[:max_keywords]]


# ---------------------------------------------------------------
# Recommendation Result
# ---------------------------------------------------------------

@dataclass
class RecommendationResult:
    """
    A single recommended product with full metadata, reasoning, and explainability.

    Core fields (always set):
        product_id, product_name, score, confidence_score, rank,
        strategy, recommendation_reason

    Explainability fields (set by RecommendationEngine._enrich_explanations):
        confidence_grade: 'High' | 'Medium' | 'Low'
        matched_keywords: Terms shared between query and recommended product.
        reason_dict: Structured explanation dict for API responses.

    Enrichment fields (set by _enrich_result from DataFrame row):
        category, brand, price, rating, etc.
    """

    product_id: str
    product_name: str
    score: float
    confidence_score: float
    rank: int
    strategy: str
    recommendation_reason: str

    # Explainability (populated by post-processor)
    confidence_grade: str = ""               # 'High' | 'Medium' | 'Low'
    matched_keywords: List[str] = field(default_factory=list)
    reason_dict: Dict[str, Any] = field(default_factory=dict)

    # Reason tags (short signals list)
    reason_tags: List[str] = field(default_factory=list)

    # Enrichment from product catalog
    category: str = ""
    category_l1: str = ""
    category_l2: str = ""
    brand: str = ""
    price: float = 0.0
    actual_price: float = 0.0
    discount_percentage: float = 0.0
    rating: float = 0.0
    rating_count: int = 0
    popularity_score: float = 0.0
    price_bucket: str = ""
    img_link: str = ""
    product_link: str = ""


def _enrich_result(result: RecommendationResult, row: pd.Series) -> RecommendationResult:
    """
    Populate enrichment fields on a RecommendationResult from a DataFrame row.

    Args:
        result: RecommendationResult with core fields set.
        row: DataFrame row for the recommended product.

    Returns:
        RecommendationResult with all metadata populated.
    """
    result.category = str(row.get("category", ""))
    result.category_l1 = str(row.get("category_l1", ""))
    result.category_l2 = str(row.get("category_l2", ""))
    result.brand = str(row.get("brand", ""))
    result.price = float(row.get("discounted_price", 0.0) or 0.0)
    result.actual_price = float(row.get("actual_price", 0.0) or 0.0)
    result.discount_percentage = float(row.get("discount_percentage", 0.0) or 0.0)
    result.rating = float(row.get("rating", 0.0) or 0.0)
    result.rating_count = int(row.get("rating_count", 0) or 0)
    result.popularity_score = float(row.get("popularity_score", 0.0) or 0.0)
    result.price_bucket = str(row.get("price_bucket", ""))
    result.img_link = str(row.get("img_link", ""))
    result.product_link = str(row.get("product_link", ""))
    return result


# ---------------------------------------------------------------
# Abstract Base Engine
# ---------------------------------------------------------------

class BaseEngine(ABC):
    """
    Abstract base for all recommendation engines.

    All engines receive a fully-featured products DataFrame (from features.csv)
    and optional model artifacts (similarity matrix, product index) at
    construction time. They expose a unified recommend() interface.

    Concrete engines must implement:
        strategy_name  — a unique string identifier
        recommend()    — returns a ranked list of RecommendationResult
    """

    def __init__(self, products_df: pd.DataFrame) -> None:
        """
        Args:
            products_df: Feature-engineered product DataFrame (from features.csv).
                         Must contain at minimum: product_id, product_name.
        """
        self._df = products_df.copy()
        self._df = self._df.reset_index(drop=True)

        # Build a product_id → DataFrame row index lookup
        if "product_id" in self._df.columns:
            self._id_to_idx: Dict[str, int] = {
                pid: idx
                for idx, pid in self._df["product_id"].items()
                if pd.notna(pid)
            }
        else:
            self._id_to_idx = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Unique short name for this engine (e.g. 'content_based')."""

    @abstractmethod
    def recommend(
        self,
        product_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Generate a ranked recommendation list for a given product.

        Args:
            product_id: Query product identifier.
            top_k: Maximum number of results to return.
            **kwargs: Engine-specific optional parameters.

        Returns:
            List of RecommendationResult, sorted by rank ascending.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_product_row(self, product_id: str) -> Optional[pd.Series]:
        """Return the DataFrame row for a product_id, or None if not found."""
        idx = self._id_to_idx.get(product_id)
        if idx is None:
            return None
        return self._df.iloc[idx]

    def _normalise_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalise a score list to [0, 1]."""
        if not scores:
            return []
        mn, mx = min(scores), max(scores)
        if mx == mn:
            return [1.0] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    def _build_result(
        self,
        product_id: str,
        score: float,
        confidence_score: float,
        rank: int,
        recommendation_reason: str,
        reason_tags: Optional[List[str]] = None,
    ) -> Optional[RecommendationResult]:
        """
        Construct a fully-enriched RecommendationResult.

        Returns None if the product_id is not in the catalog.
        """
        row = self._get_product_row(product_id)
        if row is None:
            return None

        result = RecommendationResult(
            product_id=product_id,
            product_name=str(row.get("product_name", product_id)),
            score=score,
            confidence_score=min(1.0, max(0.0, confidence_score)),
            rank=rank,
            strategy=self.strategy_name,
            recommendation_reason=recommendation_reason,
            reason_tags=reason_tags or [],
        )
        return _enrich_result(result, row)
