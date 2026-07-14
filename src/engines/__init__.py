"""
Recommendation Engines Package
================================
All 8 recommendation engines for the Product Recommendation System.

Engines:
    BaseEngine              — Abstract base with RecommendationResult
    ContentBasedEngine      — TF-IDF cosine similarity
    PopularityEngine        — Popularity score ranking
    TrendingEngine          — Trending score (popularity x recency proxy)
    CategorySimilarityEngine — Category hierarchy matching
    PriceSimilarityEngine   — Price bucket and range matching
    BrandSimilarityEngine   — Same-brand recommendations
    FrequentlyBoughtTogether — Category-complementarity proxy
    HybridEngine            — Weighted combination of all engines
"""

from src.engines.base_engine import BaseEngine, RecommendationResult
from src.engines.brand_similarity import BrandSimilarityEngine
from src.engines.category_similarity import CategorySimilarityEngine
from src.engines.content_based import ContentBasedEngine
from src.engines.frequently_bought import FrequentlyBoughtTogetherEngine
from src.engines.hybrid import HybridEngine
from src.engines.popularity import PopularityEngine
from src.engines.price_similarity import PriceSimilarityEngine
from src.engines.trending import TrendingEngine

__all__ = [
    "BaseEngine",
    "RecommendationResult",
    "ContentBasedEngine",
    "PopularityEngine",
    "TrendingEngine",
    "CategorySimilarityEngine",
    "PriceSimilarityEngine",
    "BrandSimilarityEngine",
    "FrequentlyBoughtTogetherEngine",
    "HybridEngine",
]
