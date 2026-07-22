"""
Recommendation Engine
======================
Unified orchestration layer that loads all trained model artifacts and
exposes the full set of recommendation APIs used by the FastAPI backend.

Public API:
    recommend(product_id, strategy, top_k)
    recommend_by_name(product_name, strategy, top_k)
    search(query, top_k)
    similar_products(product_id, top_k)
    frequently_bought_together(product_id, top_k)
    trending_products(category, top_k)
    related_products(product_id, top_k)
    brand_products(product_id, top_k)
    popular_products(category, top_k)

All methods return List[RecommendationResult].

Thread-safety: The engine is stateless after initialization.
The load() class method initialises from disk once; subsequent calls
return the cached singleton.
"""

from __future__ import annotations

import math
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.engines.base_engine import (
    BaseEngine,
    RecommendationResult,
    extract_matched_keywords,
    get_confidence_grade,
)
from src.engines.brand_similarity import BrandSimilarityEngine
from src.engines.category_similarity import CategorySimilarityEngine
from src.engines.content_based import ContentBasedEngine
from src.engines.frequently_bought import FrequentlyBoughtTogetherEngine
from src.engines.hybrid import HybridEngine
from src.engines.personalized import PersonalizedEngine
from src.engines.popularity import PopularityEngine
from src.engines.price_similarity import PriceSimilarityEngine
from src.engines.trending import TrendingEngine
from src.exception.exception import CustomException
from src.logger.training_logger import get_training_logger

logger = get_training_logger()

# Singleton cache
_engine_instance: Optional["RecommendationEngine"] = None


class RecommendationEngine:
    """
    Unified recommendation engine loaded from trained model artifacts.

    Usage:
        engine = RecommendationEngine.load("models/")
        recs = engine.recommend("B0CHX...")
    """

    def __init__(
        self,
        products_df: pd.DataFrame,
        similarity_matrix: np.ndarray,
        product_index: Dict[str, int],
        idx_to_id: Dict[int, str],
        model_version: str,
        vocab_size: int,
        tfidf_vectorizer=None,
    ) -> None:
        self._df = products_df
        self._sim_matrix = similarity_matrix
        self._id_to_idx = product_index
        self._idx_to_id = idx_to_id
        self.model_version = model_version
        self.vocab_size = vocab_size
        self._tfidf_vectorizer = tfidf_vectorizer

        # Initialise all engines
        logger.info("Initialising recommendation engines...")

        self._engines: Dict[str, BaseEngine] = {
            "content_based": ContentBasedEngine(
                products_df, similarity_matrix, product_index
            ),
            "popularity": PopularityEngine(products_df),
            "trending": TrendingEngine(products_df),
            "category_similarity": CategorySimilarityEngine(products_df),
            "price_similarity": PriceSimilarityEngine(products_df),
            "brand_similarity": BrandSimilarityEngine(products_df),
            "frequently_bought_together": FrequentlyBoughtTogetherEngine(products_df),
            "hybrid": HybridEngine(
                products_df,
                similarity_matrix=similarity_matrix,
                product_index=product_index,
            ),
        }

        # Personalized engine (requires vectorizer for user profile construction)
        if tfidf_vectorizer is not None:
            self._personalized_engine = PersonalizedEngine(
                products_df=products_df,
                vectorizer=tfidf_vectorizer,
                similarity_matrix=similarity_matrix,
                product_index=product_index,
            )
            logger.info("PersonalizedEngine: initialized.")
        else:
            self._personalized_engine = None
            logger.warning("PersonalizedEngine: no vectorizer supplied — personalization disabled.")

        # Build text search index
        self._name_to_ids: Dict[str, List[str]] = {}
        self._build_search_index()

        logger.info(
            f"RecommendationEngine ready: "
            f"{len(products_df)} products, "
            f"{len(self._engines)} engines, "
            f"vocab={vocab_size:,}"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        models_dir: str = "models",
        force_reload: bool = False,
    ) -> "RecommendationEngine":
        """
        Load (or return cached) RecommendationEngine from disk artifacts.

        Args:
            models_dir: Directory containing trained model files.
            force_reload: If True, discard cached instance and reload.

        Returns:
            Initialised RecommendationEngine.

        Raises:
            FileNotFoundError: If any required artifact is missing.
        """
        global _engine_instance
        if _engine_instance is not None and not force_reload:
            return _engine_instance

        models_path = Path(models_dir)
        required_files = {
            "vectorizer": models_path / "tfidf.pkl",
            "similarity_matrix": models_path / "similarity.pkl",
            "product_index": models_path / "product_index.pkl",
            "processed_products": models_path / "processed_products.csv",
        }

        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required model artifact missing: {path}\n"
                    f"Run training first: python -m src.pipelines.training_pipeline"
                )

        # Load the TF-IDF vectorizer (needed for PersonalizedEngine profile construction)
        with open(required_files["vectorizer"], "rb") as f:
            tfidf_vectorizer = pickle.load(f)

        logger.info(f"Loading model artifacts from: {models_path}")
        t0 = time.perf_counter()

        with open(required_files["similarity_matrix"], "rb") as f:
            sim_matrix: np.ndarray = pickle.load(f)

        with open(required_files["product_index"], "rb") as f:
            index_data = pickle.load(f)

        id_to_idx: Dict[str, int] = index_data["id_to_idx"]
        idx_to_id: Dict[int, str] = {
            int(k): v for k, v in index_data["idx_to_id"].items()
        }

        df = pd.read_csv(
            required_files["processed_products"], encoding="utf-8", low_memory=False
        )

        # Read metadata
        meta_path = models_path / "metadata.json"
        model_version = "unknown"
        vocab_size = 0
        if meta_path.exists():
            import json
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            model_version = meta.get("model_version", "unknown")
            vocab_size = int(meta.get("vocabulary_size", 0))

        load_time = time.perf_counter() - t0
        logger.info(f"Artifacts loaded in {load_time:.2f}s")

        _engine_instance = cls(
            products_df=df,
            similarity_matrix=sim_matrix,
            product_index=id_to_idx,
            idx_to_id=idx_to_id,
            model_version=model_version,
            vocab_size=vocab_size,
            tfidf_vectorizer=tfidf_vectorizer,
        )
        return _engine_instance

    # ------------------------------------------------------------------
    # Search index
    # ------------------------------------------------------------------

    def _build_search_index(self) -> None:
        """Build an in-memory text search index from product names."""
        for _, row in self._df.iterrows():
            pid = str(row.get("product_id", ""))
            name = str(row.get("product_name", "")).lower()
            if not pid or not name:
                continue
            # Index every word
            for word in name.split():
                if len(word) >= 3:
                    self._name_to_ids.setdefault(word, []).append(pid)

    # ------------------------------------------------------------------
    # Explainability post-processor
    # ------------------------------------------------------------------

    def _enrich_explanations(
        self,
        results: List[RecommendationResult],
        query_product_id: str = "",
    ) -> List[RecommendationResult]:
        """
        Post-process a recommendation list to add:
          - confidence_grade ('High' | 'Medium' | 'Low')
          - matched_keywords (common terms between query and recommended product)
          - reason_dict (structured explanation for API responses)

        This is the single source of truth for explainability — called
        from every public recommendation method.

        Args:
            results: Raw recommendation list from any engine.
            query_product_id: Optional query product ID for keyword extraction.

        Returns:
            Same list with explainability fields populated in-place.
        """
        # Pre-load query text for keyword extraction
        query_text = ""
        if query_product_id:
            q_idx = self._id_to_idx.get(query_product_id)
            if q_idx is not None:
                q_row = self._df.iloc[q_idx]
                # Prefer the pre-cleaned combined_text_tfidf column
                query_text = str(
                    q_row.get("combined_text_tfidf")
                    or q_row.get("combined_text")
                    or q_row.get("product_name", "")
                )

        for result in results:
            # 1. Confidence grade
            result.confidence_grade = get_confidence_grade(result.confidence_score)

            # 2. Matched keywords via text overlap
            if query_text:
                c_idx = self._id_to_idx.get(result.product_id)
                if c_idx is not None:
                    c_row = self._df.iloc[c_idx]
                    candidate_text = str(
                        c_row.get("combined_text_tfidf")
                        or c_row.get("combined_text")
                        or c_row.get("product_name", "")
                    )
                    result.matched_keywords = extract_matched_keywords(
                        query_text, candidate_text, max_keywords=5
                    )

            def _clean_confidence(val, default=0.0) -> float:
                try:
                    f = float(val)
                    if math.isnan(f) or math.isinf(f):
                        return float(default)
                    return float(round(f, 3))
                except (ValueError, TypeError):
                    return float(default)

            conf_val = _clean_confidence(result.confidence_score)

            # 3. Structured reason_dict
            result.reason_dict = {
                "product": str(result.product_name),
                "score": conf_val,
                "reason": {
                    "category": str(result.category_l1 or result.category_l2 or "N/A"),
                    "brand": str(result.brand if result.brand not in ("Unknown", "", "nan") else "N/A"),
                    "matched_keywords": result.matched_keywords,
                    "engine": str(result.strategy.replace("_", " ").title()),
                    "confidence": str(result.confidence_grade),
                    "confidence_score": conf_val,
                    "reason_tags": result.reason_tags,
                    "primary_reason": str(result.recommendation_reason),
                },
            }

        return results

    # ------------------------------------------------------------------
    # Core recommendation methods
    # ------------------------------------------------------------------

    def recommend(
        self,
        product_id: str,
        strategy: str = "hybrid",
        top_k: int = 10,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Generate recommendations for a product using the specified strategy.

        Args:
            product_id: Query product identifier.
            strategy: Engine name (content_based | popularity | hybrid | ...).
            top_k: Maximum results.

        Returns:
            Ranked list of RecommendationResult with full explainability.
        """
        engine = self._engines.get(strategy) or self._engines["hybrid"]
        try:
            results = engine.recommend(product_id, top_k=top_k, **kwargs)
            return self._enrich_explanations(results, query_product_id=product_id)
        except Exception as e:
            raise CustomException(
                f"RecommendationEngine.recommend failed: {e}", sys.exc_info()
            ) from e

    def recommend_by_name(
        self,
        product_name: str,
        strategy: str = "hybrid",
        top_k: int = 10,
    ) -> List[RecommendationResult]:
        """
        Recommend products similar to a given product name.

        Searches for the closest matching product and delegates to recommend().

        Args:
            product_name: Query product name (fuzzy match).
            strategy: Engine name.
            top_k: Maximum results.

        Returns:
            Recommendations for the best matching product.
        """
        matches = self.search(product_name, top_k=1)
        if not matches:
            # Fallback: return popular products
            return self.popular_products(top_k=top_k)

        best_match_id = matches[0].product_id
        return self.recommend(best_match_id, strategy=strategy, top_k=top_k)

    def search(
        self,
        query: str,
        top_k: int = 20,
        category_filter: Optional[str] = None,
    ) -> List[RecommendationResult]:
        """
        Text search over the product catalog.

        Implements TF-IDF score-weighted token matching with optional
        category filtering and typo tolerance (prefix matching).

        Args:
            query: Search string.
            top_k: Maximum results.
            category_filter: Optional category_l1 filter.

        Returns:
            Matching products ranked by relevance.
        """
        query_lower = query.lower().strip()
        tokens = [t for t in query_lower.split() if len(t) >= 2]

        if not tokens:
            return []

        # Score products by token hits
        scores: Dict[str, float] = {}
        for token in tokens:
            # Exact match
            for pid in self._name_to_ids.get(token, []):
                scores[pid] = scores.get(pid, 0) + 2.0
            # Prefix match (typo tolerance)
            for indexed_word, pids in self._name_to_ids.items():
                if indexed_word.startswith(token) and indexed_word != token:
                    for pid in pids:
                        scores[pid] = scores.get(pid, 0) + 0.5

        if not scores:
            return []

        # Sort by score, then popularity
        pop_lookup = {}
        if "popularity_score" in self._df.columns:
            pop_lookup = dict(zip(
                self._df["product_id"].astype(str),
                self._df["popularity_score"].fillna(0)
            ))

        ranked = sorted(
            scores.items(),
            key=lambda x: (x[1], pop_lookup.get(x[0], 0)),
            reverse=True,
        )

        results: List[RecommendationResult] = []
        seen_ids = set()
        for pid, score in ranked:
            if len(results) >= top_k:
                break
            if pid in seen_ids:
                continue

            row_idx = self._id_to_idx.get(pid)
            if row_idx is None:
                continue

            row = self._df.iloc[row_idx]

            # Apply category filter
            if category_filter:
                if str(row.get("category_l1", "")).lower() != category_filter.lower():
                    continue

            normalised = min(1.0, score / (len(tokens) * 2.0))
            pop = float(pop_lookup.get(pid, 0))
            confidence = normalised * 0.7 + pop * 0.3

            from src.engines.base_engine import _enrich_result
            result = RecommendationResult(
                product_id=pid,
                product_name=str(row.get("product_name", pid)),
                score=score,
                confidence_score=confidence,
                rank=len(results) + 1,
                strategy="search",
                recommendation_reason=f"Matched '{query}' in product catalog",
                reason_tags=["Search Result"],
            )
            _enrich_result(result, row)
            results.append(result)
            seen_ids.add(pid)

        return self._enrich_explanations(results)

    def recommend_personalized(
        self,
        history_ids: List[str],
        interaction_types: Optional[Dict[str, str]] = None,
        structured_interactions: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 8,
        candidate_pool: int = 50,
    ) -> List[RecommendationResult]:
        """
        Generate personalized recommendations from a user's session history.

        Uses the PersonalizedEngine to:
            1. Build a recency-weighted TF-IDF user profile vector.
            2. Score all catalog products via cosine similarity.
        Generate personalized recommendations from user interaction history.

        Args:
            history_ids:             Product IDs in viewing order, newest-first.
            interaction_types:       Optional dict mapping product_id → interaction_type string.
            structured_interactions: Optional list of interaction dicts/objects.
            top_k:                   Number of recommendations to return.
            candidate_pool:          Intermediate candidate pool size before MMR reranking.

        Returns:
            List of RecommendationResult with personalized reasoning and scores.
        """
        if not self._personalized_engine:
            logger.warning(
                "recommend_personalized called but PersonalizedEngine is not available. "
                "Falling back to empty list."
            )
            return []

        if not history_ids and not structured_interactions:
            return []

        logger.info(
            f"[RecommendationEngine] recommend_personalized("
            f"history={history_ids}, interactions={len(structured_interactions or [])}, k={top_k})"
        )

        results = self._personalized_engine.recommend_for_session(
            history_ids=history_ids,
            interaction_types=interaction_types,
            structured_interactions=structured_interactions,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )

        # Enrich with confidence grades and reason_dict for API response
        return self._enrich_explanations(results)

    def get_evaluator(self):
        """
        Return a RecommendationEvaluator configured with this engine's artifacts.
        Suitable for offline evaluation and MLflow metric logging.
        """
        from src.engines.recommendation_evaluator import RecommendationEvaluator
        return RecommendationEvaluator(
            catalog_size=len(self._df),
            sim_matrix=self._sim_matrix,
            product_index=self._id_to_idx,
        )

    def similar_products(
        self,
        product_id: str,
        top_k: int = 8,
    ) -> List[RecommendationResult]:
        """Return products similar to the given product (content-based)."""
        return self.recommend(product_id, strategy="content_based", top_k=top_k)

    def frequently_bought_together(
        self,
        product_id: str,
        top_k: int = 4,
    ) -> List[RecommendationResult]:
        """Return products frequently bought with the given product."""
        return self.recommend(
            product_id, strategy="frequently_bought_together", top_k=top_k
        )

    def trending_products(
        self,
        category: Optional[str] = None,
        top_k: int = 12,
    ) -> List[RecommendationResult]:
        """Return currently trending products, optionally in a category."""
        engine = self._engines["trending"]
        dummy_id = next(iter(self._id_to_idx), "")
        results = engine.recommend(dummy_id, top_k=top_k, category_filter=category)
        return self._enrich_explanations(results)

    def related_products(
        self,
        product_id: str,
        top_k: int = 8,
    ) -> List[RecommendationResult]:
        """Return related products from the same or similar category."""
        return self.recommend(product_id, strategy="category_similarity", top_k=top_k)

    def brand_products(
        self,
        product_id: str,
        top_k: int = 8,
    ) -> List[RecommendationResult]:
        """Return other products from the same brand."""
        return self.recommend(product_id, strategy="brand_similarity", top_k=top_k)

    def popular_products(
        self,
        category: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RecommendationResult]:
        """Return the most popular products, optionally in a category."""
        engine = self._engines["popularity"]
        dummy_id = next(iter(self._id_to_idx), "")
        results = engine.recommend(dummy_id, top_k=top_k, category_filter=category)
        return self._enrich_explanations(results)

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """
        Retrieve full product details by ID.

        Returns:
            Dict with all product fields, or None if not found.
        """
        idx = self._id_to_idx.get(product_id)
        if idx is None:
            return None
        row = self._df.iloc[idx]
        return row.where(pd.notna(row), None).to_dict()

    def get_all_categories(self) -> List[str]:
        """Return sorted list of all unique category_l1 values."""
        if "category_l1" not in self._df.columns:
            return []
        return sorted(self._df["category_l1"].dropna().unique().tolist())

    def get_catalog_size(self) -> int:
        """Return number of products in the catalog."""
        return len(self._df)
