"""
Personalized Session-Based Recommendation Engine
=================================================
Constructs a genuine TF-IDF user-profile vector from the user's browsing
history and generates ranked, diverse recommendations using Maximal
Marginal Relevance (MMR).

Algorithm:
    1. For each product in the session history (sorted newest-first):
       a. Retrieve its combined_text_tfidf field from the product catalog.
       b. Transform via the fitted TF-IDF vectorizer → sparse feature vector.
       c. Apply recency decay:  weight_i = interaction_weight × (decay_factor ** i)
    2. Weighted average of all feature vectors → user_profile_vector (1 × vocab).
    3. Compute cosine_similarity(user_profile_vector, all_product_vectors).
    4. Filter out products already in history.
    5. Apply MMR reranking to balance relevance and diversity.
    6. Return top-K enriched RecommendationResult objects.

Interaction Weights (configurable class constants):
    view        = 1.0
    view_repeat = 1.5
    wishlist    = 2.0   (future signal)
    cart        = 3.0   (future signal)
    purchase    = 5.0   (future signal)

MMR Lambda:
    0.0 = pure diversity
    1.0 = pure relevance
    0.6 = balanced default
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.engines.base_engine import BaseEngine, RecommendationResult, _enrich_result
from src.exception.exception import CustomException

logger = logging.getLogger(__name__)


class PersonalizedEngine(BaseEngine):
    """
    Session-based personalization engine using TF-IDF user profile construction
    and MMR-based diversity reranking.

    Interaction signal weights are configurable via INTERACTION_WEIGHTS.
    The engine is forward-compatible: adding 'wishlist', 'cart', 'purchase'
    signals to INTERACTION_WEIGHTS requires no engine rewrite.

    Args:
        products_df:       Feature-engineered product DataFrame (processed_products.csv).
        vectorizer:        Fitted sklearn TfidfVectorizer from models/tfidf.pkl.
        similarity_matrix: Precomputed cosine similarity matrix (n × n) from models/similarity.pkl.
        product_index:     Dict mapping product_id → row index in the matrix.
    """

    # -----------------------------------------------------------------------
    # Configurable weights — edit here, not inline
    # -----------------------------------------------------------------------
    INTERACTION_WEIGHTS: Dict[str, float] = {
        "view":        1.0,
        "view_repeat": 1.5,
        "wishlist":    2.0,   # future signal
        "cart":        3.0,   # future signal
        "purchase":    5.0,   # future signal
    }

    DECAY_FACTOR: float = 0.85      # recency decay per position (0 = newest)
    MMR_LAMBDA:   float = 0.60      # 0=pure diversity  |  1=pure relevance
    MIN_PROFILE_NORM: float = 1e-8  # guard against zero-norm profile vectors

    def __init__(
        self,
        products_df: pd.DataFrame,
        vectorizer,                        # sklearn TfidfVectorizer
        similarity_matrix: np.ndarray,
        product_index: Dict[str, int],
    ) -> None:
        super().__init__(products_df)
        self._vectorizer = vectorizer
        self._sim_matrix = similarity_matrix
        self._product_index = product_index

        # Pre-transform the entire product catalog to a dense TF-IDF matrix.
        # Shape: (n_products, vocab_size)  — sparse internally, materialised on demand.
        logger.info("PersonalizedEngine: building product TF-IDF matrix...")
        texts = self._get_catalog_texts()
        self._tfidf_matrix = vectorizer.transform(texts)  # sparse (n × vocab)
        logger.info(
            f"PersonalizedEngine ready: "
            f"{self._tfidf_matrix.shape[0]} products × "
            f"{self._tfidf_matrix.shape[1]} vocab features"
        )

    @property
    def strategy_name(self) -> str:
        return "personalized"

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_catalog_texts(self) -> List[str]:
        """Return the combined text representation for every product row."""
        if "combined_text_tfidf" in self._df.columns:
            col = "combined_text_tfidf"
        elif "combined_text" in self._df.columns:
            col = "combined_text"
        else:
            col = "product_name"
        return self._df[col].fillna("").astype(str).tolist()

    def _get_product_text(self, product_id: str) -> Optional[str]:
        """Return the combined text for a single product."""
        idx = self._product_index.get(product_id)
        if idx is None:
            return None
        row = self._df.iloc[idx]
        for col in ("combined_text_tfidf", "combined_text", "product_name"):
            val = row.get(col)
            if val and str(val).strip():
                return str(val)
        return None

    # -----------------------------------------------------------------------
    # Core algorithm: Build User Profile Vector
    # -----------------------------------------------------------------------

    def build_user_profile(
        self,
        history_ids: List[str],
        interaction_types: Optional[Dict[str, str]] = None,
    ) -> Optional[np.ndarray]:
        """
        Construct a recency-decayed, interaction-weighted TF-IDF user profile.

        Args:
            history_ids:        Product IDs in viewing order, newest-first.
            interaction_types:  Optional dict mapping product_id → interaction type.
                                e.g. {"B01...": "view", "B02...": "wishlist"}
                                Defaults to "view" for all items if not supplied.

        Returns:
            User profile vector of shape (1, vocab_size), or None if no valid
            product texts could be retrieved.
        """
        if not history_ids:
            return None

        interaction_types = interaction_types or {}
        profile_vectors: List[np.ndarray] = []
        total_weight: float = 0.0

        logger.info(
            f"[PersonalizedEngine] Building user profile from "
            f"{len(history_ids)} history items: {history_ids}"
        )

        for position, pid in enumerate(history_ids):
            text = self._get_product_text(pid)
            if text is None:
                logger.warning(f"  [profile] Skipping unknown product: {pid}")
                continue

            # Determine base interaction weight
            itype = interaction_types.get(pid, "view")
            base_weight = self.INTERACTION_WEIGHTS.get(itype, 1.0)

            # Apply recency decay: newer items (smaller position) get higher weight
            decay = self.DECAY_FACTOR ** position
            item_weight = base_weight * decay

            # TF-IDF transform and weight
            vec = self._vectorizer.transform([text])  # sparse (1 × vocab)
            profile_vectors.append(item_weight * vec)
            total_weight += item_weight

            logger.info(
                f"  [profile] pos={position} pid={pid} "
                f"type={itype} base={base_weight:.2f} "
                f"decay={decay:.4f} weight={item_weight:.4f}"
            )

        if not profile_vectors:
            logger.warning("[PersonalizedEngine] No valid profile vectors built.")
            return None

        # Sum and normalise
        profile = sum(profile_vectors) / total_weight  # sparse (1 × vocab)

        # Convert to dense row vector for downstream cosine similarity
        profile_dense = np.asarray(profile.todense())  # (1 × vocab)
        norm = np.linalg.norm(profile_dense)
        if norm < self.MIN_PROFILE_NORM:
            logger.warning("[PersonalizedEngine] User profile vector has near-zero norm.")
            return None

        profile_dense = profile_dense / norm  # unit vector
        logger.info(
            f"  [profile] Final unit vector computed. "
            f"norm_before_normalisation={norm:.6f} "
            f"nonzero_features={np.count_nonzero(profile_dense)}"
        )
        return profile_dense  # (1 × vocab)

    # -----------------------------------------------------------------------
    # MMR Reranking for Diversity
    # -----------------------------------------------------------------------

    def mmr_rerank(
        self,
        candidate_ids: List[str],
        candidate_scores: Dict[str, float],
        top_k: int,
        lambda_: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Maximal Marginal Relevance reranking.

        Balances relevance (cosine similarity to user profile) with diversity
        (dissimilarity to already-selected items).

        MMR(i) = λ × relevance(i) - (1-λ) × max_j(similarity(i, j))
        where j iterates over already-selected items.

        Args:
            candidate_ids:    Ordered list of candidate product IDs.
            candidate_scores: Dict mapping product_id → relevance score.
            top_k:            Number of items to select.
            lambda_:          Trade-off parameter. Defaults to MMR_LAMBDA.

        Returns:
            List of (product_id, mmr_score) tuples in selection order.
        """
        λ = lambda_ if lambda_ is not None else self.MMR_LAMBDA
        if not candidate_ids:
            return []

        # Build a lookup: product_id → matrix row index
        idx_lookup = {pid: self._product_index[pid]
                      for pid in candidate_ids
                      if pid in self._product_index}

        remaining = [pid for pid in candidate_ids if pid in idx_lookup]
        selected: List[Tuple[str, float]] = []

        logger.info(
            f"[MMR] Reranking {len(remaining)} candidates → top-{top_k} "
            f"(λ={λ:.2f})"
        )

        while len(selected) < top_k and remaining:
            best_pid: Optional[str] = None
            best_mmr: float = -np.inf

            for pid in remaining:
                relevance = candidate_scores.get(pid, 0.0)

                if not selected:
                    # First item: pure relevance
                    mmr = relevance
                else:
                    # Similarity to already-selected items via precomputed matrix
                    selected_idxs = [self._product_index[s[0]] for s in selected
                                     if s[0] in self._product_index]
                    cand_idx = idx_lookup[pid]
                    sim_to_selected = max(
                        float(self._sim_matrix[cand_idx, si])
                        for si in selected_idxs
                    ) if selected_idxs else 0.0

                    mmr = λ * relevance - (1 - λ) * sim_to_selected

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_pid = pid

            if best_pid:
                selected.append((best_pid, best_mmr))
                remaining.remove(best_pid)
                logger.info(
                    f"  [MMR] Selected #{len(selected)}: {best_pid} "
                    f"(relevance={candidate_scores.get(best_pid, 0):.4f}, "
                    f"mmr={best_mmr:.4f})"
                )
            else:
                break

        return selected

    # -----------------------------------------------------------------------
    # Primary Recommendation Method
    # -----------------------------------------------------------------------

    def recommend_for_session(
        self,
        history_ids: List[str],
        interaction_types: Optional[Dict[str, str]] = None,
        top_k: int = 8,
        candidate_pool: int = 50,
    ) -> List[RecommendationResult]:
        """
        Full personalization pipeline:
            1. Build TF-IDF user profile vector.
            2. Score all catalog products via cosine similarity.
            3. Exclude already-viewed products.
            4. MMR rerank top candidate_pool items.
            5. Return top_k enriched RecommendationResult objects.

        Args:
            history_ids:       Product IDs in viewing order, newest-first.
            interaction_types: Optional interaction type per product.
            top_k:             Final number of recommendations.
            candidate_pool:    Pool size before MMR reranking (should be >> top_k).

        Returns:
            List of RecommendationResult with genuine reasons derived from signals.
        """
        try:
            if not history_ids:
                logger.info("[PersonalizedEngine] Empty history — caller should route to cold start.")
                return []

            # Step 1: Build user profile
            user_profile = self.build_user_profile(history_ids, interaction_types)
            if user_profile is None:
                logger.warning("[PersonalizedEngine] Profile construction failed — returning empty.")
                return []

            # Step 2: Cosine similarity against all catalog products
            # _tfidf_matrix is sparse (n × vocab), user_profile is dense (1 × vocab)
            all_scores = cosine_similarity(user_profile, self._tfidf_matrix)  # (1 × n)
            all_scores = all_scores.flatten()  # (n,)

            logger.info(
                f"[PersonalizedEngine] Score distribution: "
                f"max={all_scores.max():.4f} mean={all_scores.mean():.4f} "
                f"nonzero={np.count_nonzero(all_scores > 0)}"
            )

            # Step 3: Exclude history items
            history_set = set(history_ids)
            excluded_idxs = {
                self._product_index[pid]
                for pid in history_set
                if pid in self._product_index
            }
            for idx in excluded_idxs:
                all_scores[idx] = 0.0

            # Step 4: Take top candidate_pool items by raw cosine score
            top_idxs = np.argsort(all_scores)[::-1][:candidate_pool]
            candidate_ids: List[str] = []
            candidate_scores: Dict[str, float] = {}
            for idx in top_idxs:
                score = float(all_scores[idx])
                if score <= 0.0:
                    break
                pid = str(self._df.iloc[idx]["product_id"])
                candidate_ids.append(pid)
                candidate_scores[pid] = score

            logger.info(
                f"[PersonalizedEngine] Candidate pool: "
                f"{len(candidate_ids)} products (top score={max(candidate_scores.values(), default=0):.4f})"
            )

            # Step 5: MMR rerank
            reranked = self.mmr_rerank(candidate_ids, candidate_scores, top_k=top_k)

            # Step 6: Build RecommendationResult objects
            results: List[RecommendationResult] = []
            # Derive dominant interests from history for reason generation
            interest_categories = self._extract_interests(history_ids)
            logger.info(f"[PersonalizedEngine] Interest categories: {interest_categories}")

            for rank, (pid, mmr_score) in enumerate(reranked, start=1):
                idx = self._product_index.get(pid)
                if idx is None:
                    continue
                row = self._df.iloc[idx]

                raw_relevance = float(candidate_scores.get(pid, 0.0))
                reason, tags = self._build_personalized_reason(row, interest_categories, raw_relevance)

                result = RecommendationResult(
                    product_id=str(pid),
                    product_name=str(row.get("product_name", pid)),
                    score=float(raw_relevance),
                    confidence_score=float(min(1.0, raw_relevance)),
                    rank=int(rank),
                    strategy=str(self.strategy_name),
                    recommendation_reason=str(reason),
                    reason_tags=tags,
                )
                _enrich_result(result, row)
                results.append(result)

                logger.info(
                    f"  [result] rank={rank} pid={pid} "
                    f"relevance={raw_relevance:.4f} mmr={mmr_score:.4f} "
                    f"reason='{reason}'"
                )

            return results

        except Exception as e:
            raise CustomException(
                f"PersonalizedEngine.recommend_for_session failed: {e}", sys.exc_info()
            ) from e

    def _extract_interests(self, history_ids: List[str]) -> List[str]:
        """Extract dominant category interests from history (for reason generation)."""
        cat_counts: Dict[str, int] = {}
        for pid in history_ids:
            idx = self._product_index.get(pid)
            if idx is None:
                continue
            row = self._df.iloc[idx]
            cat = str(row.get("category_l2") or row.get("category_l1") or "")
            if cat and cat not in ("nan", ""):
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        # Return top 2 categories
        return sorted(cat_counts, key=lambda c: -cat_counts[c])[:2]

    def _build_personalized_reason(
        self,
        candidate_row: pd.Series,
        interest_categories: List[str],
        score: float,
    ) -> Tuple[str, List[str]]:
        """
        Build a meaningful recommendation reason from actual signals.
        Avoids the generic 'Based on your recently viewed items' for every result.
        """
        tags: List[str] = ["Personalized"]
        cat_l1 = str(candidate_row.get("category_l1", "") or "")
        cat_l2 = str(candidate_row.get("category_l2", "") or "")

        # Check if candidate matches a dominant interest category
        if cat_l2 and cat_l2 in interest_categories:
            reason = f"Matches your recent interest in {cat_l2}"
            tags.append("Category Match")
        elif cat_l1 and cat_l1 in interest_categories:
            reason = f"Recommended based on your interest in {cat_l1}"
            tags.append("Category Match")
        elif score >= 0.40:
            reason = "Similar to products you recently explored"
            tags.append("High Similarity")
        elif score >= 0.20:
            reason = "Related to your recent browsing session"
            tags.append("Session Affinity")
        else:
            reason = "Curated pick based on your session"
            tags.append("Session Curated")

        rating = float(candidate_row.get("rating", 0) or 0)
        if rating >= 4.3:
            tags.append("Top Rated")

        return reason, tags

    # -----------------------------------------------------------------------
    # Required abstract method implementation
    # -----------------------------------------------------------------------

    def recommend(
        self,
        product_id: str,
        top_k: int = 8,
        **kwargs,
    ) -> List[RecommendationResult]:
        """
        Fallback: treat the given product_id as the sole history entry.
        Prefer recommend_for_session() for real personalization.
        """
        return self.recommend_for_session(
            history_ids=[product_id],
            top_k=top_k,
        )
