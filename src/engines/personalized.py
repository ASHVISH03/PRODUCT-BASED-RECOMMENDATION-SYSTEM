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
        structured_interactions: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[np.ndarray]:
        """
        Construct a recency-decayed, interaction-weighted TF-IDF user profile.

        Args:
            history_ids:        Product IDs in viewing order, newest-first.
            interaction_types:  Optional dict mapping product_id → interaction type string.
            structured_interactions: Optional list of interaction dicts:
                                     [{"product_id": "...", "event_type": "cart", "quantity": 1}, ...]

        Returns:
            User profile vector of shape (1, vocab_size), or None if no valid
            product texts could be retrieved.
        """
        if not history_ids and not structured_interactions:
            return None

        # Build normalized interaction sequence: List of (product_id, event_type)
        items_to_process: List[Tuple[str, str]] = []

        if structured_interactions:
            # Deduplicate structured interactions preserving latest order
            seen_pids = set()
            for item in structured_interactions:
                if isinstance(item, dict):
                    pid = str(item.get("product_id", "")).strip()
                    etype = str(item.get("event_type", "view")).strip().lower()
                else:
                    pid = getattr(item, "product_id", str(item)).strip()
                    etype = getattr(item, "event_type", "view").strip().lower()
                
                if pid and pid not in seen_pids:
                    seen_pids.add(pid)
                    items_to_process.append((pid, etype))
        else:
            interaction_types = interaction_types or {}
            seen_pids = set()
            for pid in history_ids:
                pid_str = str(pid).strip()
                if pid_str and pid_str not in seen_pids:
                    seen_pids.add(pid_str)
                    itype = interaction_types.get(pid_str, "view")
                    items_to_process.append((pid_str, itype))

        if not items_to_process:
            return None

        profile_vectors: List[np.ndarray] = []
        total_weight: float = 0.0

        logger.info(
            f"[PersonalizedEngine] Building multi-signal user profile from "
            f"{len(items_to_process)} interaction items: {items_to_process}"
        )

        for position, (pid, itype) in enumerate(items_to_process):
            text = self._get_product_text(pid)
            if text is None:
                logger.warning(f"  [profile] Skipping unknown product: {pid}")
                continue

            # Determine base interaction weight
            base_weight = self.INTERACTION_WEIGHTS.get(itype, 1.0)

            # Apply recency decay: newer items (smaller position index) get higher weight
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

        if not profile_vectors or total_weight <= 0:
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
        structured_interactions: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 8,
        candidate_pool: int = 50,
    ) -> List[RecommendationResult]:
        """
        Generate multi-signal personalized recommendations for a session.
        """
        try:
            # Determine dominant signal type across interactions
            dominant_signal = "view"
            if structured_interactions:
                signal_counts: Dict[str, float] = {}
                for item in structured_interactions:
                    etype = item.get("event_type", "view") if isinstance(item, dict) else getattr(item, "event_type", "view")
                    w = self.INTERACTION_WEIGHTS.get(etype, 1.0)
                    signal_counts[etype] = signal_counts.get(etype, 0.0) + w
                if signal_counts:
                    dominant_signal = max(signal_counts, key=lambda s: signal_counts[s])
            elif interaction_types:
                signal_counts: Dict[str, float] = {}
                for itype in interaction_types.values():
                    w = self.INTERACTION_WEIGHTS.get(itype, 1.0)
                    signal_counts[itype] = signal_counts.get(itype, 0.0) + w
                if signal_counts:
                    dominant_signal = max(signal_counts, key=lambda s: signal_counts[s])

            # Step 1: Build user profile
            user_profile = self.build_user_profile(
                history_ids=history_ids,
                interaction_types=interaction_types,
                structured_interactions=structured_interactions,
            )
            if user_profile is None:
                logger.warning("[PersonalizedEngine] Profile construction failed — returning empty.")
                return []

            # Step 2: Cosine similarity against all catalog products
            all_scores = cosine_similarity(user_profile, self._tfidf_matrix)  # (1 × n)
            all_scores = all_scores.flatten()  # (n,)

            # Zero out items in history to prevent self-recommendation
            history_set = set(history_ids or [])
            if structured_interactions:
                for item in structured_interactions:
                    pid = item.get("product_id") if isinstance(item, dict) else getattr(item, "product_id", None)
                    if pid:
                        history_set.add(str(pid))

            for pid in history_set:
                idx = self._product_index.get(pid)
                if idx is not None:
                    all_scores[idx] = 0.0

            # Step 3: Select top candidate pool
            candidate_pool = max(candidate_pool, top_k * 4)
            top_idxs = np.argsort(all_scores)[::-1][:candidate_pool]

            candidate_ids: List[str] = []
            candidate_scores: Dict[str, float] = {}
            for idx in top_idxs:
                score = float(all_scores[idx])
                if score <= 0.0:
                    continue
                pid = self._df.iloc[idx]["product_id"]
                candidate_ids.append(pid)
                candidate_scores[pid] = score

            if not candidate_ids:
                logger.warning("[PersonalizedEngine] No candidate products with score > 0.")
                return []

            # Step 4: MMR Reranking
            reranked = self.mmr_rerank(
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
                top_k=top_k,
            )

            # Step 5: Format RecommendationResult objects
            results: List[RecommendationResult] = []
            all_pids_in_history = list(history_set)
            interest_categories = self._extract_interests(all_pids_in_history)

            for rank, (pid, mmr_score) in enumerate(reranked, start=1):
                idx = self._product_index.get(pid)
                if idx is None:
                    continue
                row = self._df.iloc[idx]

                raw_relevance = float(candidate_scores.get(pid, 0.0))
                reason, tags = self._build_personalized_reason(
                    row, interest_categories, raw_relevance, dominant_signal
                )

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
        dominant_signal: str = "view",
    ) -> Tuple[str, List[str]]:
        """
        Build a meaningful recommendation reason from actual signals.
        Adapts reason dynamically based on interaction signal type (wishlist, cart, purchase, view).
        """
        tags: List[str] = ["Personalized"]
        cat_l1 = str(candidate_row.get("category_l1", "") or "")
        cat_l2 = str(candidate_row.get("category_l2", "") or "")
        category_name = cat_l2 or cat_l1 or "your interests"

        # Multi-signal explanation mapping
        if dominant_signal == "wishlist":
            tags.append("Wishlist Affinity")
            if category_name in interest_categories:
                reason = f"Based on products in your wishlist ({category_name})"
            else:
                reason = "Recommended based on your wishlist items"
        elif dominant_signal == "cart":
            tags.append("Cart Affinity")
            if category_name in interest_categories:
                reason = f"Related to items in your cart ({category_name})"
            else:
                reason = "Complements items in your shopping cart"
        elif dominant_signal == "purchase":
            tags.append("Purchase Affinity")
            if category_name in interest_categories:
                reason = f"Inspired by your recent purchase ({category_name})"
            else:
                reason = "Recommended based on your recent purchase history"
        elif cat_l2 and cat_l2 in interest_categories:
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
