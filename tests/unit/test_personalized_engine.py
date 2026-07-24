"""
Unit tests for PersonalizedEngine's core math: recency decay, interaction-type
weighting, MMR diversity reranking, and history exclusion.

Uses a small synthetic catalog (no DB, no trained model files) so these run
anywhere, including a bare CI checkout.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.engines.personalized import PersonalizedEngine

# Deliberately near-disjoint vocabularies so each product's TF-IDF vector
# points in a distinct direction — makes profile-alignment easy to assert on.
CATALOG = pd.DataFrame(
    [
        {"product_id": "COFFEE", "product_name": "Espresso Coffee Maker",
         "combined_text": "espresso coffee maker brewer machine barista",
         "category_l1": "Kitchen", "category_l2": "Coffee Makers", "rating": 4.5},
        {"product_id": "LAPTOP", "product_name": "Gaming Laptop",
         "combined_text": "gaming laptop keyboard mouse graphics processor",
         "category_l1": "Electronics", "category_l2": "Laptops", "rating": 4.2},
        {"product_id": "YOGA", "product_name": "Yoga Mat",
         "combined_text": "yoga mat exercise fitness stretching pilates",
         "category_l1": "Sports", "category_l2": "Fitness", "rating": 4.0},
        {"product_id": "COFFEE2", "product_name": "Coffee Grinder",
         "combined_text": "espresso coffee grinder burr barista beans",
         "category_l1": "Kitchen", "category_l2": "Coffee Makers", "rating": 4.4},
        {"product_id": "MOUSE", "product_name": "Gaming Mouse",
         "combined_text": "gaming mouse laptop keyboard rgb precision",
         "category_l1": "Electronics", "category_l2": "Laptops", "rating": 4.1},
    ]
).reset_index(drop=True)


def build_engine(catalog: pd.DataFrame = CATALOG) -> PersonalizedEngine:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(catalog["combined_text"])
    tfidf_matrix = vectorizer.transform(catalog["combined_text"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    product_index = {pid: idx for idx, pid in enumerate(catalog["product_id"])}
    return PersonalizedEngine(catalog, vectorizer, sim_matrix, product_index)


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------

def test_recency_decay_matches_formula():
    engine = build_engine()
    assert engine.DECAY_FACTOR == 0.85
    for position in range(4):
        expected = 0.85 ** position
        assert engine.DECAY_FACTOR ** position == pytest.approx(expected)


def test_earlier_history_item_dominates_profile_when_weights_equal():
    """Two 'view' events, same weight → the newer (position 0) item should
    dominate the profile vector, since its decay multiplier is larger."""
    engine = build_engine()
    profile = engine.build_user_profile(history_ids=["COFFEE", "LAPTOP"])
    assert profile is not None

    scores = cosine_similarity(profile, engine._tfidf_matrix).flatten()
    coffee_score = scores[engine._product_index["COFFEE"]]
    laptop_score = scores[engine._product_index["LAPTOP"]]
    assert coffee_score > laptop_score


# ---------------------------------------------------------------------------
# Interaction-type weighting
# ---------------------------------------------------------------------------

def test_higher_intent_signal_can_outweigh_recency_decay():
    """A 'purchase' (weight 5.0) at position 1 (decay 0.85) has effective
    weight 4.25, which should dominate a 'view' (weight 1.0) at position 0
    (effective weight 1.0)."""
    engine = build_engine()
    structured = [
        {"product_id": "LAPTOP", "event_type": "view"},       # pos 0: 1.0 * 1.0    = 1.0
        {"product_id": "COFFEE", "event_type": "purchase"},   # pos 1: 5.0 * 0.85   = 4.25
    ]
    profile = engine.build_user_profile(
        history_ids=[], structured_interactions=structured
    )
    assert profile is not None

    scores = cosine_similarity(profile, engine._tfidf_matrix).flatten()
    coffee_score = scores[engine._product_index["COFFEE"]]
    laptop_score = scores[engine._product_index["LAPTOP"]]
    assert coffee_score > laptop_score


# ---------------------------------------------------------------------------
# History exclusion
# ---------------------------------------------------------------------------

def test_recommend_for_session_excludes_history_items():
    engine = build_engine()
    results = engine.recommend_for_session(
        history_ids=["COFFEE", "COFFEE2"], top_k=5,
    )
    returned_ids = {r.product_id for r in results}
    assert "COFFEE" not in returned_ids
    assert "COFFEE2" not in returned_ids


# ---------------------------------------------------------------------------
# MMR reranking
# ---------------------------------------------------------------------------

def test_mmr_first_pick_is_always_highest_relevance():
    """Regardless of lambda, the first MMR selection has no 'already selected'
    items to diversify against, so it must be the highest-relevance candidate."""
    engine = build_engine()
    candidate_scores = {"A": 0.9, "B": 0.8, "C": 0.5}
    # Reuse the 5x5 sim matrix positions arbitrarily via product_index remap
    engine._product_index = {"A": 0, "B": 1, "C": 2}

    for lam in (0.0, 0.6, 1.0):
        selected = engine.mmr_rerank(
            candidate_ids=["A", "B", "C"],
            candidate_scores=candidate_scores,
            top_k=1,
            lambda_=lam,
        )
        assert selected[0][0] == "A"


def test_mmr_diversity_avoids_near_duplicate_second_pick():
    """With pure diversity (lambda=0) and a near-duplicate top candidate,
    MMR should prefer the more dissimilar item as the second pick over the
    equally-relevant near-duplicate."""
    engine = build_engine()
    # A and B are near-identical (sim=0.95); C is dissimilar to both.
    sim_matrix = np.array([
        [1.00, 0.95, 0.10],
        [0.95, 1.00, 0.12],
        [0.10, 0.12, 1.00],
    ])
    engine._sim_matrix = sim_matrix
    engine._product_index = {"A": 0, "B": 1, "C": 2}
    candidate_scores = {"A": 0.90, "B": 0.89, "C": 0.70}

    selected = engine.mmr_rerank(
        candidate_ids=["A", "B", "C"],
        candidate_scores=candidate_scores,
        top_k=2,
        lambda_=0.0,  # pure diversity after the first pick
    )
    picked = [pid for pid, _ in selected]
    assert picked[0] == "A"        # highest relevance goes first
    assert picked[1] == "C"        # diversity penalises near-duplicate B
