"""
Backend regression tests — pytest port of scripts/test_backend_regression.py.

Exercises the real RecommendationService against the trained model artifacts
and SQLite database, covering the 3 scenarios that were previously only
runnable manually:
  TEST A: Personalized engine with a real 15-item browsing history
  TEST B: Hybrid engine strategy on a known product
  TEST C: Category similarity engine strategy on a known product
"""
import json

from tests.conftest import requires_trained_artifacts

REAL_BROWSER_HISTORY_IDS = [
    "B0B4HJNPV4", "B0B1YZX72F", "B08L879JSN", "B0B9959XF3", "B0BC9BW512",
    "B0B2RBP83P", "B09PTT8DZF", "B08LW31NQ6", "B09P22HXH6", "B0B1YY6JJL",
    "B0B997FBZT", "B00EDJJ7FS", "B08BJN4MP3", "B09XXZXQC1", "B096MSW6CT",
]
KNOWN_PRODUCT_ID = "B07KSMBL2H"


@requires_trained_artifacts
def test_personalized_recommendations_no_history_leakage(recommendation_service):
    result = recommendation_service.get_personalized_recommendations(
        history_ids=REAL_BROWSER_HISTORY_IDS,
        k=8,
        session_id="regression_test_a",
    )

    data = result.get("data", [])
    returned_ids = [item["product_id"] for item in data]
    leakage = set(returned_ids) & set(REAL_BROWSER_HISTORY_IDS)

    assert result.get("mode") == "personalized"
    assert len(data) == 8
    assert not leakage, f"History leakage detected: {leakage}"
    json.dumps(result)  # must be JSON-serialisable for the API layer


@requires_trained_artifacts
def test_hybrid_strategy_returns_recommendations(recommendation_service):
    results = recommendation_service.get_recommendations(
        product_id=KNOWN_PRODUCT_ID,
        strategy="hybrid",
        k=10,
        session_id="regression_test_b",
    )

    assert len(results) > 0
    json.dumps(results)


@requires_trained_artifacts
def test_category_similarity_strategy_returns_recommendations(recommendation_service):
    results = recommendation_service.get_recommendations(
        product_id=KNOWN_PRODUCT_ID,
        strategy="category_similarity",
        k=10,
        session_id="regression_test_c",
    )

    assert len(results) > 0
    json.dumps(results)
