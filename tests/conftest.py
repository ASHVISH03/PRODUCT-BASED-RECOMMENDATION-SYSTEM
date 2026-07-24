"""
Shared pytest fixtures.

Integration tests need the trained model artifacts (models/*.pkl) and the
SQLite database (data/external/ecommerce.db) on disk — these are produced
by the training pipeline / dataset setup and are gitignored, so they won't
exist on a bare CI checkout until the pipeline has run once. Tests that need
them are skipped (not failed) when they're absent.
"""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "external" / "ecommerce.db"
MODEL_FILES = [
    PROJECT_ROOT / "models" / "tfidf.pkl",
    PROJECT_ROOT / "models" / "similarity.pkl",
    PROJECT_ROOT / "models" / "product_index.pkl",
]

requires_trained_artifacts = pytest.mark.skipif(
    not DB_PATH.exists() or not all(p.exists() for p in MODEL_FILES),
    reason="Trained model artifacts / database not present — run the training "
    "pipeline (python -m src.pipelines.training_pipeline) first.",
)


@pytest.fixture
def db_session():
    from app.database import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def recommendation_service(db_session):
    from app.services.recommendation_service import RecommendationService

    return RecommendationService(db_session)
