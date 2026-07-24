from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import time
import json
from pathlib import Path

from app.database import get_db
from app.schemas.recommendations import PersonalizedRecommendationRequest
from app.services.product_service import ProductService
from app.services.recommendation_service import RecommendationService
from app.services.search_service import SearchService
from app.services.training_service import run_training_pipeline

router = APIRouter()

@router.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "Product Recommendation API is running."}

@router.get("/metrics", tags=["System"])
def get_metrics():
    try:
        eval_path = Path("artifacts/evaluation_report.json")
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"status": "success", "data": data}
        return {"status": "error", "message": "No metrics available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", tags=["System"])
def get_dataset_stats(db: Session = Depends(get_db)):
    try:
        import pandas as pd
        df = pd.read_sql_query('SELECT * FROM products', db.bind)
        categories = df['category'].str.split('|').str[0]
        stats = {
            'total_products': len(df),
            'unique_categories': int(df['category'].nunique()),
            'top_level_count': int(categories.nunique()),
            'top_levels': categories.value_counts().to_dict(),
            'missing_images': int(df['img_link'].isna().sum() + (df['img_link'] == '').sum()),
            'missing_prices': int(df['discounted_price'].isna().sum() + (df['discounted_price'] == '').sum()),
            'missing_ratings': int(df['rating'].isna().sum() + (df['rating'] == '').sum()),
            'duplicate_ids': int(df.duplicated(subset=['product_id']).sum())
        }
        return stats
    except Exception as e:
        return {"error": str(e)}

@router.get("/products/categories", tags=["Products"])
def get_categories(db: Session = Depends(get_db)):
    svc = ProductService(db)
    return {"status": "success", "data": svc.get_product_categories()}

@router.get("/products", tags=["Products"])
def get_products(
    skip: int = Query(0, ge=0), 
    limit: int = Query(50, ge=1, le=100),
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    svc = ProductService(db)
    products = svc.get_products(skip, limit, category)
    return {"status": "success", "count": len(products), "data": products}

@router.get("/products/{product_id}", tags=["Products"])
def get_product(product_id: str, db: Session = Depends(get_db)):
    svc = ProductService(db)
    product = svc.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"status": "success", "data": product}

@router.get("/search", tags=["Search"])
def search_products(
    q: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    svc = SearchService(db)
    results = svc.search(q, limit)
    return {"status": "success", "count": len(results), "data": results}

@router.get("/recommendations/personalized", tags=["Recommendations"])
def get_personalized_recommendations(
    history: str = Query("", description="Comma-separated product IDs in viewing order, newest-first. Empty triggers cold start."),
    k: int = Query(8, ge=1, le=20, description="Number of recommendations"),
    session_id: str = Query("anonymous"),
    db: Session = Depends(get_db)
):
    """
    Generate personalized AI Picks from the user's session browsing history.

    - history: comma-separated product_ids (newest first). Empty string triggers cold-start.
    - Mode 'personalized': genuine TF-IDF user profile + MMR reranking.
    - Mode 'cold_start': trending products (no history available).
    - Mode 'fallback': trending (personalization returned empty — rare edge case).
    """
    import logging
    logger = logging.getLogger(__name__)
    history_ids = [pid.strip() for pid in history.split(",") if pid.strip()]
    logger.info(f"[API Route /recommendations/personalized] Received history param: '{history}' -> parsed IDs: {history_ids}")
    svc = RecommendationService(db)
    try:
        result = svc.get_personalized_recommendations(
            history_ids=history_ids,
            k=k,
            session_id=session_id,
        )
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"[API Route /recommendations/personalized] ERROR: {str(e)}\n{tb_str}")
        raise HTTPException(status_code=500, detail=f"Personalization error: {str(e)}\n{tb_str}")
    return {
        "status": "success",
        "mode": result["mode"],
        "history_used": result["history_used"],
        "count": result["count"],
        "data": result["data"],
    }

@router.post("/recommendations/personalized", tags=["Recommendations"])
def post_personalized_recommendations(
    payload: PersonalizedRecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate multi-signal personalized AI Picks from structured user interaction events.

    - Supports event types: view (1.0), repeat_view (1.5), wishlist (2.0), cart (3.0), purchase (5.0).
    - Recency decay factor: 0.85 per position.
    - MMR diversity reranking: lambda = 0.60.
    """
    import logging
    logger = logging.getLogger(__name__)
    svc = RecommendationService(db)
    try:
        raw_interactions = [item.model_dump() for item in payload.interactions]
        result = svc.get_personalized_recommendations_multi_signal(
            interactions=raw_interactions,
            history_ids=payload.history,
            k=payload.k,
            session_id=payload.session_id,
        )
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"[API Route POST /recommendations/personalized] ERROR: {str(e)}\n{tb_str}")
        raise HTTPException(status_code=500, detail=f"Multi-signal personalization error: {str(e)}\n{tb_str}")

    return {
        "status": "success",
        "mode": result["mode"],
        "history_used": result["history_used"],
        "dominant_signal": result.get("dominant_signal"),
        "count": result["count"],
        "data": result["data"],
    }

@router.get("/recommendations/{product_id}", tags=["Recommendations"])
def get_recommendations(
    product_id: str,
    strategy: str = Query("hybrid", description="Engine strategy: hybrid, content_based, popularity, trending, category_similarity, price_similarity, brand_similarity, frequently_bought"),
    k: int = Query(5, ge=1, le=20),
    session_id: str = Query("anonymous"),
    db: Session = Depends(get_db)
):
    svc = RecommendationService(db)
    results = svc.get_recommendations(product_id, strategy, k, session_id)
    if not results:
        raise HTTPException(status_code=404, detail="Product not found or no recommendations available")
    return {"status": "success", "strategy": strategy, "count": len(results), "data": results}

@router.post("/train", tags=["MLOps"])
def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_pipeline)
    return {"status": "accepted", "message": "Training pipeline started in the background."}
