from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import time
import json
from pathlib import Path

from app.database import get_db
from app.services.product_service import ProductService
from app.services.recommendation_service import RecommendationService
from app.services.search_service import SearchService
from src.pipelines.training_pipeline import TrainingPipeline

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

def run_training_pipeline():
    try:
        pipeline = TrainingPipeline()
        pipeline.run()
    except Exception as e:
        import logging
        logging.error(f"Background training failed: {e}")

@router.post("/train", tags=["MLOps"])
def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_pipeline)
    return {"status": "accepted", "message": "Training pipeline started in the background."}
