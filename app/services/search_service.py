from sqlalchemy.orm import Session
from src.components.recommendation.recommendation_engine import RecommendationEngine
from src.config.configuration import ConfigurationManager
from app.models import Product
from typing import List, Dict, Any

class SearchService:
    def __init__(self, db: Session):
        self.db = db
        config_manager = ConfigurationManager()
        model_dir = config_manager.get_model_config().artifacts_dir
        self.engine = RecommendationEngine.load(models_dir=model_dir)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Uses the ML engine's search capability which includes typo correction
        if rapidfuzz is available.
        """
        results = self.engine.search(query, top_k=limit)
        
        output = []
        for res in results:
            product = self.db.query(Product).filter(Product.product_id == res.product_id).first()
            if product:
                item = {
                    "product_id": product.product_id,
                    "product_name": product.product_name,
                    "img_link": product.img_link,
                    "price": product.discounted_price,
                    "rating": product.rating,
                    "score": res.confidence_score,
                    "confidence_grade": res.confidence_grade,
                    "reason": res.reason_dict
                }
                output.append(item)
                
        return output
