from sqlalchemy.orm import Session
from src.components.recommendation.recommendation_engine import RecommendationEngine
from src.config.configuration import ConfigurationManager
from app.models import RecommendationLog, Product
from typing import List, Dict, Any, Optional

class RecommendationService:
    def __init__(self, db: Session):
        self.db = db
        # Initialize the core engine
        config_manager = ConfigurationManager()
        model_dir = config_manager.get_model_config().artifacts_dir
        self.engine = RecommendationEngine.load(models_dir=model_dir)

    def get_recommendations(
        self, 
        product_id: str, 
        strategy: str = "hybrid", 
        k: int = 5,
        session_id: str = "anonymous"
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for a product and log them.
        """
        # Ensure product exists in our DB
        product = self.db.query(Product).filter(Product.product_id == product_id).first()
        if not product:
            return []

        # Get recommendations from the ML engine
        results = self.engine.recommend(product_id, strategy=strategy, top_k=k)
        
        output = []
        logs = []
        
        for res in results:
            # We want to return enriched product details from DB
            rec_product = self.db.query(Product).filter(Product.product_id == res.product_id).first()
            
            if rec_product:
                item = {
                    "product_id": rec_product.product_id,
                    "product_name": rec_product.product_name,
                    "img_link": rec_product.img_link,
                    "price": rec_product.discounted_price,
                    "rating": rec_product.rating,
                    "score": res.confidence_score,
                    "confidence_grade": res.confidence_grade,
                    "reason": res.reason_dict
                }
                output.append(item)
                
                # Create a log entry
                log_entry = RecommendationLog(
                    session_id=session_id,
                    source_product_id=product_id,
                    recommended_product_id=res.product_id,
                    engine=strategy,
                    score=res.confidence_score
                )
                logs.append(log_entry)
                
        # Bulk save the logs to track CTR later
        if logs:
            self.db.bulk_save_objects(logs)
            self.db.commit()
            
        return output
