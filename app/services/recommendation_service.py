from sqlalchemy.orm import Session
from src.components.recommendation.recommendation_engine import RecommendationEngine
from src.config.configuration import ConfigurationManager
from app.models import RecommendationLog, Product
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

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

    def get_personalized_recommendations(
        self,
        history_ids: List[str],
        k: int = 8,
        session_id: str = "anonymous",
        interaction_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate personalized recommendations from a session browsing history.

        Cold-start: if history_ids is empty or all IDs are unknown,
        falls back to trending_products() and signals cold_start=True in the response.

        Args:
            history_ids:       Product IDs in viewing order, newest-first.
            k:                 Number of recommendations.
            session_id:        Session identifier for telemetry logging.
            interaction_types: Optional dict mapping product_id → interaction type.

        Returns:
            Dict with keys: 'data' (list of enriched dicts), 'mode' ('personalized'|'cold_start'),
                            'history_used' (list of valid IDs sent to engine),
                            'count' (int).
        """
        # Validate history against the database
        valid_ids: List[str] = []
        for pid in history_ids:
            product = self.db.query(Product).filter(Product.product_id == pid).first()
            if product:
                valid_ids.append(pid)
            else:
                logger.warning(f"[PersonalizedRecommendationService] Unknown product_id in history: {pid}")

        logger.info(
            f"[PersonalizedRecommendationService] history={history_ids} "
            f"valid_ids={valid_ids} session={session_id}"
        )

        # Cold-start detection
        if not valid_ids:
            logger.info("[PersonalizedRecommendationService] Cold start: routing to trending products.")
            results = self.engine.trending_products(top_k=k)
            output = self._format_results(results, source_id=None, session_id=session_id)
            return {"data": output, "mode": "cold_start", "history_used": [], "count": len(output)}

        # Personalized recommendations
        results = self.engine.recommend_personalized(
            history_ids=valid_ids,
            interaction_types=interaction_types,
            top_k=k,
            candidate_pool=max(k * 6, 50),
        )

        if not results:
            # Fallback: personalization returned nothing (e.g. all scores 0)
            logger.warning(
                "[PersonalizedRecommendationService] Engine returned empty — "
                "falling back to trending."
            )
            results = self.engine.trending_products(top_k=k)
            output = self._format_results(results, source_id=None, session_id=session_id)
            return {"data": output, "mode": "fallback", "history_used": valid_ids, "count": len(output)}

        output = self._format_results(results, source_id=valid_ids[0], session_id=session_id)
        return {"data": output, "mode": "personalized", "history_used": valid_ids, "count": len(output)}

    def _format_results(
        self,
        results,
        source_id: str,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Format RecommendationResult list into API-ready dicts and log telemetry."""
        import math
        import pandas as pd
        output: List[Dict[str, Any]] = []
        logs: List[RecommendationLog] = []

        def _clean_float(val, default=0.0) -> float:
            try:
                if val is None or pd.isna(val):
                    return float(default)
                f = float(val)
                return float(default) if math.isnan(f) or math.isinf(f) else f
            except (ValueError, TypeError):
                return float(default)

        def _clean_int(val, default=0) -> int:
            try:
                if val is None or pd.isna(val):
                    return int(default)
                f = float(val)
                return int(default) if math.isnan(f) or math.isinf(f) else int(f)
            except (ValueError, TypeError):
                return int(default)

        def _clean_str(val, default="") -> str:
            if val is None or pd.isna(val):
                return default
            s = str(val).strip()
            return default if s.lower() == "nan" else s

        for res in results:
            rec_product = self.db.query(Product).filter(
                Product.product_id == res.product_id
            ).first()

            if rec_product:
                score_val = _clean_float(res.confidence_score)
                item = {
                    "product_id": _clean_str(rec_product.product_id),
                    "product_name": _clean_str(rec_product.product_name),
                    "img_link": _clean_str(rec_product.img_link),
                    "discounted_price": _clean_float(rec_product.discounted_price),
                    "actual_price": _clean_float(rec_product.actual_price),
                    "discount_percentage": _clean_float(rec_product.discount_percentage),
                    "rating": _clean_float(rec_product.rating),
                    "rating_count": _clean_int(rec_product.rating_count),
                    "brand": _clean_str(getattr(rec_product, "brand", res.brand)),
                    "category": _clean_str(rec_product.category),
                    "score": score_val,
                    "confidence_grade": _clean_str(res.confidence_grade, "Medium"),
                    "reason": res.reason_dict if isinstance(res.reason_dict, dict) else {},
                    "recommendation_reason": _clean_str(res.recommendation_reason),
                }
                output.append(item)

                logs.append(RecommendationLog(
                    session_id=str(session_id),
                    source_product_id=source_id,  # None is allowed (nullable=True)
                    recommended_product_id=str(res.product_id),
                    engine="personalized",
                    score=score_val,
                ))

        if logs:
            try:
                self.db.bulk_save_objects(logs)
                self.db.commit()
            except Exception as e:
                logger.warning(f"[RecommendationService] Telemetry logging failed: {e}")
                self.db.rollback()

        return output
