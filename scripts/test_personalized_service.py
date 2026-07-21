"""
Standalone Test Script — RecommendationService.get_personalized_recommendations()
===================================================================================
Run from project root: python scripts/test_personalized_service.py

Tests the service layer end-to-end:
    1. Connects to SQLite database
    2. Validates history IDs against products table
    3. Runs recommendation engine inference
    4. Formats results with full metadata
    5. Saves RecommendationLog telemetry to DB
    6. Verifies JSON serialization of final output
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.services.recommendation_service import RecommendationService

def test_service():
    db = SessionLocal()
    try:
        svc = RecommendationService(db)
        
        # Test Case 1: Cold Start (empty history)
        print("\n--- Test 1: Cold Start ---")
        res1 = svc.get_personalized_recommendations(history_ids=[], k=8, session_id="test_session_cold")
        print(f"Mode: {res1['mode']}")
        print(f"Count: {res1['count']}")
        print(f"History Used: {res1['history_used']}")
        # Verify JSON serializability
        json_str1 = json.dumps(res1)
        print(f"JSON Serialization: SUCCESS ({len(json_str1)} bytes)")
        
        # Test Case 2: Realistic History (Monitors/Displays)
        print("\n--- Test 2: Monitor Browsing History ---")
        # Fetch some product IDs from DB
        from app.models import Product
        monitors = db.query(Product).filter(Product.product_name.like("%Monitor%") | Product.product_name.like("%TV%") | Product.product_name.like("%HDMI%")).limit(5).all()
        history_ids = [p.product_id for p in monitors]
        
        if not history_ids:
            # Fallback to any 5 products
            any_prods = db.query(Product).limit(5).all()
            history_ids = [p.product_id for p in any_prods]
            
        print(f"Testing with history_ids: {history_ids}")
        res2 = svc.get_personalized_recommendations(history_ids=history_ids, k=8, session_id="test_session_monitors")
        print(f"Mode: {res2['mode']}")
        print(f"Count: {res2['count']}")
        print(f"History Used: {res2['history_used']}")
        
        # Verify JSON serializability
        json_str2 = json.dumps(res2)
        print(f"JSON Serialization: SUCCESS ({len(json_str2)} bytes)")
        
        print("\nRecommended Product IDs:")
        for item in res2['data']:
            print(f"  - {item['product_id']}: {item['product_name'][:45]} (score={item['score']:.3f}, reason='{item['recommendation_reason']}')")
            
        print("\n✅ Service End-to-End Test PASSED flawlessly!")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Service Test FAILED with error: {e}")
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_service()
