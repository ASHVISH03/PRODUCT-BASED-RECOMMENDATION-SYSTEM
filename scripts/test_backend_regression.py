"""
Backend Regression Test Suite
==============================
Run from project root: python scripts/test_backend_regression.py

Executes 3 mandatory backend regression tests:
  TEST A: Personalized Engine with exact 15 browser history IDs
  TEST B: Existing Hybrid Engine strategy=hybrid on product B07KSMBL2H
  TEST C: Existing Category Similarity Engine strategy=category_similarity on product B07KSMBL2H
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.services.recommendation_service import RecommendationService

REAL_BROWSER_HISTORY_IDS = [
    "B0B4HJNPV4", "B0B1YZX72F", "B08L879JSN", "B0B9959XF3", "B0BC9BW512",
    "B0B2RBP83P", "B09PTT8DZF", "B08LW31NQ6", "B09P22HXH6", "B0B1YY6JJL",
    "B0B997FBZT", "B00EDJJ7FS", "B08BJN4MP3", "B09XXZXQC1", "B096MSW6CT"
]

def run_tests():
    db = SessionLocal()
    results_summary = {}
    try:
        svc = RecommendationService(db)
        
        # ─────────────────────────────────────────────────────────────
        # TEST A: Personalized Endpoint with Real 15 History IDs
        # ─────────────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("  TEST A: Personalized Endpoint (15 Real Browser IDs)")
        print("="*60)
        
        res_a = svc.get_personalized_recommendations(
            history_ids=REAL_BROWSER_HISTORY_IDS,
            k=8,
            session_id="regression_test_a"
        )
        
        mode_a = res_a.get("mode")
        history_used_a = res_a.get("history_used", [])
        data_a = res_a.get("data", [])
        rec_ids_a = [item["product_id"] for item in data_a]
        
        # Check history leakage
        history_set = set(REAL_BROWSER_HISTORY_IDS)
        leakage_a = [pid for pid in rec_ids_a if pid in history_set]
        
        print(f"Mode:             {mode_a}")
        print(f"Validated IDs:    {len(history_used_a)} / {len(REAL_BROWSER_HISTORY_IDS)}")
        print(f"Recommendations:  {len(data_a)}")
        print(f"History Leakage:  {len(leakage_a)} items")
        print("\nReturned Recommendation Details:")
        for i, item in enumerate(data_a, 1):
            print(f"  #{i} [{item['product_id']}] {item['product_name'][:40]} | score={item['score']:.4f} | reason: {item['recommendation_reason']}")

        json_str_a = json.dumps(res_a)
        print(f"\nJSON Serializability: PASSED ({len(json_str_a)} bytes)")

        assert mode_a == "personalized", f"Expected mode 'personalized', got '{mode_a}'"
        assert len(data_a) == 8, f"Expected 8 recommendations, got {len(data_a)}"
        assert len(leakage_a) == 0, f"Expected 0 history leakage, got {len(leakage_a)}"

        results_summary["TEST_A"] = {
            "status": "PASSED",
            "mode": mode_a,
            "validated_history_count": len(history_used_a),
            "recommendation_count": len(data_a),
            "history_leakage_count": len(leakage_a),
            "returned_ids": rec_ids_a,
        }

        # ─────────────────────────────────────────────────────────────
        # TEST B: Hybrid Strategy for Product B07KSMBL2H
        # ─────────────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("  TEST B: Hybrid Engine (B07KSMBL2H)")
        print("="*60)
        
        res_b = svc.get_recommendations(
            product_id="B07KSMBL2H",
            strategy="hybrid",
            k=10,
            session_id="regression_test_b"
        )
        rec_ids_b = [item["product_id"] for item in res_b]
        print(f"Recommendations: {len(res_b)}")
        for i, item in enumerate(res_b, 1):
            print(f"  #{i} [{item['product_id']}] {item['product_name'][:40]} | score={item['score']:.4f}")

        json_str_b = json.dumps(res_b)
        print(f"\nJSON Serializability: PASSED ({len(json_str_b)} bytes)")
        assert len(res_b) > 0, "Hybrid engine returned 0 recommendations"

        results_summary["TEST_B"] = {
            "status": "PASSED",
            "recommendation_count": len(res_b),
            "returned_ids": rec_ids_b,
        }

        # ─────────────────────────────────────────────────────────────
        # TEST C: Category Similarity Strategy for Product B07KSMBL2H
        # ─────────────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("  TEST C: Category Similarity Engine (B07KSMBL2H)")
        print("="*60)
        
        res_c = svc.get_recommendations(
            product_id="B07KSMBL2H",
            strategy="category_similarity",
            k=10,
            session_id="regression_test_c"
        )
        rec_ids_c = [item["product_id"] for item in res_c]
        print(f"Recommendations: {len(res_c)}")
        for i, item in enumerate(res_c, 1):
            print(f"  #{i} [{item['product_id']}] {item['product_name'][:40]} | score={item['score']:.4f}")

        json_str_c = json.dumps(res_c)
        print(f"\nJSON Serializability: PASSED ({len(json_str_c)} bytes)")
        assert len(res_c) > 0, "Category similarity engine returned 0 recommendations"

        results_summary["TEST_C"] = {
            "status": "PASSED",
            "recommendation_count": len(res_c),
            "returned_ids": rec_ids_c,
        }

        print("\n" + "="*60)
        print("  ALL BACKEND REGRESSION TESTS PASSED 100%")
        print("="*60 + "\n")
        return results_summary

    except Exception as e:
        import traceback
        print(f"\n❌ REGRESSION TEST FAILED: {e}")
        traceback.print_exc()
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    run_tests()
