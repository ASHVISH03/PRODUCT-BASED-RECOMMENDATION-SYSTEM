"""
Milestone 5 Repair & Regression Verification Script
===================================================
Run from project root: python scripts/test_milestone5_repairs.py
"""

import sys
import os
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.services.recommendation_service import RecommendationService

REAL_15_IDS = [
    "B0B4HJNPV4", "B0B1YZX72F", "B08L879JSN", "B0B9959XF3", "B0BC9BW512",
    "B0B2RBP83P", "B09PTT8DZF", "B08LW31NQ6", "B09P22HXH6", "B0B1YY6JJL",
    "B0B997FBZT", "B00EDJJ7FS", "B08BJN4MP3", "B09XXZXQC1", "B096MSW6CT"
]

def make_http_post(url, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

def make_http_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

def run_repairs_test():
    print("\n" + "="*70)
    print("  MANDATORY ACCEPTANCE & REGRESSION VERIFICATION")
    print("="*70)

    base_url = "http://127.0.0.1:8000/api/v1/recommendations/personalized"

    # Test A: GET personalized with 15 real history IDs
    print("\n[TEST A] GET personalized with 15 real history IDs")
    get_url = f"{base_url}?history={','.join(REAL_15_IDS)}&k=8"
    status_a, res_a = make_http_get(get_url)
    recs_a = res_a.get("data", [])
    rec_ids_a = [p["product_id"] for p in recs_a]
    leakage_a = len([pid for pid in rec_ids_a if pid in set(REAL_15_IDS)])
    print(f"  HTTP Status: {status_a} | Mode: {res_a.get('mode')} | Count: {res_a.get('count')} | History Leakage: {leakage_a}")
    assert status_a == 200
    assert res_a.get("mode") == "personalized"
    assert len(recs_a) == 8
    assert leakage_a == 0

    # Test B: POST with history + interactions=[]
    print("\n[TEST B] POST with history + empty interactions")
    payload_b = {"interactions": [], "history": REAL_15_IDS, "k": 8}
    status_b, res_b = make_http_post(base_url, payload_b)
    recs_b = res_b.get("data", [])
    print(f"  HTTP Status: {status_b} | Mode: {res_b.get('mode')} | Count: {res_b.get('count')}")
    assert status_b == 200
    assert res_b.get("mode") == "personalized"
    assert len(recs_b) == 8

    # Test C: POST with structured interactions (view, wishlist, cart)
    print("\n[TEST C] POST with structured interactions (view, wishlist, cart)")
    payload_c = {
        "interactions": [
            {"product_id": "B09WNJ6LXZ", "event_type": "view"},
            {"product_id": "B08L879JSN", "event_type": "wishlist"},
            {"product_id": "B0B9959XF3", "event_type": "cart"}
        ],
        "k": 8
    }
    status_c, res_c = make_http_post(base_url, payload_c)
    recs_c = res_c.get("data", [])
    print(f"  HTTP Status: {status_c} | Mode: {res_c.get('mode')} | Dominant Signal: {res_c.get('dominant_signal')} | Count: {res_c.get('count')}")
    assert status_c == 200
    assert res_c.get("mode") == "personalized"
    assert res_c.get("dominant_signal") == "cart"
    assert len(recs_c) == 8

    # Test D: POST / GET with no history or interactions (cold start)
    print("\n[TEST D] Cold start (empty history/interactions)")
    status_d1, res_d1 = make_http_get(base_url)
    print(f"  GET Cold Start HTTP Status: {status_d1} | Mode: {res_d1.get('mode')} | Count: {res_d1.get('count')}")
    assert status_d1 == 200
    assert res_d1.get("mode") == "cold_start"

    status_d2, res_d2 = make_http_post(base_url, {"interactions": [], "history": [], "k": 8})
    print(f"  POST Cold Start HTTP Status: {status_d2} | Mode: {res_d2.get('mode')} | Count: {res_d2.get('count')}")
    assert status_d2 == 200
    assert res_d2.get("mode") == "cold_start"

    # Test E: Verify no NaN / Infinity in JSON
    print("\n[TEST E] Verify JSON serializability & score finiteness")
    for name, res in [("A", res_a), ("B", res_b), ("C", res_c), ("D", res_d1)]:
        dumped = json.dumps(res)
        assert "NaN" not in dumped, f"NaN found in Test {name}"
        assert "Infinity" not in dumped, f"Infinity found in Test {name}"
    print("  ✓ All responses contain valid finite scores and no NaN/Infinity!")

    # Test F: Milestone 4 Hybrid & Category similarity regression check
    print("\n[TEST F] Milestone 4 Hybrid & Category similarity engines regression test")
    status_f1, res_f1 = make_http_get("http://127.0.0.1:8000/api/v1/recommendations/B07KSMBL2H?strategy=hybrid&k=10")
    print(f"  Hybrid Engine Status: {status_f1} | Count: {len(res_f1.get('data', []))}")
    assert status_f1 == 200
    assert len(res_f1.get("data", [])) == 10

    status_f2, res_f2 = make_http_get("http://127.0.0.1:8000/api/v1/recommendations/B07KSMBL2H?strategy=category_similarity&k=10")
    print(f"  Category Similarity Status: {status_f2} | Count: {len(res_f2.get('data', []))}")
    assert status_f2 == 200
    assert len(res_f2.get("data", [])) == 10

    print("\n" + "="*70)
    print("  ALL MANDATORY TESTS (A, B, C, D, E, F) PASSED 100%")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_repairs_test()
