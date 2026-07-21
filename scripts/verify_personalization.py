"""
Verification Script — Personalization Engine Scenarios A, B, C
================================================================
Run from the project root: python scripts/verify_personalization.py

This script tests the full personalization pipeline with 3 controlled scenarios:
    Scenario A: Cold Start (no history)
    Scenario B: Monitor/Display interest (4-5 monitor products)
    Scenario C: Home & Kitchen interest (4-5 different products)

Each scenario logs:
    - history_ids
    - mode (personalized / cold_start / fallback)
    - raw relevance scores
    - final recommendation IDs
    - recommendation reasons
    - diversity metric (intra-list diversity)

Expected result: Scenario B and C should produce materially different recommendation sets.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.recommendation.recommendation_engine import RecommendationEngine
from src.engines.recommendation_evaluator import RecommendationEvaluator

MODELS_DIR = "models"

print("=" * 70)
print("  PERSONALIZATION ENGINE VERIFICATION")
print("=" * 70)

# Load engine
print("\n[1/4] Loading recommendation engine...")
engine = RecommendationEngine.load(models_dir=MODELS_DIR)
print(f"  PersonalizedEngine initialized: {engine._personalized_engine is not None}")
print(f"  Catalog size: {engine.get_catalog_size()}")

evaluator = engine.get_evaluator()

# ─── Helper ────────────────────────────────────────────────────────────────
def run_scenario(name, history_ids, engine, evaluator):
    print(f"\n{'─'*60}")
    print(f"  SCENARIO {name}")
    print(f"  History: {history_ids}")
    print(f"{'─'*60}")

    if not history_ids:
        # Cold start: get trending
        results = engine.trending_products(top_k=8)
        mode = "cold_start"
    else:
        results = engine.recommend_personalized(history_ids=history_ids, top_k=8)
        mode = "personalized"

    if not results:
        print("  [!] Engine returned EMPTY — possible fallback needed")
        return []

    print(f"\n  Mode: {mode}")
    print(f"  Recommendations ({len(results)} total):")
    rec_ids = []
    for i, r in enumerate(results, 1):
        rec_ids.append(r.product_id)
        print(
            f"    #{i:2d} {r.product_id:15s} | score={r.confidence_score:.4f} "
            f"| {r.product_name[:45]}"
        )
        print(f"         reason: {r.recommendation_reason}")

    # Evaluation metrics
    metrics = evaluator.evaluate_session(
        history_ids=history_ids,
        recommended_ids=rec_ids,
    )
    print(f"\n  Metrics:")
    print(f"    Intra-list Diversity:   {metrics['diversity_intra_list']:.4f}")
    print(f"    History Leakage:        {int(metrics['history_leakage'])} (should be 0)")
    print(f"    Coverage (this session):{metrics['coverage_session']:.6f}")

    return rec_ids


# ─── SCENARIO A: Cold Start ─────────────────────────────────────────────────
print("\n[2/4] Scenario A — Cold Start (no history)...")
scenario_a_ids = run_scenario("A — Cold Start", [], engine, evaluator)

# ─── Find Monitor/Display products in catalog ─────────────────────────────
print("\n[3/4] Finding monitor/display products in catalog...")
monitor_results = engine.search("monitor display screen", top_k=10)
if not monitor_results:
    monitor_results = engine.search("television TV LED", top_k=10)

if monitor_results:
    monitor_history = [r.product_id for r in monitor_results[:5]]
    print(f"  Found monitor products: {monitor_history}")
    for r in monitor_results[:5]:
        print(f"    {r.product_id}: {r.product_name[:60]}")
else:
    # Use first 5 products from Electronics
    monitor_history = [r.product_id for r in engine.trending_products(top_k=5)]
    print(f"  Fallback monitor history: {monitor_history}")

# ─── SCENARIO B: Monitor Interest ──────────────────────────────────────────
scenario_b_ids = run_scenario("B — Monitor/Display Interest", monitor_history, engine, evaluator)

# ─── Find Home & Kitchen products ──────────────────────────────────────────
print("\n[4/4] Finding Home & Kitchen products...")
kitchen_results = engine.search("kitchen water bottle heater", top_k=10)
if kitchen_results:
    kitchen_history = [r.product_id for r in kitchen_results[:5]]
    print(f"  Found kitchen products: {kitchen_history}")
    for r in kitchen_results[:5]:
        print(f"    {r.product_id}: {r.product_name[:60]}")
else:
    kitchen_history = []
    print("  No kitchen products found in search.")

# ─── SCENARIO C: Home & Kitchen Interest ────────────────────────────────────
if kitchen_history:
    scenario_c_ids = run_scenario("C — Home & Kitchen Interest", kitchen_history, engine, evaluator)
else:
    scenario_c_ids = []
    print("\n  Scenario C skipped (no kitchen products found)")

# ─── COMPARISON ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  SCENARIO COMPARISON")
print(f"{'=' * 70}")

set_a = set(scenario_a_ids)
set_b = set(scenario_b_ids)
set_c = set(scenario_c_ids)

print(f"\n  Scenario A (cold start): {len(set_a)} unique IDs")
print(f"  Scenario B (monitors):   {len(set_b)} unique IDs")
print(f"  Scenario C (kitchen):    {len(set_c)} unique IDs")

overlap_ab = len(set_a & set_b)
overlap_ac = len(set_a & set_c)
overlap_bc = len(set_b & set_c)

print(f"\n  A ∩ B overlap: {overlap_ab}/8 items (< 4 means personalization changed picks)")
print(f"  A ∩ C overlap: {overlap_ac}/8 items")
print(f"  B ∩ C overlap: {overlap_bc}/8 items (< 4 means B & C produced different interests)")

if overlap_bc < 4:
    print("\n  ✅ PASS: Scenario B and C produced materially different recommendations.")
    print("  ✅ Personalization is interest-specific, not random.")
else:
    print("\n  ⚠️  WARNING: B and C overlap significantly. May indicate:")
    print("     - Dataset too homogeneous (cable-heavy Dataset V1)")
    print("     - Profile vectors too similar across categories")
    print("     - Increase candidate_pool or reduce MMR lambda")

if overlap_ab < 4:
    print("  ✅ PASS: Personalized picks (B) differ from cold-start (A).")
else:
    print("  ⚠️  Cold start and personalized picks are too similar.")

print(f"\n{'=' * 70}")
print("  Verification complete. Check uvicorn logs for detailed debug output.")
print(f"{'=' * 70}\n")
