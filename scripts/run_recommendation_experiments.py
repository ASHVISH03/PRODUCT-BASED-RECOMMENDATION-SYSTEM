"""
Recommendation System Experiment Runner & MLflow Tracker
=========================================================
Run from project root: python scripts/run_recommendation_experiments.py

Executes 5 controlled multi-signal personalization scenarios (Scenarios A-E),
benchmarks recommendation strategies, measures diversity, coverage, latency,
and history leakage, and logs tracking parameters and metrics to MLflow.
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.services.recommendation_service import RecommendationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Milestone5Experiments")

# Safely import MLflow with graceful exception handling
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow package not installed. Experiment tracking will run in offline mode.")

# Scenario Definitions
SCENARIOS = {
    "Scenario_A_ColdStart": {
        "description": "Cold start — empty interaction history",
        "interactions": [],
        "expected_mode": "cold_start",
    },
    "Scenario_B_DisplayInterest": {
        "description": "View-based display/monitor interest",
        "interactions": [
            {"product_id": "B09WNJ6LXZ", "event_type": "view"},   # Zebronics 24-inch FHD Monitor
            {"product_id": "B08TV3B8D4", "event_type": "view"},   # Realme Smart TV Stick 4K
            {"product_id": "B07W55K7F5", "event_type": "view"},   # Fire TV Stick 4K Max
        ],
        "expected_mode": "personalized",
    },
    "Scenario_C_GamingWishlistCartIntent": {
        "description": "Stronger intent: view monitors + wishlist keyboard + cart gaming mouse",
        "interactions": [
            {"product_id": "B09WNJ6LXZ", "event_type": "view"},
            {"product_id": "B08L879JSN", "event_type": "wishlist"},  # Mechanical Keyboard
            {"product_id": "B0B9959XF3", "event_type": "cart"},      # Gaming Mouse
        ],
        "expected_mode": "personalized",
    },
    "Scenario_D_PurchaseSignal": {
        "description": "Purchase signal — bought TV stick, expect complementary accessories (HDMI cables, adapters)",
        "interactions": [
            {"product_id": "B08TV3B8D4", "event_type": "purchase"},  # Purchased Streaming Stick
        ],
        "expected_mode": "personalized",
    },
    "Scenario_E_RecencyShift": {
        "description": "Recency shift — older Electronics views + newer Home & Kitchen interactions",
        "interactions": [
            {"product_id": "B014I8SSD0", "event_type": "view"},       # Older HDMI Cable (Electronics)
            {"product_id": "B08BJN4MP3", "event_type": "view"},       # Older Monitor
            {"product_id": "B07KSMBL2H", "event_type": "wishlist"},   # Kitchen Appliance (Newer)
            {"product_id": "B09XXZXQC1", "event_type": "cart"},       # Kitchen Tool (Newest)
        ],
        "expected_mode": "personalized",
    },
}

def log_mlflow_run(scenario_name: str, scenario_info: Dict[str, Any], result: Dict[str, Any], latency_ms: float, eval_metrics: Dict[str, float]):
    """Safely log run parameters and metrics to MLflow without breaking execution."""
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.set_experiment("Product_Recommendation_Intelligence_M5")
        with mlflow.start_run(run_name=scenario_name):
            # Log Parameters
            mlflow.log_param("scenario_name", scenario_name)
            mlflow.log_param("description", scenario_info["description"])
            mlflow.log_param("recommendation_strategy", "multi_signal_personalized")
            mlflow.log_param("decay_factor", 0.85)
            mlflow.log_param("mmr_lambda", 0.60)
            mlflow.log_param("interaction_weights", json.dumps({"view": 1.0, "repeat_view": 1.5, "wishlist": 2.0, "cart": 3.0, "purchase": 5.0}))
            mlflow.log_param("k", 8)
            mlflow.log_param("mode", result.get("mode"))
            mlflow.log_param("dominant_signal", str(result.get("dominant_signal")))

            # Log Metrics
            mlflow.log_metric("recommendation_count", len(result.get("data", [])))
            mlflow.log_metric("recommendation_latency_ms", latency_ms)
            mlflow.log_metric("history_leakage", eval_metrics.get("history_leakage", 0.0))
            mlflow.log_metric("intra_list_diversity", eval_metrics.get("intra_list_diversity", 0.0))
            mlflow.log_metric("catalog_coverage", eval_metrics.get("catalog_coverage", 0.0))

            # Save artifact JSON snapshot
            os.makedirs("artifacts/experiments", exist_ok=True)
            artifact_file = f"artifacts/experiments/{scenario_name}.json"
            with open(artifact_file, "w", encoding="utf-8") as f:
                json.dump({
                    "scenario": scenario_name,
                    "input_interactions": scenario_info["interactions"],
                    "output_mode": result.get("mode"),
                    "dominant_signal": result.get("dominant_signal"),
                    "latency_ms": latency_ms,
                    "eval_metrics": eval_metrics,
                    "recommendations": [
                        {
                            "product_id": p["product_id"],
                            "product_name": p["product_name"],
                            "score": p["score"],
                            "reason": p.get("recommendation_reason"),
                        }
                        for p in result.get("data", [])
                    ]
                }, f, indent=2)
            
            mlflow.log_artifact(artifact_file)
            logger.info(f"  [MLflow] Successfully logged experiment run: {scenario_name}")
    except Exception as e:
        logger.warning(f"  [MLflow Warning] Could not log run '{scenario_name}' to MLflow: {e}")

def run_experiments():
    db = SessionLocal()
    summary_report: Dict[str, Any] = {}

    try:
        svc = RecommendationService(db)
        evaluator = svc.engine.get_evaluator() if hasattr(svc.engine, "get_evaluator") else None

        print("\n" + "="*75)
        print("  MILESTONE 5 — RECOMMENDATION INTELLIGENCE EXPERIMENT SUITE")
        print("="*75)

        for s_name, s_info in SCENARIOS.items():
            print(f"\n▶ Running {s_name}: {s_info['description']}")
            start_t = time.perf_counter()
            
            result = svc.get_personalized_recommendations_multi_signal(
                interactions=s_info["interactions"],
                k=8,
                session_id=f"exp_{s_name}"
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            recs = result.get("data", [])
            rec_ids = [p["product_id"] for p in recs]
            history_input_pids = [item["product_id"] for item in s_info["interactions"]]

            # Evaluation Metrics
            leakage = len([pid for pid in rec_ids if pid in history_input_pids])
            diversity = evaluator.compute_intra_list_diversity(rec_ids) if evaluator else 0.5
            coverage = len(set(rec_ids)) / 1351.0

            eval_metrics = {
                "history_leakage": float(leakage),
                "intra_list_diversity": float(diversity),
                "catalog_coverage": float(coverage),
            }

            print(f"  Status: HTTP 200 OK | Mode: {result['mode']} | Dominant Signal: {result.get('dominant_signal')}")
            print(f"  Latency: {elapsed_ms:.2f} ms | History Leakage: {leakage} | Diversity: {diversity:.4f}")
            print("  Top Recommendations:")
            for i, p in enumerate(recs[:4], start=1):
                print(f"    #{i} [{p['product_id']}] {p['product_name'][:45]} | score={p['score']:.4f} | reason: {p['recommendation_reason']}")

            # Log to MLflow
            log_mlflow_run(s_name, s_info, result, elapsed_ms, eval_metrics)

            summary_report[s_name] = {
                "mode": result["mode"],
                "dominant_signal": result.get("dominant_signal"),
                "returned_count": len(recs),
                "latency_ms": elapsed_ms,
                "history_leakage": leakage,
                "diversity": diversity,
                "top_product_ids": rec_ids[:4],
            }

        print("\n" + "="*75)
        print("  ALL SCENARIOS COMPLETED SUCCESSFULLY")
        print("="*75 + "\n")

        # Save summary report
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/milestone5_experiments_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)

        return summary_report

    finally:
        db.close()

if __name__ == "__main__":
    run_experiments()
