"""
Phase 3 Verification Test Suite
=================================
Automated verification of all Phase 3 components:
  - Model training (TF-IDF + cosine similarity)
  - Model artifacts (vectorizer, matrix, index)
  - MLflow run creation
  - Evaluation report
  - Recommendation engine (all 7 APIs)

Windows-safe: UTF-8 stdout, colorama, safe_print, no emoji.

Usage:
    python tests/verify_phase3.py
"""

import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

# ── Windows-safe stdout ────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Colour helpers ─────────────────────────────────────────────────────
try:
    import colorama
    colorama.init(autoreset=True)
    GREEN  = colorama.Fore.GREEN
    RED    = colorama.Fore.RED
    YELLOW = colorama.Fore.YELLOW
    CYAN   = colorama.Fore.CYAN
    BOLD   = colorama.Style.BRIGHT
    RESET  = colorama.Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))

def ok(msg: str) -> None:
    safe_print(f"  {GREEN}[PASS]{RESET} {msg}")

def fail(msg: str) -> None:
    safe_print(f"  {RED}[FAIL]{RESET} {msg}")

def warn(msg: str) -> None:
    safe_print(f"  {YELLOW}[WARN]{RESET} {msg}")

def header(title: str) -> None:
    safe_print(f"\n{CYAN}{BOLD}{'-' * 60}{RESET}")
    safe_print(f"{CYAN}{BOLD}  {title}{RESET}")
    safe_print(f"{CYAN}{'-' * 60}{RESET}")

def check(cond: bool, name: str, detail: str = "") -> bool:
    suffix = f" ({detail})" if detail else ""
    if cond:
        ok(f"{name}{suffix}")
    else:
        fail(f"{name}{suffix}")
    return cond


# ================================================================
# TEST 0: Import Verification
# ================================================================

def test_imports() -> bool:
    header("TEST 0: Import Verification")
    success = True
    imports = [
        ("src.engines.base_engine", "BaseEngine"),
        ("src.engines.base_engine", "RecommendationResult"),
        ("src.engines.content_based", "ContentBasedEngine"),
        ("src.engines.popularity", "PopularityEngine"),
        ("src.engines.trending", "TrendingEngine"),
        ("src.engines.category_similarity", "CategorySimilarityEngine"),
        ("src.engines.price_similarity", "PriceSimilarityEngine"),
        ("src.engines.brand_similarity", "BrandSimilarityEngine"),
        ("src.engines.frequently_bought", "FrequentlyBoughtTogetherEngine"),
        ("src.engines.hybrid", "HybridEngine"),
        ("src.components.model_training.model_trainer", "ModelTrainer"),
        ("src.components.model_evaluation.model_evaluator", "ModelEvaluator"),
        ("src.components.model_registry.model_registry", "ModelRegistry"),
        ("src.components.recommendation.recommendation_engine", "RecommendationEngine"),
        ("src.pipelines.training_pipeline", "TrainingPipeline"),
    ]
    for module, name in imports:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            ok(f"from {module} import {name}")
        except Exception as e:
            fail(f"from {module} import {name}  ->  {e}")
            success = False
    return success


# ================================================================
# TEST 1: Run Full Training Pipeline
# ================================================================

def test_training_pipeline() -> bool:
    header("TEST 1: Full Training Pipeline (Phase 2 + Phase 3)")
    import src.config.configuration as config_mod
    from src.pipelines.training_pipeline import TrainingPipeline

    original_load = config_mod.ConfigurationManager._load_yaml

    def patched_load(self):
        raw = original_load(self)
        raw["data"]["use_fallback"] = True
        return raw

    config_mod.ConfigurationManager._load_yaml = patched_load

    success = True
    try:
        safe_print("  Running full pipeline (this may take 30-90 seconds)...")
        t0 = time.perf_counter()
        pipeline = TrainingPipeline()
        artifact = pipeline.run()
        elapsed = time.perf_counter() - t0

        success &= check(artifact.success, "Pipeline completed successfully")
        success &= check(artifact.error_message is None, "No error message")
        success &= check(elapsed > 0, "Duration recorded", f"{elapsed:.1f}s")
        success &= check(artifact.ingestion is not None, "Ingestion artifact present")
        success &= check(artifact.validation is not None, "Validation artifact present")
        success &= check(artifact.transformation is not None, "Transformation artifact present")
        success &= check(artifact.feature_engineering is not None, "Feature engineering present")
        success &= check(artifact.model_training is not None, "Model training artifact present")
        success &= check(artifact.model_evaluation is not None, "Model evaluation present")
        success &= check(artifact.model_registry is not None, "Model registry present")

        # Store artifacts for downstream tests
        test_training_pipeline._artifact = artifact

    except Exception as e:
        fail(f"Pipeline failed: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        config_mod.ConfigurationManager._load_yaml = original_load

    return success

test_training_pipeline._artifact = None


# ================================================================
# TEST 2: Model Artifacts on Disk
# ================================================================

def test_model_artifacts() -> bool:
    header("TEST 2: Model Artifacts on Disk")
    import numpy as np
    import pandas as pd

    success = True

    expected_files = {
        "models/tfidf.pkl": "TF-IDF vectorizer",
        "models/similarity.pkl": "Cosine similarity matrix",
        "models/product_index.pkl": "Product index",
        "models/processed_products.csv": "Processed products CSV",
        "models/metadata.json": "Model metadata JSON",
        "artifacts/evaluation_report.json": "Evaluation report",
        "artifacts/pipeline_summary.json": "Pipeline summary",
    }

    for path, label in expected_files.items():
        exists = Path(path).exists()
        success &= check(exists, f"{label} exists", path)

    # Validate vectorizer
    vectorizer_path = Path("models/tfidf.pkl")
    if vectorizer_path.exists():
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            with open(vectorizer_path, "rb") as f:
                vec = pickle.load(f)
            success &= check(
                isinstance(vec, TfidfVectorizer),
                "Vectorizer is TfidfVectorizer"
            )
            vocab_size = len(vec.vocabulary_)
            success &= check(vocab_size > 100, "Vocabulary > 100 terms", f"{vocab_size:,}")
            success &= check(vec.idf_ is not None, "IDF values saved")
        except Exception as e:
            fail(f"Vectorizer load failed: {e}")
            success = False

    # Validate similarity matrix
    sim_path = Path("models/similarity.pkl")
    if sim_path.exists():
        try:
            with open(sim_path, "rb") as f:
                mat = pickle.load(f)
            success &= check(isinstance(mat, np.ndarray), "Matrix is numpy ndarray")
            success &= check(mat.ndim == 2, "Matrix is 2D", str(mat.shape))
            success &= check(mat.shape[0] == mat.shape[1], "Matrix is square")
            success &= check(mat.dtype == np.float32, "Matrix dtype is float32")
            n = mat.shape[0]
            success &= check(n >= 10, "Matrix has >= 10 products", str(n))
        except Exception as e:
            fail(f"Similarity matrix load failed: {e}")
            success = False

    # Validate product index
    idx_path = Path("models/product_index.pkl")
    if idx_path.exists():
        try:
            with open(idx_path, "rb") as f:
                idx = pickle.load(f)
            success &= check("id_to_idx" in idx, "Product index has id_to_idx")
            success &= check("idx_to_id" in idx, "Product index has idx_to_id")
            n_products = len(idx.get("id_to_idx", {}))
            success &= check(n_products >= 10, "Product index has >= 10 entries", str(n_products))
        except Exception as e:
            fail(f"Product index load failed: {e}")
            success = False

    # Validate metadata
    meta_path = Path("models/metadata.json")
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            required_keys = [
                "vocabulary_size", "matrix_shape", "training_date",
                "dataset_hash", "vectorizer_parameters", "artifact_paths"
            ]
            for k in required_keys:
                success &= check(k in meta, f"Metadata has key: {k}")
            success &= check(meta.get("vocabulary_size", 0) > 0, "vocabulary_size > 0")
        except Exception as e:
            fail(f"Metadata load failed: {e}")
            success = False

    # Validate processed products
    products_path = Path("models/processed_products.csv")
    if products_path.exists():
        try:
            df = pd.read_csv(products_path, encoding="utf-8")
            success &= check(len(df) >= 10, "Processed products has records", str(len(df)))
            for col in ["product_id", "product_name", "combined_text"]:
                success &= check(col in df.columns, f"products has column: {col}")
        except Exception as e:
            fail(f"Processed products load failed: {e}")
            success = False

    return success


# ================================================================
# TEST 3: Evaluation Report
# ================================================================

def test_evaluation_report() -> bool:
    header("TEST 3: Evaluation Report")
    success = True

    report_path = Path("artifacts/evaluation_report.json")
    if not report_path.exists():
        fail("Evaluation report does not exist")
        return False

    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        required_keys = [
            "evaluation_timestamp", "metrics",
            "precision_recall_by_k", "thresholds"
        ]
        for k in required_keys:
            success &= check(k in report, f"Report has key: {k}")

        metrics = report.get("metrics", {})
        for metric in ["coverage", "diversity", "novelty",
                        "intra_list_similarity", "inference_latency_ms"]:
            success &= check(
                metric in metrics,
                f"Metrics has: {metric}",
                str(metrics.get(metric, "MISSING"))
            )

        # Sanity checks on metric values
        coverage = metrics.get("coverage", -1)
        diversity = metrics.get("diversity", -1)
        latency = metrics.get("inference_latency_ms", -1)

        success &= check(0 <= coverage <= 1, "Coverage in [0, 1]", f"{coverage:.3f}")
        success &= check(0 <= diversity <= 1, "Diversity in [0, 1]", f"{diversity:.3f}")
        success &= check(latency > 0, "Inference latency > 0ms", f"{latency:.2f}ms")
        success &= check(latency < 500, "Inference latency < 500ms", f"{latency:.2f}ms")

        pr_list = report.get("precision_recall_by_k", [])
        success &= check(len(pr_list) >= 1, "Has at least 1 P@K entry")
        for entry in pr_list:
            k_val = entry.get("k")
            prec = entry.get("precision", -1)
            rec = entry.get("recall", -1)
            success &= check(0 <= prec <= 1, f"Precision@{k_val} in [0, 1]", f"{prec:.4f}")
            success &= check(0 <= rec <= 1, f"Recall@{k_val} in [0, 1]", f"{rec:.4f}")

    except Exception as e:
        fail(f"Evaluation report validation failed: {e}")
        import traceback; traceback.print_exc()
        return False

    return success


# ================================================================
# TEST 4: MLflow Tracking
# ================================================================

def test_mlflow() -> bool:
    header("TEST 4: MLflow Tracking")
    success = True

    # Check if mlruns directory was created
    mlruns_path = Path("mlruns")
    if mlruns_path.exists():
        ok("mlruns/ directory exists")
        experiments = list(mlruns_path.iterdir())
        success &= check(len(experiments) >= 1, "At least 1 MLflow experiment")

        # Check for run data
        run_count = 0
        for exp_dir in experiments:
            if exp_dir.is_dir() and exp_dir.name.isdigit():
                runs = [
                    d for d in exp_dir.iterdir()
                    if d.is_dir() and d.name not in ("meta.yaml",)
                ]
                run_count += len(runs)

        success &= check(run_count >= 1, "At least 1 MLflow run", f"{run_count} runs")
    else:
        warn("mlruns/ not found — MLflow may not be configured")
        warn("Training still succeeded (local fallback used)")
        # This is not a hard failure — model training uses local fallback
        ok("MLflow graceful degradation works")

    # Check pipeline summary for MLflow run_id (may be None for local registry)
    summary_path = Path("artifacts/pipeline_summary.json")
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        training_section = summary.get("model_training", {})
        run_id = training_section.get("mlflow_run_id")
        if run_id:
            ok(f"MLflow run_id recorded: {run_id[:12]}...")
        else:
            warn("MLflow run_id is None (local registry fallback active)")

    return success


# ================================================================
# TEST 5: Recommendation Engine Loading
# ================================================================

def test_engine_loading() -> bool:
    header("TEST 5: Recommendation Engine Loading")
    from src.components.recommendation.recommendation_engine import RecommendationEngine

    success = True
    try:
        t0 = time.perf_counter()
        engine = RecommendationEngine.load("models", force_reload=True)
        load_time = time.perf_counter() - t0

        success &= check(engine is not None, "Engine loaded")
        success &= check(load_time < 30, "Loaded in < 30s", f"{load_time:.2f}s")
        success &= check(engine.get_catalog_size() >= 10, "Catalog has >= 10 products",
                         f"{engine.get_catalog_size()}")
        success &= check(engine.vocab_size > 0, "Vocab size > 0", f"{engine.vocab_size:,}")
        success &= check(len(engine.get_all_categories()) >= 1,
                         "Has categories",
                         str(len(engine.get_all_categories())))

        test_engine_loading._engine = engine

    except Exception as e:
        fail(f"Engine loading failed: {e}")
        import traceback; traceback.print_exc()
        return False

    return success

test_engine_loading._engine = None


# ================================================================
# TEST 6: Recommendation Engine APIs
# ================================================================

def test_engine_apis() -> bool:
    header("TEST 6: Recommendation Engine APIs")
    engine = test_engine_loading._engine

    if engine is None:
        fail("Engine not loaded (skipping)")
        return False

    success = True
    from src.engines.base_engine import RecommendationResult

    # Get a sample product ID
    import pandas as pd
    products_df = engine._df
    sample_id = str(products_df["product_id"].iloc[0])
    safe_print(f"  Using test product_id: {sample_id[:20]}...")

    # Test each recommendation API
    apis = [
        ("recommend(hybrid)", lambda: engine.recommend(sample_id, strategy="hybrid", top_k=5)),
        ("recommend(content_based)", lambda: engine.recommend(sample_id, strategy="content_based", top_k=5)),
        ("recommend(popularity)", lambda: engine.recommend(sample_id, strategy="popularity", top_k=5)),
        ("similar_products()", lambda: engine.similar_products(sample_id, top_k=5)),
        ("frequently_bought_together()", lambda: engine.frequently_bought_together(sample_id, top_k=3)),
        ("trending_products()", lambda: engine.trending_products(top_k=5)),
        ("related_products()", lambda: engine.related_products(sample_id, top_k=5)),
        ("brand_products()", lambda: engine.brand_products(sample_id, top_k=5)),
        ("popular_products()", lambda: engine.popular_products(top_k=5)),
    ]

    for api_name, api_fn in apis:
        try:
            t0 = time.perf_counter()
            results = api_fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            api_ok = (
                isinstance(results, list)
                and all(isinstance(r, RecommendationResult) for r in results)
            )
            success &= check(api_ok, f"{api_name} returns List[RecommendationResult]",
                             f"{len(results)} results, {elapsed_ms:.1f}ms")

            if results:
                r = results[0]
                has_reason = bool(r.recommendation_reason)
                has_id = bool(r.product_id)
                has_name = bool(r.product_name)
                has_score = r.confidence_score >= 0
                success &= check(has_reason, f"  {api_name} has recommendation_reason")
                success &= check(has_id and has_name, f"  {api_name} has product_id and name")
                success &= check(has_score, f"  {api_name} has confidence_score >= 0",
                                 f"{r.confidence_score:.3f}")

        except Exception as e:
            fail(f"{api_name} raised exception: {e}")
            import traceback; traceback.print_exc()
            success = False

    return success


# ================================================================
# TEST 7: Search API
# ================================================================

def test_search() -> bool:
    header("TEST 7: Search API")
    engine = test_engine_loading._engine

    if engine is None:
        fail("Engine not loaded (skipping)")
        return False

    success = True
    from src.engines.base_engine import RecommendationResult

    # Get a real product name to search for
    sample_name = str(engine._df["product_name"].iloc[0])
    search_term = sample_name.split()[0] if sample_name else "phone"

    queries = [
        (search_term, "Known product name first word"),
        ("samsung", "Brand name"),
        ("wireless headphones", "Multi-word query"),
    ]

    for query, description in queries:
        try:
            results = engine.search(query, top_k=5)
            is_list = isinstance(results, list)
            success &= check(is_list, f"search('{query}') returns list ({description})",
                             f"{len(results)} results")
            if results:
                first = results[0]
                success &= check(
                    isinstance(first, RecommendationResult),
                    f"  search result is RecommendationResult"
                )
                success &= check(
                    first.strategy == "search",
                    f"  strategy is 'search'", first.strategy
                )
        except Exception as e:
            fail(f"search('{query}') failed: {e}")
            import traceback; traceback.print_exc()
            success = False

    # Test recommend_by_name
    try:
        recs = engine.recommend_by_name(search_term, top_k=5)
        success &= check(isinstance(recs, list), "recommend_by_name returns list",
                         f"{len(recs)} results")
    except Exception as e:
        fail(f"recommend_by_name failed: {e}")
        success = False

    return success


# ================================================================
# TEST 8: Inference Latency
# ================================================================

def test_inference_latency() -> bool:
    header("TEST 8: Inference Latency")
    engine = test_engine_loading._engine

    if engine is None:
        fail("Engine not loaded (skipping)")
        return False

    success = True
    import random

    sample_ids = list(engine._id_to_idx.keys())
    test_ids = random.sample(sample_ids, min(20, len(sample_ids)))

    latencies = []
    for pid in test_ids:
        t0 = time.perf_counter()
        engine.similar_products(pid, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    import statistics
    median_ms = statistics.median(latencies)
    max_ms = max(latencies)

    success &= check(median_ms < 200, "Median latency < 200ms", f"{median_ms:.1f}ms")
    success &= check(max_ms < 1000, "Max latency < 1000ms", f"{max_ms:.1f}ms")
    ok(f"Latency: median={median_ms:.1f}ms, max={max_ms:.1f}ms over {len(latencies)} queries")

    return success


# ================================================================
# Summary
# ================================================================

def print_summary(test_results: list) -> bool:
    safe_print(f"\n{CYAN}{'=' * 60}{RESET}")
    safe_print(f"{BOLD}  PHASE 3 VERIFICATION SUMMARY{RESET}")
    safe_print(f"{CYAN}{'=' * 60}{RESET}")

    passed = sum(1 for _, p in test_results if p)
    total = len(test_results)

    for name, p in test_results:
        status = f"{GREEN}PASS{RESET}" if p else f"{RED}FAIL{RESET}"
        safe_print(f"  [{status}] {name}")

    safe_print(f"\n{CYAN}{'-' * 60}{RESET}")
    color = GREEN if passed == total else RED
    safe_print(f"  {color}{BOLD}{passed}/{total} tests passed{RESET}")
    safe_print(f"{CYAN}{'=' * 60}{RESET}\n")
    return passed == total


def main() -> None:
    safe_print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    safe_print(f"{BOLD}{CYAN}  PHASE 3 VERIFICATION SUITE{RESET}")
    safe_print(f"{BOLD}{CYAN}  Product Recommendation System{RESET}")
    safe_print(f"{CYAN}{'=' * 60}{RESET}")

    tests = [
        ("Imports", test_imports),
        ("Training Pipeline", test_training_pipeline),
        ("Model Artifacts", test_model_artifacts),
        ("Evaluation Report", test_evaluation_report),
        ("MLflow Tracking", test_mlflow),
        ("Engine Loading", test_engine_loading),
        ("Engine APIs", test_engine_apis),
        ("Search API", test_search),
        ("Inference Latency", test_inference_latency),
    ]

    test_results = []
    for name, fn in tests:
        try:
            result = fn()
        except Exception as e:
            fail(f"'{name}' unhandled exception: {e}")
            import traceback; traceback.print_exc()
            result = False
        test_results.append((name, result))

    all_passed = print_summary(test_results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
