"""
Phase 2 Verification Test Suite
=================================
Runs automated verification of all Phase 2 components using the
synthetic fallback dataset (no Kaggle credentials required).

Windows-safe: no emoji, ANSI codes only via colorama, all print()
calls go through safe_print() to avoid UnicodeEncodeError.

Usage:
    python tests/verify_phase2.py
"""

import io
import json
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------
# Windows-safe stdout (must happen before any print)
# ----------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Change to project root so config relative paths work
os.chdir(PROJECT_ROOT)


# ----------------------------------------------------------------
# Colour helpers (colorama for cross-platform ANSI)
# ----------------------------------------------------------------
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


# ----------------------------------------------------------------
# Global pass/fail tracker
# ----------------------------------------------------------------

def check(condition: bool, name: str, detail: str = "") -> bool:
    suffix = f" ({detail})" if detail else ""
    if condition:
        ok(f"{name}{suffix}")
    else:
        fail(f"{name}{suffix}")
    return condition


# ================================================================
# TEST 0: Import Verification
# ================================================================

def test_imports() -> bool:
    header("TEST 0: Import Verification")
    success = True
    imports = [
        ("src.config.configuration", "ConfigurationManager"),
        ("src.exception.exception", "CustomException"),
        ("src.exception.exception", "DatasetNotFoundError"),
        ("src.exception.exception", "DataValidationError"),
        ("src.logger.training_logger", "get_training_logger"),
        ("src.utils.common", "save_json"),
        ("src.entity.config_entity", "DataConfig"),
        ("src.entity.artifact_entity", "DataIngestionArtifact"),
        ("src.components.data_ingestion.data_ingestion", "DataIngestion"),
        ("src.components.data_validation.data_validation", "DataValidation"),
        ("src.components.data_validation.validation_rules", "ALL_RULES"),
        ("src.components.data_transformation.data_transformation", "DataTransformation"),
        ("src.components.feature_engineering.feature_engineering", "FeatureEngineering"),
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
# TEST 1: Fallback dataset generation
# ================================================================

def test_fallback_generation() -> bool:
    header("TEST 1: Fallback Dataset Generation")

    fallback_path = PROJECT_ROOT / "data" / "external" / "sample_products.csv"

    if not fallback_path.exists():
        safe_print("  Generating synthetic fallback dataset...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/generate_fallback.py"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            fail(f"generate_fallback.py failed (exit {result.returncode})")
            safe_print(f"  STDOUT: {result.stdout[:500]}")
            safe_print(f"  STDERR: {result.stderr[:500]}")
            return False
        safe_print(f"  Script output: {result.stdout[:200]}")
    else:
        safe_print(f"  Fallback file already exists: {fallback_path}")

    success = True
    success &= check(fallback_path.exists(), "data/external/sample_products.csv created")

    if not fallback_path.exists():
        return False

    import pandas as pd
    try:
        df = pd.read_csv(fallback_path, encoding="utf-8")
        success &= check(len(df) >= 50, "Fallback has >=50 records", f"{len(df)} rows")
        success &= check(len(df.columns) >= 10, "Has >=10 columns", f"{len(df.columns)} cols")

        required_cols = ["product_id", "product_name", "category",
                         "discounted_price", "rating", "rating_count", "about_product"]
        for col in required_cols:
            success &= check(col in df.columns, f"Column present: {col}")

        # Check no completely empty critical columns
        for col in ["product_id", "product_name", "category"]:
            if col in df.columns:
                null_pct = df[col].isnull().mean()
                success &= check(null_pct < 0.01, f"{col} has no nulls", f"{null_pct:.0%}")

    except Exception as e:
        fail(f"Cannot read fallback CSV: {e}")
        import traceback; traceback.print_exc()
        return False

    return success


# ================================================================
# TEST 2: Configuration loading
# ================================================================

def test_config() -> bool:
    header("TEST 2: Configuration Loading")
    from src.config.configuration import ConfigurationManager

    success = True
    try:
        cfg = ConfigurationManager()
        data_cfg = cfg.get_data_config()
        feat_cfg = cfg.get_feature_config()
        model_cfg = cfg.get_model_config()
        rec_cfg = cfg.get_recommendation_config()

        success &= check(data_cfg.raw_filepath != "", "DataConfig.raw_filepath set")
        success &= check(data_cfg.fallback_filepath != "", "DataConfig.fallback_filepath set")
        success &= check(len(feat_cfg.combined_text_fields) > 0,
                         "FeatureConfig.combined_text_fields set",
                         str(feat_cfg.combined_text_fields))
        success &= check(len(feat_cfg.price_buckets) > 0,
                         "FeatureConfig.price_buckets set",
                         f"{len(feat_cfg.price_buckets)} buckets")
        success &= check(model_cfg.tfidf.max_features > 0,
                         "TFIDFConfig.max_features set",
                         str(model_cfg.tfidf.max_features))
        success &= check(rec_cfg.active_strategy != "",
                         "RecommendationConfig.active_strategy set",
                         rec_cfg.active_strategy)
    except Exception as e:
        fail(f"Config loading failed: {e}")
        import traceback; traceback.print_exc()
        return False

    return success


# ================================================================
# TEST 3: Data Ingestion
# ================================================================

def test_data_ingestion() -> bool:
    header("TEST 3: Data Ingestion")
    import src.config.configuration as config_mod
    from src.config.configuration import ConfigurationManager
    from src.components.data_ingestion.data_ingestion import DataIngestion
    from src.exception.exception import DatasetNotFoundError

    success = True

    # Patch config to enable fallback for this test
    original_load = config_mod.ConfigurationManager._load_yaml

    def patched_load(self):
        raw = original_load(self)
        raw["data"]["use_fallback"] = True
        return raw

    config_mod.ConfigurationManager._load_yaml = patched_load

    try:
        cfg = ConfigurationManager()
        data_cfg = cfg.get_data_config()
        ingestion = DataIngestion(config=data_cfg)
        artifact = ingestion.initiate_data_ingestion()

        success &= check(artifact.raw_filepath != "", "Artifact: raw_filepath set",
                         Path(artifact.raw_filepath).name)
        success &= check(artifact.record_count > 0, "Artifact: record_count > 0",
                         f"{artifact.record_count}")
        success &= check(artifact.column_count > 0, "Artifact: column_count > 0",
                         str(artifact.column_count))
        success &= check(artifact.dataset_version != "", "Artifact: dataset_version (hash) set")
        success &= check(artifact.ingestion_timestamp != "", "Artifact: ingestion_timestamp set")
        success &= check(artifact.source in ("kaggle", "fallback", "env_override"),
                         "Artifact: source label valid", artifact.source)
        success &= check(
            Path("artifacts/ingestion_metadata.json").exists(),
            "artifacts/ingestion_metadata.json created"
        )

        test_data_ingestion._artifact = artifact

    except DatasetNotFoundError as e:
        fail(f"DatasetNotFoundError: {str(e)[:200]}")
        return False
    except Exception as e:
        fail(f"Ingestion failed: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        config_mod.ConfigurationManager._load_yaml = original_load

    return success

test_data_ingestion._artifact = None  # type: ignore[attr-defined]


# ================================================================
# TEST 4: Data Validation
# ================================================================

def test_data_validation() -> bool:
    header("TEST 4: Data Validation")
    from src.config.configuration import ConfigurationManager
    from src.components.data_validation.data_validation import DataValidation
    from src.components.data_validation.validation_rules import ALL_RULES

    success = True
    ingestion_artifact = test_data_ingestion._artifact  # type: ignore[attr-defined]

    if ingestion_artifact is None:
        fail("Skipped: ingestion artifact not available")
        return False

    try:
        cfg = ConfigurationManager()
        data_cfg = cfg.get_data_config()
        validation = DataValidation(
            config=data_cfg,
            ingestion_artifact=ingestion_artifact,
        )
        artifact = validation.initiate_data_validation()

        success &= check(isinstance(artifact.validation_status, bool),
                         "Artifact: validation_status is bool",
                         str(artifact.validation_status))
        success &= check(artifact.validated_record_count > 0,
                         "Artifact: validated_record_count > 0",
                         f"{artifact.validated_record_count}")
        success &= check(Path(artifact.validation_report_path).exists(),
                         "Validation report JSON created")
        success &= check(isinstance(artifact.issues, list),
                         "Artifact: issues is list",
                         f"{len(artifact.issues)} issues")
        success &= check(len(ALL_RULES) > 0, "Rules loaded",
                         f"{len(ALL_RULES)} rules")

        # Verify report structure
        with open(artifact.validation_report_path, encoding="utf-8") as f:
            report = json.load(f)
        success &= check("validation_status" in report, "Report: validation_status key")
        success &= check("summary" in report, "Report: summary key")
        success &= check("issues" in report, "Report: issues key")
        success &= check("column_stats" in report, "Report: column_stats key")

        test_data_validation._artifact = artifact

    except Exception as e:
        fail(f"Validation failed: {e}")
        import traceback; traceback.print_exc()
        return False

    return success

test_data_validation._artifact = None  # type: ignore[attr-defined]


# ================================================================
# TEST 5: Data Transformation
# ================================================================

def test_data_transformation() -> bool:
    header("TEST 5: Data Transformation")
    from src.config.configuration import ConfigurationManager
    from src.components.data_transformation.data_transformation import DataTransformation
    import pandas as pd

    success = True
    ingestion_artifact = test_data_ingestion._artifact  # type: ignore[attr-defined]

    if ingestion_artifact is None:
        fail("Skipped: ingestion artifact not available")
        return False

    try:
        cfg = ConfigurationManager()
        data_cfg = cfg.get_data_config()
        feat_cfg = cfg.get_feature_config()

        transformation = DataTransformation(
            config=data_cfg,
            feature_config=feat_cfg,
            ingestion_artifact=ingestion_artifact,
        )
        artifact = transformation.initiate_data_transformation()

        success &= check(artifact.processed_filepath != "", "Artifact: processed_filepath set")
        success &= check(artifact.interim_filepath != "", "Artifact: interim_filepath set")
        success &= check(artifact.output_record_count > 0,
                         "Artifact: output_record_count > 0",
                         f"{artifact.output_record_count}")
        success &= check(artifact.input_record_count >= artifact.output_record_count,
                         "Input count >= output count")

        success &= check(Path(artifact.processed_filepath).exists(), "cleaned.csv created")
        success &= check(Path(artifact.interim_filepath).exists(), "interim.csv created")

        # Validate cleaned CSV contents
        df = pd.read_csv(artifact.processed_filepath, encoding="utf-8")
        success &= check(len(df) > 0, "cleaned.csv has records", f"{len(df)}")
        success &= check("product_name_clean" in df.columns,
                         "product_name_clean column present")
        success &= check("category_clean" in df.columns, "category_clean column present")
        success &= check("about_product_clean" in df.columns,
                         "about_product_clean column present")

        # Verify numeric parsing happened
        if "discounted_price" in df.columns:
            numeric_pct = pd.to_numeric(df["discounted_price"], errors="coerce").notna().mean()
            success &= check(numeric_pct > 0.5,
                             "discounted_price mostly numeric",
                             f"{numeric_pct:.0%} parsed")

        if "rating" in df.columns:
            numeric_pct = pd.to_numeric(df["rating"], errors="coerce").notna().mean()
            success &= check(numeric_pct > 0.7, "rating mostly numeric",
                             f"{numeric_pct:.0%} parsed")

        test_data_transformation._artifact = artifact

    except Exception as e:
        fail(f"Transformation failed: {e}")
        import traceback; traceback.print_exc()
        return False

    return success

test_data_transformation._artifact = None  # type: ignore[attr-defined]


# ================================================================
# TEST 6: Feature Engineering
# ================================================================

def test_feature_engineering() -> bool:
    header("TEST 6: Feature Engineering")
    from src.config.configuration import ConfigurationManager
    from src.components.feature_engineering.feature_engineering import FeatureEngineering
    import pandas as pd

    success = True
    transformation_artifact = test_data_transformation._artifact  # type: ignore[attr-defined]

    if transformation_artifact is None:
        fail("Skipped: transformation artifact not available")
        return False

    try:
        cfg = ConfigurationManager()
        feat_cfg = cfg.get_feature_config()

        fe = FeatureEngineering(
            feature_config=feat_cfg,
            transformation_artifact=transformation_artifact,
        )
        artifact = fe.initiate_feature_engineering()

        success &= check(artifact.feature_filepath != "", "Artifact: feature_filepath set")
        success &= check(artifact.combined_text_column != "",
                         "Artifact: combined_text_column set",
                         artifact.combined_text_column)
        success &= check(len(artifact.feature_columns) >= 8,
                         "At least 8 feature columns created",
                         f"{len(artifact.feature_columns)}")
        success &= check(artifact.vocabulary_estimate > 0,
                         "Artifact: vocabulary_estimate > 0",
                         f"{artifact.vocabulary_estimate}")

        success &= check(Path(artifact.feature_filepath).exists(), "features.csv created")

        df = pd.read_csv(artifact.feature_filepath, encoding="utf-8")

        expected_features = [
            "combined_text",
            "combined_text_tfidf",
            "category_l1",
            "category_l2",
            "brand",
            "price_bucket",
            "price_normalized",
            "popularity_score",
            "discount_tier",
        ]
        for feat in expected_features:
            success &= check(feat in df.columns, f"Feature column: {feat}")

        # combined_text should be mostly populated
        if "combined_text" in df.columns:
            empty_pct = df["combined_text"].fillna("").str.strip().eq("").mean()
            success &= check(empty_pct < 0.10,
                             "combined_text mostly non-empty",
                             f"{empty_pct:.0%} empty")

        # popularity_score should be in [0, 1]
        if "popularity_score" in df.columns:
            in_range = df["popularity_score"].between(0, 1).all()
            success &= check(in_range, "popularity_score in [0, 1]")

        # price_bucket should have known labels
        if "price_bucket" in df.columns:
            valid_buckets = {"Budget", "Mid-Range", "Premium", "Luxury", "Unknown"}
            actual_buckets = set(df["price_bucket"].unique())
            invalid = actual_buckets - valid_buckets
            success &= check(len(invalid) == 0,
                             "price_bucket labels all valid",
                             str(actual_buckets))

        test_feature_engineering._artifact = artifact

    except Exception as e:
        fail(f"Feature engineering failed: {e}")
        import traceback; traceback.print_exc()
        return False

    return success

test_feature_engineering._artifact = None  # type: ignore[attr-defined]


# ================================================================
# TEST 7: Full Pipeline (End-to-End)
# ================================================================

def test_full_pipeline() -> bool:
    header("TEST 7: Full Pipeline Execution (End-to-End)")
    import src.config.configuration as config_mod
    from src.pipelines.training_pipeline import TrainingPipeline

    success = True

    original_load = config_mod.ConfigurationManager._load_yaml

    def patched_load(self):
        raw = original_load(self)
        raw["data"]["use_fallback"] = True
        return raw

    config_mod.ConfigurationManager._load_yaml = patched_load

    try:
        pipeline = TrainingPipeline()
        artifact = pipeline.run()

        success &= check(artifact.success, "Pipeline completed successfully")
        success &= check(artifact.error_message is None,
                         "No error message",
                         str(artifact.error_message))
        success &= check(artifact.pipeline_duration_seconds > 0,
                         "Duration recorded",
                         f"{artifact.pipeline_duration_seconds:.2f}s")
        success &= check(artifact.ingestion is not None, "Ingestion artifact present")
        success &= check(artifact.validation is not None, "Validation artifact present")
        success &= check(artifact.transformation is not None, "Transformation artifact present")
        success &= check(artifact.feature_engineering is not None,
                         "Feature engineering artifact present")

        # Verify all output files exist
        expected_files = [
            ("artifacts/ingestion_metadata.json", "ingestion_metadata.json"),
            ("artifacts/validation_report.json", "validation_report.json"),
            ("artifacts/pipeline_summary.json", "pipeline_summary.json"),
            ("data/processed/cleaned.csv", "cleaned.csv"),
            ("data/processed/features.csv", "features.csv"),
            ("data/interim/interim.csv", "interim.csv"),
        ]
        for filepath, label in expected_files:
            success &= check(Path(filepath).exists(), f"Output file exists: {label}")

        # Validate pipeline summary JSON structure
        with open("artifacts/pipeline_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        success &= check(summary.get("pipeline_status") == "success",
                         "pipeline_summary: status=success")
        success &= check("ingestion" in summary, "pipeline_summary: ingestion section")
        success &= check("validation" in summary, "pipeline_summary: validation section")
        success &= check("transformation" in summary, "pipeline_summary: transformation section")
        success &= check("feature_engineering" in summary,
                         "pipeline_summary: feature_engineering section")

        # Spot-check key values
        success &= check(
            summary["transformation"]["output_records"] > 0,
            "pipeline_summary: output_records > 0",
            str(summary["transformation"]["output_records"])
        )
        success &= check(
            summary["feature_engineering"]["feature_count"] >= 8,
            "pipeline_summary: feature_count >= 8",
            str(summary["feature_engineering"]["feature_count"])
        )

    except Exception as e:
        fail(f"Full pipeline failed: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        config_mod.ConfigurationManager._load_yaml = original_load

    return success


# ================================================================
# TEST 8: DatasetNotFoundError behavior
# ================================================================

def test_error_handling() -> bool:
    header("TEST 8: Error Handling (DatasetNotFoundError)")
    import src.config.configuration as config_mod
    from src.config.configuration import ConfigurationManager
    from src.components.data_ingestion.data_ingestion import DataIngestion
    from src.exception.exception import DatasetNotFoundError

    original_load = config_mod.ConfigurationManager._load_yaml

    def patched_load(self):
        raw = original_load(self)
        raw["data"]["raw_filepath"] = "data/raw/__nonexistent_test_file__.csv"
        raw["data"]["use_fallback"] = False
        return raw

    config_mod.ConfigurationManager._load_yaml = patched_load

    success = True
    try:
        cfg = ConfigurationManager()
        data_cfg = cfg.get_data_config()
        ingestion = DataIngestion(config=data_cfg)
        ingestion.initiate_data_ingestion()
        fail("Expected DatasetNotFoundError was NOT raised")
        success = False
    except DatasetNotFoundError as e:
        ok("DatasetNotFoundError correctly raised")
        msg = str(e)
        success &= check(len(msg) > 50, "Error message is descriptive",
                         f"{len(msg)} chars")
        success &= check(
            "setup_dataset" in msg or "scripts" in msg or "fallback" in msg,
            "Error message references fix instructions"
        )
    except Exception as e:
        fail(f"Wrong exception type: {type(e).__name__}: {e}")
        success = False
    finally:
        config_mod.ConfigurationManager._load_yaml = original_load

    return success


# ================================================================
# Summary
# ================================================================

def print_summary(test_results: list) -> bool:
    safe_print(f"\n{CYAN}{'=' * 60}{RESET}")
    safe_print(f"{BOLD}  PHASE 2 VERIFICATION SUMMARY{RESET}")
    safe_print(f"{CYAN}{'=' * 60}{RESET}")

    passed = sum(1 for _, p in test_results if p)
    total = len(test_results)

    for name, passed_flag in test_results:
        status = f"{GREEN}PASS{RESET}" if passed_flag else f"{RED}FAIL{RESET}"
        safe_print(f"  [{status}] {name}")

    safe_print(f"\n{CYAN}{'-' * 60}{RESET}")
    color = GREEN if passed == total else RED
    safe_print(f"  {color}{BOLD}{passed}/{total} tests passed{RESET}")
    safe_print(f"{CYAN}{'=' * 60}{RESET}\n")

    return passed == total


def main() -> None:
    safe_print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    safe_print(f"{BOLD}{CYAN}  PHASE 2 VERIFICATION SUITE{RESET}")
    safe_print(f"{BOLD}{CYAN}  Product Recommendation System{RESET}")
    safe_print(f"{CYAN}{'=' * 60}{RESET}")

    tests = [
        ("Imports", test_imports),
        ("Fallback Dataset", test_fallback_generation),
        ("Configuration", test_config),
        ("Data Ingestion", test_data_ingestion),
        ("Data Validation", test_data_validation),
        ("Data Transformation", test_data_transformation),
        ("Feature Engineering", test_feature_engineering),
        ("Full Pipeline", test_full_pipeline),
        ("Error Handling", test_error_handling),
    ]

    test_results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
        except Exception as e:
            fail(f"Test '{name}' unhandled exception: {e}")
            import traceback; traceback.print_exc()
            result = False
        test_results.append((name, result))

    all_passed = print_summary(test_results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
