"""
Dataset Setup Script
=====================
Downloads the primary Kaggle Amazon product dataset and validates
it against the expected schema.

Usage:
    python scripts/setup_dataset.py

Requirements:
    KAGGLE_USERNAME and KAGGLE_KEY must be set in .env or environment.

Dataset:
    karkavelrajaj/amazon-sales-dataset
    Expected file: data/raw/amazon.csv

If credentials are not available, see scripts/generate_fallback.py
for the development fallback.
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------
# Expected schema for validation
# ---------------------------------------------------------------
EXPECTED_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "about_product",
    "img_link",
    "product_link",
]

KAGGLE_DATASET = "karkavelrajaj/amazon-sales-dataset"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_FILE = RAW_DIR / "amazon.csv"


def check_credentials() -> tuple[str, str]:
    """
    Read and validate Kaggle credentials from environment.

    Returns:
        Tuple of (username, key).

    Raises:
        SystemExit: If credentials are missing.
    """
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    if not username or not key:
        print(
            "\n❌ Kaggle credentials not found.\n"
            "\nTo set up the dataset:\n"
            "  1. Go to https://www.kaggle.com/account\n"
            "  2. Click 'Create New Token' to download kaggle.json\n"
            "  3. Add these values to your .env file:\n"
            "     KAGGLE_USERNAME=your_username\n"
            "     KAGGLE_KEY=your_api_key\n"
            "  4. Re-run: python scripts/setup_dataset.py\n"
            "\nAlternative (development only):\n"
            "  python scripts/generate_fallback.py\n"
            "  Then set 'use_fallback: true' in src/config/config.yaml"
        )
        sys.exit(1)

    return username, key


def download_dataset(username: str, key: str) -> None:
    """
    Download the Kaggle dataset using the kaggle Python API.

    Args:
        username: Kaggle username.
        key: Kaggle API key.
    """
    print(f"📥 Downloading dataset: {KAGGLE_DATASET}")

    # Set credentials for the kaggle library
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended  # type: ignore
        api = KaggleApiExtended()
        api.authenticate()

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        api.dataset_download_files(
            dataset=KAGGLE_DATASET,
            path=str(RAW_DIR),
            unzip=True,
            quiet=False,
        )
        print(f"✅ Download complete → {RAW_DIR}")

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  - Verify your KAGGLE_USERNAME and KAGGLE_KEY in .env")
        print("  - Check internet connectivity")
        print(f"  - Try manually downloading: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"  - Place amazon.csv in: {RAW_DIR}")
        sys.exit(1)


def find_csv_file() -> Path | None:
    """
    Search RAW_DIR for a CSV file that matches our target.

    Returns:
        Path to the found CSV file, or None.
    """
    if TARGET_FILE.exists():
        return TARGET_FILE

    # Look for any CSV in the raw directory
    csv_files = list(RAW_DIR.glob("*.csv"))
    if csv_files:
        print(f"  Found CSV: {csv_files[0].name}")
        if csv_files[0] != TARGET_FILE:
            print(f"  Renaming to: {TARGET_FILE.name}")
            shutil.move(str(csv_files[0]), str(TARGET_FILE))
        return TARGET_FILE

    return None


def validate_schema(csv_path: Path) -> bool:
    """
    Validate that the downloaded CSV has the expected columns.

    Args:
        csv_path: Path to the CSV file to validate.

    Returns:
        True if schema is valid, False otherwise.
    """
    try:
        import pandas as pd  # type: ignore

        print(f"\n🔍 Validating schema: {csv_path.name}")

        # Read only the header to avoid loading the entire file
        df_head = pd.read_csv(csv_path, nrows=5)
        found_columns = set(df_head.columns.str.strip().str.lower())
        expected_columns = {col.lower() for col in EXPECTED_COLUMNS}

        missing = expected_columns - found_columns
        extra = found_columns - expected_columns

        if missing:
            print(f"  ⚠️  Missing columns: {missing}")
        if extra:
            print(f"  ℹ️  Extra columns (will be ignored): {extra}")

        # Count records
        record_count = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        print(f"  ✅ Records: {record_count:,}")
        print(f"  ✅ Columns: {len(df_head.columns)}")

        if missing:
            print(
                "\n⚠️  Schema mismatch detected. The pipeline may still work if "
                "critical columns are present. Check src/components/data_ingestion/"
                "data_ingestion.py for column mapping."
            )
            return False

        print("  ✅ Schema validation passed")
        return True

    except ImportError:
        print("  ⚠️  pandas not installed — skipping schema validation")
        print("     Run: pip install -r requirements.txt")
        return True
    except Exception as e:
        print(f"  ⚠️  Schema validation error: {e}")
        return False


def print_next_steps() -> None:
    """Print instructions for the next steps after dataset setup."""
    print("\n" + "=" * 60)
    print("✅ Dataset ready!")
    print("=" * 60)
    print(f"\n📁 Dataset location: {TARGET_FILE}")
    print("\nNext steps:")
    print("  1. Train the models:")
    print("     python -m src.pipelines.training_pipeline")
    print("\n  2. Start the API:")
    print("     uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print("\n  3. Open the application:")
    print("     http://localhost:8000")
    print("\n  4. Start MLflow UI (optional):")
    print("     mlflow ui --host 0.0.0.0 --port 5000")


def main() -> None:
    """Main entry point for dataset setup."""
    print("=" * 60)
    print("  Product Recommendation System — Dataset Setup")
    print("=" * 60)

    # Check if dataset already exists
    if TARGET_FILE.exists():
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        print(f"\n✅ Dataset already exists: {TARGET_FILE}")
        print(f"   Size: {size_mb:.1f} MB")

        user_input = input("\nRedownload? [y/N]: ").strip().lower()
        if user_input != "y":
            validate_schema(TARGET_FILE)
            print_next_steps()
            return

    # Get credentials and download
    username, key = check_credentials()
    print(f"\n🔑 Credentials found for: {username}")

    download_dataset(username, key)

    # Find the downloaded file
    csv_path = find_csv_file()
    if csv_path is None:
        print(
            f"\n❌ No CSV file found in {RAW_DIR}.\n"
            "Please manually download the dataset and place amazon.csv in data/raw/"
        )
        sys.exit(1)

    # Validate schema
    validate_schema(csv_path)
    print_next_steps()


if __name__ == "__main__":
    main()
