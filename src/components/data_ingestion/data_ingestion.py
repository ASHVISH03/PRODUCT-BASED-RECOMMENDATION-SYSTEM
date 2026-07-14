"""
Data Ingestion Component
=========================
Loads the product dataset following a strict priority chain.

Priority Order:
    1. ENV: DATA_RAW_FILEPATH  →  load if file exists
    2. config.yaml: data.raw_filepath  →  load if file exists
    3. DatasetNotFoundError raised with clear setup instructions
       UNLESS use_fallback: true → load fallback with WARNING

This component NEVER silently falls back to synthetic data.
The fallback only activates when use_fallback is explicitly set to
true in config.yaml AND the fallback file exists.

Outputs:
    DataIngestionArtifact containing path, record count, hash, and metadata.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config.configuration import ConfigurationManager
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataConfig
from src.exception.exception import CustomException, DatasetNotFoundError
from src.logger.training_logger import get_training_logger
from src.utils.common import (
    create_directories,
    get_file_hash,
    get_file_size_mb,
    get_iso_timestamp,
    save_json,
)

logger = get_training_logger()


class DataIngestion:
    """
    Loads the raw product dataset into a Pandas DataFrame and
    returns a structured DataIngestionArtifact.

    Args:
        config: Typed DataConfig entity from ConfigurationManager.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_dataset_path(self) -> tuple[Path, str]:
        """
        Walk the priority chain to determine which dataset file to load.

        Returns:
            (resolved_path, source_label)
            source_label is one of: 'env_override' | 'kaggle' | 'fallback'

        Raises:
            DatasetNotFoundError: When no dataset is found and use_fallback=False.
        """
        # --- Priority 1: Environment variable override ---
        env_path_str = os.environ.get("DATA_RAW_FILEPATH", "").strip()
        if env_path_str:
            env_path = Path(env_path_str)
            if env_path.exists():
                logger.info(f"Dataset source: ENV override → {env_path}")
                return env_path, "env_override"
            logger.warning(
                f"DATA_RAW_FILEPATH is set but file not found: {env_path}. "
                "Continuing to config.yaml path."
            )

        # --- Priority 2: Config-specified primary path ---
        primary_path = Path(self.config.raw_filepath)
        if primary_path.exists():
            logger.info(f"Dataset source: primary (Kaggle) → {primary_path}")
            return primary_path, "kaggle"

        logger.warning(f"Primary dataset not found: {primary_path}")

        # --- Priority 3: Fallback (only if explicitly opted-in) ---
        if self.config.use_fallback:
            fallback_path = Path(self.config.fallback_filepath)
            if fallback_path.exists():
                logger.warning(
                    "\n"
                    "  ⚠️  ========================================================\n"
                    "  ⚠️   DEVELOPMENT FALLBACK DATASET IN USE\n"
                    "  ⚠️   This is synthetic data — NOT suitable for production.\n"
                    "  ⚠️   To use the real dataset:\n"
                    "  ⚠️     python scripts/setup_dataset.py\n"
                    "  ⚠️  ========================================================"
                )
                return fallback_path, "fallback"
            raise DatasetNotFoundError(
                f"use_fallback=true but fallback file missing: {fallback_path}.\n"
                "Run: python scripts/generate_fallback.py"
            )

        # --- No dataset found → explicit failure ---
        raise DatasetNotFoundError(
            f"\n\n"
            f"  Dataset not found at: {primary_path}\n\n"
            f"  ┌─ To set up the Kaggle dataset ──────────────────────────────┐\n"
            f"  │  1. Add credentials to .env:                                │\n"
            f"  │       KAGGLE_USERNAME=your_username                         │\n"
            f"  │       KAGGLE_KEY=your_api_key                               │\n"
            f"  │  2. Run: python scripts/setup_dataset.py                    │\n"
            f"  └─────────────────────────────────────────────────────────────┘\n\n"
            f"  ┌─ Development fallback (NOT for production) ─────────────────┐\n"
            f"  │  1. Run: python scripts/generate_fallback.py                │\n"
            f"  │  2. Set 'use_fallback: true' in src/config/config.yaml      │\n"
            f"  └─────────────────────────────────────────────────────────────┘"
        )

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """
        Load a CSV file with encoding detection.

        Tries UTF-8 first (Kaggle dataset encoding), falls back to latin-1.

        Args:
            path: Absolute path to the CSV file.

        Returns:
            Loaded DataFrame.
        """
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            logger.info(f"CSV loaded with UTF-8 encoding")
            return df
        except UnicodeDecodeError:
            logger.warning(
                f"UTF-8 decoding failed for {path.name} — retrying with latin-1"
            )
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
            logger.info(f"CSV loaded with latin-1 encoding")
            return df

    def _save_ingestion_metadata(
        self,
        artifact: DataIngestionArtifact,
        df: pd.DataFrame,
    ) -> None:
        """Persist ingestion metadata as a JSON artifact for audit and reproducibility."""
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "raw_filepath": artifact.raw_filepath,
            "source": artifact.source,
            "is_fallback": artifact.is_fallback,
            "dataset_version": artifact.dataset_version,
            "record_count": artifact.record_count,
            "column_count": artifact.column_count,
            "file_size_mb": artifact.file_size_mb,
            "ingestion_timestamp": artifact.ingestion_timestamp,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
        }

        save_json(artifacts_dir / "ingestion_metadata.json", metadata)
        logger.info(f"Ingestion metadata saved → artifacts/ingestion_metadata.json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute the data ingestion step.

        Steps:
            1. Resolve dataset path (priority chain).
            2. Load CSV into DataFrame.
            3. Log summary statistics.
            4. Persist ingestion metadata.
            5. Return DataIngestionArtifact.

        Returns:
            DataIngestionArtifact with all metadata populated.

        Raises:
            DatasetNotFoundError: If no dataset is available.
            CustomException: On unexpected I/O or parsing errors.
        """
        logger.info("=" * 60)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 60)

        try:
            create_directories([self.config.raw_dir, self.config.processed_dir])

            dataset_path, source = self._resolve_dataset_path()

            logger.info(f"Loading: {dataset_path}")
            df = self._load_csv(dataset_path)

            record_count = len(df)
            column_count = len(df.columns)
            file_size_mb = get_file_size_mb(dataset_path)
            dataset_version = get_file_hash(str(dataset_path))[:12]
            ingestion_timestamp = get_iso_timestamp()

            logger.info(f"  Records    : {record_count:,}")
            logger.info(f"  Columns    : {column_count}")
            logger.info(f"  File size  : {file_size_mb:.2f} MB")
            logger.info(f"  Version    : {dataset_version}")
            logger.info(f"  Source     : {source}")
            logger.info(f"  Columns    : {list(df.columns)}")

            artifact = DataIngestionArtifact(
                raw_filepath=str(dataset_path.resolve()),
                is_fallback=(source == "fallback"),
                dataset_version=dataset_version,
                record_count=record_count,
                column_count=column_count,
                file_size_mb=file_size_mb,
                ingestion_timestamp=ingestion_timestamp,
                source=source,
            )

            self._save_ingestion_metadata(artifact, df)
            logger.info("STEP 1 COMPLETE: Data Ingestion")
            return artifact

        except DatasetNotFoundError:
            raise
        except Exception as e:
            raise CustomException(
                f"Data ingestion failed: {e}", sys.exc_info()
            ) from e
