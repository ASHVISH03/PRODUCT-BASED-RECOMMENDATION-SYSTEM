"""
Data Transformation Component
===============================
Cleans and normalizes the validated raw dataset, producing a
structured processed CSV ready for feature engineering.

Transformations applied:
    1. Column standardization — strip whitespace from headers
    2. Price parsing         — strip ₹ and commas → float
    3. Discount parsing      — strip % → float
    4. Rating parsing        — handle string ratings → float
    5. Rating count parsing  — strip commas → int
    6. Duplicate removal     — drop fully duplicate rows first,
                               then deduplicate product_ids (keep first)
    7. Null handling         — fill or drop based on column criticality
    8. Text normalization    — product_name, category, about_product:
                               lowercase → whitespace normalization →
                               remove special chars → strip
    9. Category normalization — split pipe-delimited category hierarchy
   10. About product cleanup  — remove list brackets and quotes
   11. Combined text field    — merge product_name + category + about_product
   12. Save to data/processed/cleaned.csv

Output:
    DataTransformationArtifact with paths and stats.
"""

import re
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.entity.config_entity import DataConfig, FeatureConfig
from src.exception.exception import CustomException, DataTransformationError
from src.logger.training_logger import get_training_logger
from src.utils.common import create_directories, get_iso_timestamp

logger = get_training_logger()

# Extended stop words: sklearn's set + a few domain-specific ones
_STOP_WORDS: Set[str] = set(ENGLISH_STOP_WORDS) | {
    "product", "amazon", "brand", "buy", "new", "best", "get",
    "pack", "set", "also", "free", "offer", "deal", "sale",
}

# Regex patterns (compiled once at module level for performance)
_RE_CURRENCY = re.compile(r"[₹$€£¥,\s]")
_RE_PERCENT = re.compile(r"[%\s]")
_RE_SPECIAL_CHARS = re.compile(r"[^a-z0-9\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_LIST_ARTIFACTS = re.compile(r"[\[\]'\"{}()]")


class DataTransformation:
    """
    Cleans, normalizes, and structures the raw dataset.

    Args:
        config: DataConfig from ConfigurationManager.
        feature_config: FeatureConfig from ConfigurationManager.
        ingestion_artifact: DataIngestionArtifact from prior step.
    """

    def __init__(
        self,
        config: DataConfig,
        feature_config: FeatureConfig,
        ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        self.config = config
        self.feature_config = feature_config
        self.ingestion_artifact = ingestion_artifact

    # ------------------------------------------------------------------
    # Static text utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_price(value: object) -> float:
        """
        Parse a price string like '₹26,900' or '₹1,09,990' into a float.

        Args:
            value: Raw price value (may be string or NaN).

        Returns:
            Parsed float, or NaN on failure.
        """
        if pd.isna(value):
            return float("nan")
        cleaned = _RE_CURRENCY.sub("", str(value)).strip()
        try:
            return float(cleaned)
        except ValueError:
            return float("nan")

    @staticmethod
    def _clean_percentage(value: object) -> float:
        """
        Parse a percentage string like '45%' or '10%off' into a float.

        Args:
            value: Raw percentage value.

        Returns:
            Parsed float (0–100), or NaN on failure.
        """
        if pd.isna(value):
            return float("nan")
        cleaned = _RE_PERCENT.sub("", str(value)).replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return float("nan")

    @staticmethod
    def _clean_rating(value: object) -> float:
        """
        Parse a rating string. Handles '4.1', '4.1 out of 5', 'Get', etc.

        Args:
            value: Raw rating value.

        Returns:
            Parsed float (0.0–5.0), or NaN on failure.
        """
        if pd.isna(value):
            return float("nan")
        # Extract first valid-looking float from the string
        match = re.search(r"\d+\.?\d*", str(value).replace(",", "."))
        if match:
            parsed = float(match.group())
            # Ratings above 5 usually mean something was parsed wrong
            return parsed if 0.0 <= parsed <= 5.0 else float("nan")
        return float("nan")

    @staticmethod
    def _clean_rating_count(value: object) -> float:
        """
        Parse a rating count string like '1,091' or '10,234 ratings' into int.

        Args:
            value: Raw rating count value.

        Returns:
            Parsed float (cast-able to int), or NaN on failure.
        """
        if pd.isna(value):
            return float("nan")
        cleaned = re.sub(r"[^0-9]", "", str(value))
        try:
            return float(cleaned) if cleaned else float("nan")
        except ValueError:
            return float("nan")

    @staticmethod
    def _normalize_text(text: object) -> str:
        """
        Normalize a text field for ML use:
            lowercase → remove special chars → collapse whitespace → strip

        Args:
            text: Raw string value.

        Returns:
            Normalized string, or empty string if value is null/empty.
        """
        if pd.isna(text) or str(text).strip() in ("", "nan", "None"):
            return ""
        s = str(text).lower()
        s = _RE_LIST_ARTIFACTS.sub(" ", s)   # remove list brackets/quotes
        s = _RE_SPECIAL_CHARS.sub(" ", s)     # remove punctuation
        s = _RE_MULTI_SPACE.sub(" ", s)       # collapse whitespace
        return s.strip()

    @staticmethod
    def _remove_stop_words(text: str) -> str:
        """
        Remove English stop words and domain-specific noise words.

        Args:
            text: Normalized (lowercased) text string.

        Returns:
            Text with stop words removed.
        """
        if not text:
            return ""
        tokens = text.split()
        filtered = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
        return " ".join(filtered)

    @staticmethod
    def _clean_about_product(text: object) -> str:
        """
        Clean the about_product field, which often contains Python list
        representation: ['Point 1|Point 2|Point 3']

        Args:
            text: Raw about_product value.

        Returns:
            Clean, joined text string.
        """
        if pd.isna(text) or str(text).strip() in ("", "nan"):
            return ""
        s = str(text)
        # Remove list-formatting characters
        s = _RE_LIST_ARTIFACTS.sub(" ", s)
        # Replace pipe separators with spaces
        s = s.replace("|", " ")
        # Collapse whitespace
        s = _RE_MULTI_SPACE.sub(" ", s).strip()
        return s

    # ------------------------------------------------------------------
    # Transformation steps
    # ------------------------------------------------------------------

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from column names."""
        df.columns = df.columns.str.strip()
        logger.info(f"  Columns standardized: {list(df.columns)}")
        return df

    def _parse_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse all string-encoded numeric columns into proper floats."""
        df["discounted_price"] = df["discounted_price"].apply(self._clean_price)
        df["actual_price"] = df["actual_price"].apply(self._clean_price)
        df["discount_percentage"] = df["discount_percentage"].apply(self._clean_percentage)
        df["rating"] = df["rating"].apply(self._clean_rating)
        df["rating_count"] = df["rating_count"].apply(self._clean_rating_count)

        logger.info(
            f"  Numeric parsing complete — "
            f"price nulls: {df['discounted_price'].isnull().sum():,}, "
            f"rating nulls: {df['rating'].isnull().sum():,}"
        )
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        """
        Remove fully duplicate rows, then deduplicate by product_id.

        Returns:
            (cleaned_df, full_dup_count, product_id_dup_count)
        """
        initial_count = len(df)

        # Step 1: Drop fully duplicate rows
        df = df.drop_duplicates()
        full_dup_count = initial_count - len(df)

        # Step 2: Deduplicate by product_id (keep first occurrence)
        if "product_id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["product_id"], keep="first")
            product_id_dup_count = before - len(df)
        else:
            product_id_dup_count = 0

        logger.info(
            f"  Duplicates removed — "
            f"full rows: {full_dup_count:,}, product_id: {product_id_dup_count:,}"
        )
        return df, full_dup_count, product_id_dup_count

    def _handle_nulls(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        """
        Fill or drop nulls by column criticality.

        Critical columns (must have value):
            product_id, product_name, category

        Semi-critical (fill with defaults):
            about_product → ""
            rating → median
            rating_count → 0
            discounted_price → actual_price
            actual_price → discounted_price
            discount_percentage → 0

        Returns:
            (cleaned_df, filled_count, dropped_count)
        """
        filled_count = 0
        initial_count = len(df)

        # Fill semi-critical columns
        if "about_product" in df.columns:
            n = df["about_product"].isnull().sum()
            df["about_product"] = df["about_product"].fillna("")
            filled_count += n

        if "rating" in df.columns:
            median_rating = df["rating"].median()
            n = df["rating"].isnull().sum()
            df["rating"] = df["rating"].fillna(
                median_rating if not pd.isna(median_rating) else 4.0
            )
            filled_count += n

        if "rating_count" in df.columns:
            n = df["rating_count"].isnull().sum()
            df["rating_count"] = df["rating_count"].fillna(0)
            filled_count += n

        if "discount_percentage" in df.columns:
            n = df["discount_percentage"].isnull().sum()
            df["discount_percentage"] = df["discount_percentage"].fillna(0)
            filled_count += n

        # Cross-fill prices
        if "discounted_price" in df.columns and "actual_price" in df.columns:
            n1 = df["discounted_price"].isnull().sum()
            df["discounted_price"] = df["discounted_price"].fillna(df["actual_price"])
            n2 = df["actual_price"].isnull().sum()
            df["actual_price"] = df["actual_price"].fillna(df["discounted_price"])
            filled_count += n1 + n2

        # Drop rows where critical columns are still null
        critical_cols = [c for c in ["product_id", "product_name", "category"] if c in df.columns]
        before_drop = len(df)
        df = df.dropna(subset=critical_cols)
        dropped_count = before_drop - len(df)

        logger.info(
            f"  Null handling — filled: {filled_count:,}, dropped: {dropped_count:,}"
        )
        return df, filled_count, dropped_count

    def _normalize_text_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """
        Apply text normalization to all text columns.

        Returns:
            (df_with_normalized_columns, list_of_normalized_column_names)
        """
        text_cols_cleaned: List[str] = []

        # Product name: normalize but preserve readability (no stop word removal)
        if "product_name" in df.columns:
            df["product_name_clean"] = df["product_name"].apply(self._normalize_text)
            text_cols_cleaned.append("product_name")

        # Category: normalize, preserve hierarchy separator |
        if "category" in df.columns:
            df["category_clean"] = (
                df["category"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
            text_cols_cleaned.append("category")

        # About product: clean list formatting, normalize
        if "about_product" in df.columns:
            df["about_product_clean"] = df["about_product"].apply(
                lambda x: self._normalize_text(self._clean_about_product(x))
            )
            text_cols_cleaned.append("about_product")

        logger.info(f"  Text columns normalized: {text_cols_cleaned}")
        return df, text_cols_cleaned

    def _save_interim(self, df: pd.DataFrame) -> str:
        """Save the post-normalization DataFrame to data/interim/."""
        interim_dir = Path(self.config.interim_dir)
        interim_dir.mkdir(parents=True, exist_ok=True)
        interim_path = interim_dir / "interim.csv"
        df.to_csv(interim_path, index=False, encoding="utf-8")
        logger.info(f"  Interim data saved → {interim_path}")
        return str(interim_path.resolve())

    def _save_processed(self, df: pd.DataFrame) -> str:
        """Save the final cleaned DataFrame to data/processed/cleaned.csv."""
        processed_dir = Path(self.config.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "cleaned.csv"
        df.to_csv(processed_path, index=False, encoding="utf-8")
        logger.info(f"  Processed data saved → {processed_path}")
        return str(processed_path.resolve())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Execute the complete data transformation pipeline.

        Returns:
            DataTransformationArtifact with paths and statistics.

        Raises:
            DataTransformationError: On transformation failures.
            CustomException: On unexpected errors.
        """
        logger.info("=" * 60)
        logger.info("STEP 3: DATA TRANSFORMATION")
        logger.info("=" * 60)

        try:
            create_directories([self.config.processed_dir, self.config.interim_dir])

            # Load validated raw data
            df = pd.read_csv(
                self.ingestion_artifact.raw_filepath,
                encoding="utf-8",
                low_memory=False,
            )
            input_count = len(df)
            logger.info(f"Loaded {input_count:,} records for transformation")

            # --- Step 1: Standardize column names ---
            df = self._standardize_columns(df)

            # --- Step 2: Parse numeric columns ---
            df = self._parse_numeric_columns(df)

            # --- Step 3: Remove duplicates ---
            df, full_dups, pid_dups = self._remove_duplicates(df)
            removed_duplicates = full_dups + pid_dups

            # --- Step 4: Handle nulls ---
            df, filled_nulls, dropped_nulls = self._handle_nulls(df)

            # --- Step 5: Save interim snapshot ---
            interim_path = self._save_interim(df)

            # --- Step 6: Normalize text columns ---
            df, text_cols = self._normalize_text_columns(df)

            output_count = len(df)
            logger.info(f"  Input records  : {input_count:,}")
            logger.info(f"  Output records : {output_count:,}")
            logger.info(f"  Removed (dups) : {removed_duplicates:,}")
            logger.info(f"  Filled nulls   : {filled_nulls:,}")
            logger.info(f"  Dropped (null) : {dropped_nulls:,}")

            # --- Step 7: Save processed data ---
            processed_path = self._save_processed(df)

            artifact = DataTransformationArtifact(
                processed_filepath=processed_path,
                interim_filepath=interim_path,
                input_record_count=input_count,
                output_record_count=output_count,
                removed_duplicates=removed_duplicates,
                filled_nulls=filled_nulls,
                dropped_nulls=dropped_nulls,
                text_columns_cleaned=text_cols,
            )

            logger.info("STEP 3 COMPLETE: Data Transformation")
            return artifact

        except Exception as e:
            if isinstance(e, (DataTransformationError, CustomException)):
                raise
            raise CustomException(
                f"Data transformation failed: {e}", sys.exc_info()
            ) from e
