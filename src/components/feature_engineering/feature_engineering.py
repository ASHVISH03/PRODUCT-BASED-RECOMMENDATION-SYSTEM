"""
Feature Engineering Component
===============================
Transforms the cleaned dataset into a feature-rich DataFrame
ready for model training and all 7 recommendation engines.

Features produced:
    combined_text       — product_name_clean + category_clean + about_product_clean
                          (TF-IDF input — the primary content signal)
    combined_text_tfidf — stop-word-removed version of combined_text
                          (used for TF-IDF vectorization)
    category_l1         — first pipe-delimited category level (e.g. "Electronics")
    category_l2         — second level (e.g. "Smartphones & Accessories")
    category_l3         — third level if present, else ""
    brand               — extracted from product_name heuristics
    price_bucket        — "Budget" | "Mid-Range" | "Premium" | "Luxury"
    price_normalized    — discounted_price / max(discounted_price) ∈ [0, 1]
    popularity_score    — rating^w1 * log1p(rating_count)^w2 (configurable)
    discount_tier       — "No Discount" | "Low" | "Medium" | "High"

All features are configuration-driven via FeatureConfig.
Output saved to data/processed/features.csv.
"""

import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.entity.artifact_entity import DataTransformationArtifact, FeatureEngineeringArtifact
from src.entity.config_entity import FeatureConfig, PriceBucket
from src.exception.exception import CustomException, FeatureEngineeringError
from src.logger.training_logger import get_training_logger
from src.utils.common import create_directories, get_iso_timestamp

logger = get_training_logger()

_STOP_WORDS: Set[str] = set(ENGLISH_STOP_WORDS)
_RE_MULTI_SPACE = re.compile(r"\s+")

# Discount tier boundaries (%)
_DISCOUNT_TIERS = [
    (0.0, "No Discount"),
    (10.0, "Low"),       # 1–19%
    (20.0, "Medium"),    # 20–39%
    (40.0, "High"),      # 40%+
]

# Common brand indicators to strip from product names when extracting brand
_BRAND_STOP_WORDS = {
    "for", "with", "and", "or", "by", "in", "on", "the", "a", "an",
    "combo", "pack", "set", "kit", "bundle",
}


class FeatureEngineering:
    """
    Produces all ML-ready features for the recommendation system.

    Args:
        feature_config: FeatureConfig from ConfigurationManager.
        transformation_artifact: DataTransformationArtifact from prior step.
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        transformation_artifact: DataTransformationArtifact,
    ) -> None:
        self.feature_config = feature_config
        self.transformation_artifact = transformation_artifact

    # ------------------------------------------------------------------
    # Feature: Combined text
    # ------------------------------------------------------------------

    def _build_combined_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate product_name_clean, category_clean, and about_product_clean
        into a single combined_text field for TF-IDF vectorization.

        Also produces combined_text_tfidf — a stop-word-removed version.
        """
        fields = self.feature_config.combined_text_fields

        # Map configured field names to _clean columns where available
        field_map = {
            "product_name": "product_name_clean",
            "category": "category_clean",
            "about_product": "about_product_clean",
        }

        parts: List[pd.Series] = []
        used_fields: List[str] = []
        for field in fields:
            clean_col = field_map.get(field, field)
            if clean_col in df.columns:
                parts.append(df[clean_col].fillna("").astype(str))
                used_fields.append(clean_col)
            elif field in df.columns:
                parts.append(df[field].fillna("").astype(str))
                used_fields.append(field)

        if not parts:
            logger.warning("No text columns found for combined_text. Using product_name.")
            parts = [df.get("product_name", pd.Series([""] * len(df))).fillna("").astype(str)]

        df["combined_text"] = parts[0]
        for p in parts[1:]:
            df["combined_text"] = df["combined_text"] + " " + p
        df["combined_text"] = df["combined_text"].str.strip()

        # Stop-word-removed version for TF-IDF
        df["combined_text_tfidf"] = df["combined_text"].apply(
            lambda text: " ".join(
                t for t in text.split() if t not in _STOP_WORDS and len(t) > 1
            )
        )

        vocab_estimate = (
            len(set(" ".join(df["combined_text_tfidf"].dropna()).split()))
        )

        logger.info(
            f"  Combined text built from: {used_fields}. "
            f"Vocabulary estimate: {vocab_estimate:,}"
        )
        return df, vocab_estimate

    # ------------------------------------------------------------------
    # Feature: Category hierarchy
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_category_levels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Split the pipe-delimited category into up to 3 hierarchy levels.

        Amazon category format: "Electronics|Headphones|Over-Ear"
        """
        if "category" not in df.columns:
            df["category_l1"] = ""
            df["category_l2"] = ""
            df["category_l3"] = ""
            return df

        cat_split = df["category"].astype(str).str.split("|", expand=False)

        df["category_l1"] = cat_split.apply(
            lambda parts: parts[0].strip() if parts and len(parts) > 0 else ""
        )
        df["category_l2"] = cat_split.apply(
            lambda parts: parts[1].strip() if parts and len(parts) > 1 else ""
        )
        df["category_l3"] = cat_split.apply(
            lambda parts: parts[2].strip() if parts and len(parts) > 2 else ""
        )

        unique_l1 = df["category_l1"].nunique()
        unique_l2 = df["category_l2"].nunique()
        logger.info(
            f"  Category hierarchy — L1: {unique_l1} unique, L2: {unique_l2} unique"
        )
        return df

    # ------------------------------------------------------------------
    # Feature: Brand extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_brand(product_name: str, category_l2: str) -> str:
        """
        Heuristically extract the brand from a product name.

        Strategy:
        1. Take the first 1–2 tokens of the product name.
        2. Filter out stop words and generic terms.
        3. Return the first valid token as the brand.

        This is approximate — the Amazon dataset does not have a dedicated
        brand column. For Kaggle `amazon-sales-dataset`, the brand is usually
        the first meaningful word(s) in the product name.

        Args:
            product_name: Full product name string.
            category_l2: Second-level category (fallback context).

        Returns:
            Brand string, or "Unknown" if extraction fails.
        """
        if not product_name or str(product_name).strip() == "":
            return "Unknown"

        tokens = str(product_name).split()
        for token in tokens[:3]:
            cleaned = re.sub(r"[^a-zA-Z0-9]", "", token)
            if cleaned and cleaned.lower() not in _BRAND_STOP_WORDS and len(cleaned) > 1:
                return cleaned.title()

        return "Unknown"

    def _add_brand_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a brand column by extracting it from product_name."""
        if "product_name" not in df.columns:
            df["brand"] = "Unknown"
            return df

        cat_l2 = df.get("category_l2", pd.Series([""] * len(df)))

        df["brand"] = [
            self._extract_brand(name, cat)
            for name, cat in zip(df["product_name"], cat_l2)
        ]

        unique_brands = df["brand"].nunique()
        logger.info(f"  Brand extraction complete — {unique_brands} unique brands")
        return df

    # ------------------------------------------------------------------
    # Feature: Price bucket
    # ------------------------------------------------------------------

    def _assign_price_bucket(self, price: float) -> str:
        """
        Assign a human-readable price bucket label.

        Uses the price_buckets defined in config.yaml.

        Args:
            price: Numeric price value (discounted_price).

        Returns:
            Bucket label string, or "Unknown" for NaN/invalid.
        """
        if pd.isna(price) or price < 0:
            return "Unknown"

        for bucket in self.feature_config.price_buckets:
            if bucket.min_price <= price < bucket.max_price:
                return bucket.label

        # If above all buckets, assign the last bucket's label
        if self.feature_config.price_buckets:
            return self.feature_config.price_buckets[-1].label
        return "Unknown"

    def _add_price_features(self, df: pd.DataFrame) -> tuple:
        """Add price_bucket and price_normalized columns."""
        if "discounted_price" not in df.columns:
            df["price_bucket"] = "Unknown"
            df["price_normalized"] = 0.0
            return df, {}

        # Price bucket
        df["price_bucket"] = df["discounted_price"].apply(self._assign_price_bucket)

        bucket_dist: Dict[str, int] = df["price_bucket"].value_counts().to_dict()
        logger.info(f"  Price buckets: {bucket_dist}")

        # Price normalized (0–1 min-max)
        max_price = df["discounted_price"].max()
        if pd.isna(max_price) or max_price == 0:
            df["price_normalized"] = 0.0
        else:
            df["price_normalized"] = (df["discounted_price"] / max_price).clip(0, 1)

        return df, bucket_dist


    # ------------------------------------------------------------------
    # Feature: Popularity score
    # ------------------------------------------------------------------

    def _compute_popularity_score(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
        """
        Compute a popularity score for each product.

        Formula (configurable via config.yaml):
            score = rating^rating_weight * log1p(rating_count)^review_count_weight

        The score is normalized to [0, 1] by dividing by the maximum.

        Args:
            df: DataFrame with rating and rating_count columns.

        Returns:
            (df_with_popularity_score, stats_dict)
        """
        formula = self.feature_config.popularity_formula
        rw = formula.rating_weight
        cw = formula.review_count_weight

        if "rating" not in df.columns or "rating_count" not in df.columns:
            df["popularity_score"] = 0.0
            return df, {}

        # Clamp rating to [0, 5] to guard against parsing artifacts
        rating = df["rating"].clip(0, 5).fillna(0)
        rating_count = df["rating_count"].clip(0).fillna(0)

        raw_score = (rating ** rw) * (np.log1p(rating_count) ** cw)
        max_score = raw_score.max()

        if max_score > 0:
            df["popularity_score"] = (raw_score / max_score).round(6)
        else:
            df["popularity_score"] = 0.0

        stats = {
            "min": round(float(df["popularity_score"].min()), 4),
            "max": round(float(df["popularity_score"].max()), 4),
            "mean": round(float(df["popularity_score"].mean()), 4),
            "median": round(float(df["popularity_score"].median()), 4),
        }
        logger.info(f"  Popularity score stats: {stats}")
        return df, stats

    # ------------------------------------------------------------------
    # Feature: Discount tier
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_discount_tier(discount_pct: float) -> str:
        """
        Classify a discount percentage into a named tier.

        Tiers: No Discount (0%), Low (1–19%), Medium (20–39%), High (40%+)
        """
        if pd.isna(discount_pct) or discount_pct <= 0:
            return "No Discount"
        if discount_pct < 20:
            return "Low"
        if discount_pct < 40:
            return "Medium"
        return "High"

    def _add_discount_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add discount_tier column."""
        if "discount_percentage" in df.columns:
            df["discount_tier"] = df["discount_percentage"].apply(self._assign_discount_tier)
        else:
            df["discount_tier"] = "Unknown"
        return df

    # ------------------------------------------------------------------
    # Feature: Product metadata for hybrid engine
    # ------------------------------------------------------------------

    @staticmethod
    def _add_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary / computed metadata columns used by the hybrid engine
        and explanation layer:
            is_highly_rated   — rating >= 4.0
            is_budget         — price_bucket == "Budget"
            has_description   — about_product non-empty
            has_image         — img_link non-empty
        """
        if "rating" in df.columns:
            df["is_highly_rated"] = (df["rating"] >= 4.0).astype(int)
        else:
            df["is_highly_rated"] = 0

        if "price_bucket" in df.columns:
            df["is_budget"] = (df["price_bucket"] == "Budget").astype(int)
        else:
            df["is_budget"] = 0

        if "about_product" in df.columns:
            df["has_description"] = (
                df["about_product"].fillna("").str.strip().str.len() > 10
            ).astype(int)
        else:
            df["has_description"] = 0

        if "img_link" in df.columns:
            df["has_image"] = (
                df["img_link"].fillna("").str.strip().str.startswith("http")
            ).astype(int)
        else:
            df["has_image"] = 0

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        """
        Execute the complete feature engineering pipeline.

        Returns:
            FeatureEngineeringArtifact with output paths and statistics.

        Raises:
            FeatureEngineeringError: On feature construction failures.
            CustomException: On unexpected errors.
        """
        logger.info("=" * 60)
        logger.info("STEP 4: FEATURE ENGINEERING")
        logger.info("=" * 60)

        try:
            create_directories(["data/processed"])

            # Load cleaned data from transformation step
            df = pd.read_csv(
                self.transformation_artifact.processed_filepath,
                encoding="utf-8",
                low_memory=False,
            )
            logger.info(f"Loaded {len(df):,} cleaned records for feature engineering")

            # --- Feature 1: Combined text (TF-IDF input) ---
            df, vocab_estimate = self._build_combined_text(df)

            # --- Feature 2: Category hierarchy ---
            df = self._extract_category_levels(df)

            # --- Feature 3: Brand extraction ---
            df = self._add_brand_feature(df)

            # --- Feature 4: Price features (bucket + normalized) ---
            df, bucket_dist = self._add_price_features(df)

            # --- Feature 5: Popularity score ---
            df, pop_stats = self._compute_popularity_score(df)

            # --- Feature 6: Discount tier ---
            df = self._add_discount_tier(df)

            # --- Feature 7: Metadata columns ---
            df = self._add_metadata_columns(df)

            # Collect feature column list
            feature_columns = [
                "combined_text",
                "combined_text_tfidf",
                "category_l1",
                "category_l2",
                "category_l3",
                "brand",
                "price_bucket",
                "price_normalized",
                "popularity_score",
                "discount_tier",
                "is_highly_rated",
                "is_budget",
                "has_description",
                "has_image",
            ]
            actual_features = [c for c in feature_columns if c in df.columns]

            logger.info(f"  Features created: {len(actual_features)}")
            logger.info(f"  Feature columns: {actual_features}")
            logger.info(f"  Final dataset shape: {df.shape}")

            # Save features CSV
            features_path = Path("data/processed/features.csv")
            df.to_csv(features_path, index=False, encoding="utf-8")
            logger.info(f"  Features saved → {features_path}")

            artifact = FeatureEngineeringArtifact(
                feature_filepath=str(features_path.resolve()),
                combined_text_column="combined_text_tfidf",
                feature_columns=actual_features,
                vocabulary_estimate=vocab_estimate,
                price_bucket_distribution=bucket_dist,
                popularity_score_stats=pop_stats,
            )

            logger.info("STEP 4 COMPLETE: Feature Engineering")
            return artifact

        except Exception as e:
            if isinstance(e, (FeatureEngineeringError, CustomException)):
                raise
            raise CustomException(
                f"Feature engineering failed: {e}", sys.exc_info()
            ) from e
