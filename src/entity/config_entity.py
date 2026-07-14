"""
Configuration Entity Definitions
=================================
Typed dataclass entities for every section of config.yaml.
These are the output types of ConfigurationManager.

All entities use Python dataclasses for:
- Type safety
- IDE autocompletion
- Easy serialization
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------
# Project
# ---------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Top-level project metadata."""
    name: str
    version: str
    description: str
    environment: str


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------

@dataclass
class DataValidationThresholds:
    """Acceptable data quality thresholds."""
    max_missing_ratio: float
    max_duplicate_ratio: float
    min_records: int
    min_price: float
    max_price: float
    min_rating: float
    max_rating: float
    min_rating_count: int


@dataclass
class DataConfig:
    """Dataset paths, schema, and validation settings."""
    raw_dir: str
    processed_dir: str
    interim_dir: str
    external_dir: str
    kaggle_dataset: str
    raw_filename: str
    raw_filepath: str
    fallback_filepath: str
    use_fallback: bool
    expected_columns: List[str]
    validation: DataValidationThresholds


# ---------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------

@dataclass
class PriceBucket:
    """A single price bucket definition."""
    min_price: float
    max_price: float
    label: str


@dataclass
class PopularityFormula:
    """Weights for the popularity score formula."""
    rating_weight: float
    review_count_weight: float


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    combined_text_fields: List[str]
    price_buckets: List[PriceBucket]
    popularity_formula: PopularityFormula


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

@dataclass
class TFIDFConfig:
    """TF-IDF vectorizer hyperparameters."""
    max_features: int
    ngram_range: Tuple[int, int]
    min_df: int
    max_df: float
    stop_words: str
    sublinear_tf: bool
    analyzer: str


@dataclass
class ModelConfig:
    """Model artifact paths and hyperparameters."""
    artifacts_dir: str
    tfidf_model_path: str
    similarity_matrix_path: str
    product_index_path: str
    metadata_path: str
    tfidf: TFIDFConfig


# ---------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------

@dataclass
class HybridWeights:
    """Weights for the hybrid recommendation engine."""
    content_based: float
    category_similarity: float
    brand_similarity: float
    price_similarity: float
    popularity: float
    rating: float


@dataclass
class ConfidenceWeights:
    """Weights for computing recommendation confidence scores."""
    text_similarity: float
    category_similarity: float
    brand_similarity: float
    price_similarity: float
    popularity_contribution: float
    rating_score: float


@dataclass
class ConfidenceThresholds:
    """Score boundaries for confidence grade labels."""
    excellent: float
    very_good: float
    good: float
    fair: float


@dataclass
class ReasonTagThresholds:
    """Minimum scores to activate each reason tag."""
    same_brand: float
    similar_category: float
    similar_specifications: float
    frequently_purchased_together: int
    highly_rated: float
    popular_choice: float
    great_price_match: float


@dataclass
class TrendingConfig:
    """Trending engine configuration."""
    window_hours: int
    min_rating_count: int
    decay_factor: float


@dataclass
class DealsConfig:
    """Deals filter configuration."""
    min_discount_percentage: float


@dataclass
class NewArrivalsConfig:
    """New arrivals filter configuration."""
    days_threshold: int


@dataclass
class FBTConfig:
    """Frequently Bought Together configuration."""
    min_co_occurrence: int


@dataclass
class RecommendationConfig:
    """Complete recommendation system configuration."""
    active_strategy: str
    default_n: int
    max_n: int
    min_n: int
    similar_products_n: int
    fbt_n: int
    trending_n: int
    recently_viewed_n: int
    homepage_sections_n: int
    top_rated_n: int
    new_arrivals_n: int
    deals_n: int
    best_sellers_n: int
    customers_also_bought_n: int
    because_you_viewed_n: int
    recommended_for_you_n: int
    hybrid_weights: HybridWeights
    confidence_weights: ConfidenceWeights
    confidence_thresholds: ConfidenceThresholds
    reason_tag_thresholds: ReasonTagThresholds
    trending: TrendingConfig
    deals: DealsConfig
    new_arrivals: NewArrivalsConfig
    fbt: FBTConfig


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Model evaluation settings."""
    k_values: List[int]
    min_precision_at_5: float
    diversity_sample_size: int
    novelty_sample_size: int


# ---------------------------------------------------------------
# API
# ---------------------------------------------------------------

@dataclass
class CORSConfig:
    """CORS policy configuration."""
    allow_origins: List[str]
    allow_credentials: bool
    allow_methods: List[str]
    allow_headers: List[str]


@dataclass
class RateLimitConfig:
    """Rate limits per endpoint group."""
    default: str
    search: str
    training: str
    explain: str
    analytics: str


@dataclass
class APIConfig:
    """FastAPI application configuration."""
    host: str
    port: int
    title: str
    description: str
    version: str
    docs_url: str
    redoc_url: str
    cors: CORSConfig
    rate_limits: RateLimitConfig


# ---------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------

@dataclass
class MLflowStages:
    """MLflow model registry stage names."""
    staging: str
    production: str
    archived: str


@dataclass
class MLflowConfig:
    """MLflow tracking and registry configuration."""
    tracking_uri: str
    experiment_name: str
    model_name: str
    artifacts_dir: str
    stages: MLflowStages
    auto_promote_threshold: float


# ---------------------------------------------------------------
# Database
# ---------------------------------------------------------------

@dataclass
class DatabaseConfig:
    """SQLAlchemy database configuration."""
    url: str
    echo: bool
    pool_pre_ping: bool
    connect_args: Dict[str, Any]


# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------

@dataclass
class LogFileConfig:
    """Paths for each domain logger."""
    api: str
    training: str
    prediction: str
    experiment: str
    deployment: str
    database: str
    security: str


@dataclass
class LoggingConfig:
    """Logging framework configuration."""
    level: str
    format: str
    date_format: str
    logs_dir: str
    max_bytes: int
    backup_count: int
    files: LogFileConfig


# ---------------------------------------------------------------
# Cache
# ---------------------------------------------------------------

@dataclass
class CacheTTLs:
    """TTL values (seconds) per cache key pattern."""
    similar: int
    trending: int
    fbt: int
    products_list: int
    top_rated: int
    new_arrivals: int
    deals: int
    best_sellers: int
    search: int
    categories: int


@dataclass
class CacheConfig:
    """In-memory cache configuration."""
    enabled: bool
    max_size: int
    default_ttl: int
    ttls: CacheTTLs


# ---------------------------------------------------------------
# Session
# ---------------------------------------------------------------

@dataclass
class SessionConfig:
    """Anonymous session management configuration."""
    expiry_days: int
    recently_viewed_limit: int
    recently_searched_limit: int
    cleanup_interval_hours: int
    session_id_header: str


# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------

@dataclass
class TypoCorrectionConfig:
    """rapidfuzz typo correction settings."""
    enabled: bool
    min_score: int
    max_suggestions: int


@dataclass
class AutocompleteConfig:
    """Autocomplete settings."""
    limit: int
    min_chars: int


@dataclass
class SearchConfig:
    """Search service configuration."""
    typo_correction: TypoCorrectionConfig
    autocomplete: AutocompleteConfig
    min_query_length: int
    max_query_length: int
    results_per_page: int


# ---------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------

@dataclass
class MonitoringConfig:
    """System monitoring configuration."""
    enabled: bool
    drift_threshold: float
    latency_warning_ms: int
    latency_critical_ms: int
    max_latency_history: int
    system_check_interval: int


# ---------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------

@dataclass
class AnalyticsConfig:
    """Recommendation analytics configuration."""
    enabled: bool
    ctr_window_hours: int
    conversion_window_minutes: int
    min_impressions_for_ctr: int


# ---------------------------------------------------------------
# Background Jobs
# ---------------------------------------------------------------

@dataclass
class BackgroundConfig:
    """Background job executor configuration."""
    max_workers: int
    training_timeout: int
    job_cleanup_hours: int
