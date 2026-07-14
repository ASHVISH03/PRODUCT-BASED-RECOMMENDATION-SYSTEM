"""
Configuration Manager
======================
Loads and validates config.yaml, applies environment variable overrides,
and returns strongly-typed dataclass entities for each configuration section.

Usage:
    from src.config.configuration import ConfigurationManager

    config = ConfigurationManager()
    data_cfg = config.get_data_config()
    model_cfg = config.get_model_config()
    api_cfg = config.get_api_config()
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.entity.config_entity import (
    AnalyticsConfig,
    APIConfig,
    AutocompleteConfig,
    BackgroundConfig,
    CacheConfig,
    CacheTTLs,
    CORSConfig,
    ConfidenceThresholds,
    ConfidenceWeights,
    DataConfig,
    DataValidationThresholds,
    DatabaseConfig,
    DealsConfig,
    EvaluationConfig,
    FBTConfig,
    FeatureConfig,
    HybridWeights,
    LogFileConfig,
    LoggingConfig,
    MLflowConfig,
    MLflowStages,
    ModelConfig,
    MonitoringConfig,
    NewArrivalsConfig,
    PopularityFormula,
    PriceBucket,
    ProjectConfig,
    RateLimitConfig,
    ReasonTagThresholds,
    RecommendationConfig,
    SearchConfig,
    SessionConfig,
    TFIDFConfig,
    TrendingConfig,
    TypoCorrectionConfig,
)
from src.exception.exception import ConfigurationError

# Load .env file if present (does not override existing environment variables)
load_dotenv()

# Default config file location — relative to the project root
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class ConfigurationManager:
    """
    Centralized configuration loader.

    Reads config.yaml from the specified path, applies environment
    variable overrides for selected keys, and returns typed entities.

    Args:
        config_path: Path to config.yaml. Defaults to src/config/config.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._raw: Dict[str, Any] = self._load_yaml()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_yaml(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file."""
        if not self._config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self._config_path}. "
                "Ensure src/config/config.yaml exists."
            )
        with open(self._config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ConfigurationError(
                f"Invalid configuration format in {self._config_path}. "
                "Expected a YAML mapping (dict)."
            )
        return config

    @staticmethod
    def _env(key: str, default: Any = None) -> Any:
        """Read an environment variable, returning default if not set."""
        return os.environ.get(key, default)

    @staticmethod
    def _get(cfg: Dict, *keys: str, default: Any = None) -> Any:
        """Safely navigate nested dict keys."""
        for key in keys:
            if not isinstance(cfg, dict):
                return default
            cfg = cfg.get(key, default)
        return cfg

    # ------------------------------------------------------------------
    # Public section accessors
    # ------------------------------------------------------------------

    def get_project_config(self) -> ProjectConfig:
        """Return project metadata configuration."""
        p = self._raw.get("project", {})
        return ProjectConfig(
            name=p.get("name", "Product Recommendation System"),
            version=p.get("version", "1.0.0"),
            description=p.get("description", ""),
            environment=self._env("APP_ENV", p.get("environment", "development")),
        )

    def get_data_config(self) -> DataConfig:
        """Return dataset paths and validation thresholds."""
        d = self._raw.get("data", {})
        v = d.get("validation", {})

        # Environment variable can override the raw filepath
        raw_filepath = (
            self._env("DATA_RAW_FILEPATH") or d.get("raw_filepath", "data/raw/amazon.csv")
        )

        return DataConfig(
            raw_dir=d.get("raw_dir", "data/raw"),
            processed_dir=d.get("processed_dir", "data/processed"),
            interim_dir=d.get("interim_dir", "data/interim"),
            external_dir=d.get("external_dir", "data/external"),
            kaggle_dataset=d.get("kaggle_dataset", "karkavelrajaj/amazon-sales-dataset"),
            raw_filename=d.get("raw_filename", "amazon.csv"),
            raw_filepath=raw_filepath,
            fallback_filepath=d.get("fallback_filepath", "data/external/sample_products.csv"),
            use_fallback=d.get("use_fallback", False),
            expected_columns=d.get("expected_columns", []),
            validation=DataValidationThresholds(
                max_missing_ratio=v.get("max_missing_ratio", 0.30),
                max_duplicate_ratio=v.get("max_duplicate_ratio", 0.10),
                min_records=v.get("min_records", 100),
                min_price=v.get("min_price", 0.0),
                max_price=v.get("max_price", 1_000_000.0),
                min_rating=v.get("min_rating", 0.0),
                max_rating=v.get("max_rating", 5.0),
                min_rating_count=v.get("min_rating_count", 0),
            ),
        )

    def get_feature_config(self) -> FeatureConfig:
        """Return feature engineering configuration."""
        f = self._raw.get("features", {})
        raw_buckets = f.get("price_buckets", [])
        pf = f.get("popularity_formula", {})
        return FeatureConfig(
            combined_text_fields=f.get("combined_text_fields", ["product_name", "category"]),
            price_buckets=[
                PriceBucket(min_price=b[0], max_price=b[1], label=b[2])
                for b in raw_buckets
            ],
            popularity_formula=PopularityFormula(
                rating_weight=pf.get("rating_weight", 0.7),
                review_count_weight=pf.get("review_count_weight", 0.3),
            ),
        )

    def get_model_config(self) -> ModelConfig:
        """Return model artifact paths and TF-IDF hyperparameters."""
        m = self._raw.get("model", {})
        t = m.get("tfidf", {})
        ngram_raw = t.get("ngram_range", [1, 2])
        return ModelConfig(
            artifacts_dir=m.get("artifacts_dir", "models"),
            tfidf_model_path=m.get("tfidf_model_path", "models/tfidf.pkl"),
            similarity_matrix_path=m.get("similarity_matrix_path", "models/similarity.pkl"),
            product_index_path=m.get("product_index_path", "models/product_index.pkl"),
            metadata_path=m.get("metadata_path", "models/metadata.json"),
            tfidf=TFIDFConfig(
                max_features=t.get("max_features", 5000),
                ngram_range=(ngram_raw[0], ngram_raw[1]),
                min_df=t.get("min_df", 2),
                max_df=t.get("max_df", 0.95),
                stop_words=t.get("stop_words", "english"),
                sublinear_tf=t.get("sublinear_tf", True),
                analyzer=t.get("analyzer", "word"),
            ),
        )

    def get_recommendation_config(self) -> RecommendationConfig:
        """Return full recommendation engine configuration."""
        r = self._raw.get("recommendation", {})
        hw = r.get("hybrid_weights", {})
        cw = r.get("confidence_weights", {})
        ct = r.get("confidence_thresholds", {})
        rt = r.get("reason_tag_thresholds", {})
        tr = r.get("trending", {})
        dl = r.get("deals", {})
        na = r.get("new_arrivals", {})
        ft = r.get("fbt", {})

        return RecommendationConfig(
            active_strategy=r.get("active_strategy", "hybrid"),
            default_n=r.get("default_n", 10),
            max_n=r.get("max_n", 50),
            min_n=r.get("min_n", 1),
            similar_products_n=r.get("similar_products_n", 8),
            fbt_n=r.get("fbt_n", 4),
            trending_n=r.get("trending_n", 12),
            recently_viewed_n=r.get("recently_viewed_n", 6),
            homepage_sections_n=r.get("homepage_sections_n", 10),
            top_rated_n=r.get("top_rated_n", 10),
            new_arrivals_n=r.get("new_arrivals_n", 10),
            deals_n=r.get("deals_n", 10),
            best_sellers_n=r.get("best_sellers_n", 10),
            customers_also_bought_n=r.get("customers_also_bought_n", 8),
            because_you_viewed_n=r.get("because_you_viewed_n", 8),
            recommended_for_you_n=r.get("recommended_for_you_n", 12),
            hybrid_weights=HybridWeights(
                content_based=hw.get("content_based", 0.35),
                category_similarity=hw.get("category_similarity", 0.25),
                brand_similarity=hw.get("brand_similarity", 0.15),
                price_similarity=hw.get("price_similarity", 0.10),
                popularity=hw.get("popularity", 0.10),
                rating=hw.get("rating", 0.05),
            ),
            confidence_weights=ConfidenceWeights(
                text_similarity=cw.get("text_similarity", 0.35),
                category_similarity=cw.get("category_similarity", 0.25),
                brand_similarity=cw.get("brand_similarity", 0.15),
                price_similarity=cw.get("price_similarity", 0.10),
                popularity_contribution=cw.get("popularity_contribution", 0.10),
                rating_score=cw.get("rating_score", 0.05),
            ),
            confidence_thresholds=ConfidenceThresholds(
                excellent=ct.get("excellent", 0.90),
                very_good=ct.get("very_good", 0.75),
                good=ct.get("good", 0.60),
                fair=ct.get("fair", 0.45),
            ),
            reason_tag_thresholds=ReasonTagThresholds(
                same_brand=rt.get("same_brand", 1.0),
                similar_category=rt.get("similar_category", 0.8),
                similar_specifications=rt.get("similar_specifications", 0.7),
                frequently_purchased_together=rt.get("frequently_purchased_together", 3),
                highly_rated=rt.get("highly_rated", 0.85),
                popular_choice=rt.get("popular_choice", 0.8),
                great_price_match=rt.get("great_price_match", 0.9),
            ),
            trending=TrendingConfig(
                window_hours=tr.get("window_hours", 24),
                min_rating_count=tr.get("min_rating_count", 5),
                decay_factor=tr.get("decay_factor", 0.95),
            ),
            deals=DealsConfig(
                min_discount_percentage=dl.get("min_discount_percentage", 20.0),
            ),
            new_arrivals=NewArrivalsConfig(
                days_threshold=na.get("days_threshold", 30),
            ),
            fbt=FBTConfig(
                min_co_occurrence=ft.get("min_co_occurrence", 2),
            ),
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        """Return model evaluation settings."""
        e = self._raw.get("evaluation", {})
        return EvaluationConfig(
            k_values=e.get("k_values", [5, 10]),
            min_precision_at_5=e.get("min_precision_at_5", 0.50),
            diversity_sample_size=e.get("diversity_sample_size", 100),
            novelty_sample_size=e.get("novelty_sample_size", 100),
        )

    def get_api_config(self) -> APIConfig:
        """Return FastAPI application configuration."""
        a = self._raw.get("api", {})
        c = a.get("cors", {})
        rl = a.get("rate_limits", {})
        return APIConfig(
            host=self._env("APP_HOST", a.get("host", "0.0.0.0")),
            port=int(self._env("APP_PORT", a.get("port", 8000))),
            title=a.get("title", "Product Recommendation System API"),
            description=a.get("description", ""),
            version=a.get("version", "1.0.0"),
            docs_url=a.get("docs_url", "/docs"),
            redoc_url=a.get("redoc_url", "/redoc"),
            cors=CORSConfig(
                allow_origins=c.get("allow_origins", ["*"]),
                allow_credentials=c.get("allow_credentials", False),
                allow_methods=c.get("allow_methods", ["*"]),
                allow_headers=c.get("allow_headers", ["*"]),
            ),
            rate_limits=RateLimitConfig(
                default=rl.get("default", "100/minute"),
                search=rl.get("search", "60/minute"),
                training=rl.get("training", "5/hour"),
                explain=rl.get("explain", "30/minute"),
                analytics=rl.get("analytics", "100/minute"),
            ),
        )

    def get_mlflow_config(self) -> MLflowConfig:
        """Return MLflow tracking and registry configuration."""
        m = self._raw.get("mlflow", {})
        s = m.get("stages", {})
        return MLflowConfig(
            tracking_uri=self._env("MLFLOW_TRACKING_URI", m.get("tracking_uri", "mlruns")),
            experiment_name=self._env(
                "MLFLOW_EXPERIMENT_NAME",
                m.get("experiment_name", "product-recommendation"),
            ),
            model_name=m.get("model_name", "recommendation-model"),
            artifacts_dir=m.get("artifacts_dir", "artifacts"),
            stages=MLflowStages(
                staging=s.get("staging", "Staging"),
                production=s.get("production", "Production"),
                archived=s.get("archived", "Archived"),
            ),
            auto_promote_threshold=m.get("auto_promote_threshold", 0.02),
        )

    def get_database_config(self) -> DatabaseConfig:
        """Return SQLAlchemy database configuration."""
        d = self._raw.get("database", {})
        return DatabaseConfig(
            url=self._env("DATABASE_URL", d.get("url", "sqlite:///data/recommendation.db")),
            echo=d.get("echo", False),
            pool_pre_ping=d.get("pool_pre_ping", True),
            connect_args=d.get("connect_args", {"check_same_thread": False}),
        )

    def get_logging_config(self) -> LoggingConfig:
        """Return logging framework configuration."""
        lg = self._raw.get("logging", {})
        files = lg.get("files", {})
        return LoggingConfig(
            level=self._env("LOG_LEVEL", lg.get("level", "INFO")),
            format=lg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
            date_format=lg.get("date_format", "%Y-%m-%d %H:%M:%S"),
            logs_dir=lg.get("logs_dir", "logs"),
            max_bytes=lg.get("max_bytes", 10 * 1024 * 1024),
            backup_count=lg.get("backup_count", 5),
            files=LogFileConfig(
                api=files.get("api", "logs/api.log"),
                training=files.get("training", "logs/training.log"),
                prediction=files.get("prediction", "logs/prediction.log"),
                experiment=files.get("experiment", "logs/experiment.log"),
                deployment=files.get("deployment", "logs/deployment.log"),
                database=files.get("database", "logs/database.log"),
                security=files.get("security", "logs/security.log"),
            ),
        )

    def get_cache_config(self) -> CacheConfig:
        """Return in-memory cache configuration."""
        c = self._raw.get("cache", {})
        t = c.get("ttls", {})
        return CacheConfig(
            enabled=c.get("enabled", True),
            max_size=c.get("max_size", 1000),
            default_ttl=int(self._env("CACHE_DEFAULT_TTL", c.get("default_ttl", 3600))),
            ttls=CacheTTLs(
                similar=t.get("similar", 3600),
                trending=t.get("trending", 900),
                fbt=t.get("fbt", 14400),
                products_list=t.get("products_list", 1800),
                top_rated=t.get("top_rated", 3600),
                new_arrivals=t.get("new_arrivals", 1800),
                deals=t.get("deals", 900),
                best_sellers=t.get("best_sellers", 3600),
                search=t.get("search", 300),
                categories=t.get("categories", 7200),
            ),
        )

    def get_session_config(self) -> SessionConfig:
        """Return anonymous session configuration."""
        s = self._raw.get("session", {})
        return SessionConfig(
            expiry_days=s.get("expiry_days", 30),
            recently_viewed_limit=s.get("recently_viewed_limit", 50),
            recently_searched_limit=s.get("recently_searched_limit", 20),
            cleanup_interval_hours=s.get("cleanup_interval_hours", 24),
            session_id_header=s.get("session_id_header", "X-Session-ID"),
        )

    def get_search_config(self) -> SearchConfig:
        """Return search service configuration."""
        s = self._raw.get("search", {})
        tc = s.get("typo_correction", {})
        ac = s.get("autocomplete", {})
        return SearchConfig(
            typo_correction=TypoCorrectionConfig(
                enabled=tc.get("enabled", True),
                min_score=tc.get("min_score", 75),
                max_suggestions=tc.get("max_suggestions", 3),
            ),
            autocomplete=AutocompleteConfig(
                limit=ac.get("limit", 10),
                min_chars=ac.get("min_chars", 2),
            ),
            min_query_length=s.get("min_query_length", 2),
            max_query_length=s.get("max_query_length", 200),
            results_per_page=s.get("results_per_page", 20),
        )

    def get_monitoring_config(self) -> MonitoringConfig:
        """Return system monitoring configuration."""
        m = self._raw.get("monitoring", {})
        return MonitoringConfig(
            enabled=m.get("enabled", True),
            drift_threshold=m.get("drift_threshold", 0.15),
            latency_warning_ms=m.get("latency_warning_ms", 500),
            latency_critical_ms=m.get("latency_critical_ms", 2000),
            max_latency_history=m.get("max_latency_history", 1000),
            system_check_interval=m.get("system_check_interval", 60),
        )

    def get_analytics_config(self) -> AnalyticsConfig:
        """Return recommendation analytics configuration."""
        a = self._raw.get("analytics", {})
        return AnalyticsConfig(
            enabled=a.get("enabled", True),
            ctr_window_hours=a.get("ctr_window_hours", 24),
            conversion_window_minutes=a.get("conversion_window_minutes", 30),
            min_impressions_for_ctr=a.get("min_impressions_for_ctr", 10),
        )

    def get_background_config(self) -> BackgroundConfig:
        """Return background job executor configuration."""
        b = self._raw.get("background", {})
        return BackgroundConfig(
            max_workers=b.get("max_workers", 2),
            training_timeout=b.get("training_timeout", 3600),
            job_cleanup_hours=b.get("job_cleanup_hours", 72),
        )

    def get_raw(self) -> Dict[str, Any]:
        """Return the raw parsed YAML dictionary (read-only)."""
        return dict(self._raw)
