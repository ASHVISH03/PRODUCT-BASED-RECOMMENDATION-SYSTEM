"""
Artifact Entity Definitions
============================
Typed dataclass entities representing the OUTPUT of each pipeline
component. These flow between pipeline stages and are logged by
each component upon completion.

Design principle:
- Each pipeline component returns exactly one Artifact entity.
- Entities are immutable snapshots of what was produced.
- Downstream components receive upstream artifacts as input.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------
# Data Ingestion Artifact
# ---------------------------------------------------------------

@dataclass
class DataIngestionArtifact:
    """
    Output of the Data Ingestion component.

    Attributes:
        raw_filepath: Absolute path to the ingested CSV file.
        is_fallback: True if synthetic fallback was used instead of real data.
        dataset_version: Hash or timestamp identifying this data version.
        record_count: Number of rows in the raw dataset.
        column_count: Number of columns in the raw dataset.
        file_size_mb: File size in megabytes.
        ingestion_timestamp: ISO 8601 timestamp of when ingestion occurred.
        source: 'kaggle' | 'local' | 'fallback'
    """
    raw_filepath: str
    is_fallback: bool
    dataset_version: str
    record_count: int
    column_count: int
    file_size_mb: float
    ingestion_timestamp: str
    source: str  # 'kaggle' | 'local' | 'fallback'


# ---------------------------------------------------------------
# Data Validation Artifact
# ---------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A single validation finding."""
    column: str
    issue_type: str  # 'missing' | 'duplicate' | 'schema' | 'range' | 'dtype'
    severity: str    # 'error' | 'warning' | 'info'
    count: int
    message: str


@dataclass
class DataValidationArtifact:
    """
    Output of the Data Validation component.

    Attributes:
        validation_status: True if data passed all error-level checks.
        validation_report_path: Path to the saved validation report (JSON).
        missing_values_count: Total null values across all columns.
        duplicate_count: Number of duplicate rows found.
        schema_valid: True if all expected columns are present with correct types.
        issues: List of all validation findings.
        validated_record_count: Records remaining after validation.
    """
    validation_status: bool
    validation_report_path: str
    missing_values_count: int
    duplicate_count: int
    schema_valid: bool
    issues: List[ValidationIssue]
    validated_record_count: int


# ---------------------------------------------------------------
# Data Transformation Artifact
# ---------------------------------------------------------------

@dataclass
class DataTransformationArtifact:
    """
    Output of the Data Transformation component.

    Attributes:
        processed_filepath: Path to the cleaned, processed CSV.
        interim_filepath: Path to intermediate transformation output.
        input_record_count: Records before transformation.
        output_record_count: Records after transformation (may be fewer).
        removed_duplicates: How many duplicates were dropped.
        filled_nulls: How many null values were filled.
        dropped_nulls: How many records were dropped due to nulls.
        text_columns_cleaned: Columns that underwent text normalization.
    """
    processed_filepath: str
    interim_filepath: str
    input_record_count: int
    output_record_count: int
    removed_duplicates: int
    filled_nulls: int
    dropped_nulls: int
    text_columns_cleaned: List[str]


# ---------------------------------------------------------------
# Feature Engineering Artifact
# ---------------------------------------------------------------

@dataclass
class FeatureEngineeringArtifact:
    """
    Output of the Feature Engineering component.

    Attributes:
        feature_filepath: Path to the feature-engineered CSV.
        combined_text_column: Name of the column containing merged text.
        feature_columns: All feature columns created.
        vocabulary_estimate: Estimated vocabulary size from combined text.
        price_bucket_distribution: Count of records per price bucket.
        popularity_score_stats: Min/max/mean of the popularity score.
    """
    feature_filepath: str
    combined_text_column: str
    feature_columns: List[str]
    vocabulary_estimate: int
    price_bucket_distribution: Dict[str, int]
    popularity_score_stats: Dict[str, float]


# ---------------------------------------------------------------
# Model Training Artifact
# ---------------------------------------------------------------

@dataclass
class ModelTrainingArtifact:
    """
    Output of the Model Training component.

    Attributes:
        tfidf_model_path: Path to saved TF-IDF vectorizer pickle.
        similarity_matrix_path: Path to saved cosine similarity matrix pickle.
        product_index_path: Path to saved product ID → index mapping.
        metadata_path: Path to saved model metadata JSON.
        training_duration_seconds: Wall-clock training time.
        vocabulary_size: Number of features in the TF-IDF vocabulary.
        matrix_shape: Shape of the similarity matrix (n, n).
        mlflow_run_id: MLflow run ID for this training session.
        mlflow_experiment_id: MLflow experiment ID.
        dataset_version: Version of dataset used for training.
    """
    tfidf_model_path: str
    similarity_matrix_path: str
    product_index_path: str
    metadata_path: str
    training_duration_seconds: float
    vocabulary_size: int
    matrix_shape: tuple
    mlflow_run_id: Optional[str]
    mlflow_experiment_id: Optional[str]
    dataset_version: str


# ---------------------------------------------------------------
# Model Evaluation Artifact
# ---------------------------------------------------------------

@dataclass
class PrecisionRecallAtK:
    """Precision and Recall at a specific K."""
    k: int
    precision: float
    recall: float


@dataclass
class ModelEvaluationArtifact:
    """
    Output of the Model Evaluation component.

    Attributes:
        precision_recall: List of Precision@K and Recall@K for each K.
        coverage: Fraction of catalog products covered by recommendations.
        diversity: Average pairwise dissimilarity in recommendation lists.
        novelty: Inverse-popularity weighted novelty score.
        intra_list_similarity: Average similarity within a recommendation list.
        evaluation_report_path: Path to saved evaluation report (JSON).
        meets_minimum_threshold: True if Precision@5 ≥ configured minimum.
        mlflow_run_id: MLflow run ID where metrics were logged.
    """
    precision_recall: List[PrecisionRecallAtK]
    coverage: float
    diversity: float
    novelty: float
    intra_list_similarity: float
    evaluation_report_path: str
    meets_minimum_threshold: bool
    mlflow_run_id: str


# ---------------------------------------------------------------
# Model Registry Artifact
# ---------------------------------------------------------------

@dataclass
class ModelRegistryArtifact:
    """
    Output of the Model Registry component.

    Attributes:
        registered: True if model was successfully registered with MLflow.
        model_name: Registered model name in MLflow registry.
        model_version: Version number assigned by MLflow.
        stage: Current stage ('Staging' | 'Production' | 'Archived').
        registry_uri: Full MLflow model URI.
        promoted: True if model was auto-promoted to Production.
    """
    registered: bool
    model_name: str
    model_version: str
    stage: str
    registry_uri: str
    promoted: bool


# ---------------------------------------------------------------
# Training Pipeline Artifact (Composite)
# ---------------------------------------------------------------

@dataclass
class TrainingPipelineArtifact:
    """
    Composite artifact returned by the full training pipeline.
    Contains artifacts from every stage.

    Note: model_training, model_evaluation, and model_registry are Optional
    because Phase 2 only covers the data pipeline. These are populated in
    Phase 3 (Model Training).
    """
    ingestion: DataIngestionArtifact
    validation: DataValidationArtifact
    transformation: DataTransformationArtifact
    feature_engineering: FeatureEngineeringArtifact
    model_training: Optional["ModelTrainingArtifact"]
    model_evaluation: Optional["ModelEvaluationArtifact"]
    model_registry: Optional["ModelRegistryArtifact"]

    pipeline_duration_seconds: float
    success: bool
    error_message: Optional[str] = None


# ---------------------------------------------------------------
# Prediction Artifact
# ---------------------------------------------------------------

@dataclass
class RecommendationItem:
    """A single recommended product with its confidence score."""
    product_id: str
    product_name: str
    rank: int
    confidence_score: float
    confidence_grade: str
    reason_tags: List[str]
    primary_reason: str
    engine_used: str


@dataclass
class PredictionArtifact:
    """
    Output of a single prediction pipeline invocation.

    Attributes:
        query_product_id: Input product ID (or None for session-based).
        session_id: Browser session UUID (or None for direct queries).
        recommendations: Ranked list of recommended products.
        engine_used: Primary engine or 'hybrid'.
        model_version: Model version used for prediction.
        prediction_duration_ms: Time taken to generate recommendations.
        cache_hit: True if result was served from cache.
    """
    query_product_id: Optional[str]
    session_id: Optional[str]
    recommendations: List[RecommendationItem]
    engine_used: str
    model_version: str
    prediction_duration_ms: float
    cache_hit: bool
