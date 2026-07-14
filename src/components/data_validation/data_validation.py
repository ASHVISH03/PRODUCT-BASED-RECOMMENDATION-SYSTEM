"""
Data Validation Component
==========================
Validates the ingested dataset against all defined rules in validation_rules.py.

Checks performed:
    • Schema validation (required columns present)
    • Missing value ratios per column
    • Empty string ratios per column
    • Dtype parseability (prices, ratings as numerics)
    • Numeric range validation (rating 0–5, discount 0–100)
    • Duplicate row detection
    • Duplicate product_id detection

Output:
    • DataValidationArtifact (pass/fail status, issue list)
    • JSON validation report saved to artifacts/validation_report.json

Design:
    - Errors stop the pipeline (validation_status=False).
    - Warnings are logged and included in the report, pipeline continues.
    - Report is always written regardless of pass/fail.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.components.data_validation.validation_rules import (
    ALL_RULES,
    ValidationRule,
)
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, ValidationIssue
from src.entity.config_entity import DataConfig
from src.exception.exception import CustomException, DataValidationError
from src.logger.training_logger import get_training_logger
from src.utils.common import get_iso_timestamp, save_json

logger = get_training_logger()


class DataValidation:
    """
    Validates the raw dataset against declarative rules.

    Args:
        config: DataConfig entity from ConfigurationManager.
        ingestion_artifact: Output of the DataIngestion component.
    """

    def __init__(
        self,
        config: DataConfig,
        ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        self.config = config
        self.ingestion_artifact = ingestion_artifact
        self._issues: List[ValidationIssue] = []
        self._has_errors: bool = False

    # ------------------------------------------------------------------
    # Private: issue recording
    # ------------------------------------------------------------------

    def _add_issue(
        self,
        rule: ValidationRule,
        count: int,
        message: str,
        column: Optional[str] = None,
    ) -> None:
        """Record a validation finding and set error flag if severity='error'."""
        issue = ValidationIssue(
            column=column or rule.column or "dataset",
            issue_type=rule.check_type,
            severity=rule.severity,
            count=count,
            message=message,
        )
        self._issues.append(issue)

        if rule.severity == "error":
            self._has_errors = True
            logger.error(f"  [FAIL] {rule.name}: {message}")
        else:
            logger.warning(f"  [WARN] {rule.name}: {message}")

    # ------------------------------------------------------------------
    # Private: individual checks
    # ------------------------------------------------------------------

    def _check_schema(self, df: pd.DataFrame, rule: ValidationRule) -> None:
        """Verify all expected columns are present."""
        expected = set(self.config.expected_columns)
        found = set(df.columns.str.strip().str.lower())
        missing_cols = expected - found

        if missing_cols:
            self._add_issue(
                rule,
                count=len(missing_cols),
                message=(
                    f"Missing required columns: {sorted(missing_cols)}. "
                    f"Found columns: {sorted(found)}"
                ),
            )
        else:
            logger.info(f"  [PASS] {rule.name}: All {len(expected)} required columns present")

    def _check_missing(self, df: pd.DataFrame, rule: ValidationRule) -> Tuple[int, float]:
        """Check null ratio for a column. Returns (null_count, null_ratio)."""
        if rule.column not in df.columns:
            return 0, 0.0

        null_count = df[rule.column].isnull().sum()
        null_ratio = null_count / max(len(df), 1)

        if rule.threshold is not None and null_ratio > rule.threshold:
            self._add_issue(
                rule,
                count=int(null_count),
                message=(
                    f"{rule.column}: {null_ratio:.1%} missing "
                    f"({null_count:,} of {len(df):,} rows). "
                    f"Threshold: {rule.threshold:.1%}"
                ),
            )
        else:
            logger.info(
                f"  [PASS] {rule.name}: {rule.column} — "
                f"{null_ratio:.1%} missing ({null_count:,} rows)"
            )

        return int(null_count), null_ratio

    def _check_empty_strings(self, df: pd.DataFrame, rule: ValidationRule) -> None:
        """Check ratio of empty / whitespace-only strings."""
        if rule.column not in df.columns:
            return

        col = df[rule.column].astype(str)
        empty_mask = col.str.strip().eq("") | col.eq("nan")
        empty_count = empty_mask.sum()
        empty_ratio = empty_count / max(len(df), 1)

        if rule.threshold is not None and empty_ratio > rule.threshold:
            self._add_issue(
                rule,
                count=int(empty_count),
                message=(
                    f"{rule.column}: {empty_ratio:.1%} empty strings "
                    f"({empty_count:,} rows). Threshold: {rule.threshold:.1%}"
                ),
            )
        else:
            logger.info(
                f"  [PASS] {rule.name}: {rule.column} — "
                f"{empty_ratio:.1%} empty ({empty_count:,} rows)"
            )

    def _check_dtype(self, df: pd.DataFrame, rule: ValidationRule) -> None:
        """Check what fraction of a column is parseable as a numeric type."""
        if rule.column not in df.columns:
            return

        # Strip common currency/percent symbols before testing parseability
        cleaned = (
            df[rule.column]
            .astype(str)
            .str.replace(r"[₹,% ]", "", regex=True)
            .str.strip()
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
        unparseable_count = parsed.isnull().sum() - df[rule.column].isnull().sum()
        unparseable_count = max(0, int(unparseable_count))
        unparseable_ratio = unparseable_count / max(len(df), 1)

        if rule.threshold is not None and unparseable_ratio > rule.threshold:
            self._add_issue(
                rule,
                count=unparseable_count,
                message=(
                    f"{rule.column}: {unparseable_ratio:.1%} values not parseable "
                    f"as {rule.expected_type} ({unparseable_count:,} rows). "
                    f"Threshold: {rule.threshold:.1%}"
                ),
            )
        else:
            logger.info(
                f"  [PASS] {rule.name}: {rule.column} — "
                f"{unparseable_ratio:.1%} unparseable ({unparseable_count:,} rows)"
            )

    def _check_range(self, df: pd.DataFrame, rule: ValidationRule) -> None:
        """Check numeric range after parsing. Applied to pre-cleaned numeric values."""
        if rule.column not in df.columns:
            return

        cleaned = (
            df[rule.column]
            .astype(str)
            .str.replace(r"[₹,% ]", "", regex=True)
            .str.strip()
        )
        numeric_col = pd.to_numeric(cleaned, errors="coerce").dropna()

        if len(numeric_col) == 0:
            logger.warning(f"  [SKIP] {rule.name}: No parseable values in {rule.column}")
            return

        out_of_range = numeric_col[
            (numeric_col < rule.min_value) | (numeric_col > rule.max_value)
        ]
        out_count = len(out_of_range)
        out_ratio = out_count / max(len(numeric_col), 1)

        if out_count > 0:
            self._add_issue(
                rule,
                count=out_count,
                message=(
                    f"{rule.column}: {out_count:,} values out of range "
                    f"[{rule.min_value}, {rule.max_value}]. "
                    f"Min found: {numeric_col.min():.2f}, Max: {numeric_col.max():.2f}"
                ),
            )
        else:
            logger.info(
                f"  [PASS] {rule.name}: {rule.column} — "
                f"all values in [{rule.min_value}, {rule.max_value}]"
            )

    def _check_duplicates(self, df: pd.DataFrame, rule: ValidationRule) -> Tuple[int, float]:
        """Check for duplicate rows or duplicate values in a specific column."""
        if rule.column is not None:
            if rule.column not in df.columns:
                return 0, 0.0
            dup_count = df[rule.column].duplicated().sum()
        else:
            dup_count = df.duplicated().sum()

        dup_ratio = dup_count / max(len(df), 1)
        label = f"{rule.column} duplicates" if rule.column else "fully duplicate rows"

        if rule.threshold is not None and dup_ratio > rule.threshold:
            self._add_issue(
                rule,
                count=int(dup_count),
                message=(
                    f"{label}: {dup_ratio:.1%} duplicates "
                    f"({dup_count:,} rows). Threshold: {rule.threshold:.1%}"
                ),
            )
        else:
            logger.info(
                f"  [PASS] {rule.name}: {label} — "
                f"{dup_ratio:.1%} ({dup_count:,} rows)"
            )

        return int(dup_count), dup_ratio

    # ------------------------------------------------------------------
    # Private: report generation
    # ------------------------------------------------------------------

    def _save_validation_report(
        self,
        df: pd.DataFrame,
        validation_status: bool,
        report_path: Path,
    ) -> None:
        """Serialize validation results to a JSON report file."""
        report: Dict[str, Any] = {
            "validation_status": validation_status,
            "validated_at": get_iso_timestamp(),
            "dataset_path": self.ingestion_artifact.raw_filepath,
            "dataset_version": self.ingestion_artifact.dataset_version,
            "record_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "summary": {
                "total_issues": len(self._issues),
                "errors": sum(1 for i in self._issues if i.severity == "error"),
                "warnings": sum(1 for i in self._issues if i.severity == "warning"),
            },
            "issues": [
                {
                    "column": issue.column,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "count": issue.count,
                    "message": issue.message,
                }
                for issue in self._issues
            ],
            "column_stats": {
                col: {
                    "null_count": int(df[col].isnull().sum()),
                    "null_pct": round(df[col].isnull().mean() * 100, 2),
                    "unique_count": int(df[col].nunique()),
                    "dtype": str(df[col].dtype),
                }
                for col in df.columns
            },
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(report_path, report)
        logger.info(f"Validation report saved → {report_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Run all validation rules against the ingested dataset.

        Returns:
            DataValidationArtifact with status, issue list, and report path.

        Raises:
            DataValidationError: If the dataset fails error-level checks.
            CustomException: On unexpected internal failures.
        """
        logger.info("=" * 60)
        logger.info("STEP 2: DATA VALIDATION")
        logger.info("=" * 60)

        try:
            # Load raw dataset
            df = pd.read_csv(
                self.ingestion_artifact.raw_filepath,
                encoding="utf-8",
                low_memory=False,
            )
            logger.info(f"Loaded {len(df):,} records for validation")

            # Tracking counters
            total_missing = 0
            total_duplicates = 0

            # Run each rule
            for rule in ALL_RULES:
                logger.info(f"Checking rule: {rule.name}")

                if rule.check_type == "schema":
                    self._check_schema(df, rule)

                elif rule.check_type == "missing":
                    null_count, _ = self._check_missing(df, rule)
                    total_missing += null_count

                elif rule.check_type == "empty":
                    self._check_empty_strings(df, rule)

                elif rule.check_type == "dtype":
                    self._check_dtype(df, rule)

                elif rule.check_type == "range":
                    self._check_range(df, rule)

                elif rule.check_type == "duplicate":
                    dup_count, _ = self._check_duplicates(df, rule)
                    total_duplicates += dup_count

            # Determine overall status
            validation_status = not self._has_errors
            validated_record_count = len(df)

            # Save report regardless of pass/fail
            report_path = Path("artifacts") / "validation_report.json"
            self._save_validation_report(df, validation_status, report_path)

            # Summary log
            error_count = sum(1 for i in self._issues if i.severity == "error")
            warning_count = sum(1 for i in self._issues if i.severity == "warning")

            logger.info("-" * 60)
            logger.info(f"VALIDATION SUMMARY")
            logger.info(f"  Status   : {'PASSED ✓' if validation_status else 'FAILED ✗'}")
            logger.info(f"  Records  : {validated_record_count:,}")
            logger.info(f"  Errors   : {error_count}")
            logger.info(f"  Warnings : {warning_count}")
            logger.info(f"  Missing  : {total_missing:,} total null values")
            logger.info(f"  Dupes    : {total_duplicates:,} duplicate entries")
            logger.info("-" * 60)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                validation_report_path=str(report_path.resolve()),
                missing_values_count=total_missing,
                duplicate_count=total_duplicates,
                schema_valid=(
                    not any(
                        i.issue_type == "schema" and i.severity == "error"
                        for i in self._issues
                    )
                ),
                issues=self._issues,
                validated_record_count=validated_record_count,
            )

            if not validation_status:
                raise DataValidationError(
                    f"Dataset failed {error_count} error-level validation check(s). "
                    f"See report: {report_path}"
                )

            logger.info("STEP 2 COMPLETE: Data Validation PASSED")
            return artifact

        except (DataValidationError, CustomException):
            raise
        except Exception as e:
            raise CustomException(
                f"Data validation failed unexpectedly: {e}", sys.exc_info()
            ) from e
