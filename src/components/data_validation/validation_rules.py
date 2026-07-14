"""
Validation Rules
=================
Declarative rule definitions consumed by DataValidation.
Each ValidationRule describes one check: what column to check,
what kind of check, severity, and thresholds.

Separating rules from logic keeps DataValidation clean and makes
adding new rules a one-line change here.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ValidationRule:
    """
    Defines a single data quality check.

    Attributes:
        name: Unique rule identifier.
        description: Human-readable description of what is checked.
        column: Column name to check, or None for row-level checks.
        check_type: Category of check.
            'schema'    — required columns present
            'missing'   — null/NaN ratio below threshold
            'duplicate' — duplicate ratio below threshold
            'range'     — numeric values within [min_value, max_value]
            'empty'     — non-empty string values
            'dtype'     — column parseable as expected_type
        severity: Impact level.
            'error'   — validation fails, pipeline should stop
            'warning' — logged but pipeline continues
        threshold: Ratio threshold (0.0 – 1.0) for missing/duplicate checks.
        min_value: Minimum valid numeric value for range checks.
        max_value: Maximum valid numeric value for range checks.
        expected_type: 'numeric' | 'string' for dtype checks.
    """

    name: str
    description: str
    column: Optional[str]
    check_type: str
    severity: str
    threshold: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    expected_type: Optional[str] = None


# ---------------------------------------------------------------
# Schema Rules — required columns
# ---------------------------------------------------------------

SCHEMA_RULES: List[ValidationRule] = [
    ValidationRule(
        name="required_columns_present",
        description="All expected schema columns must be present in the dataset",
        column=None,
        check_type="schema",
        severity="error",
    ),
]

# ---------------------------------------------------------------
# Missing Value Rules
# ---------------------------------------------------------------

MISSING_VALUE_RULES: List[ValidationRule] = [
    ValidationRule(
        name="product_id_not_null",
        description="product_id must not have missing values",
        column="product_id",
        check_type="missing",
        severity="error",
        threshold=0.01,  # max 1% missing
    ),
    ValidationRule(
        name="product_name_not_null",
        description="product_name must not have excessive missing values",
        column="product_name",
        check_type="missing",
        severity="error",
        threshold=0.05,  # max 5% missing
    ),
    ValidationRule(
        name="category_not_null",
        description="category must not have excessive missing values",
        column="category",
        check_type="missing",
        severity="error",
        threshold=0.10,
    ),
    ValidationRule(
        name="discounted_price_not_null",
        description="discounted_price must not have excessive missing values",
        column="discounted_price",
        check_type="missing",
        severity="error",
        threshold=0.20,
    ),
    ValidationRule(
        name="actual_price_not_null",
        description="actual_price must not have excessive missing values",
        column="actual_price",
        check_type="missing",
        severity="warning",
        threshold=0.30,
    ),
    ValidationRule(
        name="rating_not_null",
        description="rating should not have excessive missing values",
        column="rating",
        check_type="missing",
        severity="warning",
        threshold=0.30,
    ),
    ValidationRule(
        name="about_product_not_null",
        description="about_product description should not be mostly empty",
        column="about_product",
        check_type="missing",
        severity="warning",
        threshold=0.40,
    ),
]

# ---------------------------------------------------------------
# Empty String Rules
# ---------------------------------------------------------------

EMPTY_STRING_RULES: List[ValidationRule] = [
    ValidationRule(
        name="product_name_not_empty",
        description="product_name must not contain empty strings",
        column="product_name",
        check_type="empty",
        severity="warning",
        threshold=0.05,
    ),
    ValidationRule(
        name="category_not_empty",
        description="category must not contain empty strings",
        column="category",
        check_type="empty",
        severity="warning",
        threshold=0.10,
    ),
    ValidationRule(
        name="about_product_not_empty",
        description="about_product must not be predominantly empty strings",
        column="about_product",
        check_type="empty",
        severity="warning",
        threshold=0.40,
    ),
]

# ---------------------------------------------------------------
# Range Rules — applied after parsing numerics
# ---------------------------------------------------------------

RANGE_RULES: List[ValidationRule] = [
    ValidationRule(
        name="rating_range_valid",
        description="Parsed rating values must be between 0.0 and 5.0",
        column="rating",
        check_type="range",
        severity="warning",
        min_value=0.0,
        max_value=5.0,
    ),
    ValidationRule(
        name="discount_pct_range_valid",
        description="Parsed discount_percentage must be between 0.0 and 100.0",
        column="discount_percentage",
        check_type="range",
        severity="warning",
        min_value=0.0,
        max_value=100.0,
    ),
]

# ---------------------------------------------------------------
# Duplicate Rules
# ---------------------------------------------------------------

DUPLICATE_RULES: List[ValidationRule] = [
    ValidationRule(
        name="no_duplicate_product_ids",
        description="product_id values should be unique (duplicates are warnings)",
        column="product_id",
        check_type="duplicate",
        severity="warning",
        threshold=0.20,  # error if >20% duplicates
    ),
    ValidationRule(
        name="no_duplicate_rows",
        description="Fully duplicate rows should be below threshold",
        column=None,
        check_type="duplicate",
        severity="warning",
        threshold=0.10,
    ),
]

# ---------------------------------------------------------------
# Dtype Rules
# ---------------------------------------------------------------

DTYPE_RULES: List[ValidationRule] = [
    ValidationRule(
        name="rating_parseable_numeric",
        description="rating column values must be parseable as floating point",
        column="rating",
        check_type="dtype",
        severity="warning",
        expected_type="numeric",
        threshold=0.30,  # warn if >30% unparseable
    ),
    ValidationRule(
        name="rating_count_parseable_numeric",
        description="rating_count values must be parseable as integer",
        column="rating_count",
        check_type="dtype",
        severity="warning",
        expected_type="numeric",
        threshold=0.30,
    ),
]

# ---------------------------------------------------------------
# Master rule list (order matters — schema first)
# ---------------------------------------------------------------

ALL_RULES: List[ValidationRule] = (
    SCHEMA_RULES
    + MISSING_VALUE_RULES
    + EMPTY_STRING_RULES
    + DTYPE_RULES
    + RANGE_RULES
    + DUPLICATE_RULES
)
