"""
Data Drift Detector
====================
Compares a fresh dataset snapshot against a stored baseline snapshot using
Population Stability Index (PSI) for numeric features and total variation
distance for the top-level category distribution — both standard,
interpretable drift metrics that don't require external dependencies.

The baseline is the dataset the currently-live Production model was
trained on. It only advances when a new model is actually promoted to
Production (see ModelRegistry's promotion gate) — so "drift" always means
"how far has incoming data moved from what's actually deployed".

PSI interpretation (industry-standard bands):
    < 0.10        → no significant drift
    0.10 – 0.25   → moderate drift, worth reviewing
    >= 0.25       → significant drift, investigate / retrain

Total variation distance (categorical, range 0-1):
    < 0.05        → no significant drift
    0.05 – 0.15   → moderate drift
    >= 0.15       → significant drift
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.common import save_json, get_iso_timestamp

NUMERIC_FEATURES = ["discounted_price", "actual_price", "rating", "rating_count"]
PSI_BINS = 10


def _psi(baseline: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index between two numeric samples."""
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # not enough distinct baseline values to bucket meaningfully

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    baseline_pct = np.clip(baseline_counts / max(baseline_counts.sum(), 1), 1e-4, None)
    current_pct = np.clip(current_counts / max(current_counts.sum(), 1), 1e-4, None)

    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def _total_variation_distance(baseline_counts: Dict[str, int], current_counts: Dict[str, int]) -> float:
    """Total variation distance between two categorical distributions, in [0, 1]."""
    keys = set(baseline_counts) | set(current_counts)
    b_total = sum(baseline_counts.values()) or 1
    c_total = sum(current_counts.values()) or 1
    tv = sum(
        abs(baseline_counts.get(k, 0) / b_total - current_counts.get(k, 0) / c_total)
        for k in keys
    )
    return tv / 2.0


def _grade_psi(psi: float) -> str:
    if psi < 0.10:
        return "none"
    if psi < 0.25:
        return "moderate"
    return "high"


def _grade_tvd(tvd: float) -> str:
    if tvd < 0.05:
        return "none"
    if tvd < 0.15:
        return "moderate"
    return "high"


_GRADE_RANK = {"none": 0, "moderate": 1, "high": 2}


@dataclass
class DriftDetector:
    baseline_path: Path = field(default_factory=lambda: Path("artifacts/drift_baseline.json"))
    report_path: Path = field(default_factory=lambda: Path("artifacts/drift_report.json"))

    def _snapshot(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_samples = {}
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                numeric_samples[col] = series.tolist()

        category_counts: Dict[str, int] = {}
        if "category" in df.columns:
            top_level = df["category"].astype(str).str.split("|").str[0]
            category_counts = top_level.value_counts().to_dict()

        return {
            "record_count": len(df),
            "numeric_samples": numeric_samples,
            "category_counts": category_counts,
            "snapshot_at": get_iso_timestamp(),
        }

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare df against the stored baseline and save a drift report.

        If no baseline exists yet, this snapshot *becomes* the baseline and
        an empty (no-drift) report is returned — there's nothing to compare
        against on the very first run.
        """
        current = self._snapshot(df)

        if not self.baseline_path.exists():
            save_json(self.baseline_path, current)
            report = {
                "status": "baseline_established",
                "checked_at": current["snapshot_at"],
                "record_count": current["record_count"],
                "features": {},
                "overall_grade": "none",
            }
            save_json(self.report_path, report)
            return report

        baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))

        features: Dict[str, Any] = {}
        overall_grade = "none"

        for col, current_samples in current["numeric_samples"].items():
            baseline_samples = baseline.get("numeric_samples", {}).get(col, [])
            if not baseline_samples or not current_samples:
                continue
            psi = _psi(np.array(baseline_samples, dtype=float), np.array(current_samples, dtype=float))
            grade = _grade_psi(psi)
            features[col] = {"metric": "psi", "value": round(psi, 4), "grade": grade}
            if _GRADE_RANK[grade] > _GRADE_RANK[overall_grade]:
                overall_grade = grade

        tvd = _total_variation_distance(
            baseline.get("category_counts", {}), current["category_counts"]
        )
        cat_grade = _grade_tvd(tvd)
        features["category"] = {
            "metric": "total_variation_distance", "value": round(tvd, 4), "grade": cat_grade,
        }
        if _GRADE_RANK[cat_grade] > _GRADE_RANK[overall_grade]:
            overall_grade = cat_grade

        report = {
            "status": "compared",
            "checked_at": current["snapshot_at"],
            "baseline_snapshot_at": baseline.get("snapshot_at"),
            "record_count": current["record_count"],
            "baseline_record_count": baseline.get("record_count"),
            "features": features,
            "overall_grade": overall_grade,
        }
        save_json(self.report_path, report)
        return report

    def update_baseline(self, df: pd.DataFrame) -> None:
        """Advance the baseline to this dataset — call only after a model
        trained on it is actually promoted to Production."""
        save_json(self.baseline_path, self._snapshot(df))
