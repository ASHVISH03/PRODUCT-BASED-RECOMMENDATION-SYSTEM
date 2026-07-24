"""
Unit tests for DriftDetector (PSI + categorical total variation distance).
Uses tmp_path for baseline/report files so it never touches artifacts/.
"""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector


def make_df(n, price_mean, price_std, categories, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "discounted_price": rng.normal(price_mean, price_std, n).clip(min=1),
            "actual_price": rng.normal(price_mean * 1.2, price_std, n).clip(min=1),
            "rating": rng.uniform(3.0, 5.0, n),
            "rating_count": rng.integers(1, 500, n),
            "category": rng.choice(categories, n),
        }
    )


@pytest.fixture
def detector(tmp_path):
    return DriftDetector(
        baseline_path=tmp_path / "baseline.json",
        report_path=tmp_path / "report.json",
    )


def test_first_run_establishes_baseline_with_no_drift(detector):
    df = make_df(200, price_mean=500, price_std=50, categories=["Electronics", "Books"])
    report = detector.run(df)

    assert report["status"] == "baseline_established"
    assert report["overall_grade"] == "none"
    assert detector.baseline_path.exists()


def test_identical_distribution_reports_no_drift(detector):
    # Large n keeps PSI sampling noise well below the "moderate" threshold —
    # at small n, two draws from the *same* distribution can still land
    # PSI > 0.1 purely from bin-count noise (~30 samples/bin at n=300).
    df1 = make_df(3000, price_mean=500, price_std=50, categories=["Electronics", "Books"], seed=1)
    detector.run(df1)

    df2 = make_df(3000, price_mean=500, price_std=50, categories=["Electronics", "Books"], seed=2)
    report = detector.run(df2)

    assert report["status"] == "compared"
    assert report["overall_grade"] == "none"
    assert report["features"]["discounted_price"]["grade"] == "none"


def test_large_price_shift_is_detected_as_drift(detector):
    baseline_df = make_df(300, price_mean=500, price_std=50, categories=["Electronics"], seed=1)
    detector.run(baseline_df)

    shifted_df = make_df(300, price_mean=5000, price_std=50, categories=["Electronics"], seed=2)
    report = detector.run(shifted_df)

    assert report["overall_grade"] == "high"
    assert report["features"]["discounted_price"]["grade"] == "high"
    assert report["features"]["discounted_price"]["value"] > 0.25


def test_category_distribution_shift_is_detected(detector):
    baseline_df = make_df(
        300, price_mean=500, price_std=50,
        categories=["Electronics"] * 9 + ["Books"], seed=1,
    )
    detector.run(baseline_df)

    shifted_df = make_df(
        300, price_mean=500, price_std=50,
        categories=["Books"] * 9 + ["Electronics"], seed=2,
    )
    report = detector.run(shifted_df)

    assert report["features"]["category"]["grade"] == "high"


def test_update_baseline_advances_stored_snapshot(detector):
    df1 = make_df(200, price_mean=500, price_std=50, categories=["Electronics"], seed=1)
    detector.run(df1)

    df2 = make_df(200, price_mean=9000, price_std=50, categories=["Electronics"], seed=2)
    detector.update_baseline(df2)

    # Comparing df2 against the just-updated baseline (itself) → no drift.
    report = detector.run(df2)
    assert report["overall_grade"] == "none"
