"""
Model drift detection - compares current prediction distribution against baseline.
Simulates what you'd run periodically with a monitoring service.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DRIFT_REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"


def load_baseline_data() -> pd.DataFrame:
    """Load original dataset as baseline reference."""
    data_path = PROJECT_ROOT / "data" / "creditcard.csv"
    return pd.read_csv(data_path)


def compute_feature_stats(df: pd.DataFrame) -> dict:
    """Compute per-feature mean and std for PSI calculation."""
    feature_cols = [c for c in df.columns if c.startswith("V")]
    stats_dict = {}
    for col in feature_cols:
        stats_dict[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "median": float(df[col].median()),
        }
    return stats_dict


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1: no significant change
    0.1 <= PSI < 0.2: moderate drift
    PSI >= 0.2: significant drift
    """
    def create_bins(base):
        return np.histogram_bin_edges(base, bins=buckets)

    edges = create_bins(expected)
    expected_bins = np.histogram(expected, bins=edges)[0]
    actual_bins = np.histogram(actual, bins=edges)[0]

    # Normalize to proportions
    expected_pct = (expected_bins + 1e-5) / (expected_bins.sum() + 1e-5)
    actual_pct = (actual_bins + 1e-5) / (actual_bins.sum() + 1e-5)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def detect_drift() -> dict:
    """Run drift detection and return results."""
    logger.info("Running drift detection...")

    baseline = load_baseline_data()
    baseline_stats = compute_feature_stats(baseline)

    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    feature_cols = [c for c in baseline.columns if c.startswith("V")]
    X_baseline_scaled = scaler.transform(baseline[feature_cols])
    baseline_scores = model.decision_function(X_baseline_scaled)

    # Simulate a "current" batch with a small perturbation to detect drift
    perturbed = baseline.copy()
    for col in feature_cols:
        perturbed[col] = perturbed[col] * np.random.uniform(0.95, 1.05, size=len(perturbed))

    X_current_scaled = scaler.transform(perturbed[feature_cols])
    current_scores = model.decision_function(X_current_scaled)

    psi_score = compute_psi(baseline_scores, current_scores)

    # Per-feature PSI
    feature_psi = {}
    for col in feature_cols:
        feature_psi[col] = compute_psi(
            baseline[col].values, perturbed[col].values, buckets=10
        )

    drift_status = "stable"
    if psi_score >= 0.2:
        drift_status = "critical_drift"
    elif psi_score >= 0.1:
        drift_status = "moderate_drift"

    report = {
        "drift_status": drift_status,
        "overall_psi": psi_score,
        "baseline_n": len(baseline),
        "features_monitored": len(feature_cols),
        "max_feature_psi": max(feature_psi.values()),
        "mean_feature_psi": float(np.mean(list(feature_psi.values()))),
    }

    logger.info(f"Drift Status: {drift_status}")
    logger.info(f"Overall PSI: {psi_score:.4f}")

    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DRIFT_REPORTS_DIR / "latest_drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    return report


if __name__ == "__main__":
    detect_drift()
