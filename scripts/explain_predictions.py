"""
SHAP-based explainability for Isolation Forest predictions.
Computes feature contributions for individual predictions.
"""

import logging
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def explain_transactions(n_samples: int = 100):
    """
    Use SHAP to explain Isolation Forest predictions.
    For IsolationForest, we use KernelExplainer on the decision_function.
    """
    logger.info("Loading model artifacts...")
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    data_path = PROJECT_ROOT / "data" / "creditcard.csv"
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c.startswith("V")]
    X = df[feature_cols].sample(n=n_samples, random_state=42)
    X_scaled = scaler.transform(X)

    logger.info(f"Computing SHAP values for {n_samples} samples (this may take a minute)...")

    def predict_func(x):
        """Wrapper for SHAP: returns anomaly scores."""
        return model.decision_function(x)

    # Use a background sample for baseline
    background = X_scaled[:20]
    explainer = shap.KernelExplainer(predict_func, background)
    shap_values = explainer.shap_values(X_scaled, nsamples=100)

    # Summary statistics
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(feature_cols, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.info("Top 10 most influential features:")
    for feat, imp in feature_importance[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    # Save results
    importance_data = {
        "feature_importance": [{"feature": f, "mean_abs_shap": float(i)} for f, i in feature_importance],
        "n_samples": n_samples,
        "model": "IsolationForest",
    }

    output_path = ARTIFACTS_DIR / "shap_importance.json"
    with open(output_path, "w") as f:
        json.dump(importance_data, f, indent=2)

    logger.info(f"SHAP importance saved to {output_path}")

    # Explain one fraud transaction if any sampled
    fraud_mask = df.loc[X.index, "Class"] == 1
    if fraud_mask.any():
        fraud_idx = fraud_mask[fraud_mask].index[0]
        fraud_pos = X.index.get_loc(fraud_idx)
        logger.info(f"\nExplaining fraud transaction {fraud_idx}:")
        for feat, val in zip(feature_cols, shap_values[fraud_pos]):
            if abs(val) > 0.01:
                logger.info(f"  {feat}: SHAP = {val:.4f}")

    return feature_importance


if __name__ == "__main__":
    explain_transactions(n_samples=100)
