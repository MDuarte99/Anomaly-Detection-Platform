"""Training pipeline for Isolation Forest fraud detection model."""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the credit card dataset."""
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ["Time", "Class"]]
    logger.info(f"Dataset shape: {df.shape}, Features: {len(feature_cols)}, Fraud rate: {df['Class'].mean():.4f}")
    return df


def prepare_features(df: pd.DataFrame):
    """Scale features and split into train/test sets."""
    feature_cols = [c for c in df.columns if c.startswith("V")]
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Train: {X_train_scaled.shape[0]} samples, Test: {X_test_scaled.shape[0]} samples")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def train_model(X_train: np.ndarray, contamination: float = 0.0017) -> IsolationForest:
    """Train Isolation Forest with MLflow tracking."""
    tracking_uri = str(PROJECT_ROOT / "mlruns").replace("\\", "/")
    if "://" not in tracking_uri:
        tracking_uri = f"file:///{tracking_uri}" if not tracking_uri.startswith("file") else tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-isolation-forest")

    with mlflow.start_run() as run:
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("max_samples", "auto")
        mlflow.sklearn.log_model(model, "model")
        logger.info(f"MLflow run ID: {run.info.run_id}")

        return model


def evaluate_model(model, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """Evaluate and return metrics dict."""
    scores = model.decision_function(X_test)
    predictions = model.predict(X_test)
    preds_binary = (predictions == -1).astype(int)

    auc = roc_auc_score(y_test, -scores)
    cm = confusion_matrix(y_test, preds_binary).tolist()
    report = classification_report(y_test, preds_binary, output_dict=True)

    metrics = {
        "roc_auc": auc,
        "confusion_matrix": cm,
        "precision_fraud": report.get("1", {}).get("precision", 0),
        "recall_fraud": report.get("1", {}).get("recall", 0),
        "f1_fraud": report.get("1", {}).get("f1-score", 0),
        "accuracy": report.get("accuracy", 0),
    }

    logger.info(f"AUC-ROC: {auc:.4f}")
    logger.info(f"Precision (fraud): {metrics['precision_fraud']:.4f}")
    logger.info(f"Recall (fraud): {metrics['recall_fraud']:.4f}")
    logger.info(f"F1 (fraud): {metrics['f1_fraud']:.4f}")

    with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id):
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision_fraud", metrics["precision_fraud"])
        mlflow.log_metric("recall_fraud", metrics["recall_fraud"])
        mlflow.log_metric("f1_fraud", metrics["f1_fraud"])

    return metrics


def save_artifacts(model, scaler, metrics, feature_cols, contamination: float = 0.0017):
    """Persist model, scaler, and metrics to disk."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "features": feature_cols,
        "metrics": metrics,
        "contamination": contamination,
        "model_type": "IsolationForest",
        "scikit_learn_version": __import__("sklearn").__version__,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Artifacts saved to {ARTIFACTS_DIR}")


def main():
    """Execute full training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting fraud detection training pipeline")
    logger.info("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_features(df)
    contamination = 0.0017
    model = train_model(X_train, contamination)
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, scaler, metrics, feature_cols, contamination=contamination)

    logger.info("Training pipeline complete")
    return metrics


if __name__ == "__main__":
    main()
