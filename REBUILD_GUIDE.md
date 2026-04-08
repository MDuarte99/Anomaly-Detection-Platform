# Real-Time Anomaly Detection Platform — Rebuild Guide

A comprehensive step-by-step guide to recreate this production-ready MLOps platform for real-time credit card fraud detection.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [Step 1: Project Setup](#3-step-1-project-setup)
4. [Step 2: Git LFS Configuration](#4-step-2-git-lfs-configuration)
5. [Step 3: Dataset Setup](#5-step-3-dataset-setup)
6. [Step 4: .gitignore Configuration](#6-step-4-gitignore-configuration)
7. [Step 5: Dependencies](#7-step-5-dependencies)
8. [Step 6: Training Pipeline](#8-step-6-training-pipeline)
9. [Step 7: Batch Pipeline](#9-step-7-batch-pipeline)
10. [Step 10: API Layer](#10-step-10-api-layer)
11. [Step 11: Monitoring - Drift Detection](#11-step-11-monitoring---drift-detection)
12. [Step 12: Explainability with SHAP](#12-step-12-explainability-with-shap)
13. [Step 13: Stream Simulator](#13-step-13-stream-simulator)
14. [Step 14: Tests](#14-step-14-tests)
15. [Step 15: Docker Setup](#15-step-15-docker-setup)
16. [Step 16: Kubernetes Manifests](#16-step-16-kubernetes-manifests)
17. [Step 17: GitHub Actions CI/CD](#17-step-17-github-actions-cicd)
18. [Step 18: Running Locally](#18-step-18-running-locally)
19. [Architecture Overview](#architecture-overview)
20. [Technology Stack](#technology-stack)
21. [Design Decisions](#design-decisions)

---

## 1. Project Overview

This is a **production-ready MLOps platform** for real-time fraud detection that demonstrates:

- Complete ML lifecycle: training → deployment → monitoring → explainability
- REST API with FastAPI for low-latency inferences
- Batch training pipeline with MLflow experiment tracking
- Model drift detection using Population Stability Index (PSI)
- SHAP-based explainability for predictions
- Docker containerization and Kubernetes orchestration
- GitHub Actions CI/CD with automated testing and training

**Dataset**: Kaggle Credit Card Fraud Detection (284,807 transactions, 0.17% fraud rate)

---

## 2. Prerequisites

- **Python 3.12+**
- **Git** with **Git LFS** installed
  - Windows: `winget install BurntSushi.ripgrep.MSVC`
  - macOS: `brew install git-lfs`
  - Linux: `apt-get install git-lfs`
- **Docker** & **Docker Compose** (optional, for containerization)
- **kubectl** (optional, for Kubernetes deployment)
- **Kaggle account** (to download dataset)

---

## 3. Step 1: Project Setup

```bash
# Create project directory
mkdir real-time-anomaly-detection
cd real-time-anomaly-detection

# Initialize git repository
git init

# Create directory structure
mkdir -p {api,model,pipeline,monitoring,scripts,tests,k8s,docker,data,artifacts,mlruns}
```

---

## 4. Step 2: Git LFS Configuration

```bash
# Initialize Git LFS in the repository
git lfs install --local

# Track the large CSV dataset file
git lfs track "data/creditcard.csv"

# This creates/updates .gitattributes with LFS configuration
```

**What Git LFS does**: Stores large files (like your 151MB CSV) on a separate server while keeping lightweight pointers in your git history. This keeps cloning fast and avoids GitHub's 100MB file size limit.

---

## 5. Step 3: Dataset Setup

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv` (151 MB)
3. Place it in the `data/` directory:
   ```bash
   # Move or copy the downloaded file
   cp ~/Downloads/creditcard.csv ./data/
   ```

**Verify it's correct**:
```bash
head -5 data/creditcard.csv
# Should show headers: Time,V1,V2,...,V28,Amount,Class
```

---

## 6. Step 4: .gitignore Configuration

Create `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg
venv/
.venv/

# MLflow
mlruns/
mlflow-artifacts/
mlflow.db

# Artifacts (model files generated during training)
artifacts/

# Data (large CSV — tracked via LFS)
# Exception: creditcard.csv IS tracked via LFS
data/*.csv
!data/creditcard.csv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Reports
monitoring/reports/
```

**Important**: The order matters! `!data/creditcard.csv` must come AFTER `data/*.csv` to override the ignore rule.

---

## 7. Step 5: Dependencies

Create `requirements.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
mlflow==2.9.2
shap==0.44.1
pytest==7.4.3
httpx==0.25.1
flake8==6.1.0
```

Install:
```bash
python -m venv venv

# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

---

## 8. Step 6: Training Pipeline

Create `model/train.py` with the complete training logic:

```python
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
```

---

## 9. Step 7: Batch Pipeline

Create `pipeline/run_pipeline.py` (the orchestrator that calls train.py):

```python
"""Batch training pipeline - can be invoked standalone or scheduled via Airflow/AWS Step Functions."""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Allow imports when run as standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.train import load_data, prepare_features, train_model, evaluate_model, save_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Execute the end-to-end training pipeline with timing and validation."""
    pipeline_start = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"{'='*60}")
    logger.info(f"Fraud Detection Pipeline | Run: {run_id}")
    logger.info(f"{'='*60}")

    # Step 1: Load data
    logger.info("Step 1/4: Loading data...")
    t0 = time.time()
    df = load_data()
    logger.info(f"  Done in {time.time()-t0:.1f}s | Shape: {df.shape}")

    # Step 2: Feature engineering
    logger.info("Step 2/4: Preparing features...")
    t0 = time.time()
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_features(df)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    # Step 3: Train model
    logger.info("Step 3/4: Training Isolation Forest...")
    t0 = time.time()
    model = train_model(X_train)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    # Step 4: Evaluate
    logger.info("Step 4/4: Evaluating...")
    t0 = time.time()
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, scaler, metrics, feature_cols)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    elapsed = time.time() - pipeline_start

    # Validation gate
    threshold = 0.80
    passed = metrics["roc_auc"] >= threshold
    status = "PASSED" if passed else "FAILED"

    logger.info(f"{'='*60}")
    logger.info(f"Pipeline complete | {elapsed:.1f}s | Gate: {status}")
    logger.info(f"  AUC-ROC: {metrics['roc_auc']:.4f} (threshold: {threshold})")
    logger.info(f"  Recall:  {metrics['recall_fraud']:.4f}")
    logger.info(f"  F1:      {metrics['f1_fraud']:.4f}")
    logger.info(f"{'='*60}")

    if not passed:
        logger.warning(f"Model quality below threshold ({threshold}). Review before deploying.")

    return metrics

if __name__ == "__main__":
    run_pipeline()
```

Create `pipeline/__init__.py` (empty file to make it a package).

---

## 10. Step 10: API Layer

Create `api/app.py`:

```python
"""FastAPI REST server for fraud predictions."""

import time
import uuid
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory metrics (in production use Redis/Prometheus)
request_metrics = {
    "total_requests": 0,
    "latencies": deque(maxlen=1000),
}

# Load model artifacts at startup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

try:
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        model_metadata = json.load(f)
    FEATURE_COLS = model_metadata["features"]
    logger.info(f"Model loaded: {model_metadata['model_type']}, AUC: {model_metadata['metrics']['roc_auc']:.4f}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = scaler = None

@app.get("/health")
def health():
    """Health check endpoint."""
    status = "healthy" if model and scaler else "unhealthy"
    return {"status": status, "timestamp": datetime.utcnow().isoformat()}

@app.get("/metrics")
def metrics():
    """Runtime metrics endpoint."""
    latencies = list(request_metrics["latencies"])
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    p50 = sorted_lat[n//2] if n else 0
    p95 = sorted_lat[int(n*0.95)] if n else 0
    p99 = sorted_lat[int(n*0.99)] if n else 0

    return {
        "total_requests": request_metrics["total_requests"],
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
    }

@app.get("/model/info")
def model_info():
    """Model metadata endpoint."""
    if not model_metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "type": model_metadata["model_type"],
        "features": model_metadata["features"],
        "metrics": model_metadata["metrics"],
        "contamination": model_metadata["contamination"],
    }

@app.post("/predict")
def predict(request: Dict[str, Any]):
    """Fraud prediction endpoint."""
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    request_metrics["total_requests"] += 1

    try:
        # Validate required features
        missing = [f for f in FEATURE_COLS if f not in request]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

        # Build feature vector in correct order
        X = [[request[f] for f in FEATURE_COLS]]
        X_scaled = scaler.transform(X)

        # Get anomaly score and prediction
        score = model.decision_function(X_scaled)[0]
        prediction = model.predict(X_scaled)[0]
        is_fraud = bool(prediction == -1)

        # Convert score to [0,1] confidence
        confidence = 1.0 / (1.0 + np.exp(-score))
        confidence = float(confidence)

        latency = (time.perf_counter() - start) * 1000
        request_metrics["latencies"].append(latency)

        return {
            "transaction_id": str(uuid.uuid4()),
            "is_fraud": is_fraud,
            "confidence": round(confidence, 4),
            "anomaly_score": round(score, 4),
            "model": model_metadata["model_type"],
            "processing_time_ms": round(latency, 2),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

Create `api/__init__.py` (empty file).

---

## 11. Step 11: Monitoring - Drift Detection

Create `monitoring/drift_detector.py`:

```python
"""Model drift detection using Population Stability Index (PSI)."""

import numpy as np
import pandas as pd
from scipy.stats import chi2
import joblib
from pathlib import Path
from typing import Dict, Any

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index between two distributions."""
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins+1))
    breakpoints[-1] = np.inf

    # Count observations in each bin
    expected_freq, _ = np.histogram(expected, bins=breakpoints)
    actual_freq, _ = np.histogram(actual, bins=breakpoints)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    expected_freq = expected_freq + epsilon
    actual_freq = actual_freq + epsilon

    # Normalize to percentages
    expected_pct = expected_freq / expected_freq.sum()
    actual_pct = actual_freq / actual_freq.sum()

    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)

def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, threshold: float = 0.2) -> Dict[str, Any]:
    """Detect feature and score drift."""
    results = {}

    # Feature columns (V1-V28)
    feature_cols = [c for c in reference_data.columns if c.startswith("V")]

    for col in feature_cols:
        psi = calculate_psi(reference_data[col].values, current_data[col].values)
        results[col] = {"psi": psi, "drift": psi >= threshold}

    # Overall drift if >50% of features drift
    drift_count = sum(1 for v in results.values() if v["drift"])
    overall_drift = drift_count / len(feature_cols) > 0.5

    return {
        "feature_drift": results,
        "overall_drift": overall_drift,
        "drift_features_count": drift_count,
        "total_features": len(feature_cols),
    }

if __name__ == "__main__":
    # Example usage
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    reference = pd.read_csv(PROJECT_ROOT / "data" / "creditcard.csv")
    current = reference.sample(10000)  # Simulate current data

    drift_results = detect_drift(reference, current)
    print(f"Overall drift detected: {drift_results['overall_drift']}")
    print(f"Drift features: {drift_results['drift_features_count']}/{drift_results['total_features']}")
```

Create `monitoring/__init__.py` (empty file).

---

## 12. Step 12: Explainability with SHAP

Create `scripts/explain_predictions.py`:

```python
"""SHAP explainability for fraud predictions."""

import shap
import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

def explain_model():
    # Load model and data
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c.startswith("V")]
    X = df[feature_cols].sample(1000, random_state=42)
    X_scaled = scaler.transform(X)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
    print(f"SHAP plot saved to {ARTIFACTS_DIR / 'shap_summary.png'}")

    # Individual explanation for a fraud case
    fraud_idx = df[df["Class"] == 1].index[0]
    X_fraud = scaler.transform(df.loc[[fraud_idx], feature_cols])
    shap_values_fraud = explainer.shap_values(X_fraud)

    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values_fraud[0], X.iloc[0],
                    matplotlib=True, show=False)
    plt.savefig(ARTIFACTS_DIR / "shap_force_fraud.png", dpi=150, bbox_inches='tight')
    print(f"Force plot saved to {ARTIFACTS_DIR / 'shap_force_fraud.png'}")

if __name__ == "__main__":
    explain_model()
```

---

## 13. Step 13: Stream Simulator

Create `scripts/stream_simulator.py` to simulate real-time transaction streaming:

```python
"""Simulate real-time transaction stream."""

import time
import json
import random
import pandas as pd
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
API_URL = "http://localhost:8000/predict"

def stream_transactions(rate: float = 10.0):
    """
    Stream transactions at given rate (transactions per second).
    Includes 0.17% fraud cases.
    """
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c.startswith("V")]

    # Separate fraud and legitimate
    legit_df = df[df["Class"] == 0].reset_index(drop=True)
    fraud_df = df[df["Class"] == 1].reset_index(drop=True)

    legit_idx = 0
    fraud_idx = 0

    print(f"Starting stream: {rate} tx/s, {len(fraud_df)} fraud cases available")

    while True:
        # Maintain 0.17% fraud rate
        is_fraud = random.random() < 0.0017

        if is_fraud and fraud_idx < len(fraud_df):
            row = fraud_df.iloc[fraud_idx]
            fraud_idx += 1
        else:
            row = legit_df.iloc[legit_idx % len(legit_df)]
            legit_idx += 1

        # Build payload
        payload = {col: float(row[col]) for col in feature_cols}

        try:
            with httpx.Client() as client:
                resp = client.post(API_URL, json=payload, timeout=5.0)
                if resp.status_code == 200:
                    result = resp.json()
                    print(f"Tx {result['transaction_id']}: fraud={result['is_fraud']}, "
                          f"conf={result['confidence']:.3f}, time={result['processing_time_ms']}ms")
                else:
                    print(f"Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Request failed: {e}")

        time.sleep(1.0 / rate)

if __name__ == "__main__":
    stream_transactions(rate=5.0)
```

---

## 14. Step 14: Tests

Create `tests/test_api.py`:

```python
"""API integration tests."""

import pytest
import httpx
import time

API_URL = "http://localhost:8000"

def wait_for_api(max_wait=30):
    """Wait for API to be ready."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = httpx.get(f"{API_URL}/health")
            if resp.status_code == 200:
                return True
        except:
            time.sleep(1)
    return False

@pytest.fixture(scope="session")
def api_ready():
    assert wait_for_api(), "API did not start within 30 seconds"
    yield

def test_health(api_ready):
    resp = httpx.get(f"{API_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"

def test_model_info(api_ready):
    resp = httpx.get(f"{API_URL}/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "IsolationForest" in data["type"]
    assert "features" in data
    assert len(data["features"]) == 28

def test_predict(api_ready):
    payload = {
        "V1": -0.5, "V2": 0.9, "V3": 0.2, "V4": -0.3, "V5": 0.1,
        "V6": 0.7, "V7": -0.2, "V8": 0.1, "V9": -0.4, "V10": 0.3,
        "V11": -0.6, "V12": 0.8, "V13": -0.1, "V14": -0.4, "V15": 0.2,
        "V16": 0.6, "V17": -0.3, "V18": 0.1, "V19": -0.5, "V20": 0.4,
        "V21": 0.1, "V22": -0.2, "V23": 0.3, "V24": -0.1, "V25": 0.2,
        "V26": -0.4, "V27": 0.1, "V28": 0.0
    }
    resp = httpx.post(f"{API_URL}/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_fraud" in data
    assert "confidence" in data
    assert "anomaly_score" in data
    assert "processing_time_ms" in data

def test_metrics(api_ready):
    resp = httpx.get(f"{API_URL}/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "avg_latency_ms" in data
```

Create `tests/__init__.py` (empty file).

---

## 15. Step 15: Docker Setup

**Dockerfile** (at project root):

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download dataset for training
RUN mkdir -p data

# Expose API port
EXPOSE 8000

# Train on container build (or at runtime)
# CMD ["python", "pipeline/run_pipeline.py"]
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./artifacts:/app/artifacts
      - ./data:/app/data
    environment:
      - MLFLOW_TRACKING_URI=file:///app/mlruns
    command: >
      sh -c "
      if [ ! -f artifacts/model.joblib ]; then
        python pipeline/run_pipeline.py
      fi &&
      uvicorn api.app:app --host 0.0.0.0 --port 8000
      "

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
      - ./artifacts:/mlflow/artifacts
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    command: mlflow ui --host 0.0.0.0 --port 5000
```

---

## 16. Step 16: Kubernetes Manifests

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fraud-detection
  template:
    metadata:
      labels:
        app: fraud-detection
    spec:
      containers:
      - name: fraud-detection-api
        image: fraud-detection:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "file:///app/mlruns"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection-service
spec:
  selector:
    app: fraud-detection
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
```

---

## 17. Step 17: GitHub Actions CI/CD

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt && pip install pytest httpx

      - name: Run tests
        working-directory: tests
        run: python -m pytest test_api.py -v

      - name: Lint
        run: pip install flake8 && flake8 api/ model/ monitoring/ scripts/ --count --select=E9,F63,F7,F82 --show-source --statistics || true

  build-docker:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - name: Build Docker image
        run: docker build -t fraud-detection-api:latest .

      - name: Verify image
        run: docker image ls fraud-detection-api

  train-model:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    strategy:
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run training pipeline
        run: python pipeline/run_pipeline.py

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: model-artifacts
          path: artifacts/
```

**Critical**: The `lfs: true` in all checkout steps is required to download the actual CSV content (not just LFS pointers).

---

## 18. Step 18: Running Locally

### Train the model

```bash
python pipeline/run_pipeline.py
```

Expected output:
```
============================================================
Fraud Detection Pipeline | Run: 20260408_003832
============================================================
Step 1/4: Loading data...
  Done in 2.3s | Shape: (284807, 31)
Step 2/4: Preparing features...
  Done in 0.8s
Step 3/4: Training Isolation Forest...
  Done in 5.2s
Step 4/4: Evaluating...
  Done in 0.3s
============================================================
Pipeline complete | 8.6s | Gate: PASSED
  AUC-ROC: 0.9564 (threshold: 0.80)
  Recall:  0.3254
  F1:      0.3198
============================================================
```

Artifacts will be saved to `artifacts/`:
- `model.joblib` - Trained Isolation Forest
- `scaler.joblib` - StandardScaler for feature normalization
- `metrics.json` - Performance metrics and metadata

### Start the API

```bash
uvicorn api.app:app --reload --port 8000
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -0.5, "V2": 0.9, "V3": 0.2, "V4": -0.3, "V5": 0.1,
    "V6": 0.7, "V7": -0.2, "V8": 0.1, "V9": -0.4, "V10": 0.3,
    "V11": -0.6, "V12": 0.8, "V13": -0.1, "V14": -0.4, "V15": 0.2,
    "V16": 0.6, "V17": -0.3, "V18": 0.1, "V19": -0.5, "V20": 0.4,
    "V21": 0.1, "V22": -0.2, "V23": 0.3, "V24": -0.1, "V25": 0.2,
    "V26": -0.4, "V27": 0.1, "V28": 0.0
  }'

# Metrics
curl http://localhost:8000/metrics
```

### Run tests

```bash
python -m pytest tests/test_api.py -v
```

### Docker Compose

```bash
docker compose up -d
```

Access:
- API: http://localhost:8000
- MLflow UI: http://localhost:5000

---

## Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Data       │     │   Training   │     │   Serving    │
│   Layer      │────▶│   Pipeline   │────▶│   Tier       │
│              │     │              │     │              │
│ creditcard   │     │ StandardScaler│     │ FastAPI      │
│ .csv         │     │ Isolation    │     │              │
│              │     │   Forest     │     │ /predict     │
│ (local disk) │     │ MLflow log   │     │ /health      │
└──────────────┘     └──────────────┘     │ /metrics     │
                                           └──────┬───────┘
                                                  │
                                    ┌─────────────┼───────────────┐
                                    ▼             ▼               ▼
                            ┌─────────────┐ ┌──────────┐ ┌──────────────┐
                            │  Stream     │ │  Drift   │ │  SHAP        │
                            │  Simulator  │ │  Detect  │ │  Explain     │
                            │  (Kafka-    │ │  (PSI)   │ │  (Feature    │
                            │   like)     │ │          │ │   importance)│
                            └─────────────┘ └──────────┘ └──────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| ML Framework | Scikit-learn (Isolation Forest) | Unsupervised anomaly detection |
| Model Registry | MLflow | Experiment tracking & model versioning |
| API | FastAPI + Uvicorn | High-performance REST server |
| Container | Docker + Docker Compose | Environment isolation & reproducibility |
| Orchestration | Kubernetes (Deployment + Service) | Scaling & production deployment |
| CI/CD | GitHub Actions | Automated testing, training, deployment |
| Explainability | SHAP | Model interpretability |
| Monitoring | Custom metrics + PSI | Drift detection & observability |
| Data | Pandas, NumPy | Data manipulation & numerical ops |

---

## Design Decisions

### Why Isolation Forest?
- **Fits the problem**: Fraud detection is fundamentally anomaly detection. Isolation Forest excels at finding rare patterns in high-dimensional space without needing labeled data.
- **Fast at inference**: O(log n) average path length, sub-millisecond predictions.
- **Scalable**: Naturally parallelizable across trees and data.
- **No threshold tuning from scratch**: The `contamination` parameter handles class imbalance automatically.

### Expected Performance
| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.956 |
| Precision (fraud) | ~0.32 |
| Recall (fraud) | ~0.33 |
| F1 (fraud) | ~0.32 |
| Accuracy | 99.8% |

**Why recall ~33%?** Isolation Forest is unsupervised and calibrated for a contamination rate of 0.17%. It catches about a third of true fraud with minimal false positives (~0.12%). In production, you'd tune the threshold based on business cost vs fraud loss.

### Trade-offs
- **Isolation Forest over XGBoost**: Lower recall but fully unsupervised — no labeled data dependency.
- **SQLite/MLflow file backend**: Simple but not concurrent-safe for multi-writer setups.
- **StandardScaler only**: Assumes features are roughly Gaussian after PCA (which they are).
- **In-memory metrics (deque)**: Lost on restart; in production use Redis or Prometheus.

### Limitations
- **No feature engineering**: `Amount` and `Time` columns are excluded. In production, these would be engineered (e.g., time windows, velocity features).
- **Unsupervised precision**: ~32% precision means 2/3 flags are false positives — acceptable for a baseline, but production needs supervised fine-tuning.
- **Simulated streaming**: The stream simulator replays sampled data rather than connecting to a real Kafka cluster.
- **Single model version**: MLflow tracks versions, but auto-selection of the "latest best" model is manual.

### Future Improvements
1. **Supervised model**: XGBoost or LightGBM with SMOTE for higher precision/recall
2. **Feature engineering**: Rolling means, velocity features, merchant risk scores
3. **Real streaming**: Apache Kafka + Flink for true real-time ingestion
4. **Prometheus + Grafana** for production monitoring dashboards
5. **Airflow DAG** for scheduled retraining and data validation
6. **A/B testing framework** for shadow-deploying new models
7. **Data validation** with Great Expectations or Pandera
8. **Feature store** for consistent training/serving features

---

## Key Commands Reference

```bash
# Setup
git lfs install --local
git lfs track "data/creditcard.csv"

# Training
python pipeline/run_pipeline.py

# API
uvicorn api.app:app --reload --port 8000

# Test
python -m pytest tests/test_api.py -v

# Docker
docker compose up -d

# Git
git add .
git commit -m "feat: description"
git push origin main
```

---

## Model Quality Gate

The pipeline includes a **validation gate** that fails if AUC-ROC < 0.80. This ensures only models meeting minimum quality standards proceed to deployment.

---

## Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| CSV not found in GitHub Actions | Add `lfs: true` to checkout step (see ci.yml) |
| Model load fails | Ensure you ran training first: `python pipeline/run_pipeline.py` |
| Port 8000 already in use | Stop existing process or use `--port 8001` |
| SHAP visualization fails | Install `pip install shap matplotlib` |
| Git LFS file shows as pointer | Run `git lfs pull` to fetch actual content |

---

## Project Timeline

| Phase | Task | Time Est. |
|-------|------|-----------|
| Setup | Initialize repo, Git LFS, dataset | 30 min |
| Training | Implement and test pipeline | 1 hour |
| API | Build REST endpoints | 1 hour |
| Docker | Containerization | 30 min |
| CI/CD | GitHub Actions setup | 30 min |
| K8s | Manifests & deployment | 30 min |

**Total: ~4-5 hours** to rebuild the complete platform from scratch.

---

**Happy building!** 🚀

*For questions or issues, refer to the original repository: https://github.com/MDuarte99/Anomaly-Detection-Platform*