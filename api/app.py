"""FastAPI REST API for real-time fraud detection predictions."""

import time
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from collections import deque
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

model = None
scaler = None
metadata = None

# In-memory metrics store
request_latencies = deque(maxlen=1000)
request_count = 0
prediction_count = 0
fraud_count = 0


# =========================
# INPUT SCHEMAS
# =========================

class TransactionFeatures(BaseModel):
    """Full transaction with PCA features (production format)."""

    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float
    V7: float; V8: float; V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float; V17: float; V18: float
    V19: float; V20: float; V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float


class SimpleFeatures(BaseModel):
    """Simplified input for demo purposes."""
    features: list[float] = Field(..., min_items=28, max_items=28)


# =========================
# OUTPUT SCHEMAS
# =========================

class PredictionResponse(BaseModel):
    transaction_id: Optional[str] = None
    is_fraud: bool
    confidence: float
    anomaly_score: float
    model: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    uptime_seconds: float
    memory_mb: float
    total_requests: int
    total_predictions: int
    total_frauds_detected: int
    avg_latency_ms: float


class MetricsResponse(BaseModel):
    total_requests: int
    total_predictions: int
    total_frauds_detected: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    model_type: str
    contamination: float


START_TIME = time.time()


# =========================
# MODEL LOADING
# =========================

def load_model_artifacts():
    global model, scaler, metadata

    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    model_path = ARTIFACTS_DIR / "model.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run 'python model/train.py' first."
        )

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    with open(metrics_path) as f:
        metadata = json.load(f)

    logger.info(f"Model loaded: {metadata['model_type']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_artifacts()
    logger.info("API ready")
    yield


# =========================
# APP INIT
# =========================

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using Isolation Forest",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================
# ROOT → DOCS
# =========================

@app.get("/")
def root():
    return RedirectResponse(url="/docs")


# =========================
# CORE PREDICTION LOGIC
# =========================

def run_prediction(transaction: TransactionFeatures, transaction_id: Optional[str] = None):
    global request_count, prediction_count, fraud_count

    start = time.time()
    request_count += 1

    features = [getattr(transaction, f"V{i}") for i in range(1, 29)]
    feature_cols = metadata["features"] if metadata else [f"V{i}" for i in range(1, 29)]
    X = pd.DataFrame([features], columns=feature_cols)

    try:
        X_scaled = scaler.transform(X)
        raw_pred = model.predict(X_scaled)[0]
        score = model.decision_function(X_scaled)[0]

        is_fraud = bool(raw_pred == -1)
        confidence = round(float(np.clip(1 - score / 0.18, 0, 1)), 4)

        prediction_count += 1
        if is_fraud:
            fraud_count += 1

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = round((time.time() - start) * 1000, 2)
    request_latencies.append(elapsed_ms)

    return PredictionResponse(
        transaction_id=transaction_id,
        is_fraud=is_fraud,
        confidence=confidence,
        anomaly_score=round(float(score), 4),
        model=metadata["model_type"] if metadata else "IsolationForest",
        processing_time_ms=elapsed_ms,
    )


# =========================
# ENDPOINTS
# =========================

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionFeatures, transaction_id: Optional[str] = None):
    return run_prediction(transaction, transaction_id)


@app.post("/predict_simple", response_model=PredictionResponse)
async def predict_simple(data: SimpleFeatures):
    transaction = TransactionFeatures(
        **{f"V{i+1}": v for i, v in enumerate(data.features)}
    )
    return run_prediction(transaction)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = time.time() - START_TIME
    avg_lat = sum(request_latencies) / len(request_latencies) if request_latencies else 0

    return HealthResponse(
        status="healthy" if model and scaler else "degraded",
        model_loaded=model is not None,
        model_type=metadata["model_type"] if metadata else "unknown",
        uptime_seconds=round(uptime, 1),
        memory_mb=round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        total_requests=request_count,
        total_predictions=prediction_count,
        total_frauds_detected=fraud_count,
        avg_latency_ms=round(avg_lat, 2),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    latencies = sorted(request_latencies) if request_latencies else [0]

    return MetricsResponse(
        total_requests=request_count,
        total_predictions=prediction_count,
        total_frauds_detected=fraud_count,
        avg_latency_ms=round(np.mean(latencies), 2),
        p50_latency_ms=round(float(np.percentile(latencies, 50)), 2),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
        p99_latency_ms=round(float(np.percentile(latencies, 99)), 2),
        model_type=metadata["model_type"] if metadata else "IsolationForest",
        contamination=metadata.get("contamination", 0.0017),
    )


@app.get("/model/info")
async def model_info():
    if not metadata:
        return {"error": "Model not loaded"}
    return metadata