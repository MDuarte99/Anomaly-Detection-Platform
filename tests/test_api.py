"""Tests for the FastAPI fraud detection API."""

import sys
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

# Add project root so imports work when test is run
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We test via the FastAPI test client pattern
from fastapi.testclient import TestClient
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def ensure_model_loaded():
    """Ensure model is trained before running API tests."""
    artifacts = PROJECT_ROOT / "artifacts"
    if not (artifacts / "model.joblib").exists():
        pytest.skip("Model not trained. Run: python model/train.py")


def test_health_endpoint():
    """Test /health returns expected structure."""
    from api.app import app, load_model_artifacts, START_TIME
    from api.app import model, scaler

    # Ensure model is loaded
    if model is None:
        load_model_artifacts()

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "memory_mb" in data


def test_predict_endpoint():
    """Test /predict with a sample transaction."""
    from api.app import app, load_model_artifacts, model

    if model is None:
        load_model_artifacts()

    features = {f"V{i}": float(np.random.randn()) for i in range(1, 29)}

    with TestClient(app) as client:
        response = client.post("/predict", json=features)
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert isinstance(data["is_fraud"], bool)
        assert "confidence" in data
        assert "anomaly_score" in data
        assert "processing_time_ms" in data
        assert data["model"] == "IsolationForest"


def test_predict_with_transaction_id():
    """Test /predict includes transaction_id in response."""
    from api.app import app, load_model_artifacts, model

    if model is None:
        load_model_artifacts()

    features = {f"V{i}": float(np.random.randn()) for i in range(1, 29)}

    with TestClient(app) as client:
        response = client.post("/predict", json=features, params={"transaction_id": "abc-123"})
        assert response.status_code == 200
        assert response.json()["transaction_id"] == "abc-123"


def test_metrics_endpoint():
    """Test /metrics returns latency and request stats."""
    from api.app import app, load_model_artifacts, model

    if model is None:
        load_model_artifacts()

    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data
        assert "p95_latency_ms" in data
        assert "p99_latency_ms" in data


def test_model_info():
    """Test /model/info returns metadata."""
    from api.app import app, load_model_artifacts, model

    if model is None:
        load_model_artifacts()

    with TestClient(app) as client:
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert model is not None


def test_predict_invalid_data():
    """Test /predict rejects malformed requests."""
    from api.app import app, load_model_artifacts, model

    if model is None:
        load_model_artifacts()

    with TestClient(app) as client:
        # Missing required fields
        response = client.post("/predict", json={"V1": 1.0})
        assert response.status_code == 422
