# Real-Time Fraud Detection System

[![CI/CD](https://github.com/username/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/username/fraud-detection-mlops/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Production-ready MLOps platform for real-time credit card fraud detection. Demonstrates a complete ML lifecycle — training, deployment, monitoring, and explainability — in a single, self-contained repo.

---

## 📊 Dataset

Kaggle's [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset:

| Detail | Value |
|---|---|
| Size | 284,807 transactions |
| Features | V1–V28 (PCA-transformed), Amount, Time |
| Fraud rate | 0.17% (492 positive cases) |
| Location | `data/creditcard.csv` |

---

## 🏗️ Architecture

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

**Project structure:**

```
├── api/app.py                    # FastAPI REST server
├── model/train.py                 # Training pipeline with MLflow
├── pipeline/run_pipeline.py       # End-to-end batch pipeline
├── monitoring/drift_detector.py   # Model drift detection (PSI)
├── scripts/
│   ├── stream_simulator.py        # Real-time stream simulation
│   └── explain_predictions.py     # SHAP explainability
├── tests/test_api.py             # API integration tests
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment manifest
│   └── service.yaml               # Kubernetes service manifest
├── docker/                        # (Dockerfile at root)
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── docker-compose.yml             # Docker Compose (API + MLflow UI)
├── requirements.txt
├── Dockerfile
└── artifacts/                     # Serialized model, scaler, metrics
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Framework | Scikit-learn (Isolation Forest) |
| Model Registry | MLflow |
| API | FastAPI + Uvicorn |
| Container | Docker + Docker Compose |
| Orchestration | Kubernetes (Deployment + Service) |
| CI/CD | GitHub Actions |
| Explainability | SHAP |
| Monitoring | Built-in metrics + PSI drift detection |
| Data | Pandas, NumPy |

---

## ▶️ How to Run Locally

### Prerequisites

```bash
python -m venv venv && source venv/bin/activate  # Linux/macOS
# Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 1. Train the model

```bash
python model/train.py
```

Output (artifacts saved to `artifacts/`):

| Metric | Value |
|---|---|
| AUC-ROC | 0.956 |
| Precision (fraud) | 0.32 |
| Recall (fraud) | 0.33 |
| F1 (fraud) | 0.32 |
| Accuracy | 99.8% |

> **Why recall ~33%?** Isolation Forest is unsupervised and calibrated for a contamination rate of 0.17%. It catches about a third of true fraud with minimal false positives (~0.12%). In production, you'd tune the threshold based on business cost of vs fraud loss.

### 2. Start the API

```bash
uvicorn api.app:app --reload --port 8000
```

### 3. Make predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -0.5, "V2": 0.9, "V3": 0.2, "V4": -0.3, "V5": 0.1,
       "V6": 0.7, "V7": -0.2, "V8": 0.1, "V9": -0.4, "V10": 0.3,
       "V11": -0.6, "V12": 0.8, "V13": -0.1, "V14": -0.4, "V15": 0.2,
       "V16": 0.6, "V17": -0.3, "V18": 0.1, "V19": -0.5, "V20": 0.4,
       "V21": 0.1, "V22": -0.2, "V23": 0.3, "V24": -0.1, "V25": 0.2,
       "V26": -0.4, "V27": 0.1, "V28": 0.0}'
```

Response:

```json
{
  "transaction_id": null,
  "is_fraud": false,
  "confidence": 0.12,
  "anomaly_score": 0.1562,
  "model": "IsolationForest",
  "processing_time_ms": 2.5
}
```

### 4. Check health & metrics

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/model/info
```

### 5. Run tests

```bash
python -m pytest tests/test_api.py -v
```

---

## 🐳 Docker

### Build and run

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

### Docker Compose (API + MLflow UI)

```bash
docker compose up -d
```

MLflow UI available at `http://localhost:5000`

---

## ☸️ Kubernetes

Apply manifests after building and pushing Docker image:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc fraud-detection-service
```

---

## 🔄 Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                        │
│                                                                 │
│  1. Load Data        →  creditcard.csv into Pandas DataFrame    │
│  2. Prepare Features →  Select V1-V28, StandardScaler, 80/20   │
│  3. Train Model      →  IsolationForest, contamination=0.0017  │
│  4. Evaluate         →  AUC-ROC, Precision, Recall, F1         │
│  5. Save Artifacts   →  model.joblib, scaler.joblib, metrics   │
│  6. Log to MLflow    →  Params, metrics, model versioning      │
└─────────────────────────────────────────────────────────────────┘
```

**Quality gate:** Training fails if AUC-ROC drops below 0.80.

### CI/CD Pipeline (GitHub Actions)

```yaml
push → main
├── test          →  pytest on API endpoints
├── lint          →  flake8 static analysis
├── build-docker  →  build Docker image
└── train-model   →  run pipeline, upload artifacts
```

---

## 📈 Monitoring

### Runtime metrics

The `/metrics` endpoint exposes request-level stats:

```json
{
  "total_requests": 1523,
  "avg_latency_ms": 2.8,
  "p50_latency_ms": 2.1,
  "p95_latency_ms": 5.4,
  "p99_latency_ms": 8.9
}
```

### Model drift detection

```bash
python monitoring/drift_detector.py
```

Uses **Population Stability Index (PSI)** to detect when prediction distributions shift:

| PSI | Status |
|---|---|
| < 0.10 | Stable |
| 0.10 - 0.20 | Moderate drift |
| >= 0.20 | Significant drift — retrain |

---

## 💡 Design Decisions

### Why Isolation Forest?

- **Fits the problem**: Fraud detection is fundamentally anomaly detection — Isolation Forest excels at finding rare patterns in high-dimensional space without labeled data
- **Fast at inference**: O(log n) average path length, sub-millisecond predictions
- **Scalable**: Splits the dataset across trees naturally; can be parallelized or distributed
- **No threshold tuning from scratch**: Contamination parameter handles class imbalance

### Trade-offs

| Decision | Trade-off |
|---|---|
| Isolation Forest over XGBoost | Lower recall but fully unsupervised — no labeled data dependency |
| SQLite/MLflow file backend | Simple but not concurrent-safe for multi-writer setups |
| StandardScaler only | Assumes features are roughly Gaussian after PCA (which they are) |
| In-memory metrics (deque) | Lost on restart; in production use Redis or Prometheus |

### Limitations

- **No feature engineering**: `Amount` and `Time` columns are excluded. In production, these would be engineered (e.g., time windows, velocity features).
- **Unsupervised precision**: ~32% precision means 2/3 flags are false positives — acceptable for a baseline, but production needs supervised fine-tuning.
- **Simulated streaming**: The stream simulator replays sampled data rather than connecting to a real Kafka cluster.
- **Single model version**: MLflow tracks versions, but auto-selection of the "latest best" model is manual.

### Future improvements

1. **Supervised model**: XGBoost or LightGBM with SMOTE for higher precision/recall
2. **Feature engineering**: Rolling means, velocity features, merchant risk scores
3. **Real streaming**: Apache Kafka + Flink for true real-time ingestion
4. **Prometheus + Grafana** for production monitoring dashboards
5. **Airflow DAG** for scheduled retraining and data validation
6. **A/B testing framework** for shadow-deploying new models
7. **Data validation** with Great Expectations or Pandera
8. **Feature store** for consistent training/serving features
