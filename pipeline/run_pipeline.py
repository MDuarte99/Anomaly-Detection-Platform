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
