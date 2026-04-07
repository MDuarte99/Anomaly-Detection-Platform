"""
Simulates a real-time Kafka-like data stream.
Generates random transactions and sends them to the FastAPI predict endpoint,
mimicking a streaming ingestion pattern.
"""

import json
import time
import random
import logging
from pathlib import Path

import requests
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


def load_sample_features() -> list[dict]:
    """Sample transactions from the dataset to replay as stream."""
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c.startswith("V")]

    records = []
    for _, row in df.sample(n=min(500, len(df)), random_state=42).iterrows():
        records.append({"Features_{i}".format(i=i): row[f"V{i}"] for i in range(1, 29)})

    return records, feature_cols


def send_transaction(features: dict, session: requests.Session):
    """Send a single transaction to the predict endpoint."""
    payload = {f"V{k}": v for k, v in enumerate(features.values(), 1)}
    tx_id = f"tx_{int(time.time()*1000)}_{random.randint(1000, 9999)}"

    try:
        resp = session.post(
            f"{BASE_URL}/predict",
            json=payload,
            params={"transaction_id": tx_id},
            timeout=5,
        )
        result = resp.json()
        status = "FRAUD" if result["is_fraud"] else "OK"
        logger.info(f"[{status}] {tx_id} | score={result['anomaly_score']:.4f} | {result['processing_time_ms']:.1f}ms")
        return result
    except requests.ConnectionError:
        logger.error(f"API not reachable at {BASE_URL}. Start it first.")
        return None
    except Exception as e:
        logger.error(f"Error sending transaction: {e}")
        return None


def run_stream(duration_sec: int = 60, rate_hz: int = 10):
    """
    Simulate a streaming data source.
    Sends transactions at approximately `rate_hz` per second for `duration_sec`.
    """
    samples, feature_cols = load_sample_features()
    session = requests.Session()

    logger.info(f"Starting stream simulation: {rate_hz} tx/sec for {duration_sec}s")
    logger.info(f"Using {len(samples)} sampled transactions as source")

    start = time.time()
    total_sent = 0
    fraud_detected = 0

    try:
        while time.time() - start < duration_sec:
            batch_start = time.time()
            features = random.choice(samples)
            result = send_transaction(features, session)
            if result and result.get("is_fraud"):
                fraud_detected += 1
            total_sent += 1

            elapsed = time.time() - batch_start
            sleep_time = max(0, (1.0 / rate_hz) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user.")

    logger.info(f"Stream complete: {total_sent} transactions sent, {fraud_detected} flagged as fraud")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate transaction stream")
    parser.add_argument("--duration", type=int, default=60, help="Stream duration in seconds")
    parser.add_argument("--rate", type=int, default=10, help="Transactions per second")
    args = parser.parse_args()

    run_stream(duration_sec=args.duration, rate_hz=args.rate)
