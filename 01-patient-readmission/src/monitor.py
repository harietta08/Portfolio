"""
monitor.py — Prediction monitoring and GCS logging
====================================================
Logs every prediction input + output + timestamp to:
  1. Local JSONL file (logs/predictions.jsonl)
  2. GCS bucket (if configured in .env)

Called by: api/main.py after every /predict request
Read by:   scripts/check_drift.py for weekly drift detection
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "predictions.jsonl"


def log_prediction(input_data: dict, output: dict, latency_ms: float = 0.0):
    """
    Log a single prediction to local JSONL and optionally to GCS.

    Each line is one complete JSON record:
    {
        "timestamp": "2026-03-29T12:00:00Z",
        "input": { ...patient features... },
        "output": { probability, risk_level, flagged },
        "latency_ms": 45.2
    }

    Args:
        input_data: Patient feature dict from PredictRequest
        output: Prediction result dict from predict_single
        latency_ms: Inference latency in milliseconds
    """
    LOG_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "output": {
            "readmission_probability": output.get("readmission_probability"),
            "risk_level": output.get("risk_level"),
            "flagged_for_intervention": output.get("flagged_for_intervention"),
        },
        "latency_ms": latency_ms,
    }

    # Write to local JSONL
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Upload to GCS if configured
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if bucket_name:
        _upload_to_gcs(record, bucket_name)


def _upload_to_gcs(record: dict, bucket_name: str):
    """
    Append prediction record to GCS JSONL file.
    Silently fails if GCS credentials are not configured.

    Args:
        record: Prediction log record dict
        bucket_name: GCS bucket name from env
    """
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        date_str = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        blob_name = f"predictions/{date_str}/predictions.jsonl"
        blob = bucket.blob(blob_name)

        # Append to existing blob or create new
        try:
            existing = blob.download_as_text()
            new_content = existing + json.dumps(record) + "\n"
        except Exception:
            new_content = json.dumps(record) + "\n"

        blob.upload_from_string(new_content)
        logger.debug(f"Prediction logged to GCS: gs://{bucket_name}/{blob_name}")

    except ImportError:
        logger.debug("google-cloud-storage not installed, skipping GCS upload")
    except Exception as e:
        logger.warning(f"GCS upload failed (non-fatal): {e}")


def load_prediction_logs(log_file: Path = LOG_FILE) -> list:
    """
    Load all prediction logs from local JSONL file.

    Args:
        log_file: Path to JSONL log file

    Returns:
        List of prediction record dicts
    """
    if not log_file.exists():
        return []

    records = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def get_monitoring_stats() -> dict:
    """
    Compute basic monitoring statistics from prediction logs.
    Used by /health endpoint and drift detection script.

    Returns:
        Dict with prediction counts, risk distribution, avg latency
    """
    records = load_prediction_logs()

    if not records:
        return {"total_predictions": 0, "message": "No predictions logged yet"}

    probabilities = [r["output"]["readmission_probability"]
                     for r in records
                     if r["output"]["readmission_probability"] is not None]

    risk_levels = [r["output"]["risk_level"]
                   for r in records
                   if r["output"]["risk_level"] is not None]

    latencies = [r["latency_ms"] for r in records if r["latency_ms"] > 0]

    from collections import Counter
    risk_dist = dict(Counter(risk_levels))

    return {
        "total_predictions": len(records),
        "avg_probability": round(sum(probabilities) / len(probabilities), 4) if probabilities else 0,
        "risk_distribution": risk_dist,
        "flagged_rate": round(
            sum(1 for r in records if r["output"].get("flagged_for_intervention")) / len(records), 4
        ),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
    }
