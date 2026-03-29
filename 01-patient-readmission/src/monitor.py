"""
monitor.py — Prediction monitoring
====================================
Logs every prediction to a local JSONL file.
In production this writes to GCS.
scripts/check_drift.py reads these logs for weekly drift detection.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "predictions.jsonl"


def log_prediction(input_data: dict, output: dict, latency_ms: float = 0.0):
    """
    Log a single prediction to JSONL file.

    Each line is one JSON record containing:
    - timestamp
    - input features
    - output (probability, risk level)
    - latency

    Args:
        input_data: Patient feature dict from PredictRequest
        output: Prediction result dict from predict_single
        latency_ms: Inference latency in milliseconds
    """
    LOG_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "output": {
            "readmission_probability": output.get("readmission_probability"),
            "risk_level": output.get("risk_level"),
            "flagged_for_intervention": output.get("flagged_for_intervention"),
        },
        "latency_ms": latency_ms,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
