"""
check_drift.py — Weekly drift detection script
================================================
Compares live feature distributions from prediction logs
against the training data baseline.

Uses Population Stability Index (PSI) to measure drift.
PSI < 0.1  : No significant drift
PSI 0.1-0.2: Moderate drift — monitor closely
PSI > 0.2  : Significant drift — consider retraining

Run with: python scripts/check_drift.py
Or:        make drift-check
"""

import os
import sys
import json
import logging
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.monitor import load_prediction_logs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

# PSI thresholds
PSI_LOW      = 0.1
PSI_MODERATE = 0.2

# Numeric features to monitor for drift
NUMERIC_FEATURES_TO_MONITOR = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_medications',
    'number_inpatient',
    'number_emergency',
    'number_outpatient',
    'number_diagnoses',
]


# ─────────────────────────────────────────────
# PSI CALCULATION
# ─────────────────────────────────────────────

def compute_psi(baseline: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Compute Population Stability Index between baseline and current distributions.

    PSI measures how much a feature distribution has shifted.
    Originally used in credit risk modeling, now standard in MLOps.

    Formula: PSI = sum((current% - baseline%) * ln(current% / baseline%))

    Args:
        baseline: Array of feature values from training data
        current: Array of feature values from recent predictions
        n_bins: Number of bins for histogram

    Returns:
        PSI value (float)
    """
    # Create bins from baseline distribution
    bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    # Compute bin frequencies
    baseline_counts, _ = np.histogram(baseline, bins=bins)
    current_counts, _  = np.histogram(current, bins=bins)

    # Convert to percentages with small epsilon to avoid log(0)
    eps = 1e-6
    baseline_pct = baseline_counts / len(baseline) + eps
    current_pct  = current_counts  / len(current)  + eps

    # PSI formula
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return round(float(psi), 4)


def classify_drift(psi: float) -> str:
    """Classify drift severity based on PSI value."""
    if psi < PSI_LOW:
        return "STABLE"
    elif psi < PSI_MODERATE:
        return "MODERATE DRIFT"
    else:
        return "SIGNIFICANT DRIFT — CONSIDER RETRAINING"


# ─────────────────────────────────────────────
# BASELINE COMPUTATION
# ─────────────────────────────────────────────

def compute_training_baseline(data_path: str = None) -> dict:
    """
    Compute baseline feature statistics from training data.

    Returns dict of {feature_name: array_of_values}
    """
    if data_path is None:
        data_path = os.getenv('DATA_PATH', 'data/raw/diabetic_data.csv')

    logger.info(f"Loading training baseline from {data_path}")

    from src.features import deduplicate_patients, engineer_features

    raw = pd.read_csv(data_path, na_values='?')
    raw = deduplicate_patients(raw)
    X = engineer_features(raw)

    baseline = {}
    for feat in NUMERIC_FEATURES_TO_MONITOR:
        if feat in X.columns:
            baseline[feat] = X[feat].dropna().values

    logger.info(f"Baseline computed for {len(baseline)} features")
    return baseline


# ─────────────────────────────────────────────
# LIVE DISTRIBUTION EXTRACTION
# ─────────────────────────────────────────────

def extract_live_distributions(records: list) -> dict:
    """
    Extract feature distributions from recent prediction logs.

    Args:
        records: List of prediction log dicts from monitor.py

    Returns:
        Dict of {feature_name: array_of_values}
    """
    live = {feat: [] for feat in NUMERIC_FEATURES_TO_MONITOR}

    for record in records:
        input_data = record.get("input", {})
        for feat in NUMERIC_FEATURES_TO_MONITOR:
            val = input_data.get(feat)
            if val is not None:
                try:
                    live[feat].append(float(val))
                except (ValueError, TypeError):
                    continue

    return {k: np.array(v) for k, v in live.items() if len(v) > 0}


# ─────────────────────────────────────────────
# MAIN DRIFT CHECK
# ─────────────────────────────────────────────

def run_drift_check():
    """
    Main drift detection function.

    Loads baseline from training data, loads live distributions
    from prediction logs, computes PSI for each feature,
    and reports findings.
    """
    print("=" * 65)
    print("Weekly Drift Detection Report")
    print(f"Run time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 65)

    # Load prediction logs
    records = load_prediction_logs()

    if len(records) < 50:
        print(f"\nInsufficient predictions for drift analysis: {len(records)} records")
        print("Need at least 50 predictions. Run more inference requests first.")
        return

    print(f"\nAnalyzing {len(records)} prediction records")

    # Compute baseline
    try:
        baseline = compute_training_baseline()
    except FileNotFoundError as e:
        logger.error(f"Cannot load training data: {e}")
        print("ERROR: Training data not found. Set DATA_PATH in .env")
        return

    # Extract live distributions
    live = extract_live_distributions(records)

    if not live:
        print("No numeric features found in prediction logs.")
        return

    # Compute PSI for each feature
    print(f"\n{'Feature':<30} {'PSI':<10} {'Status'}")
    print("-" * 65)

    alerts = []
    results = []

    for feat in NUMERIC_FEATURES_TO_MONITOR:
        if feat not in baseline or feat not in live:
            continue
        if len(live[feat]) < 10:
            continue

        psi = compute_psi(baseline[feat], live[feat])
        status = classify_drift(psi)

        print(f"{feat:<30} {psi:<10.4f} {status}")
        results.append({"feature": feat, "psi": psi, "status": status})

        if psi >= PSI_MODERATE:
            alerts.append(feat)

    # Summary
    print("\n" + "=" * 65)
    if alerts:
        print(f"DRIFT ALERTS ({len(alerts)} features):")
        for feat in alerts:
            print(f"  ⚠  {feat} — PSI above threshold")
        print("\nRECOMMENDATION: Review model performance on recent data.")
        print("If recall has dropped, trigger retraining with recent data.")
    else:
        print("No significant drift detected. Model distributions are stable.")

    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_predictions": len(records),
        "features_checked": len(results),
        "alerts": alerts,
        "results": results,
    }

    report_path = Path("logs/drift_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to {report_path}")
    print("=" * 65)

    return report


if __name__ == "__main__":
    run_drift_check()
