"""
predict.py — Inference logic
============================
Loads trained model and preprocessor, runs feature pipeline,
returns risk probability for a single patient.

Called by: api/main.py, app/streamlit_app.py, tests/test_predict.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import (
    engineer_features,
    build_preprocessor,
    map_icd9_to_group,
    MED_COLS,
)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

_model = None
_preprocessor = None
_model_info = None


def load_model():
    """
    Load trained model and preprocessor from MLflow artifacts.
    Uses lazy loading — model is loaded once on first call.

    Returns:
        Tuple of (model, preprocessor, model_info dict)
    """
    global _model, _preprocessor, _model_info

    if _model is not None:
        return _model, _preprocessor, _model_info

    # Load model info
    model_info_path = Path('mlflow/best_model.json')
    if not model_info_path.exists():
        raise FileNotFoundError(
            "mlflow/best_model.json not found. Run src/train.py first."
        )

    with open(model_info_path) as f:
        _model_info = json.load(f)

    # Load model via MLflow
    import mlflow
    from dotenv import load_dotenv
    load_dotenv()

    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'mlflow/mlruns')
    mlflow.set_tracking_uri(tracking_uri)

    model_name = f"readmission-classifier-{_model_info['model_name']}"
    model_uri = f"models:/{model_name}/latest"

    try:
        _model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        # Fallback: load directly from run artifacts
        run_id = _model_info['run_id']
        artifact_uri = f"{tracking_uri}/0/{run_id}/artifacts/model"
        _model = mlflow.sklearn.load_model(artifact_uri)

    # Rebuild preprocessor
    # We refit on a small sample to get the correct schema
    # In production this would be saved and loaded as an artifact
    _preprocessor = _build_inference_preprocessor()

    return _model, _preprocessor, _model_info


def _build_inference_preprocessor():
    """
    Build and fit preprocessor on training data for inference.
    Called once during model loading.
    """
    from src.features import (
        deduplicate_patients, build_target,
        build_preprocessor as _build_preprocessor
    )

    data_path = os.getenv('DATA_PATH', 'data/raw/diabetic_data.csv')

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    raw = pd.read_csv(data_path, na_values='?')
    raw = deduplicate_patients(raw)
    X = engineer_features(raw)

    preprocessor = _build_preprocessor()
    preprocessor.fit(X)
    return preprocessor


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def predict_single(patient_data: dict) -> dict:
    """
    Generate readmission risk prediction for a single patient.

    Args:
        patient_data: Dictionary matching PredictRequest schema fields

    Returns:
        Dictionary with probability, risk level, and top risk factors
    """
    model, preprocessor, model_info = load_model()
    threshold = model_info.get('optimal_threshold', 0.5)

    # Convert to DataFrame
    df = pd.DataFrame([patient_data])

    # Apply feature engineering
    df = _prepare_single_patient(df)

    # Preprocess
    X = preprocessor.transform(df)

    # Predict
    proba = model.predict_proba(X)[0][1]
    flagged = bool(proba >= threshold)

    # Risk level
    if proba >= 0.6:
        risk_level = "HIGH"
    elif proba >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Top risk factors via feature importances
    top_factors = _get_top_risk_factors(model, X, preprocessor)

    return {
        "readmission_probability": round(float(proba), 4),
        "risk_level": risk_level,
        "decision_threshold": threshold,
        "flagged_for_intervention": flagged,
        "top_risk_factors": top_factors,
        "model_version": f"{model_info['model_name']}-v1",
    }


def _prepare_single_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations to a single patient record.
    Mirrors the same logic as engineer_features() for training data.
    """
    df = df.copy()

    # Map diagnosis codes to groups
    for diag_col, group_col in [('diag_1', 'diag1_group'),
                                  ('diag_2', 'diag2_group'),
                                  ('diag_3', 'diag3_group')]:
        if diag_col in df.columns:
            df[group_col] = df[diag_col].apply(map_icd9_to_group)
            df = df.drop(columns=[diag_col])
        else:
            df[group_col] = 'Other'

    # Insulin flag
    if 'insulin' in df.columns:
        df['on_insulin'] = (df['insulin'] != 'No').astype(int)

    # Count diabetes medications
    med_cols_present = [c for c in MED_COLS if c in df.columns]
    if med_cols_present:
        df['num_diabetes_meds'] = (df[med_cols_present] != 'No').sum(axis=1)
        df = df.drop(columns=med_cols_present)
    else:
        df['num_diabetes_meds'] = 0

    # Fill missing categoricals
    if 'medical_specialty' in df.columns:
        df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Unknown')

    # Convert IDs to string
    for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Drop columns not used in training
    drop_cols = ['encounter_id', 'patient_nbr', 'weight',
                 'payer_code', 'readmitted', 'insulin']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def _get_top_risk_factors(model, X: np.ndarray, preprocessor) -> list:
    """
    Extract top risk factors from model feature importances.

    Args:
        model: Trained model with feature_importances_ attribute
        X: Processed feature array for this patient
        preprocessor: Fitted ColumnTransformer

    Returns:
        List of top risk factor dicts
    """
    try:
        from src.features import get_feature_names
        feature_names = get_feature_names(preprocessor)
        importances = model.feature_importances_

        top_idx = np.argsort(importances)[::-1][:5]
        factors = []
        for idx in top_idx:
            if idx < len(feature_names):
                factors.append({
                    "feature": feature_names[idx],
                    "importance": round(float(importances[idx]), 4),
                    "patient_value": round(float(X[0][idx]), 4),
                })
        return factors
    except Exception:
        return []
