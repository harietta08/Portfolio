"""
test_predict.py — Unit tests for src/predict.py
================================================
Run with: pytest tests/test_predict.py -v
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from src.predict import predict_single, _prepare_single_patient
import pandas as pd


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def valid_patient_dict():
    return {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 5,
        "num_lab_procedures": 44,
        "num_procedures": 1,
        "num_medications": 14,
        "number_diagnoses": 9,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 2,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "diag_1": "410",
        "diag_2": "250",
        "diag_3": "401",
        "insulin": "Steady",
        "medical_specialty": "Cardiology",
        "change": "Ch",
        "diabetesMed": "Yes",
    }


# ─────────────────────────────────────────────
# PREDICT SINGLE TESTS
# ─────────────────────────────────────────────

class TestPredictSingle:

    def test_returns_dict(self, valid_patient_dict):
        result = predict_single(valid_patient_dict)
        assert isinstance(result, dict)

    def test_probability_is_float(self, valid_patient_dict):
        result = predict_single(valid_patient_dict)
        assert isinstance(result["readmission_probability"], float)

    def test_probability_between_0_and_1(self, valid_patient_dict):
        result = predict_single(valid_patient_dict)
        prob = result["readmission_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_level_valid(self, valid_patient_dict):
        result = predict_single(valid_patient_dict)
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_flagged_is_boolean(self, valid_patient_dict):
        result = predict_single(valid_patient_dict)
        assert isinstance(result["flagged_for_intervention"], bool)

    def test_model_version_is_string(self, valid_patient_dict):
        result = predict_single(valid_patient_dict)
        assert isinstance(result["model_version"], str)
        assert len(result["model_version"]) > 0

    def test_top_risk_factors_is_list(self, valid_patient_dict):
        result = predict_single(valid_patient_dict)
        assert isinstance(result["top_risk_factors"], list)

    def test_threshold_in_result(self, valid_patient_dict):
        result = predict_single(valid_patient_dict)
        assert "decision_threshold" in result
        assert 0.0 < result["decision_threshold"] < 1.0

    def test_high_inpatient_visits_increases_risk(self, valid_patient_dict):
        low_risk = valid_patient_dict.copy()
        low_risk["number_inpatient"] = 0
        low_risk["number_emergency"] = 0

        high_risk = valid_patient_dict.copy()
        high_risk["number_inpatient"] = 5
        high_risk["number_emergency"] = 4

        result_low = predict_single(low_risk)
        result_high = predict_single(high_risk)

        assert result_high["readmission_probability"] >= result_low["readmission_probability"]

    def test_deterministic_output(self, valid_patient_dict):
        result1 = predict_single(valid_patient_dict)
        result2 = predict_single(valid_patient_dict)
        assert result1["readmission_probability"] == result2["readmission_probability"]


# ─────────────────────────────────────────────
# PREPARE SINGLE PATIENT TESTS
# ─────────────────────────────────────────────

class TestPrepareSinglePatient:

    def test_creates_diag_groups(self, valid_patient_dict):
        df = pd.DataFrame([valid_patient_dict])
        result = _prepare_single_patient(df)
        assert 'diag1_group' in result.columns
        assert 'diag2_group' in result.columns
        assert 'diag3_group' in result.columns

    def test_removes_raw_diag_columns(self, valid_patient_dict):
        df = pd.DataFrame([valid_patient_dict])
        result = _prepare_single_patient(df)
        assert 'diag_1' not in result.columns
        assert 'diag_2' not in result.columns

    def test_creates_on_insulin(self, valid_patient_dict):
        df = pd.DataFrame([valid_patient_dict])
        result = _prepare_single_patient(df)
        assert 'on_insulin' in result.columns
        assert result['on_insulin'].values[0] in [0, 1]

    def test_creates_num_diabetes_meds(self, valid_patient_dict):
        df = pd.DataFrame([valid_patient_dict])
        result = _prepare_single_patient(df)
        assert 'num_diabetes_meds' in result.columns
        assert result['num_diabetes_meds'].values[0] >= 0
