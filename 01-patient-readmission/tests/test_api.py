"""
test_api.py — Unit tests for api/main.py and api/schemas.py
============================================================
Run with: pytest tests/test_api.py -v
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def valid_patient():
    """Valid patient payload matching PredictRequest schema."""
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
        "diabetesMed": "Yes"
    }


# ─────────────────────────────────────────────
# HEALTH ENDPOINT TESTS
# ─────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_has_model_name(self):
        response = client.get("/health")
        data = response.json()
        assert "model_name" in data
        assert len(data["model_name"]) > 0

    def test_health_has_optimal_threshold(self):
        response = client.get("/health")
        data = response.json()
        assert "optimal_threshold" in data
        assert 0 < data["optimal_threshold"] < 1

    def test_health_has_auc_roc(self):
        response = client.get("/health")
        data = response.json()
        assert "auc_roc" in data
        assert data["auc_roc"] > 0


# ─────────────────────────────────────────────
# PREDICT ENDPOINT TESTS
# ─────────────────────────────────────────────

class TestPredictEndpoint:

    def test_valid_request_returns_200(self, valid_patient):
        response = client.post("/predict", json=valid_patient)
        assert response.status_code == 200

    def test_response_has_probability(self, valid_patient):
        response = client.post("/predict", json=valid_patient)
        data = response.json()
        assert "readmission_probability" in data

    def test_probability_between_0_and_1(self, valid_patient):
        response = client.post("/predict", json=valid_patient)
        data = response.json()
        prob = data["readmission_probability"]
        assert 0.0 <= prob <= 1.0

    def test_response_has_risk_level(self, valid_patient):
        response = client.post("/predict", json=valid_patient)
        data = response.json()
        assert "risk_level" in data
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_response_has_flagged_field(self, valid_patient):
        response = client.post("/predict", json=valid_patient)
        data = response.json()
        assert "flagged_for_intervention" in data
        assert isinstance(data["flagged_for_intervention"], bool)

    def test_response_has_model_version(self, valid_patient):
        response = client.post("/predict", json=valid_patient)
        data = response.json()
        assert "model_version" in data

    def test_invalid_gender_returns_422(self, valid_patient):
        bad_patient = valid_patient.copy()
        bad_patient["gender"] = "Unknown"
        response = client.post("/predict", json=bad_patient)
        assert response.status_code == 422

    def test_invalid_age_returns_422(self, valid_patient):
        bad_patient = valid_patient.copy()
        bad_patient["age"] = "[999-1000)"
        response = client.post("/predict", json=bad_patient)
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self, valid_patient):
        bad_patient = valid_patient.copy()
        del bad_patient["gender"]
        response = client.post("/predict", json=bad_patient)
        assert response.status_code == 422

    def test_time_in_hospital_out_of_range_returns_422(self, valid_patient):
        bad_patient = valid_patient.copy()
        bad_patient["time_in_hospital"] = 99
        response = client.post("/predict", json=bad_patient)
        assert response.status_code == 422

    def test_high_risk_patient_flagged(self, valid_patient):
        # Patient with many prior inpatient visits should be higher risk
        high_risk = valid_patient.copy()
        high_risk["number_inpatient"] = 5
        high_risk["number_emergency"] = 3
        response = client.post("/predict", json=high_risk)
        data = response.json()
        assert data["readmission_probability"] > 0.0


# ─────────────────────────────────────────────
# ROOT ENDPOINT TEST
# ─────────────────────────────────────────────

class TestRootEndpoint:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_docs_link(self):
        response = client.get("/")
        data = response.json()
        assert "docs" in data
