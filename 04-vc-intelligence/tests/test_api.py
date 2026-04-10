# ── tests/test_api.py ─────────────────────────────────────────────────────────
# Tests FastAPI endpoints without loading real models
# Uses dependency overrides to mock MODEL and CHROMA

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def client():
    """TestClient with mocked model dependencies."""
    with patch("api.main.MODEL") as mock_model, \
         patch("api.main.TFIDF") as mock_tfidf, \
         patch("api.main.LABEL_ENCODER") as mock_le, \
         patch("api.main.CHROMA_COLLECTION") as mock_chroma, \
         patch("api.main.extract_startup_fields") as mock_llm:

        # Mock classifier
        mock_tfidf.transform.return_value = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
        mock_le.classes_ = ["Climate Tech", "Fintech", "Healthcare AI"]

        # Mock LLM extraction
        from llm.validate_output import ExtractionResult, StartupExtraction
        mock_llm.return_value = ExtractionResult(
            success=True,
            data=StartupExtraction(
                sector="Fintech",
                traction_score=7,
                key_metrics=["$2M ARR", "50 customers"],
                business_model="B2B SaaS",
                target_customer="Mid-market banks",
                moat="Proprietary data network",
                risk_flags=["regulatory risk"],
                investment_signal="Strong traction",
            ),
        )

        # Mock ChromaDB
        mock_chroma.count.return_value = 50
        mock_chroma.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["AI platform description", "Carbon accounting tool"]],
            "metadatas": [[
                {"name": "StartupA", "sector": "Fintech", "stage": "Series A"},
                {"name": "StartupB", "sector": "Climate Tech", "stage": "Seed"},
            ]],
            "distances": [[0.1, 0.3]],
        }

        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_predict_success(client):
    response = client.post("/predict", json={
        "description": "AI-powered carbon accounting platform for mid-market companies. SOC2 certified. $1.1M ARR from 85 customers."
    })
    assert response.status_code == 200
    data = response.json()
    assert "sector_prediction" in data
    assert "llm_extraction" in data
    assert "latency_ms" in data
    assert data["sector_prediction"]["confidence"] > 0


def test_predict_short_description(client):
    response = client.post("/predict", json={"description": "short"})
    assert response.status_code == 422


def test_search_success(client):
    response = client.post("/search", json={
        "query": "climate tech carbon accounting",
        "top_k": 2,
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= 2


def test_search_missing_query(client):
    response = client.post("/search", json={"top_k": 5})
    assert response.status_code == 422
