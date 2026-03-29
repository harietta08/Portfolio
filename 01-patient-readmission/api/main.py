"""
main.py — FastAPI application
==============================
Endpoints:
  POST /predict  — returns readmission risk score for a patient
  GET  /health   — returns model version and API status

Run locally: uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
Or:          make api
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.schemas import PredictRequest, PredictResponse, HealthResponse
from src.predict import predict_single, load_model
from src.monitor import log_prediction

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# APP LIFECYCLE
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup so first request isn't slow."""
    logger.info("Loading model on startup...")
    try:
        model, preprocessor, model_info = load_model()
        logger.info(f"Model loaded: {model_info['model_name']} | "
                    f"AUC: {model_info.get('auc_roc', 'N/A')}")
    except Exception as e:
        logger.warning(f"Model preload failed: {e}. Will load on first request.")
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Patient Readmission Prediction API",
    description=(
        "Predicts 30-day hospital readmission risk for diabetic patients. "
        "Built on 70,000 real clinical encounters from 130 US hospitals (1999-2008). "
        "Each prevented readmission saves approximately $15,000 in hospital costs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.

    Returns model version and status.
    Used by load balancers, monitoring systems, and CI/CD pipelines.
    A production API always has this endpoint.
    """
    try:
        _, _, model_info = load_model()
        return HealthResponse(
            status="healthy",
            model_name=model_info.get("model_name", "unknown"),
            model_version=f"{model_info.get('model_name', 'unknown')}-v1",
            optimal_threshold=model_info.get("optimal_threshold", 0.5),
            auc_roc=model_info.get("auc_roc", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict 30-day readmission risk for a single patient.

    Takes patient clinical data at time of discharge.
    Returns risk probability, risk level, and top contributing factors.

    Clinical note: threshold is optimized for clinical cost, not F1.
    A false negative (missed high-risk patient) costs $15,000.
    A false positive (unnecessary follow-up) costs $500.
    """
    start_time = time.time()

    try:
        # Convert Pydantic model to dict
        patient_data = request.model_dump()

        # Run inference
        result = predict_single(patient_data)

        # Log prediction for monitoring
        latency_ms = round((time.time() - start_time) * 1000, 2)
        try:
            log_prediction(
                input_data=patient_data,
                output=result,
                latency_ms=latency_ms,
            )
        except Exception as log_err:
            logger.warning(f"Monitoring log failed (non-fatal): {log_err}")

        logger.info(
            f"Prediction: prob={result['readmission_probability']:.3f} "
            f"risk={result['risk_level']} latency={latency_ms}ms"
        )

        return PredictResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Info"])
async def root():
    """API root — returns basic info and links."""
    return {
        "name": "Patient Readmission Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
