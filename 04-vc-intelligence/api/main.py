# ── api/main.py ───────────────────────────────────────────────────────────────
# Purpose: FastAPI serving layer — /predict /search /health
# Loads model from MLflow Model Registry at startup
# Interview answer: "Training happens in Databricks. Serving happens here.
#                   The registry is the contract between the two."

import os
import time
import pickle
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from loguru import logger
from dotenv import load_dotenv

import mlflow.pyfunc
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictRequest, PredictResponse,
    SearchRequest, SearchResponse,
    SearchResult, SectorPrediction,
    LLMExtraction, HealthResponse,
)
from llm.extract_fields import extract_startup_fields

load_dotenv()

# ── Global state — loaded once at startup ─────────────────────────────────────
MODEL = None
TFIDF = None
LABEL_ENCODER = None
CHROMA_COLLECTION = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and connections at startup."""
    global MODEL, TFIDF, LABEL_ENCODER, CHROMA_COLLECTION

    logger.info("Loading sector classifier from MLflow...")
    try:
        model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/vc-sector-classifier/Production")
        MODEL = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded: {model_uri}")
    except Exception as e:
        logger.warning(f"MLflow load failed: {e} — loading local fallback")
        # Local fallback for development
        local_model_path = Path("data/processed/sector_classifier.pkl")
        if local_model_path.exists():
            with open(local_model_path, "rb") as f:
                MODEL = pickle.load(f)

    logger.info("Loading TF-IDF vectorizer...")
    try:
        tfidf_path = Path("data/processed/tfidf_vectorizer.pkl")
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                TFIDF = pickle.load(f)
        le_path = Path("data/processed/label_encoder.pkl")
        if le_path.exists():
            with open(le_path, "rb") as f:
                LABEL_ENCODER = pickle.load(f)
    except Exception as e:
        logger.warning(f"Vectorizer load failed: {e}")

    logger.info("Connecting to ChromaDB...")
    try:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "startups")
        CHROMA_COLLECTION = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready — {CHROMA_COLLECTION.count()} documents")
    except Exception as e:
        logger.warning(f"ChromaDB connection failed: {e}")

    yield

    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Startup Funding Intelligence API",
    description=(
        "Screens startup deals in 30 seconds instead of 30 minutes. "
        "Sector classification, traction scoring, semantic search, "
        "and LLM-extracted structured metrics."
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


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        chromadb_ready=CHROMA_COLLECTION is not None,
        timestamp=datetime.datetime.utcnow().isoformat(),
    )


# ── /predict ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Given a startup description, return:
    - Sector classification with confidence
    - LLM-extracted structured fields (traction score, metrics, risks)
    - Latency breakdown for classifier vs LLM
    """
    if MODEL is None or TFIDF is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ── Classifier inference ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    X = TFIDF.transform([request.description])
    proba = MODEL.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    pred_sector = LABEL_ENCODER.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {"sector": LABEL_ENCODER.classes_[i], "confidence": round(float(proba[i]), 3)}
        for i in top3_idx
    ]
    classifier_ms = round((time.perf_counter() - t0) * 1000, 2)

    # ── LLM extraction ────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    extraction_result = extract_startup_fields(request.description)
    llm_ms = round((time.perf_counter() - t1) * 1000, 2)

    if extraction_result.success and extraction_result.data:
        d = extraction_result.data
        llm_out = LLMExtraction(
            success=True,
            traction_score=d.traction_score,
            key_metrics=d.key_metrics,
            business_model=d.business_model,
            target_customer=d.target_customer,
            moat=d.moat,
            risk_flags=d.risk_flags,
            investment_signal=d.investment_signal,
        )
    else:
        llm_out = LLMExtraction(
            success=False,
            error=extraction_result.error or "LLM extraction failed",
        )

    return PredictResponse(
        sector_prediction=SectorPrediction(
            sector=pred_sector,
            confidence=round(confidence, 3),
            top_3=top3,
        ),
        llm_extraction=llm_out,
        latency_ms={"classifier_ms": classifier_ms, "llm_ms": llm_ms},
    )


# ── /search ───────────────────────────────────────────────────────────────────
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Semantic similarity search over startup database using ChromaDB.
    Returns top-k most similar startups to the query.
    """
    if CHROMA_COLLECTION is None:
        raise HTTPException(status_code=503, detail="ChromaDB not ready")

    t0 = time.perf_counter()

    results = CHROMA_COLLECTION.query(
        query_texts=[request.query],
        n_results=min(request.top_k, CHROMA_COLLECTION.count()),
        include=["documents", "metadatas", "distances"],
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    search_results = []
    if results and results["ids"] and len(results["ids"][0]) > 0:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = round(1 - distance, 3)
            doc = results["documents"][0][i]

            search_results.append(SearchResult(
                id=str(doc_id),
                name=meta.get("name", "Unknown"),
                sector=meta.get("sector", "Unknown"),
                stage=meta.get("stage", "Unknown"),
                similarity_score=similarity,
                description_snippet=doc[:150] + "..." if len(doc) > 150 else doc,
            ))

    return SearchResponse(
        query=request.query,
        results=search_results,
        latency_ms=latency_ms,
    )
