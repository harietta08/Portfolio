# ── api/schemas.py ────────────────────────────────────────────────────────────
# Purpose: Pydantic request and response models for FastAPI
# Every endpoint has typed inputs and outputs — no raw dicts anywhere

from pydantic import BaseModel, Field
from typing import Optional
import datetime


# ── Request models ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Startup description text to classify",
        example="AI-powered carbon accounting platform for mid-market companies. SOC2 certified. $1.1M ARR.",
    )


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Natural language query to find similar startups",
        example="climate tech companies with enterprise SaaS revenue",
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


# ── Response models ───────────────────────────────────────────────────────────
class SectorPrediction(BaseModel):
    sector: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_3: list[dict]


class LLMExtraction(BaseModel):
    success: bool
    traction_score: Optional[int] = None
    key_metrics: list[str] = []
    business_model: Optional[str] = None
    target_customer: Optional[str] = None
    moat: Optional[str] = None
    risk_flags: list[str] = []
    investment_signal: Optional[str] = None
    error: Optional[str] = None


class PredictResponse(BaseModel):
    sector_prediction: SectorPrediction
    llm_extraction: LLMExtraction
    latency_ms: dict


class SearchResult(BaseModel):
    id: str
    name: str
    sector: str
    stage: str
    similarity_score: float
    description_snippet: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    chromadb_ready: bool
    timestamp: str
