# ── llm/validate_output.py ────────────────────────────────────────────────────
# Pydantic validation on every LLM response before storage or UI display.
# This is the single biggest differentiator between a naive LLM project
# and a production-aware one.
#
# Why this matters:
# LLMs hallucinate. Pydantic catches it before it reaches the user.
# If validation fails, we return a structured error — never crash.

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import json
import re
from loguru import logger


# ── Schemas ───────────────────────────────────────────────────────────────────
class StartupExtraction(BaseModel):
    sector: str
    stage: Optional[str] = None
    traction_score: Optional[int] = Field(None, ge=1, le=10)
    key_metrics: list[str] = []
    business_model: str
    target_customer: str
    moat: Optional[str] = None
    risk_flags: list[str] = []
    investment_signal: str

    @field_validator("traction_score")
    @classmethod
    def score_in_range(cls, v):
        if v is not None and not (1 <= v <= 10):
            raise ValueError("traction_score must be 1-10")
        return v

    @field_validator("sector")
    @classmethod
    def sector_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("sector cannot be empty")
        return v.strip()


class Comparable(BaseModel):
    name: str
    reason: str


class ComparableOutput(BaseModel):
    comparables: list[Comparable]

    @field_validator("comparables")
    @classmethod
    def at_least_one(cls, v):
        if len(v) == 0:
            raise ValueError("comparables list cannot be empty")
        return v


# ── Extraction result wrapper ─────────────────────────────────────────────────
class ExtractionResult(BaseModel):
    success: bool
    data: Optional[StartupExtraction] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None


# ── Parser ────────────────────────────────────────────────────────────────────
def parse_llm_json(raw: str) -> dict:
    """
    Safely extract JSON from LLM response.
    LLMs sometimes wrap JSON in markdown code blocks — strip those first.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```json|```", "", raw).strip()

    # Find first { and last } to extract JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")

    return json.loads(cleaned[start:end])


def validate_extraction(raw_response: str) -> ExtractionResult:
    """
    Parse and validate LLM extraction output.
    Returns ExtractionResult with success=False and error message on failure.
    Never raises — always returns a structured result.
    """
    try:
        data = parse_llm_json(raw_response)
        extraction = StartupExtraction(**data)
        return ExtractionResult(success=True, data=extraction, raw_response=raw_response)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        return ExtractionResult(
            success=False,
            error=f"LLM returned invalid JSON: {str(e)}",
            raw_response=raw_response,
        )
    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        return ExtractionResult(
            success=False,
            error=f"Schema validation failed: {str(e)}",
            raw_response=raw_response,
        )


def validate_comparables(raw_response: str) -> dict:
    """Validate comparable companies output."""
    try:
        data = parse_llm_json(raw_response)
        result = ComparableOutput(**data)
        return {"success": True, "data": result.model_dump()}
    except Exception as e:
        logger.warning(f"Comparables validation failed: {e}")
        return {"success": False, "error": str(e), "data": {"comparables": []}}
