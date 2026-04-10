# ── tests/test_llm_validation.py ─────────────────────────────────────────────
# Tests Pydantic validation layer — the most important tests in the project
# These prove the system handles LLM failure modes gracefully

import pytest
from llm.validate_output import (
    validate_extraction,
    validate_comparables,
    parse_llm_json,
    StartupExtraction,
    ExtractionResult,
)


# ── parse_llm_json ────────────────────────────────────────────────────────────
def test_parse_clean_json():
    raw = '{"sector": "Fintech", "stage": "Seed"}'
    result = parse_llm_json(raw)
    assert result["sector"] == "Fintech"


def test_parse_json_with_markdown_fences():
    """LLMs often wrap JSON in code blocks — strip them."""
    raw = '```json\n{"sector": "Climate Tech"}\n```'
    result = parse_llm_json(raw)
    assert result["sector"] == "Climate Tech"


def test_parse_json_with_preamble():
    """LLMs sometimes add text before JSON — handle it."""
    raw = 'Here is the extraction:\n{"sector": "Healthcare AI", "stage": null}'
    result = parse_llm_json(raw)
    assert result["sector"] == "Healthcare AI"


def test_parse_invalid_json_raises():
    with pytest.raises(Exception):
        parse_llm_json("this is not json at all")


# ── validate_extraction ───────────────────────────────────────────────────────
def test_valid_extraction():
    raw = '''{
        "sector": "Climate Tech",
        "stage": "Series A",
        "traction_score": 7,
        "key_metrics": ["$1M ARR", "50 customers"],
        "business_model": "B2B SaaS",
        "target_customer": "Mid-market companies",
        "moat": "Data network effects",
        "risk_flags": ["crowded market"],
        "investment_signal": "Strong traction"
    }'''
    result = validate_extraction(raw)
    assert result.success is True
    assert result.data.sector == "Climate Tech"
    assert result.data.traction_score == 7


def test_invalid_traction_score():
    """Traction score outside 1-10 should fail validation."""
    raw = '''{
        "sector": "Fintech",
        "stage": null,
        "traction_score": 99,
        "key_metrics": [],
        "business_model": "SaaS",
        "target_customer": "Banks",
        "moat": null,
        "risk_flags": [],
        "investment_signal": "Unclear"
    }'''
    result = validate_extraction(raw)
    assert result.success is False
    assert result.error is not None


def test_missing_required_field():
    """Missing sector should fail — it is required."""
    raw = '''{
        "stage": "Seed",
        "traction_score": 5,
        "key_metrics": [],
        "business_model": "SaaS",
        "target_customer": "SMBs",
        "moat": null,
        "risk_flags": [],
        "investment_signal": "Early"
    }'''
    result = validate_extraction(raw)
    assert result.success is False


def test_empty_response():
    """Empty LLM response should return structured error, not crash."""
    result = validate_extraction("")
    assert result.success is False
    assert result.error is not None


def test_hallucinated_non_json():
    """LLM returning prose instead of JSON — handled gracefully."""
    raw = "The startup appears to be in the fintech sector with good traction."
    result = validate_extraction(raw)
    assert result.success is False
    assert result.error is not None


# ── validate_comparables ──────────────────────────────────────────────────────
def test_valid_comparables():
    raw = '''{
        "comparables": [
            {"name": "Persefoni", "reason": "Carbon accounting SaaS"},
            {"name": "Watershed", "reason": "Enterprise climate platform"},
            {"name": "Sweep", "reason": "Mid-market ESG reporting"}
        ]
    }'''
    result = validate_comparables(raw)
    assert result["success"] is True
    assert len(result["data"]["comparables"]) == 3


def test_empty_comparables():
    raw = '{"comparables": []}'
    result = validate_comparables(raw)
    assert result["success"] is False
