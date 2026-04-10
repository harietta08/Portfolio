# ── tests/test_ingestion.py ───────────────────────────────────────────────────
# Tests pandera schema validation and ingestion pipeline

import pytest
import pandas as pd
from ingestion.validate_schema import validate_and_clean, STARTUP_SCHEMA


def make_valid_df():
    """Minimal valid dataframe matching the schema."""
    return pd.DataFrame([{
        "id": 1,
        "name": "TestStartup",
        "description": "An AI platform for automating clinical documentation in hospitals.",
        "sector": "Healthcare AI",
        "stage": "Series A",
        "funding_amount_usd": 12000000.0,
        "funding_year": 2023,
        "hq_country": "USA",
        "hq_city": "Boston",
        "founded_year": 2021,
        "employee_count": 45,
        "revenue_stage": "Early revenue",
        "investors": "General Catalyst",
        "traction_signal": "$1M ARR",
    }])


def test_valid_record_passes():
    df = make_valid_df()
    result = validate_and_clean(df)
    assert len(result) == 1


def test_invalid_stage_dropped():
    """Record with invalid funding stage should be dropped."""
    df = make_valid_df()
    df.loc[0, "stage"] = "Series Z"
    result = validate_and_clean(df)
    assert len(result) == 0


def test_negative_funding_dropped():
    """Negative funding amount violates contract — row dropped."""
    df = make_valid_df()
    df.loc[0, "funding_amount_usd"] = -1000.0
    result = validate_and_clean(df)
    assert len(result) == 0


def test_short_description_dropped():
    """Description under 20 chars violates contract."""
    df = make_valid_df()
    df.loc[0, "description"] = "Too short"
    result = validate_and_clean(df)
    assert len(result) == 0


def test_future_funding_year_dropped():
    """Funding year outside 2010-2026 is invalid."""
    df = make_valid_df()
    df.loc[0, "funding_year"] = 2099
    result = validate_and_clean(df)
    assert len(result) == 0


def test_multiple_records_partial_failure():
    """Valid records survive even when others fail."""
    df = make_valid_df()
    bad_row = make_valid_df()
    bad_row.loc[0, "id"] = 2
    bad_row.loc[0, "funding_amount_usd"] = -500.0
    combined = pd.concat([df, bad_row], ignore_index=True)
    result = validate_and_clean(combined)
    assert len(result) == 1


def test_missing_name_handled():
    """Null name should be caught by schema."""
    df = make_valid_df()
    df.loc[0, "name"] = None
    result = validate_and_clean(df)
    assert len(result) == 0
