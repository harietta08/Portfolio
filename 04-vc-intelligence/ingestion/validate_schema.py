# ── ingestion/validate_schema.py ──────────────────────────────────────────────
# Purpose: Enforce data contracts with pandera before anything enters the pipeline
# Why pandera: catches bad data at the source — wrong types, nulls, out-of-range values
# Two contract layers: pandera here, Pydantic at LLM output
# Interview answer: "Data contracts at ingestion mean bad data never reaches
#                   training or the vector store."

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from pathlib import Path
from loguru import logger

RAW_PATH = Path("data/raw/startups_raw.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Pandera schema — data contract ────────────────────────────────────────────
STARTUP_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(int, nullable=False),
        "name": Column(str, nullable=False, checks=Check(lambda s: s.str.len() > 0)),
        "description": Column(
            str,
            nullable=False,
            checks=[
                Check(lambda s: s.str.len() >= 20, error="Description too short"),
                Check(lambda s: s.str.len() <= 2000, error="Description too long"),
            ],
        ),
        "sector": Column(str, nullable=False),
        "stage": Column(
            str,
            nullable=False,
            checks=Check(
                lambda s: s.isin(["Seed", "Series A", "Series B", "Series C", "Pre-seed", "Growth"]),
                error="Invalid funding stage",
            ),
        ),
        "funding_amount_usd": Column(
            float,
            nullable=False,
            checks=[
                Check(lambda s: s > 0, error="Funding must be positive"),
                Check(lambda s: s <= 1e10, error="Funding amount suspiciously large"),
            ],
        ),
        "funding_year": Column(
            int,
            nullable=False,
            checks=Check(
                lambda s: s.between(2010, 2026),
                error="Funding year out of expected range",
            ),
        ),
        "hq_country": Column(str, nullable=False),
        "hq_city": Column(str, nullable=False),
        "founded_year": Column(
            int,
            nullable=False,
            checks=Check(lambda s: s.between(2000, 2026), error="Founded year out of range"),
        ),
        "employee_count": Column(
            int,
            nullable=True,
            checks=Check(lambda s: s >= 0, error="Employee count cannot be negative"),
        ),
        "revenue_stage": Column(
            str,
            nullable=True,
            checks=Check(
                lambda s: s.isin([
                    "Pre-revenue", "Early revenue", "Revenue",
                    "Growth", "Grant-funded"
                ]),
                error="Invalid revenue stage",
            ),
        ),
        "investors": Column(str, nullable=True),
        "traction_signal": Column(str, nullable=True),
    },
    coerce=True,       # auto-cast types where possible
    strict=False,      # allow extra columns
)


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run pandera validation. Log failures. Return cleaned dataframe.
    Drops invalid rows rather than failing the whole pipeline.
    """
    logger.info(f"Validating {len(df)} records against schema")

    try:
        validated = STARTUP_SCHEMA.validate(df, lazy=True)
        logger.info(f"All {len(validated)} records passed validation")
        return validated

    except pa.errors.SchemaErrors as e:
        failure_cases = e.failure_cases
        logger.warning(f"Validation failures: {len(failure_cases)} issues found")
        logger.warning(f"\n{failure_cases[['column', 'check', 'failure_case']].head(10)}")

        # Drop rows with failures rather than crashing
        failed_indices = failure_cases["index"].dropna().unique()
        df_clean = df.drop(index=failed_indices, errors="ignore")
        logger.info(f"Dropped {len(failed_indices)} invalid rows — {len(df_clean)} remain")

        if len(df_clean) == 0:
            raise ValueError("No valid records remain after validation")

        return df_clean


def run_validation(input_path: Path = RAW_PATH) -> pd.DataFrame:
    """Load raw data, validate, save processed output."""
    df = pd.read_csv(input_path)
    df_valid = validate_and_clean(df)

    output_path = PROCESSED_DIR / "startups_validated.csv"
    df_valid.to_csv(output_path, index=False)
    logger.info(f"Validated data saved: {output_path}")

    return df_valid


if __name__ == "__main__":
    df = run_validation()
    print(f"Validated: {len(df)} records")
    print(df.dtypes)
