# =============================================================================
# validate_schema.py
# Purpose: Enforce data contracts at ingestion boundary using pandera.
#          If data violates the contract, the pipeline halts here — not
#          silently downstream in a dbt test 6 hours later.
# =============================================================================

import json
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from pathlib import Path
from datetime import date

RAW_DATA_DIR   = Path(__file__).parent.parent / "data" / "raw"
INGESTION_DATE = date.today().isoformat()


# -----------------------------------------------------------------------------
# DATA CONTRACTS
# These are the minimum guarantees we require from the API.
# Any violation raises a SchemaError and stops the pipeline.
# -----------------------------------------------------------------------------

PRODUCTS_SCHEMA = DataFrameSchema({
    "id":    Column(int,   checks=Check.greater_than(0),  nullable=False),
    "title": Column(str,   checks=Check.str_length(1, 500), nullable=False),
    "price": Column(float, checks=Check.greater_than(0),  nullable=False),
    "category": Column(str, checks=Check.isin([
        "men's clothing",
        "women's clothing",
        "jewelery",
        "electronics"
    ]), nullable=False),
}, name="products_schema")

USERS_SCHEMA = DataFrameSchema({
    "id":       Column(int, checks=Check.greater_than(0), nullable=False),
    "email":    Column(str, checks=Check.str_length(3, 200), nullable=False),
    "username": Column(str, checks=Check.str_length(1, 100), nullable=False),
}, name="users_schema")

CARTS_SCHEMA = DataFrameSchema({
    "id":     Column(int, checks=Check.greater_than(0), nullable=False),
    "userId": Column(int, checks=Check.greater_than(0), nullable=False),
    "date":   Column(str, nullable=False),
    "products": Column(object, nullable=False),
}, name="carts_schema")


def load_json(name: str) -> list:
    path = RAW_DATA_DIR / f"dt={INGESTION_DATE}" / name / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"Run api_client.py first to fetch data."
        )
    with open(path) as f:
        return json.load(f)


def validate_products() -> pd.DataFrame:
    print("  Validating products...")
    data = load_json("products")
    df   = pd.DataFrame(data)

    # Flatten nested rating dict before validation
    if "rating" in df.columns:
        df["rating_rate"]  = df["rating"].apply(
            lambda x: x.get("rate") if isinstance(x, dict) else None)
        df["rating_count"] = df["rating"].apply(
            lambda x: x.get("count") if isinstance(x, dict) else None)
        df = df.drop(columns=["rating"])

    df["price"] = df["price"].astype(float)
    df["id"]    = df["id"].astype(int)

    PRODUCTS_SCHEMA.validate(df)
    print(f"  products: {len(df)} rows — PASSED")
    return df


def validate_users() -> pd.DataFrame:
    print("  Validating users...")
    data = load_json("users")
    df   = pd.DataFrame(data)
    df["id"] = df["id"].astype(int)

    USERS_SCHEMA.validate(df)
    print(f"  users: {len(df)} rows — PASSED")
    return df


def validate_carts() -> pd.DataFrame:
    print("  Validating carts...")
    data = load_json("carts")
    df   = pd.DataFrame(data)
    df["id"]     = df["id"].astype(int)
    df["userId"] = df["userId"].astype(int)

    CARTS_SCHEMA.validate(df)
    print(f"  carts: {len(df)} rows — PASSED")
    return df


def run_validation() -> bool:
    print(f"\nSchema validation started: {INGESTION_DATE}\n")
    try:
        validate_products()
        validate_users()
        validate_carts()
        print("\nAll schemas PASSED. Pipeline may continue.")
        return True
    except pa.errors.SchemaError as e:
        print(f"\nSCHEMA VALIDATION FAILED:\n{e}")
        print("Pipeline halted. Fix data quality issue before proceeding.")
        raise


if __name__ == "__main__":
    run_validation()
