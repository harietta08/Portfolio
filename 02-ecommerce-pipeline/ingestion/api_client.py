# =============================================================================
# api_client.py
# Purpose: Fetch all data from Fake Store API and save locally as JSON.
#          Called by the Airflow DAG before GCS upload.
#          Also runnable standalone for local testing.
# =============================================================================

import requests
import json
import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_URL       = os.getenv("FAKESTORE_API_BASE_URL", "https://fakestoreapi.com")
RAW_DATA_DIR   = Path(__file__).parent.parent / "data" / "raw"
INGESTION_DATE = date.today().isoformat()


def fetch_endpoint(endpoint: str) -> list:
    url  = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    print(f"  GET /{endpoint} -> {len(data)} records")
    return data


def save_raw(data: list, name: str) -> Path:
    out_dir = RAW_DATA_DIR / f"dt={INGESTION_DATE}" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{name}.json"
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {len(data)} records -> {out_file}")
    return out_file


def run_ingestion() -> dict:
    print(f"\nIngestion started: {INGESTION_DATE}")
    print(f"API base: {BASE_URL}\n")

    results = {}
    endpoints = {
        "products":   "products",
        "users":      "users",
        "carts":      "carts",
        "categories": "products/categories",
    }

    for name, endpoint in endpoints.items():
        data = fetch_endpoint(endpoint)
        path = save_raw(data if isinstance(data, list) else [data], name)
        results[name] = {"count": len(data), "path": str(path)}

    print(f"\nIngestion complete. Files saved to: {RAW_DATA_DIR}")
    return results


if __name__ == "__main__":
    results = run_ingestion()
    print("\nSummary:")
    for name, info in results.items():
        print(f"  {name:15s} {info['count']:>4} records -> {info['path']}")
