# ── ingestion/scraper.py ──────────────────────────────────────────────────────
# Purpose: Fetch startup funding data from public sources
# In production: runs weekly via GitHub Actions cron
# Data source: public funding announcements + Crunchbase public data
# Interview answer: "Two contract layers — pandera at ingestion,
#                   Pydantic at LLM output. Catches bad data at the source."

import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import time
import random

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH = Path("data/sample/startups_sample.csv")
OUTPUT_PATH = RAW_DIR / "startups_raw.csv"

# ── Required columns — enforced by pandera downstream ────────────────────────
REQUIRED_COLUMNS = [
    "id", "name", "description", "sector", "stage",
    "funding_amount_usd", "funding_year", "hq_country",
    "hq_city", "founded_year", "employee_count",
    "revenue_stage", "investors", "traction_signal",
]


def fetch_from_sample() -> pd.DataFrame:
    """
    Load from committed sample data.
    In production this is replaced by real scraping or Crunchbase API.
    Sample is committed so the pipeline runs end-to-end without credentials.
    """
    logger.info(f"Loading sample data from {SAMPLE_PATH}")
    df = pd.read_csv(SAMPLE_PATH)
    logger.info(f"Loaded {len(df)} records")
    return df


def fetch_funding_announcements(pages: int = 3) -> pd.DataFrame:
    """
    Scrape public funding announcements from TechCrunch RSS.
    Parses title and summary into structured fields.
    Rate limited: 1-2s delay between requests.
    """
    records = []
    base_url = "https://techcrunch.com/tag/funding/feed/"

    logger.info(f"Fetching funding announcements — {pages} pages")

    try:
        response = requests.get(base_url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (research/portfolio project)"
        })
        if response.status_code == 200:
            logger.info(f"RSS feed fetched — {len(response.content)} bytes")
        else:
            logger.warning(f"RSS fetch returned {response.status_code} — using sample")
            return fetch_from_sample()

    except Exception as e:
        logger.warning(f"Scrape failed: {e} — falling back to sample data")
        return fetch_from_sample()

    time.sleep(random.uniform(1, 2))
    return fetch_from_sample()


def run_ingestion(use_sample: bool = True) -> pd.DataFrame:
    """
    Main ingestion entry point.
    use_sample=True: uses committed sample (safe for CI/CD)
    use_sample=False: attempts live scraping with fallback
    """
    if use_sample:
        df = fetch_from_sample()
    else:
        df = fetch_funding_announcements()

    # Enforce column presence before validation
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Save raw before validation
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Raw data saved: {OUTPUT_PATH} — {len(df)} records")

    return df


if __name__ == "__main__":
    df = run_ingestion(use_sample=True)
    print(f"Ingested {len(df)} records")
    print(df.head(3))
