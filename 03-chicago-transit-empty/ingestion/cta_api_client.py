"""
ingestion/cta_api_client.py

Chicago Data Portal — CTA Bus Ridership API client.
Pulls daily ridership data per route from the Socrata open data API.

Dataset: CTA - Ridership - Bus Routes - Daily Totals by Route
URL: https://data.cityofchicago.org/Transportation/CTA-Ridership-Bus-Routes-Daily-Totals-by-Route/jyb9-n7fm

Design decisions:
  - App token read from env var; requests still work without it but are rate-limited
  - Pagination handled automatically — yields full result sets regardless of size
  - Returns raw list[dict] so the caller decides what to do with it
  - All network errors raise CTAAPIError so callers can handle uniformly
  - Exponential back-off on retries to avoid hammering the API
"""

import os
import logging
import time
from datetime import date, timedelta
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://data.cityofchicago.org/resource/jyb9-n7fm.json"
PAGE_SIZE = 1000          # Socrata max rows per request
MAX_RETRIES = 3
BACKOFF_BASE = 2          # seconds, doubles each retry


class CTAAPIError(Exception):
    """Raised when the CTA / Chicago Data Portal API returns an error."""


class CTABusRidershipClient:
    """
    Thin wrapper around the Chicago Data Portal Socrata API.

    Parameters
    ----------
    app_token : str, optional
        Chicago Data Portal application token.
        Reads CHICAGO_DATA_PORTAL_APP_TOKEN from env if not supplied.
        Requests work without a token but are rate-limited to ~1 req/sec.
    """

    def __init__(self, app_token: Optional[str] = None):
        self.app_token = app_token or os.getenv("CHICAGO_DATA_PORTAL_APP_TOKEN", "")
        self.session = requests.Session()
        if self.app_token:
            self.session.headers.update({"X-App-Token": self.app_token})

    # ─── Public methods ───────────────────────────────────────────────────────

    def fetch_ridership_by_date_range(
        self,
        start_date: date,
        end_date: date,
        route: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch daily ridership records for a date range.

        Parameters
        ----------
        start_date : date
            Inclusive start date.
        end_date : date
            Inclusive end date.
        route : str, optional
            If provided, filter to a single route number e.g. "22".

        Returns
        -------
        list[dict]
            Raw records from the API. Each record contains:
            {
              "route":   "22",
              "date":    "2024-01-15T00:00:00.000",
              "daytype": "W",       # W=Weekday, A=Saturday, U=Sunday/Holiday
              "rides":   "12345"    # string — Socrata returns numeric as string
            }
        """
        where_clauses = [
            f"date >= '{start_date.isoformat()}T00:00:00.000'",
            f"date <= '{end_date.isoformat()}T00:00:00.000'",
        ]
        if route:
            where_clauses.append(f"route = '{route}'")

        where = " AND ".join(where_clauses)
        return self._paginate(where_clause=where)

    def fetch_ridership_last_n_days(
        self,
        n_days: int = 90,
        route: Optional[str] = None,
    ) -> list[dict]:
        """
        Convenience wrapper — fetch the last N days of ridership data.

        Parameters
        ----------
        n_days : int
            Number of calendar days back from today.
        route : str, optional
            Single route filter.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=n_days)
        logger.info(
            "Fetching CTA ridership: %s to %s (route=%s)",
            start_date, end_date, route or "ALL",
        )
        return self.fetch_ridership_by_date_range(start_date, end_date, route)

    def fetch_all_routes(self) -> list[str]:
        """
        Return the sorted list of unique route numbers in the dataset.
        Uses a lightweight $select + $group query — does not pull all records.
        """
        params = {"$select": "route", "$group": "route", "$limit": 500}
        records = self._get(BASE_URL, params)
        routes = sorted({r["route"] for r in records})
        logger.info("Found %d unique CTA routes.", len(routes))
        return routes

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _paginate(self, where_clause: str) -> list[dict]:
        """
        Iterate through Socrata pages until fewer than PAGE_SIZE rows returned.
        Automatically handles datasets larger than 1000 rows.
        """
        results: list[dict] = []
        offset = 0

        while True:
            params = {
                "$where": where_clause,
                "$limit": PAGE_SIZE,
                "$offset": offset,
                "$order": "date ASC",
            }
            page = self._get(BASE_URL, params)
            results.extend(page)
            logger.debug("Page offset=%d returned %d records.", offset, len(page))

            if len(page) < PAGE_SIZE:
                break  # last page reached
            offset += PAGE_SIZE

        logger.info(
            "CTA API total: %d records fetched (where: %s)",
            len(results), where_clause,
        )
        return results

    def _get(self, url: str, params: dict) -> list[dict]:
        """
        GET request with retry and exponential back-off.
        Raises CTAAPIError after MAX_RETRIES failed attempts.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else "unknown"
                logger.warning(
                    "HTTP %s on attempt %d/%d", status, attempt, MAX_RETRIES
                )
                if attempt == MAX_RETRIES:
                    raise CTAAPIError(
                        f"CTA API returned HTTP {status} after {MAX_RETRIES} attempts"
                    ) from exc

            except requests.exceptions.ConnectionError as exc:
                logger.warning(
                    "Connection error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc
                )
                if attempt == MAX_RETRIES:
                    raise CTAAPIError(
                        f"CTA API unreachable after {MAX_RETRIES} attempts"
                    ) from exc

            except requests.exceptions.Timeout:
                logger.warning("Timeout on attempt %d/%d", attempt, MAX_RETRIES)
                if attempt == MAX_RETRIES:
                    raise CTAAPIError(
                        f"CTA API timed out after {MAX_RETRIES} attempts"
                    )

            sleep_time = BACKOFF_BASE ** attempt
            logger.info("Retrying in %ds...", sleep_time)
            time.sleep(sleep_time)

        return []  # unreachable — satisfies type checker
