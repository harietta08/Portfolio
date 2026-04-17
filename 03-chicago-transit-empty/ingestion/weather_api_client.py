"""
ingestion/weather_api_client.py

Open-Meteo API client — free forever, no API key required.
Fetches historical and forecast hourly weather for Chicago.

API docs: https://open-meteo.com/en/docs

Variables pulled per hour:
  - temperature_2m        (°C)
  - apparent_temperature  (feels-like, °C)
  - precipitation         (mm)
  - windspeed_10m         (km/h)
  - weathercode           (WMO code — used to derive rain/snow flags)

Design decisions:
  - Returns flat list[dict] keyed by (date, hour) for easy join to ridership
  - Historical endpoint for past data; forecast endpoint for future
  - Derived boolean flags (is_precipitation, is_heavy_precipitation) computed
    here so downstream code doesn't need to know WMO thresholds
  - No API key needed — Open-Meteo is open source and permanently free
"""

import logging
import time
from datetime import date, datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Chicago coordinates
DEFAULT_LATITUDE = 41.8781
DEFAULT_LONGITUDE = -87.6298
DEFAULT_TIMEZONE = "America/Chicago"

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "windspeed_10m",
    "weathercode",
]

MAX_RETRIES = 3
BACKOFF_BASE = 2


class WeatherAPIError(Exception):
    """Raised when Open-Meteo returns an unexpected or error response."""


class OpenMeteoClient:
    """
    Client for Open-Meteo historical and forecast weather data.

    Parameters
    ----------
    latitude : float
    longitude : float
    timezone : str
    """

    def __init__(
        self,
        latitude: float = DEFAULT_LATITUDE,
        longitude: float = DEFAULT_LONGITUDE,
        timezone: str = DEFAULT_TIMEZONE,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.session = requests.Session()

    # ─── Public methods ───────────────────────────────────────────────────────

    def fetch_historical_weather(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """
        Fetch hourly historical weather for a date range.

        Parameters
        ----------
        start_date : date
        end_date : date

        Returns
        -------
        list[dict]
            One record per hour:
            {
              "date": "2024-01-15",
              "hour": 8,
              "temperature_2m": 2.3,
              "apparent_temperature": -1.1,
              "precipitation": 0.0,
              "windspeed_10m": 14.2,
              "weathercode": 3,
              "is_precipitation": False,
              "is_heavy_precipitation": False
            }
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": self.timezone,
        }
        logger.info("Fetching historical weather %s → %s", start_date, end_date)
        raw = self._get(HISTORICAL_URL, params)
        return self._parse_hourly(raw)

    def fetch_forecast_weather(self, days_ahead: int = 7) -> list[dict]:
        """
        Fetch hourly forecast weather for the next N days.

        Parameters
        ----------
        days_ahead : int
            Number of forecast days (max 16 on free tier).

        Returns
        -------
        list[dict]  — same schema as fetch_historical_weather
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "forecast_days": days_ahead,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": self.timezone,
        }
        logger.info("Fetching %d-day weather forecast.", days_ahead)
        raw = self._get(FORECAST_URL, params)
        return self._parse_hourly(raw)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _parse_hourly(self, raw: dict) -> list[dict]:
        """
        Transform Open-Meteo nested response into a flat list of hourly dicts.
        """
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            logger.warning("Open-Meteo returned no hourly data.")
            return []

        n = len(times)
        records = []

        for i, ts_str in enumerate(times):
            # ts_str format: "2024-01-15T08:00"
            ts = datetime.fromisoformat(ts_str)
            precip = (hourly.get("precipitation") or [0.0] * n)[i] or 0.0

            record = {
                "date": ts.date().isoformat(),
                "hour": ts.hour,
                "temperature_2m": (hourly.get("temperature_2m") or [None] * n)[i],
                "apparent_temperature": (hourly.get("apparent_temperature") or [None] * n)[i],
                "precipitation": precip,
                "windspeed_10m": (hourly.get("windspeed_10m") or [None] * n)[i],
                "weathercode": (hourly.get("weathercode") or [None] * n)[i],
                # Derived flags used as Prophet regressors and dbt features
                "is_precipitation": precip > 0.1,
                "is_heavy_precipitation": precip > 5.0,
            }
            records.append(record)

        logger.info("Parsed %d hourly weather records.", len(records))
        return records

    def _get(self, url: str, params: dict) -> dict:
        """GET with retry and exponential back-off."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise WeatherAPIError(
                        f"Open-Meteo error: {data.get('reason', 'unknown')}"
                    )
                return data

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Weather API error attempt %d/%d: %s", attempt, MAX_RETRIES, exc
                )
                if attempt == MAX_RETRIES:
                    raise WeatherAPIError(
                        f"Open-Meteo unreachable after {MAX_RETRIES} attempts"
                    ) from exc

            time.sleep(BACKOFF_BASE ** attempt)

        return {}  # unreachable
