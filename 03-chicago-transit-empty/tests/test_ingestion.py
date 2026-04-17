"""
tests/test_ingestion.py

Unit tests for ingestion/cta_api_client.py and ingestion/weather_api_client.py

Tests cover:
  - Successful API responses return correct structure
  - Pagination logic iterates until last page
  - Retry logic fires on HTTP errors and connection errors
  - CTAAPIError raised after MAX_RETRIES exhausted
  - WeatherAPIError raised on Open-Meteo error response
  - Derived boolean flags computed correctly in weather client

All HTTP calls are mocked via the `responses` library — no real
network calls are made during tests. This keeps CI fast and free.
"""

import pytest
import responses as responses_mock
from datetime import date

from ingestion.cta_api_client import CTABusRidershipClient, CTAAPIError
from ingestion.weather_api_client import OpenMeteoClient, WeatherAPIError

# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_CTA_RECORD = {
    "route": "22",
    "date": "2024-01-15T00:00:00.000",
    "daytype": "W",
    "rides": "12345",
}

SAMPLE_WEATHER_RESPONSE = {
    "hourly": {
        "time": [
            "2024-01-15T06:00",
            "2024-01-15T07:00",
            "2024-01-15T08:00",
        ],
        "temperature_2m": [2.1, 2.5, 3.0],
        "apparent_temperature": [-1.0, -0.5, 0.2],
        "precipitation": [0.0, 0.0, 5.5],
        "windspeed_10m": [12.0, 14.0, 16.0],
        "weathercode": [3, 3, 61],
    }
}

CTA_BASE_URL = "https://data.cityofchicago.org/resource/jyb9-n7fm.json"
WEATHER_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"


# ─── CTA API client tests ─────────────────────────────────────────────────────

class TestCTABusRidershipClient:

    def setup_method(self):
        self.client = CTABusRidershipClient(app_token="test-token")

    @responses_mock.activate
    def test_fetch_ridership_returns_records(self):
        """Happy path: API returns one page of records."""
        responses_mock.add(
            responses_mock.GET,
            CTA_BASE_URL,
            json=[SAMPLE_CTA_RECORD],
            status=200,
        )

        records = self.client.fetch_ridership_by_date_range(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
            route="22",
        )

        assert len(records) == 1
        assert records[0]["route"] == "22"
        assert records[0]["rides"] == "12345"
        assert records[0]["daytype"] == "W"

    @responses_mock.activate
    def test_fetch_ridership_empty_response(self):
        """API returns empty list — no records for date range."""
        responses_mock.add(
            responses_mock.GET,
            CTA_BASE_URL,
            json=[],
            status=200,
        )

        records = self.client.fetch_ridership_by_date_range(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
            route="22",
        )

        assert records == []

    @responses_mock.activate
    def test_pagination_fetches_all_pages(self):
        """
        When first page returns PAGE_SIZE (1000) records, client fetches
        a second page. When second page returns fewer than PAGE_SIZE,
        pagination stops.
        """
        from ingestion.cta_api_client import PAGE_SIZE

        page_1 = [SAMPLE_CTA_RECORD] * PAGE_SIZE
        page_2 = [SAMPLE_CTA_RECORD] * 5

        # Register two responses — responses library serves them in order
        responses_mock.add(
            responses_mock.GET, CTA_BASE_URL, json=page_1, status=200
        )
        responses_mock.add(
            responses_mock.GET, CTA_BASE_URL, json=page_2, status=200
        )

        records = self.client.fetch_ridership_by_date_range(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert len(records) == PAGE_SIZE + 5
        assert len(responses_mock.calls) == 2

    @responses_mock.activate
    def test_raises_cta_api_error_on_http_500(self):
        """HTTP 500 after all retries raises CTAAPIError."""
        from ingestion.cta_api_client import MAX_RETRIES

        for _ in range(MAX_RETRIES):
            responses_mock.add(
                responses_mock.GET,
                CTA_BASE_URL,
                json={"error": "internal server error"},
                status=500,
            )

        with pytest.raises(CTAAPIError, match="HTTP 500"):
            self.client.fetch_ridership_by_date_range(
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 15),
            )

    @responses_mock.activate
    def test_raises_cta_api_error_on_404(self):
        """HTTP 404 raises CTAAPIError."""
        from ingestion.cta_api_client import MAX_RETRIES

        for _ in range(MAX_RETRIES):
            responses_mock.add(
                responses_mock.GET,
                CTA_BASE_URL,
                json={"error": "not found"},
                status=404,
            )

        with pytest.raises(CTAAPIError):
            self.client.fetch_ridership_by_date_range(
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 15),
            )

    @responses_mock.activate
    def test_retry_succeeds_on_second_attempt(self):
        """
        First request fails with 500, second succeeds.
        Client should return records without raising.
        """
        responses_mock.add(
            responses_mock.GET, CTA_BASE_URL,
            json={"error": "server error"}, status=500,
        )
        responses_mock.add(
            responses_mock.GET, CTA_BASE_URL,
            json=[SAMPLE_CTA_RECORD], status=200,
        )

        records = self.client.fetch_ridership_by_date_range(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        assert len(records) == 1

    @responses_mock.activate
    def test_fetch_ridership_last_n_days(self):
        """fetch_ridership_last_n_days calls the API and returns records."""
        responses_mock.add(
            responses_mock.GET,
            CTA_BASE_URL,
            json=[SAMPLE_CTA_RECORD],
            status=200,
        )

        records = self.client.fetch_ridership_last_n_days(n_days=7)
        assert len(records) == 1

    @responses_mock.activate
    def test_fetch_all_routes(self):
        """fetch_all_routes returns sorted list of route strings."""
        responses_mock.add(
            responses_mock.GET,
            CTA_BASE_URL,
            json=[{"route": "77"}, {"route": "22"}, {"route": "151"}],
            status=200,
        )

        routes = self.client.fetch_all_routes()
        assert routes == ["151", "22", "77"]  # sorted ascending

    @responses_mock.activate
    def test_app_token_sent_in_header(self):
        """App token is sent as X-App-Token header."""
        responses_mock.add(
            responses_mock.GET,
            CTA_BASE_URL,
            json=[SAMPLE_CTA_RECORD],
            status=200,
        )

        self.client.fetch_ridership_by_date_range(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        sent_headers = responses_mock.calls[0].request.headers
        assert sent_headers.get("X-App-Token") == "test-token"


# ─── Weather API client tests ─────────────────────────────────────────────────

class TestOpenMeteoClient:

    def setup_method(self):
        self.client = OpenMeteoClient(
            latitude=41.8781,
            longitude=-87.6298,
            timezone="America/Chicago",
        )

    @responses_mock.activate
    def test_fetch_historical_weather_returns_records(self):
        """Happy path: returns one dict per hour."""
        responses_mock.add(
            responses_mock.GET,
            WEATHER_HISTORICAL_URL,
            json=SAMPLE_WEATHER_RESPONSE,
            status=200,
        )

        records = self.client.fetch_historical_weather(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        assert len(records) == 3
        assert records[0]["date"] == "2024-01-15"
        assert records[0]["hour"] == 6
        assert records[0]["temperature_2m"] == 2.1

    @responses_mock.activate
    def test_is_precipitation_flag_set_correctly(self):
        """is_precipitation = True when precipitation > 0.1mm."""
        responses_mock.add(
            responses_mock.GET,
            WEATHER_HISTORICAL_URL,
            json=SAMPLE_WEATHER_RESPONSE,
            status=200,
        )

        records = self.client.fetch_historical_weather(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        # Hours 0 and 1 have 0.0mm precip — should be False
        assert records[0]["is_precipitation"] is False
        assert records[1]["is_precipitation"] is False

        # Hour 2 has 5.5mm — should be True
        assert records[2]["is_precipitation"] is True

    @responses_mock.activate
    def test_is_heavy_precipitation_flag_set_correctly(self):
        """is_heavy_precipitation = True when precipitation > 5.0mm."""
        responses_mock.add(
            responses_mock.GET,
            WEATHER_HISTORICAL_URL,
            json=SAMPLE_WEATHER_RESPONSE,
            status=200,
        )

        records = self.client.fetch_historical_weather(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        assert records[0]["is_heavy_precipitation"] is False
        assert records[2]["is_heavy_precipitation"] is True  # 5.5mm > 5.0

    @responses_mock.activate
    def test_empty_hourly_data_returns_empty_list(self):
        """API returns no hourly data — client returns empty list."""
        responses_mock.add(
            responses_mock.GET,
            WEATHER_HISTORICAL_URL,
            json={"hourly": {"time": []}},
            status=200,
        )

        records = self.client.fetch_historical_weather(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        assert records == []

    @responses_mock.activate
    def test_raises_weather_api_error_on_open_meteo_error(self):
        """Open-Meteo returns an error payload — raise WeatherAPIError."""
        from ingestion.weather_api_client import MAX_RETRIES

        for _ in range(MAX_RETRIES):
            responses_mock.add(
                responses_mock.GET,
                WEATHER_HISTORICAL_URL,
                json={"error": True, "reason": "Invalid date range"},
                status=200,  # Open-Meteo returns 200 even for errors
            )

        with pytest.raises(WeatherAPIError, match="Invalid date range"):
            self.client.fetch_historical_weather(
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 15),
            )

    @responses_mock.activate
    def test_raises_weather_api_error_on_http_500(self):
        """HTTP 500 from Open-Meteo raises WeatherAPIError after retries."""
        from ingestion.weather_api_client import MAX_RETRIES

        for _ in range(MAX_RETRIES):
            responses_mock.add(
                responses_mock.GET,
                WEATHER_HISTORICAL_URL,
                json={},
                status=500,
            )

        with pytest.raises(WeatherAPIError):
            self.client.fetch_historical_weather(
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 15),
            )

    @responses_mock.activate
    def test_record_schema_has_all_required_fields(self):
        """Every returned record contains all required keys."""
        responses_mock.add(
            responses_mock.GET,
            WEATHER_HISTORICAL_URL,
            json=SAMPLE_WEATHER_RESPONSE,
            status=200,
        )

        records = self.client.fetch_historical_weather(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        required_keys = {
            "date", "hour", "temperature_2m", "apparent_temperature",
            "precipitation", "windspeed_10m", "weathercode",
            "is_precipitation", "is_heavy_precipitation",
        }
        for record in records:
            assert required_keys.issubset(set(record.keys())), (
                f"Record missing keys: {required_keys - set(record.keys())}"
            )

    @responses_mock.activate
    def test_hour_parsed_correctly_from_timestamp(self):
        """Hour extracted correctly from ISO timestamp string."""
        responses_mock.add(
            responses_mock.GET,
            WEATHER_HISTORICAL_URL,
            json=SAMPLE_WEATHER_RESPONSE,
            status=200,
        )

        records = self.client.fetch_historical_weather(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        hours = [r["hour"] for r in records]
        assert hours == [6, 7, 8]
