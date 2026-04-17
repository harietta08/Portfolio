"""
tests/test_schema_validation.py

Unit tests for ingestion/validate_schema.py

Tests cover:
  - Valid records pass all three validators without raising
  - Missing required fields raise SchemaValidationError
  - Invalid field values (wrong enum, negative rides, bad date pattern) fail
  - validate_batch correctly splits valid and invalid records
  - Invalid records in batch get _validation_error key appended
  - TransitEventValidator rejects non-canonical source values
  - Empty batch returns ([], []) gracefully

These tests enforce the schema contract at the ingestion boundary.
If any of these fail, the pipeline has a schema regression.
"""

import pytest
from ingestion.validate_schema import (
    CTARidershipValidator,
    WeatherRecordValidator,
    TransitEventValidator,
    SchemaValidationError,
)


# ─── Valid record fixtures ─────────────────────────────────────────────────────

VALID_CTA_RECORD = {
    "route": "22",
    "date": "2024-01-15T00:00:00.000",
    "daytype": "W",
    "rides": "12345",
}

VALID_WEATHER_RECORD = {
    "date": "2024-01-15",
    "hour": 8,
    "temperature_2m": 2.3,
    "apparent_temperature": -1.1,
    "precipitation": 0.0,
    "windspeed_10m": 14.2,
    "weathercode": 3,
    "is_precipitation": False,
    "is_heavy_precipitation": False,
}

VALID_TRANSIT_EVENT = {
    "event_id": "550e8400-e29b-41d4-a716-446655440000",
    "route": "22",
    "service_date": "2024-01-15",
    "day_type": "W",
    "rides": 12345,
    "ingested_at": "2024-01-15T10:00:00+00:00",
    "source": "chicago_data_portal",
    "temperature_2m": 2.3,
    "precipitation": 0.0,
    "windspeed_10m": 14.2,
    "weathercode": 3,
}


# ─── CTARidershipValidator tests ──────────────────────────────────────────────

class TestCTARidershipValidator:

    def test_valid_record_passes(self):
        """A correctly formed CTA record passes without raising."""
        CTARidershipValidator.validate_one(VALID_CTA_RECORD)  # no exception

    def test_valid_saturday_record_passes(self):
        """day_type=A (Saturday) is valid."""
        record = {**VALID_CTA_RECORD, "daytype": "A"}
        CTARidershipValidator.validate_one(record)

    def test_valid_sunday_record_passes(self):
        """day_type=U (Sunday/Holiday) is valid."""
        record = {**VALID_CTA_RECORD, "daytype": "U"}
        CTARidershipValidator.validate_one(record)

    def test_missing_route_raises(self):
        """Missing 'route' field raises SchemaValidationError."""
        record = {k: v for k, v in VALID_CTA_RECORD.items() if k != "route"}
        with pytest.raises(SchemaValidationError):
            CTARidershipValidator.validate_one(record)

    def test_missing_rides_raises(self):
        """Missing 'rides' field raises SchemaValidationError."""
        record = {k: v for k, v in VALID_CTA_RECORD.items() if k != "rides"}
        with pytest.raises(SchemaValidationError):
            CTARidershipValidator.validate_one(record)

    def test_missing_date_raises(self):
        """Missing 'date' field raises SchemaValidationError."""
        record = {k: v for k, v in VALID_CTA_RECORD.items() if k != "date"}
        with pytest.raises(SchemaValidationError):
            CTARidershipValidator.validate_one(record)

    def test_invalid_daytype_raises(self):
        """daytype not in [W, A, U] raises SchemaValidationError."""
        record = {**VALID_CTA_RECORD, "daytype": "X"}
        with pytest.raises(SchemaValidationError):
            CTARidershipValidator.validate_one(record)

    def test_rides_with_letters_raises(self):
        """Non-numeric rides string raises SchemaValidationError."""
        record = {**VALID_CTA_RECORD, "rides": "abc"}
        with pytest.raises(SchemaValidationError):
            CTARidershipValidator.validate_one(record)

    def test_rides_with_decimal_raises(self):
        """Decimal rides string (e.g. '123.5') raises SchemaValidationError."""
        record = {**VALID_CTA_RECORD, "rides": "123.5"}
        with pytest.raises(SchemaValidationError):
            CTARidershipValidator.validate_one(record)

    def test_empty_route_raises(self):
        """Empty string route raises SchemaValidationError."""
        record = {**VALID_CTA_RECORD, "route": ""}
        with pytest.raises(SchemaValidationError):
            CTARidershipValidator.validate_one(record)

    def test_additional_properties_allowed(self):
        """Extra fields from the Socrata API do not cause validation failure."""
        record = {**VALID_CTA_RECORD, "extra_field": "extra_value"}
        CTARidershipValidator.validate_one(record)  # should not raise

    def test_validate_batch_splits_valid_invalid(self):
        """validate_batch correctly separates valid and invalid records."""
        invalid_record = {**VALID_CTA_RECORD, "daytype": "Z"}
        records = [VALID_CTA_RECORD, invalid_record]

        valid, invalid = CTARidershipValidator.validate_batch(records)

        assert len(valid) == 1
        assert len(invalid) == 1
        assert valid[0]["route"] == "22"

    def test_validate_batch_invalid_gets_error_key(self):
        """Invalid records in batch get '_validation_error' key appended."""
        invalid_record = {**VALID_CTA_RECORD, "daytype": "BAD"}
        _, invalid = CTARidershipValidator.validate_batch([invalid_record])

        assert len(invalid) == 1
        assert "_validation_error" in invalid[0]
        assert isinstance(invalid[0]["_validation_error"], str)

    def test_validate_batch_all_valid(self):
        """All valid records → empty invalid list."""
        records = [VALID_CTA_RECORD, {**VALID_CTA_RECORD, "route": "77"}]
        valid, invalid = CTARidershipValidator.validate_batch(records)

        assert len(valid) == 2
        assert len(invalid) == 0

    def test_validate_batch_empty_input(self):
        """Empty input returns ([], [])."""
        valid, invalid = CTARidershipValidator.validate_batch([])
        assert valid == []
        assert invalid == []

    def test_validate_batch_all_invalid(self):
        """All invalid records → empty valid list."""
        records = [
            {**VALID_CTA_RECORD, "daytype": "X"},
            {**VALID_CTA_RECORD, "rides": "not_a_number"},
        ]
        valid, invalid = CTARidershipValidator.validate_batch(records)
        assert len(valid) == 0
        assert len(invalid) == 2


# ─── WeatherRecordValidator tests ────────────────────────────────────────────

class TestWeatherRecordValidator:

    def test_valid_record_passes(self):
        """A correctly formed weather record passes without raising."""
        WeatherRecordValidator.validate_one(VALID_WEATHER_RECORD)

    def test_null_temperature_allowed(self):
        """temperature_2m can be null (nullable in schema)."""
        record = {**VALID_WEATHER_RECORD, "temperature_2m": None}
        WeatherRecordValidator.validate_one(record)

    def test_null_precipitation_allowed(self):
        """precipitation can be null."""
        record = {**VALID_WEATHER_RECORD, "precipitation": None}
        WeatherRecordValidator.validate_one(record)

    def test_missing_date_raises(self):
        """Missing 'date' raises SchemaValidationError."""
        record = {k: v for k, v in VALID_WEATHER_RECORD.items() if k != "date"}
        with pytest.raises(SchemaValidationError):
            WeatherRecordValidator.validate_one(record)

    def test_invalid_date_format_raises(self):
        """Date not matching YYYY-MM-DD raises SchemaValidationError."""
        record = {**VALID_WEATHER_RECORD, "date": "01/15/2024"}
        with pytest.raises(SchemaValidationError):
            WeatherRecordValidator.validate_one(record)

    def test_hour_out_of_range_raises(self):
        """Hour > 23 raises SchemaValidationError."""
        record = {**VALID_WEATHER_RECORD, "hour": 25}
        with pytest.raises(SchemaValidationError):
            WeatherRecordValidator.validate_one(record)

    def test_hour_negative_raises(self):
        """Negative hour raises SchemaValidationError."""
        record = {**VALID_WEATHER_RECORD, "hour": -1}
        with pytest.raises(SchemaValidationError):
            WeatherRecordValidator.validate_one(record)

    def test_negative_precipitation_raises(self):
        """Negative precipitation raises SchemaValidationError."""
        record = {**VALID_WEATHER_RECORD, "precipitation": -0.5}
        with pytest.raises(SchemaValidationError):
            WeatherRecordValidator.validate_one(record)

    def test_additional_properties_not_allowed(self):
        """Extra fields raise SchemaValidationError (additionalProperties=False)."""
        record = {**VALID_WEATHER_RECORD, "surprise_field": "oops"}
        with pytest.raises(SchemaValidationError):
            WeatherRecordValidator.validate_one(record)

    def test_validate_batch_works(self):
        """Batch validation splits correctly."""
        bad_record = {**VALID_WEATHER_RECORD, "hour": 99}
        valid, invalid = WeatherRecordValidator.validate_batch(
            [VALID_WEATHER_RECORD, bad_record]
        )
        assert len(valid) == 1
        assert len(invalid) == 1


# ─── TransitEventValidator tests ─────────────────────────────────────────────

class TestTransitEventValidator:

    def test_valid_event_passes(self):
        """A correctly formed transit event passes without raising."""
        TransitEventValidator.validate_one(VALID_TRANSIT_EVENT)

    def test_nullable_weather_fields_allowed(self):
        """Weather fields can be null (enrichment not always available)."""
        event = {
            **VALID_TRANSIT_EVENT,
            "temperature_2m": None,
            "precipitation": None,
            "windspeed_10m": None,
            "weathercode": None,
        }
        TransitEventValidator.validate_one(event)

    def test_missing_event_id_raises(self):
        """Missing event_id raises SchemaValidationError."""
        event = {k: v for k, v in VALID_TRANSIT_EVENT.items() if k != "event_id"}
        with pytest.raises(SchemaValidationError):
            TransitEventValidator.validate_one(event)

    def test_missing_service_date_raises(self):
        """Missing service_date raises SchemaValidationError."""
        event = {k: v for k, v in VALID_TRANSIT_EVENT.items() if k != "service_date"}
        with pytest.raises(SchemaValidationError):
            TransitEventValidator.validate_one(event)

    def test_invalid_service_date_format_raises(self):
        """service_date not matching YYYY-MM-DD raises SchemaValidationError."""
        event = {**VALID_TRANSIT_EVENT, "service_date": "Jan 15 2024"}
        with pytest.raises(SchemaValidationError):
            TransitEventValidator.validate_one(event)

    def test_invalid_day_type_raises(self):
        """day_type not in [W, A, U] raises SchemaValidationError."""
        event = {**VALID_TRANSIT_EVENT, "day_type": "X"}
        with pytest.raises(SchemaValidationError):
            TransitEventValidator.validate_one(event)

    def test_invalid_source_raises(self):
        """source not in allowed enum raises SchemaValidationError."""
        event = {**VALID_TRANSIT_EVENT, "source": "unknown_source"}
        with pytest.raises(SchemaValidationError):
            TransitEventValidator.validate_one(event)

    def test_negative_rides_raises(self):
        """rides < 0 raises SchemaValidationError."""
        event = {**VALID_TRANSIT_EVENT, "rides": -100}
        with pytest.raises(SchemaValidationError):
            TransitEventValidator.validate_one(event)

    def test_rides_as_string_raises(self):
        """rides must be integer, not string."""
        event = {**VALID_TRANSIT_EVENT, "rides": "12345"}
        with pytest.raises(SchemaValidationError):
            TransitEventValidator.validate_one(event)

    def test_zero_rides_allowed(self):
        """rides=0 is valid (route ran but no riders)."""
        event = {**VALID_TRANSIT_EVENT, "rides": 0}
        TransitEventValidator.validate_one(event)  # should not raise

    def test_validate_batch_returns_correct_counts(self):
        """Batch validation with mixed records returns correct split."""
        bad_event = {**VALID_TRANSIT_EVENT, "source": "bad_source"}
        records = [VALID_TRANSIT_EVENT, bad_event, VALID_TRANSIT_EVENT]

        valid, invalid = TransitEventValidator.validate_batch(records)

        assert len(valid) == 2
        assert len(invalid) == 1
        assert "_validation_error" in invalid[0]

    def test_validate_batch_empty_input(self):
        """Empty input returns ([], [])."""
        valid, invalid = TransitEventValidator.validate_batch([])
        assert valid == []
        assert invalid == []
