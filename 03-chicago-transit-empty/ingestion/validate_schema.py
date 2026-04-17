"""
ingestion/validate_schema.py

Schema contracts at the API ingestion boundary.

Every record from the CTA and weather APIs is validated here before it
travels downstream to Pub/Sub, S3, or BigQuery.

Why validate at the boundary:
  A broken schema caught at ingestion costs nothing to fix.
  The same break caught after it has propagated through Pub/Sub →
  BigQuery → dbt → Tableau costs hours of debugging and corrupts
  ML feature pipelines silently.

Design:
  - jsonschema for declarative, readable contract definitions
  - SchemaValidationError never swallowed — callers decide what to do
  - validate_batch returns (valid, invalid) so pipelines can route
    invalid records to a dead-letter path without stopping the flow
"""

import logging
from typing import Any

import jsonschema
from jsonschema import ValidationError

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when a single record fails schema validation."""


# ─── Schema definitions ───────────────────────────────────────────────────────

CTA_RIDERSHIP_SCHEMA = {
    "type": "object",
    "required": ["route", "date", "daytype", "rides"],
    "properties": {
        "route": {
            "type": "string",
            "minLength": 1,
            "maxLength": 10,
            "description": "CTA route number e.g. '22'",
        },
        "date": {
            "type": "string",
            "description": "ISO-8601 datetime string from Socrata e.g. '2024-01-15T00:00:00.000'",
        },
        "daytype": {
            "type": "string",
            "enum": ["W", "A", "U"],
            "description": "W=Weekday, A=Saturday, U=Sunday/Holiday",
        },
        "rides": {
            "type": "string",
            "pattern": "^[0-9]+$",
            "description": "Ridership count as string — Socrata returns numeric fields as strings",
        },
    },
    "additionalProperties": True,
}

WEATHER_RECORD_SCHEMA = {
    "type": "object",
    "required": ["date", "hour", "temperature_2m", "precipitation", "windspeed_10m"],
    "properties": {
        "date": {
            "type": "string",
            "pattern": r"^\d{4}-\d{2}-\d{2}$",
        },
        "hour": {
            "type": "integer",
            "minimum": 0,
            "maximum": 23,
        },
        "temperature_2m": {"type": ["number", "null"]},
        "apparent_temperature": {"type": ["number", "null"]},
        "precipitation": {
            "type": ["number", "null"],
            "minimum": 0,
        },
        "windspeed_10m": {"type": ["number", "null"]},
        "weathercode": {"type": ["integer", "null"]},
        "is_precipitation": {"type": "boolean"},
        "is_heavy_precipitation": {"type": "boolean"},
    },
    "additionalProperties": False,
}

TRANSIT_EVENT_SCHEMA = {
    "type": "object",
    "required": [
        "event_id",
        "route",
        "service_date",
        "day_type",
        "rides",
        "ingested_at",
        "source",
    ],
    "properties": {
        "event_id": {
            "type": "string",
            "description": "Deterministic UUID5 based on route + service_date",
        },
        "route": {"type": "string", "minLength": 1},
        "service_date": {
            "type": "string",
            "pattern": r"^\d{4}-\d{2}-\d{2}$",
        },
        "day_type": {
            "type": "string",
            "enum": ["W", "A", "U"],
        },
        "rides": {"type": "integer", "minimum": 0},
        "ingested_at": {
            "type": "string",
            "description": "UTC ISO-8601 timestamp of ingestion",
        },
        "source": {
            "type": "string",
            "enum": ["chicago_data_portal"],
        },
        "temperature_2m": {"type": ["number", "null"]},
        "precipitation": {"type": ["number", "null"]},
        "windspeed_10m": {"type": ["number", "null"]},
        "weathercode": {"type": ["integer", "null"]},
    },
    "additionalProperties": True,
}


# ─── Validators ───────────────────────────────────────────────────────────────

class CTARidershipValidator:
    """Validates raw CTA ridership records from the Chicago Data Portal."""

    @staticmethod
    def validate_one(record: dict[str, Any]) -> None:
        """
        Validate a single record against the CTA ridership schema.

        Raises
        ------
        SchemaValidationError
        """
        try:
            jsonschema.validate(instance=record, schema=CTA_RIDERSHIP_SCHEMA)
        except ValidationError as exc:
            raise SchemaValidationError(
                f"CTA record failed validation: {exc.message} | record={record}"
            ) from exc

    @classmethod
    def validate_batch(
        cls, records: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Validate a batch of records.

        Returns
        -------
        (valid_records, invalid_records)
            Invalid records have an added '_validation_error' key.
        """
        valid, invalid = [], []
        for record in records:
            try:
                cls.validate_one(record)
                valid.append(record)
            except SchemaValidationError as exc:
                logger.warning("Invalid CTA record: %s", exc)
                invalid.append({**record, "_validation_error": str(exc)})

        logger.info(
            "CTA validation: %d valid, %d invalid out of %d total.",
            len(valid), len(invalid), len(records),
        )
        return valid, invalid


class WeatherRecordValidator:
    """Validates weather records from Open-Meteo."""

    @staticmethod
    def validate_one(record: dict[str, Any]) -> None:
        try:
            jsonschema.validate(instance=record, schema=WEATHER_RECORD_SCHEMA)
        except ValidationError as exc:
            raise SchemaValidationError(
                f"Weather record failed validation: {exc.message} | record={record}"
            ) from exc

    @classmethod
    def validate_batch(
        cls, records: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        valid, invalid = [], []
        for record in records:
            try:
                cls.validate_one(record)
                valid.append(record)
            except SchemaValidationError as exc:
                logger.warning("Invalid weather record: %s", exc)
                invalid.append({**record, "_validation_error": str(exc)})

        logger.info(
            "Weather validation: %d valid, %d invalid out of %d total.",
            len(valid), len(invalid), len(records),
        )
        return valid, invalid


class TransitEventValidator:
    """Validates the canonical transit event record before BigQuery insert."""

    @staticmethod
    def validate_one(record: dict[str, Any]) -> None:
        try:
            jsonschema.validate(instance=record, schema=TRANSIT_EVENT_SCHEMA)
        except ValidationError as exc:
            raise SchemaValidationError(
                f"Transit event failed validation: {exc.message} | record={record}"
            ) from exc

    @classmethod
    def validate_batch(
        cls, records: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        valid, invalid = [], []
        for record in records:
            try:
                cls.validate_one(record)
                valid.append(record)
            except SchemaValidationError as exc:
                logger.warning("Invalid transit event: %s", exc)
                invalid.append({**record, "_validation_error": str(exc)})

        logger.info(
            "Transit event validation: %d valid, %d invalid out of %d total.",
            len(valid), len(invalid), len(records),
        )
        return valid, invalid
