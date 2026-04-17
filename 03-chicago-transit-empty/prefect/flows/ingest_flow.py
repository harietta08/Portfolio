"""
prefect/flows/ingest_flow.py

Prefect batch ingestion flow — historical backfill and daily batch loads.

This flow is distinct from the Lambda streaming ingestion:
  Lambda   : real-time, event-driven, processes one day at a time on schedule
  This flow: batch, idempotent, used for historical backfill and full refresh

When to use this flow vs Lambda:
  - First-time setup: backfill 2 years of historical ridership for model training
  - Recovery: re-ingest a date range after a pipeline failure
  - Scheduled daily refresh: runs at 6 AM CT after CTA API reflects prior day

Idempotency:
  event_id is deterministic (uuid5 of route + date) so re-running this flow
  produces the same BigQuery rows and S3 objects. BigQuery deduplicates
  on insert_id within the 1-minute window. S3 overwrites the same key.
  Re-running is always safe.

Failure handling:
  - Each task retries up to 3 times with exponential back-off
  - Prefect Cloud sends failure alerts via email (configure in Prefect UI)
  - Failed flow runs appear in Prefect Cloud dashboard with full logs

Run locally:
    python -m prefect.flows.ingest_flow
    or: make ingest
"""

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional
import uuid

from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

load_dotenv()

from ingestion.cta_api_client import CTABusRidershipClient, CTAAPIError
from ingestion.weather_api_client import OpenMeteoClient, WeatherAPIError
from ingestion.validate_schema import CTARidershipValidator
from aws.s3_archive import S3Archiver
from gcp.pubsub_consumer import stream_rows_to_bigquery

logger = logging.getLogger(__name__)

# ─── Routes ───────────────────────────────────────────────────────────────────
TARGET_ROUTES = [
    "22",   # Clark
    "77",   # Belmont
    "66",   # Chicago Ave
    "151",  # Sheridan
    "6",    # Jackson Park Express
    "36",   # Broadway — DiD treatment candidate
    "49",   # Western — DiD control candidate
    "82",   # Kimball-Homan — DiD control candidate
    "8",    # Halsted
    "20",   # Madison
    "56",   # Milwaukee
    "60",   # Blue Island / 26th
    "63",   # 63rd
    "79",   # 79th
]


# ─── Helper: transform raw CTA record → transit event ────────────────────────

def _to_transit_event(record: dict) -> dict:
    """
    Transform a raw CTA API record into the canonical transit event schema.
    Deterministic event_id makes every run idempotent.
    """
    service_date = record["date"][:10]
    event_id = str(uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"{record['route']}-{service_date}",
    ))
    return {
        "event_id": event_id,
        "route": record["route"],
        "service_date": service_date,
        "day_type": record["daytype"],
        "rides": int(record["rides"]),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source": "chicago_data_portal",
        "temperature_2m": None,
        "apparent_temperature": None,
        "precipitation": None,
        "windspeed_10m": None,
        "weathercode": None,
        "is_precipitation": None,
    }


# ─── Tasks ────────────────────────────────────────────────────────────────────

@task(retries=3, retry_delay_seconds=60, name="fetch-cta-ridership")
def fetch_ridership(
    route: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Fetch CTA ridership for one route over a date range.

    Retries up to 3 times with 60s delay on API failures.
    Each retry is logged in Prefect Cloud with the full exception.
    """
    flow_logger = get_run_logger()
    flow_logger.info(
        "Fetching route=%s from %s to %s", route, start_date, end_date
    )
    client = CTABusRidershipClient()
    try:
        records = client.fetch_ridership_by_date_range(
            start_date, end_date, route=route
        )
        flow_logger.info(
            "Route %s: %d records fetched.", route, len(records)
        )
        return records
    except CTAAPIError as exc:
        flow_logger.error("CTA API error for route %s: %s", route, exc)
        raise


@task(retries=2, retry_delay_seconds=30, name="fetch-weather")
def fetch_weather(start_date: date, end_date: date) -> list[dict]:
    """
    Fetch historical weather for Chicago for the full date range.
    Called once per flow run and shared across all route tasks.
    """
    flow_logger = get_run_logger()
    flow_logger.info(
        "Fetching weather from %s to %s", start_date, end_date
    )
    client = OpenMeteoClient()
    try:
        records = client.fetch_historical_weather(start_date, end_date)
        flow_logger.info(
            "Weather: %d hourly records fetched.", len(records)
        )
        return records
    except WeatherAPIError as exc:
        flow_logger.error("Weather API error: %s", exc)
        raise


@task(name="validate-and-transform")
def validate_and_transform(
    raw_records: list[dict],
    weather_records: list[dict],
    route: str,
) -> tuple[list[dict], int]:
    """
    Validate CTA records, transform to transit event schema,
    and enrich with weather.

    Returns
    -------
    (transformed_events, rejected_count)
    """
    flow_logger = get_run_logger()

    # Build weather lookup: (date_str, hour) → weather dict
    weather_lookup: dict[tuple[str, int], dict] = {
        (r["date"], r["hour"]): r for r in weather_records
    }

    valid_raw, invalid_raw = CTARidershipValidator.validate_batch(raw_records)
    flow_logger.info(
        "Route %s: %d valid, %d invalid records.",
        route, len(valid_raw), len(invalid_raw),
    )

    events = []
    for record in valid_raw:
        event = _to_transit_event(record)

        # Enrich with weather using 8 AM as daily representative hour
        weather = weather_lookup.get((event["service_date"], 8))
        if weather:
            event["temperature_2m"] = weather.get("temperature_2m")
            event["apparent_temperature"] = weather.get("apparent_temperature")
            event["precipitation"] = weather.get("precipitation")
            event["windspeed_10m"] = weather.get("windspeed_10m")
            event["weathercode"] = weather.get("weathercode")
            event["is_precipitation"] = weather.get("is_precipitation", False)

        events.append(event)

    return events, len(invalid_raw)


@task(retries=2, retry_delay_seconds=30, name="archive-to-s3")
def archive_to_s3(
    records: list[dict],
    date_str: str,
    route: str,
) -> str:
    """
    Archive raw records to S3 partitioned by date and route.
    Idempotent — same key overwrites on rerun.
    """
    flow_logger = get_run_logger()
    archiver = S3Archiver()
    key = S3Archiver.ridership_key(date_str, route)
    uri = archiver.write_json(key=key, data=records)
    flow_logger.info(
        "Archived %d records to %s", len(records), uri
    )
    return uri


@task(retries=3, retry_delay_seconds=60, name="load-to-bigquery")
def load_to_bigquery(events: list[dict], route: str) -> int:
    """
    Stream-insert transformed and enriched events to BigQuery.
    Returns the number of rows successfully inserted.
    """
    flow_logger = get_run_logger()
    if not events:
        flow_logger.info(
            "No events to insert for route %s.", route
        )
        return 0
    inserted = stream_rows_to_bigquery(events)
    flow_logger.info(
        "Route %s: %d rows inserted to BigQuery.", route, inserted
    )
    return inserted


@task(name="archive-dead-letter")
def archive_dead_letter(
    invalid_records: list[dict],
    date_str: str,
    route: str,
) -> Optional[str]:
    """
    Write invalid records to the S3 dead-letter prefix.
    Returns the S3 URI or None if no invalid records.
    """
    flow_logger = get_run_logger()
    if not invalid_records:
        return None
    archiver = S3Archiver()
    key = S3Archiver.dead_letter_key(date_str, route)
    uri = archiver.write_json(key=key, data=invalid_records)
    flow_logger.warning(
        "Route %s: %d invalid records written to DLQ at %s",
        route, len(invalid_records), uri,
    )
    return uri


# ─── Flow ─────────────────────────────────────────────────────────────────────

@flow(
    name="cta-batch-ingest",
    description=(
        "Daily batch ingest of CTA ridership → validates → enriches with "
        "weather → archives to S3 → loads to BigQuery. Idempotent."
    ),
    task_runner=ConcurrentTaskRunner(),
    log_prints=True,
)
def ingest_flow(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    routes: Optional[list[str]] = None,
) -> dict:
    """
    Batch ingest flow.

    Parameters
    ----------
    start_date : date, optional
        Defaults to yesterday.
    end_date : date, optional
        Defaults to yesterday.
    routes : list[str], optional
        Defaults to TARGET_ROUTES. Pass a subset for targeted backfill.

    Returns
    -------
    dict
        Per-route summary: {route: {fetched, inserted, rejected}}.

    Examples
    --------
    # Run with defaults (yesterday)
    ingest_flow()

    # Backfill 90 days
    ingest_flow(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
    )

    # Single route debug run
    ingest_flow(routes=["22"])
    """
    flow_logger = get_run_logger()

    # Resolve defaults
    if start_date is None:
        start_date = date.today() - timedelta(days=1)
    if end_date is None:
        end_date = date.today() - timedelta(days=1)
    if routes is None:
        routes = TARGET_ROUTES

    flow_logger.info(
        "Ingest flow started: %s → %s | %d routes",
        start_date, end_date, len(routes),
    )

    # Fetch weather once — shared across all route iterations
    weather_records = fetch_weather(start_date, end_date)

    summary = {}
    total_inserted = 0

    for route in routes:
        flow_logger.info("── Processing route %s ──", route)

        raw_records = fetch_ridership(route, start_date, end_date)

        if not raw_records:
            flow_logger.info(
                "No records returned for route %s — skipping.", route
            )
            summary[route] = {"fetched": 0, "inserted": 0, "rejected": 0}
            continue

        events, rejected_count = validate_and_transform(
            raw_records, weather_records, route
        )

        # Archive raw records to S3 regardless of validity
        archive_to_s3(raw_records, start_date.isoformat(), route)

        # Archive invalid records to dead-letter
        invalid_records = raw_records[len(events):]
        if rejected_count > 0:
            archive_dead_letter(
                raw_records[-rejected_count:], start_date.isoformat(), route
            )

        # Load valid enriched events to BigQuery
        inserted = load_to_bigquery(events, route)
        total_inserted += inserted

        summary[route] = {
            "fetched": len(raw_records),
            "inserted": inserted,
            "rejected": rejected_count,
        }

    flow_logger.info(
        "Ingest flow complete. Total inserted: %d rows across %d routes.",
        total_inserted, len(routes),
    )
    return summary


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = ingest_flow()
    print("Flow result:", result)
