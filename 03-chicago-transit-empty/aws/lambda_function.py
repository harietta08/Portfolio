"""
aws/lambda_function.py

AWS Lambda — Scheduled CTA API ingestion function.

Trigger: EventBridge cron rule (every 6 hours via EventBridge Scheduler)
Flow:
  1. Pull yesterday's CTA ridership data from Chicago Data Portal API
  2. Validate every record against schema contract
  3. Publish valid records to GCP Pub/Sub as JSON messages
  4. Archive raw payload to S3 partitioned by date/route
  5. Route invalid records to dead-letter S3 prefix

Why Lambda here:
  Lambda + EventBridge is the standard serverless scheduled ingestion
  pattern on AWS. No always-on compute needed. The function runs for
  ~30 seconds and exits. Cost is effectively zero on the free tier
  (1M requests/month free).

Retry policy:
  EventBridge retries failed Lambda invocations up to 2 times with
  exponential back-off. Lambda itself retries HTTP errors via
  CTABusRidershipClient internal retry logic.
  Configure a DLQ on the Lambda function for failed invocations.

Environment variables (set in Lambda console or Terraform):
  CHICAGO_DATA_PORTAL_APP_TOKEN
  S3_BUCKET_NAME
  GCP_PROJECT_ID
  PUBSUB_TOPIC_ID
  GOOGLE_APPLICATION_CREDENTIALS_JSON  ← full JSON string of service account key
"""

import json
import logging
import os
import uuid
from datetime import date, datetime, timedelta, timezone

import boto3
from google.cloud import pubsub_v1
from google.oauth2 import service_account

from ingestion.cta_api_client import CTABusRidershipClient, CTAAPIError
from ingestion.validate_schema import CTARidershipValidator
from aws.s3_archive import S3Archiver

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
GCP_PROJECT = os.environ["GCP_PROJECT_ID"]
PUBSUB_TOPIC = os.environ["PUBSUB_TOPIC_ID"]

# Routes ingested on every Lambda run
TARGET_ROUTES = [
    "22",   # Clark — high ridership, North Side
    "77",   # Belmont — cross-city connector
    "66",   # Chicago Avenue — West Side / downtown
    "151",  # Sheridan — lakefront express
    "6",    # Jackson Park Express — South Side
    "36",   # Broadway — DiD treatment route candidate
    "49",   # Western — DiD control route candidate
    "82",   # Kimball-Homan — DiD control route candidate
]


# ─── GCP Pub/Sub client (initialized once per cold start) ─────────────────────
def _build_pubsub_publisher() -> pubsub_v1.PublisherClient:
    """
    Build GCP Pub/Sub publisher from service account JSON stored as env var.
    Storing as a string (not a file path) is the correct pattern for Lambda —
    Lambda functions cannot rely on a mounted filesystem for credentials.
    """
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_info = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/pubsub"],
        )
        return pubsub_v1.PublisherClient(credentials=credentials)
    # Fallback: use ADC — works in local dev with gcloud auth application-default login
    return pubsub_v1.PublisherClient()


publisher = _build_pubsub_publisher()
topic_path = publisher.topic_path(GCP_PROJECT, PUBSUB_TOPIC)
s3_archiver = S3Archiver(bucket=S3_BUCKET)
cta_client = CTABusRidershipClient()


# ─── Lambda handler ───────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    """
    Lambda entry point. Called by EventBridge on schedule.

    Parameters
    ----------
    event : dict
        EventBridge event payload. Pass {"target_date": "YYYY-MM-DD"}
        to override the default (yesterday) for backfills.
    context : LambdaContext
        Lambda runtime context (used for remaining time checks if needed).

    Returns
    -------
    dict
        Run summary: routes processed, records fetched/valid/published/rejected.
    """
    run_date = _resolve_target_date(event)
    logger.info("Lambda run started. Target date: %s", run_date)

    summary = {
        "run_date": run_date.isoformat(),
        "routes_processed": 0,
        "records_fetched": 0,
        "records_valid": 0,
        "records_published": 0,
        "records_archived": 0,
        "records_rejected": 0,
    }

    try:
        for route in TARGET_ROUTES:
            route_summary = _process_route(route, run_date)
            summary["routes_processed"] += 1
            summary["records_fetched"] += route_summary["fetched"]
            summary["records_valid"] += route_summary["valid"]
            summary["records_published"] += route_summary["published"]
            summary["records_archived"] += route_summary["archived"]
            summary["records_rejected"] += route_summary["rejected"]

    except CTAAPIError as exc:
        logger.error("CTA API failure: %s", exc)
        raise  # Let Lambda / EventBridge handle retry

    logger.info("Lambda run complete. Summary: %s", summary)
    return summary


# ─── Per-route processing ─────────────────────────────────────────────────────

def _process_route(route: str, target_date: date) -> dict:
    """
    Fetch, validate, publish to Pub/Sub, and archive to S3 for one route.
    Returns a per-route count summary.
    """
    try:
        raw_records = cta_client.fetch_ridership_by_date_range(
            start_date=target_date,
            end_date=target_date,
            route=route,
        )
    except CTAAPIError as exc:
        logger.error("Failed to fetch route %s: %s", route, exc)
        return {"fetched": 0, "valid": 0, "published": 0, "archived": 0, "rejected": 0}

    fetched = len(raw_records)
    if fetched == 0:
        logger.info("No records for route %s on %s — skipping.", route, target_date)
        return {"fetched": 0, "valid": 0, "published": 0, "archived": 0, "rejected": 0}

    # Validate schema at boundary
    valid_records, invalid_records = CTARidershipValidator.validate_batch(raw_records)

    # Archive ALL raw records to S3 (valid + invalid) — immutable record
    archive_key = S3Archiver.ridership_key(target_date.isoformat(), route)
    s3_archiver.write_json(key=archive_key, data=raw_records)

    # Route invalid records to dead-letter prefix
    if invalid_records:
        dlq_key = S3Archiver.dead_letter_key(target_date.isoformat(), route)
        s3_archiver.write_json(key=dlq_key, data=invalid_records)
        logger.warning(
            "Route %s: %d invalid records written to DLQ at %s",
            route, len(invalid_records), dlq_key,
        )

    # Publish valid records to Pub/Sub
    published = 0
    for record in valid_records:
        transit_event = _to_transit_event(record)
        try:
            _publish_event(transit_event)
            published += 1
        except Exception as exc:
            logger.error(
                "Pub/Sub publish failed for event_id=%s: %s",
                transit_event.get("event_id"), exc,
            )

    logger.info(
        "Route %s: fetched=%d valid=%d published=%d archived=%d rejected=%d",
        route, fetched, len(valid_records), published,
        len(raw_records), len(invalid_records),
    )

    return {
        "fetched": fetched,
        "valid": len(valid_records),
        "published": published,
        "archived": len(raw_records),
        "rejected": len(invalid_records),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_transit_event(record: dict) -> dict:
    """
    Transform a raw CTA API record into the canonical transit event schema.

    event_id is deterministic (uuid5 of route + service_date) so that
    re-running the Lambda on the same date produces the same IDs.
    This makes the entire pipeline idempotent — BigQuery deduplicates
    on insert_id, S3 overwrites the same key, Pub/Sub attributes carry
    the event_id for consumer-side dedup.
    """
    service_date = record["date"][:10]  # "2024-01-15T00:00:00.000" → "2024-01-15"
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
        # Weather fields populated downstream by pubsub_consumer.py
        "temperature_2m": None,
        "precipitation": None,
        "windspeed_10m": None,
        "weathercode": None,
        "is_precipitation": None,
    }


def _publish_event(event: dict) -> None:
    """
    Publish a single transit event to GCP Pub/Sub as UTF-8 JSON bytes.

    Pub/Sub message attributes carry route and event_id for:
      - Subscription-side filtering by route
      - Consumer-side deduplication
    """
    data = json.dumps(event).encode("utf-8")
    future = publisher.publish(
        topic_path,
        data=data,
        event_id=event["event_id"],
        route=event["route"],
        service_date=event["service_date"],
    )
    future.result(timeout=10)  # block until ack or raise


def _resolve_target_date(event: dict) -> date:
    """
    Determine target ingestion date.
    - If EventBridge passes 'target_date' key, use it (supports backfill).
    - Default: yesterday (CTA API typically lags ~1 day).
    """
    if "target_date" in event:
        return date.fromisoformat(event["target_date"])
    return date.today() - timedelta(days=1)
