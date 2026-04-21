"""
gcp/pubsub_consumer.py

GCP Pub/Sub → BigQuery streaming insert consumer.

This is the real-time streaming leg of the pipeline.

Flow:
  1. Subscribe to cta-transit-events Pub/Sub subscription
  2. For each message: deserialize JSON → enrich with weather → validate
  3. Batch messages and stream-insert to BigQuery (max latency ~5s per batch)
  4. Ack messages on successful BigQuery insert
  5. Nack on any unrecoverable error so Pub/Sub redelivers

BigQuery streaming insert notes:
  - Uses the legacy streaming API (insert_rows_json) — available on free tier
  - Late-arriving data: BigQuery accepts out-of-order rows. The streaming
    buffer is immediately queryable. Data moves to standard storage within
    ~90 minutes.
  - Deduplication: Pub/Sub delivers at-least-once. We pass event_id as the
    BigQuery row_id for deduplication within the 1-minute dedup window.

Run locally:
    python gcp/pubsub_consumer.py
    or: make pubsub-local

Production deployment:
    Deploy as a Cloud Run service (always-on, scales to zero when idle).
    The process runs continuously, streaming events in real time.
"""

import json
import logging
import os
import signal
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery, pubsub_v1

from ingestion.validate_schema import TransitEventValidator, SchemaValidationError
from ingestion.weather_api_client import OpenMeteoClient, WeatherAPIError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
SUBSCRIPTION_ID = os.environ["PUBSUB_SUBSCRIPTION_ID"]
BQ_DATASET = os.environ.get("BIGQUERY_DATASET", "chicago_transit")
BQ_TABLE = os.environ.get("BIGQUERY_TABLE", "transit_events")
FULL_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

# Batch tuning — flush to BigQuery every N messages OR every N seconds
MAX_MESSAGES_PER_BATCH = 50
MAX_BATCH_LATENCY_S = 5


# ─── Weather cache ────────────────────────────────────────────────────────────

class WeatherCache:
    """
    In-memory cache for Open-Meteo weather records keyed by (date, hour).

    Why cache:
      The consumer receives one message per route per day. Without caching,
      every message triggers an Open-Meteo API call for the same date.
      The cache loads each date once and serves all routes from memory.
    """

    def __init__(self):
        self._cache: dict[tuple[str, int], dict] = {}
        self._loaded_dates: set[str] = set()
        self.client = OpenMeteoClient()

    def get(self, date_str: str, hour: int) -> Optional[dict]:
        """Return weather for a specific date and hour, loading if needed."""
        if date_str not in self._loaded_dates:
            self._load_date(date_str)
        return self._cache.get((date_str, hour))

    def _load_date(self, date_str: str) -> None:
        """Fetch and cache all hourly weather records for a single date."""
        target = date.fromisoformat(date_str)
        try:
            records = self.client.fetch_historical_weather(
                start_date=target,
                end_date=target,
            )
            for r in records:
                self._cache[(r["date"], r["hour"])] = r
            self._loaded_dates.add(date_str)
            logger.info(
                "Weather cache loaded for %s (%d records).", date_str, len(records)
            )
        except WeatherAPIError as exc:
            # Mark as attempted to avoid hammering the API on repeated failures
            logger.warning("Could not load weather for %s: %s", date_str, exc)
            self._loaded_dates.add(date_str)


weather_cache = WeatherCache()
bq_client = bigquery.Client(project=PROJECT_ID)


# ─── BigQuery streaming insert ────────────────────────────────────────────────

def stream_rows_to_bigquery(rows: list[dict]) -> int:
    """
    Stream-insert a batch of rows to BigQuery.

    Parameters
    ----------
    rows : list[dict]
        Transit event records. Must match bigquery_schema.json.

    Returns
    -------
    int
        Number of successfully inserted rows.

    Notes
    -----
    - row_ids set to event_id for deduplication within BQ's 1-minute window
    - Rows with insert errors are logged but do not raise — the consumer
      acks the Pub/Sub message to avoid infinite redelivery loops.
      Monitor BQ insert errors in Cloud Logging.
    """
    if not rows:
        return 0

    row_ids = [r["event_id"] for r in rows]

    errors = bq_client.insert_rows_json(
        table=FULL_TABLE,
        json_rows=rows,
        row_ids=row_ids,
    )

    if errors:
        for error in errors:
            logger.error("BigQuery insert error: %s", error)
        return len(rows) - len(errors)

    logger.info("Streamed %d rows to %s.", len(rows), FULL_TABLE)
    return len(rows)


# ─── Message processing ───────────────────────────────────────────────────────

def enrich_with_weather(event: dict) -> dict:
    """
    Join weather data to a transit event based on service_date.

    CTA ridership is daily so we use 8 AM as the representative hour —
    this captures morning commute conditions that most correlate with
    daily ridership decisions.
    """
    service_date = event.get("service_date", "")
    if not service_date:
        return event

    weather = weather_cache.get(service_date, hour=8)
    if weather:
        event["temperature_2m"] = weather.get("temperature_2m")
        event["apparent_temperature"] = weather.get("apparent_temperature")
        event["precipitation"] = weather.get("precipitation")
        event["windspeed_10m"] = weather.get("windspeed_10m")
        event["weathercode"] = weather.get("weathercode")
        event["is_precipitation"] = weather.get("is_precipitation", False)

    return event


def process_message(
    message: pubsub_v1.subscriber.message.Message,
) -> Optional[dict]:
    """
    Deserialize, enrich, and validate a single Pub/Sub message.

    Returns
    -------
    dict
        Processed event ready for BigQuery insert.
    None
        If the message is unprocessable (malformed JSON, etc.).
        Caller should nack.
    """
    try:
        event = json.loads(message.data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "Failed to deserialize message %s: %s", message.message_id, exc
        )
        return None

    event = enrich_with_weather(event)

    try:
        TransitEventValidator.validate_one(event)
    except SchemaValidationError as exc:
        # Log but do not drop — schema issues are logged for monitoring.
        # BigQuery schema is permissive (NULLABLE fields) so the row still lands.
        logger.warning(
            "Schema validation warning for message %s: %s",
            message.message_id, exc,
        )

    return event


# ─── Consumer ────────────────────────────────────────────────────────────────

class PubSubConsumer:
    """
    Streaming consumer: GCP Pub/Sub → BigQuery.

    Accumulates messages into batches for efficient BigQuery streaming inserts.
    Flushes when batch reaches MAX_MESSAGES_PER_BATCH or MAX_BATCH_LATENCY_S.
    Handles graceful shutdown on SIGINT / SIGTERM.
    """

    def __init__(self):
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(
            PROJECT_ID, SUBSCRIPTION_ID
        )
        self._pending: list[
            tuple[dict, pubsub_v1.subscriber.message.Message]
        ] = []
        self._last_flush = time.time()
        self._running = True
        self._total_processed = 0
        self._total_inserted = 0

    def callback(
        self, message: pubsub_v1.subscriber.message.Message
    ) -> None:
        """
        Called by the Pub/Sub subscriber thread for each incoming message.
        Accumulates into batch and flushes when ready.
        """
        event = process_message(message)
        if event is None:
            message.nack()
            return

        self._pending.append((event, message))

        should_flush = (
            len(self._pending) >= MAX_MESSAGES_PER_BATCH
            or time.time() - self._last_flush >= MAX_BATCH_LATENCY_S
        )
        if should_flush:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """
        Stream-insert the pending batch to BigQuery, then ack all messages.
        On BigQuery API failure, nack all messages for redelivery.
        """
        if not self._pending:
            return

        batch = self._pending.copy()
        self._pending.clear()
        self._last_flush = time.time()

        rows = [event for event, _ in batch]
        messages = [msg for _, msg in batch]

        try:
            inserted = stream_rows_to_bigquery(rows)
            self._total_inserted += inserted
        except GoogleAPICallError as exc:
            logger.error("BigQuery API error during batch flush: %s", exc)
            for msg in messages:
                msg.nack()
            return

        for msg in messages:
            msg.ack()

        self._total_processed += len(batch)
        logger.info(
            "Batch flushed: %d messages | %d inserted | "
            "totals: processed=%d inserted=%d",
            len(batch), inserted,
            self._total_processed, self._total_inserted,
        )

    def run(self) -> None:
        """Start the streaming consumer. Runs until SIGINT or SIGTERM."""
        logger.info(
            "Starting Pub/Sub consumer.\n  Subscription : %s\n  BigQuery     : %s",
            self.subscription_path,
            FULL_TABLE,
        )

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=self.callback,
        )

        logger.info(
            "Consumer running. Batch size=%d, max latency=%ds. "
            "Press Ctrl+C to stop.",
            MAX_MESSAGES_PER_BATCH,
            MAX_BATCH_LATENCY_S,
        )

        try:
            while self._running:
                # Periodic flush for low-volume periods where batch never fills
                time.sleep(MAX_BATCH_LATENCY_S)
                self._flush_batch()
        except Exception as exc:
            logger.error("Consumer error: %s", exc)
            streaming_pull_future.cancel()
            raise
        finally:
            streaming_pull_future.cancel()
            try:
                streaming_pull_future.result(timeout=5)
            except (FuturesTimeoutError, Exception):
                pass
            self._flush_batch()  # drain remaining messages on shutdown
            logger.info(
                "Consumer stopped. Final totals: processed=%d inserted=%d",
                self._total_processed,
                self._total_inserted,
            )
            self.subscriber.close()

    def _shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received (%s). Draining batch...", signum)
        self._running = False


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    consumer = PubSubConsumer()
    consumer.run()
