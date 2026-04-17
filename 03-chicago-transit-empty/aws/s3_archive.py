"""
aws/s3_archive.py

S3 archiving utility — raw event data to S3 partitioned by date.

Partition scheme (Hive-style, Athena-compatible):
  s3://<bucket>/raw/cta_ridership/date=YYYY-MM-DD/route=<route>/records.json.gz
  s3://<bucket>/raw/weather/date=YYYY-MM-DD/weather.json.gz
  s3://<bucket>/dead-letter/cta_ridership/date=YYYY-MM-DD/route=<route>/invalid.json.gz

Design decisions:
  - Writes are idempotent — same key = overwrite. Re-running Lambda on the
    same date produces the same S3 objects. This is the immutable archive
    pattern: never delete, let S3 lifecycle rules handle expiry.
  - Gzip compression by default — reduces storage cost ~70% for JSON.
  - JSONL option available for Athena / Glue Crawler compatibility.
  - key_exists() check allows callers to skip re-archiving if already done.
"""

import gzip
import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3ArchiveError(Exception):
    """Raised when an S3 operation fails."""


class S3Archiver:
    """
    Utility for writing and reading JSON data to/from S3.

    Parameters
    ----------
    bucket : str, optional
        S3 bucket name. Falls back to S3_BUCKET_NAME env var.
    region : str, optional
        AWS region. Falls back to AWS_DEFAULT_REGION env var.
    """

    def __init__(
        self,
        bucket: str = "",
        region: str = "",
    ):
        self.bucket = bucket or os.environ["S3_BUCKET_NAME"]
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.client = boto3.client("s3", region_name=self.region)

    # ─── Write operations ─────────────────────────────────────────────────────

    def write_json(
        self,
        key: str,
        data: Any,
        compress: bool = True,
        metadata: dict | None = None,
    ) -> str:
        """
        Write a JSON-serializable object to S3.

        Parameters
        ----------
        key : str
            S3 object key (path within the bucket).
            .gz suffix appended automatically if compress=True.
        data : Any
            JSON-serializable Python object (list, dict, etc.).
        compress : bool
            Gzip-compress the payload. Recommended — reduces size ~70%.
        metadata : dict, optional
            S3 object metadata key-value tags.

        Returns
        -------
        str
            Full S3 URI: s3://<bucket>/<key>
        """
        if compress and not key.endswith(".gz"):
            key = key + ".gz"

        body = json.dumps(data, indent=None, default=str).encode("utf-8")
        if compress:
            body = gzip.compress(body)

        extra_args: dict = {
            "ContentType": "application/json",
            "ContentEncoding": "gzip" if compress else "identity",
        }
        if metadata:
            extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                **extra_args,
            )
            uri = f"s3://{self.bucket}/{key}"
            logger.info("Archived to %s (%d bytes)", uri, len(body))
            return uri
        except ClientError as exc:
            raise S3ArchiveError(
                f"Failed to write s3://{self.bucket}/{key}: {exc}"
            ) from exc

    def write_jsonl(
        self,
        key: str,
        records: list[dict],
        compress: bool = True,
    ) -> str:
        """
        Write a list of dicts as newline-delimited JSON (JSON Lines).
        Preferred format for large datasets queried via Athena or Glue.

        Parameters
        ----------
        key : str
        records : list[dict]
        compress : bool

        Returns
        -------
        str  — S3 URI
        """
        if compress and not key.endswith(".gz"):
            key = key + ".gz"

        lines = "\n".join(json.dumps(r, default=str) for r in records)
        body = lines.encode("utf-8")
        if compress:
            body = gzip.compress(body)

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType="application/x-ndjson",
                ContentEncoding="gzip" if compress else "identity",
            )
            uri = f"s3://{self.bucket}/{key}"
            logger.info(
                "Archived %d records as JSONL to %s (%d bytes)",
                len(records), uri, len(body),
            )
            return uri
        except ClientError as exc:
            raise S3ArchiveError(
                f"Failed to write JSONL to s3://{self.bucket}/{key}: {exc}"
            ) from exc

    # ─── Read operations ──────────────────────────────────────────────────────

    def read_json(self, key: str) -> Any:
        """
        Read and parse a JSON (or gzipped JSON) object from S3.
        Used for backfill verification and local testing.
        """
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"].read()
            if key.endswith(".gz"):
                body = gzip.decompress(body)
            return json.loads(body.decode("utf-8"))
        except ClientError as exc:
            raise S3ArchiveError(
                f"Failed to read s3://{self.bucket}/{key}: {exc}"
            ) from exc

    def key_exists(self, key: str) -> bool:
        """
        Check whether an S3 object exists without downloading it.
        Use for idempotent writes — skip if already archived for the day.
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise S3ArchiveError(
                f"head_object failed for s3://{self.bucket}/{key}: {exc}"
            ) from exc

    def list_keys(self, prefix: str) -> list[str]:
        """
        List all S3 keys under a prefix. Handles pagination automatically.
        Useful for backfill inventory checks.
        """
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    # ─── Partition key helpers ────────────────────────────────────────────────

    @staticmethod
    def ridership_key(date_str: str, route: str, filename: str = "records.json") -> str:
        """S3 key for a ridership archive file."""
        return f"raw/cta_ridership/date={date_str}/route={route}/{filename}"

    @staticmethod
    def weather_key(date_str: str, filename: str = "weather.json") -> str:
        """S3 key for a weather archive file."""
        return f"raw/weather/date={date_str}/{filename}"

    @staticmethod
    def dead_letter_key(
        date_str: str, route: str, filename: str = "invalid.json"
    ) -> str:
        """S3 key for a dead-letter file."""
        return f"dead-letter/cta_ridership/date={date_str}/route={route}/{filename}"
