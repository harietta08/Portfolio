# ── ingestion/gcs_upload.py ───────────────────────────────────────────────────
# Purpose: Upload validated data to GCS bucket
# Why GCS: 5GB always free, Databricks reads directly from gs:// paths
# Interview answer: "Raw data lives in GCS. Databricks reads it at training time.
#                   FastAPI never touches raw data — only the registered model."

import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "vc-intelligence-data")
GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
LOCAL_PROCESSED = Path("data/processed/startups_validated.csv")


def upload_to_gcs(
    local_path: Path,
    bucket_name: str,
    destination_blob: str,
) -> str:
    """
    Upload a local file to GCS.
    Returns the gs:// URI for use in Databricks.
    """
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)

    blob.upload_from_filename(str(local_path))
    uri = f"gs://{bucket_name}/{destination_blob}"
    logger.info(f"Uploaded: {local_path} → {uri}")
    return uri


def run_upload(local_path: Path = LOCAL_PROCESSED) -> str:
    """Upload validated startup data to GCS."""
    if not local_path.exists():
        raise FileNotFoundError(
            f"{local_path} not found. Run validate_schema.py first."
        )

    destination = f"raw/startups_validated.csv"
    uri = upload_to_gcs(local_path, GCS_BUCKET, destination)

    logger.info(f"GCS URI for Databricks: {uri}")
    logger.info("In Databricks: spark.read.csv('{uri}', header=True)")
    return uri


if __name__ == "__main__":
    uri = run_upload()
    print(f"Uploaded to: {uri}")
