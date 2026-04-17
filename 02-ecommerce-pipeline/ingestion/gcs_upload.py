# =============================================================================
# gcs_upload.py
# Purpose: Upload validated raw JSON files from local to GCS.
#          Partitioned by date: gs://bucket/ecommerce/raw/dt=YYYY-MM-DD/
#          Idempotent: re-uploading same date overwrites same path.
# =============================================================================

import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_RAW_PREFIX  = os.getenv("GCS_RAW_PREFIX", "ecommerce/raw")
RAW_DATA_DIR    = Path(__file__).parent.parent / "data" / "raw"
INGESTION_DATE  = date.today().isoformat()


def get_gcs_client() -> storage.Client:
    return storage.Client()


def upload_directory(local_dir: Path, gcs_prefix: str, bucket) -> list:
    uploaded = []
    for file_path in local_dir.rglob("*.json"):
        relative  = file_path.relative_to(RAW_DATA_DIR)
        blob_name = f"{gcs_prefix}/{relative}"
        blob      = bucket.blob(blob_name)
        blob.upload_from_filename(str(file_path))
        uploaded.append(f"gs://{GCS_BUCKET_NAME}/{blob_name}")
        print(f"  Uploaded: gs://{GCS_BUCKET_NAME}/{blob_name}")
    return uploaded


def run_upload() -> list:
    print(f"\nGCS upload started: {INGESTION_DATE}")
    print(f"Source: {RAW_DATA_DIR}")
    print(f"Target: gs://{GCS_BUCKET_NAME}/{GCS_RAW_PREFIX}/\n")

    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)

    date_dir = RAW_DATA_DIR / f"dt={INGESTION_DATE}"
    if not date_dir.exists():
        raise FileNotFoundError(
            f"No raw data found for {INGESTION_DATE}. "
            f"Run api_client.py first."
        )

    uploaded = upload_directory(
        date_dir,
        f"{GCS_RAW_PREFIX}/dt={INGESTION_DATE}",
        bucket
    )

    print(f"\nUploaded {len(uploaded)} files to GCS.")
    return uploaded


if __name__ == "__main__":
    run_upload()
