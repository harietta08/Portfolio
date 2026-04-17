# =============================================================================
# gold_to_bigquery.py
# Purpose: Export Databricks Gold Delta tables to BigQuery.
#          This is the Databricks -> warehouse handoff.
#          dbt then runs on BigQuery, not on Databricks.
#
# Two export modes supported:
#   1. Via pandas (local/CE): read Gold CSV exports, upload via BQ Python client
#   2. Via JDBC (production): Databricks writes directly to BQ via connector
#
# For CE: Gold tables exported as CSV from Databricks, then this script
#         uploads them to BigQuery. Documented in README architecture diagram.
# =============================================================================

import os
import pandas as pd
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

GCP_PROJECT_ID   = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "ecommerce_analytics")
BIGQUERY_LOCATION = os.getenv("BIGQUERY_LOCATION", "US")
EXPORT_DIR       = Path(__file__).parent.parent / "data" / "processed"
INGESTION_DATE   = date.today().isoformat()

# BigQuery table schema map: local file name -> BQ table name
TABLE_MAP = {
    "gold_customer_kpis":    "gold_customer_kpis",
    "gold_product_metrics":  "gold_product_metrics",
}


def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=GCP_PROJECT_ID)


def ensure_dataset_exists(client: bigquery.Client):
    dataset_ref = bigquery.Dataset(f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}")
    dataset_ref.location = BIGQUERY_LOCATION
    try:
        client.get_dataset(dataset_ref)
        print(f"  Dataset exists: {BIGQUERY_DATASET}")
    except Exception:
        client.create_dataset(dataset_ref, exists_ok=True)
        print(f"  Created dataset: {BIGQUERY_DATASET}")


def upload_to_bigquery(df: pd.DataFrame, table_name: str,
                       client: bigquery.Client) -> int:
    full_table = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        # WRITE_TRUNCATE = idempotent: re-running replaces existing data
        # In production with incremental loads, use WRITE_APPEND with dedup
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, full_table, job_config=job_config)
    job.result()  # wait for completion

    table = client.get_table(full_table)
    print(f"  Loaded {table.num_rows} rows -> {full_table}")
    return table.num_rows


def load_gold_csv(name: str) -> pd.DataFrame:
    """
    Load a Gold export CSV. In the CE workflow:
    1. Run 03_gold_aggregation.py in Databricks
    2. Export the Delta table as CSV from Databricks UI or dbutils
    3. Save to data/processed/gold_{name}.csv
    4. This script picks it up and loads to BigQuery
    """
    path = EXPORT_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Gold export not found: {path}\n"
            f"Export the Delta table from Databricks first:\n"
            f"  spark.table('ecommerce_gold.{name.replace('gold_', '')}') \\\n"
            f"    .toPandas().to_csv('/dbfs/FileStore/exports/{name}.csv', index=False)"
        )
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {path}")
    return df


def run_export() -> dict:
    print(f"\nBigQuery export started: {INGESTION_DATE}")
    print(f"Project: {GCP_PROJECT_ID} | Dataset: {BIGQUERY_DATASET}\n")

    client = get_bq_client()
    ensure_dataset_exists(client)

    results = {}
    for file_name, table_name in TABLE_MAP.items():
        print(f"Exporting {file_name} -> {BIGQUERY_DATASET}.{table_name}")
        df   = load_gold_csv(file_name)
        rows = upload_to_bigquery(df, table_name, client)
        results[table_name] = rows

    print(f"\nExport complete. Tables loaded to BigQuery:")
    for table, rows in results.items():
        print(f"  {BIGQUERY_DATASET}.{table:35s} {rows:>6} rows")

    return results


if __name__ == "__main__":
    run_export()
